"""
LGRL PPO training on MiniGrid-DoorKey-5x5-v0.

Stage-based hierarchy with forward-only progression.
Supports both LLM (Qwen 2.5 7B) and rule-based oracle planners.

Subgoals follow the LGRL paper: search for, pickup, open, close, drop.
No "explore" or "go to".

Usage:
    python scripts/train_lgrl.py                       # LLM planner (default)
    python scripts/train_lgrl.py --planner rule_based  # oracle ablation
    python scripts/train_lgrl.py --resume
"""

import argparse
import csv
import json
import os
import sys
import time

import gymnasium as gym
import matplotlib
import minigrid  # noqa: F401
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch_ac
import torch_ac.algos.base
from utils.sequential_env import SequentialEnv
torch_ac.algos.base.ParallelEnv = SequentialEnv

from models.baseline_agent import Vocabulary
from models.lgrl_agent import LGRLAgent
from utils.env_parser import parse_env_description
from utils.llm_planner import LLMPlanner
from utils.rule_based_planner import RuleBasedPlanner
from utils.subgoal_tracker import SubgoalTracker
from utils.subgoal_logger import SubgoalLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_NAME            = "MiniGrid-DoorKey-5x5-v0"
NUM_ENVS            = 16
NUM_FRAMES_PER_PROC = 128
TOTAL_FRAMES        = 10_000_000

LR              = 1e-4
DISCOUNT        = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM   = 0.5
EPOCHS          = 4
BATCH_SIZE      = 256
RECURRENCE      = 4

R_MISSION       = 0.5
R_SUBGOAL       = 0.5
MISSION_TIME_COEF = 0.5          # coefficient in mission reward penalty
N_SUBGOALS      = RuleBasedPlanner.NUM_STAGES  # 5
MAX_ENV_STEPS   = 250            # T_max for mission reward scaling
MAX_SUBGOAL_STEPS = 250          # T_max for subgoal budget calculation

CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY = 10

LOG_DIR    = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR   = os.path.join(LOG_DIR, "plots")
PLOT_EVERY = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["llm", "rule_based"], default="llm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subgoal-log", action="store_true", default=False)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Per-env hierarchy state (stage-based, forward-only)
# ---------------------------------------------------------------------------

class HierarchyState:
    def __init__(self, num_envs: int, planner, envs: list):
        self.num_envs = num_envs
        self.planner = planner
        self.envs = envs
        self._init_lists()

    def _init_lists(self):
        self.active_subgoals: list[str] = [""] * self.num_envs
        self.stage_indices: list[int] = [0] * self.num_envs
        self.step_counters: list[int] = [0] * self.num_envs
        self.episode_steps: list[int] = [0] * self.num_envs
        self.trackers: list[SubgoalTracker] = [
            SubgoalTracker() for _ in range(self.num_envs)
        ]
        self.histories: list[list[dict]] = [[] for _ in range(self.num_envs)]

    def init_env_subgoal(self, env_idx: int, obs: dict):
        uw = self.envs[env_idx].unwrapped
        env_json = parse_env_description(obs["image"], uw.carrying)
        subgoal, new_stage = self.planner.get_subgoal(
            obs["mission"], env_json, obs.get("direction", 0), stage_index=0
        )
        self.active_subgoals[env_idx] = subgoal
        self.stage_indices[env_idx] = new_stage

    def reset_env(self, env_idx: int):
        self.active_subgoals[env_idx] = ""
        self.stage_indices[env_idx] = 0
        self.step_counters[env_idx] = 0
        self.episode_steps[env_idx] = 0
        self.trackers[env_idx].reset()
        self.histories[env_idx] = []

    def subgoal_budget(self, env_idx: int) -> float:
        i = min(self.stage_indices[env_idx] + 1, N_SUBGOALS)
        return (i / N_SUBGOALS) * MAX_SUBGOAL_STEPS

    def advance(self, env_idx: int, obs: dict):
        next_stage = self.stage_indices[env_idx] + 1
        if next_stage >= N_SUBGOALS:
            self.stage_indices[env_idx] = N_SUBGOALS
            self.active_subgoals[env_idx] = "search for the goal"
            self.step_counters[env_idx] = 0
            return

        uw = self.envs[env_idx].unwrapped
        env_json = parse_env_description(obs["image"], uw.carrying)
        subgoal, new_stage = self.planner.get_subgoal(
            obs["mission"], env_json, obs.get("direction", 0),
            stage_index=next_stage,
        )
        self.active_subgoals[env_idx] = subgoal
        self.stage_indices[env_idx] = new_stage
        self.step_counters[env_idx] = 0

# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def make_preprocess_obss(vocab, hierarchy_state, device=None):
    def preprocess_obss(obss, device=device):
        images = np.array([obs["image"] for obs in obss])
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images.permute(0, 3, 1, 2) / 255.0

        token_ids = []
        for i, obs in enumerate(obss):
            subgoal = (
                hierarchy_state.active_subgoals[i]
                if i < hierarchy_state.num_envs
                else "search for the goal"
            )
            if not subgoal:
                subgoal = "search for the key"
            combined = f"{obs['mission']} [SEP] {subgoal}"
            token_ids.append(vocab.tokenize(combined, max_len=32))

        texts = torch.tensor(token_ids, dtype=torch.long, device=device)
        return torch_ac.DictList({"image": images, "text": texts})

    return preprocess_obss

def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def make_reshape_reward(hierarchy_state, logger=None):
    def reshape_reward(obs, action, reward, done):
        env_idx = reshape_reward._current_env_idx
        reshape_reward._current_env_idx = (env_idx + 1) % hierarchy_state.num_envs

        total_reward = 0.0
        hierarchy_state.step_counters[env_idx] += 1
        hierarchy_state.episode_steps[env_idx] += 1
        mission = obs.get("mission", "")

        if done:
            success = reward > 0
            if success:
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / MAX_ENV_STEPS, 1.0)
                total_reward += R_MISSION * (1.0 - MISSION_TIME_COEF * ratio)
            if logger:
                logger.on_episode_end(
                    env_idx, mission, success,
                    hierarchy_state.episode_steps[env_idx],
                )
            hierarchy_state.reset_env(env_idx)
            return total_reward

        if hierarchy_state.stage_indices[env_idx] >= N_SUBGOALS:
            return total_reward

        if not hierarchy_state.active_subgoals[env_idx]:
            hierarchy_state.init_env_subgoal(env_idx, obs)
            if logger:
                uw = hierarchy_state.envs[env_idx].unwrapped
                env_json = parse_env_description(obs["image"], uw.carrying)
                logger.log(
                    env_idx, "init", mission=mission,
                    subgoal=hierarchy_state.active_subgoals[env_idx],
                    stage=hierarchy_state.stage_indices[env_idx],
                    budget=hierarchy_state.subgoal_budget(env_idx),
                    env_state=env_json,
                )

        uw = hierarchy_state.envs[env_idx].unwrapped
        subgoal = hierarchy_state.active_subgoals[env_idx]
        completed = hierarchy_state.trackers[env_idx].check_completion(
            subgoal, uw, action, obs_image=obs["image"],
        )

        t_used = hierarchy_state.step_counters[env_idx]
        t_budget = hierarchy_state.subgoal_budget(env_idx)
        timed_out = t_used > 2 * t_budget

        if completed:
            ratio = min(t_used / max(t_budget, 1), 2.0)
            r_i = max(R_SUBGOAL * (1.0 - 0.5 * ratio), 0.0)
            total_reward += r_i / N_SUBGOALS

            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Success", "steps": t_used,
                 "stage": hierarchy_state.stage_indices[env_idx]}
            )
            if logger:
                logger.log(
                    env_idx, "completed", mission=mission, subgoal=subgoal,
                    stage=hierarchy_state.stage_indices[env_idx],
                    steps_used=t_used, budget=t_budget,
                    reward=r_i / N_SUBGOALS,
                )
            hierarchy_state.advance(env_idx, obs)
            if logger:
                env_json = parse_env_description(
                    obs["image"], hierarchy_state.envs[env_idx].unwrapped.carrying,
                )
                logger.log(
                    env_idx, "new", mission=mission,
                    subgoal=hierarchy_state.active_subgoals[env_idx],
                    stage=hierarchy_state.stage_indices[env_idx],
                    budget=hierarchy_state.subgoal_budget(env_idx),
                    env_state=env_json,
                    raw_llm=getattr(hierarchy_state.planner, "last_raw_response", None),
                )

        elif timed_out:
            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Failed", "steps": t_used,
                 "stage": hierarchy_state.stage_indices[env_idx]}
            )
            if logger:
                logger.log(
                    env_idx, "timed_out", mission=mission, subgoal=subgoal,
                    stage=hierarchy_state.stage_indices[env_idx],
                    steps_used=t_used, budget=t_budget,
                )
            hierarchy_state.advance(env_idx, obs)
            if logger:
                env_json = parse_env_description(
                    obs["image"], hierarchy_state.envs[env_idx].unwrapped.carrying,
                )
                logger.log(
                    env_idx, "new", mission=mission,
                    subgoal=hierarchy_state.active_subgoals[env_idx],
                    stage=hierarchy_state.stage_indices[env_idx],
                    budget=hierarchy_state.subgoal_budget(env_idx),
                    env_state=env_json,
                    raw_llm=getattr(hierarchy_state.planner, "last_raw_response", None),
                )

        return total_reward

    reshape_reward._current_env_idx = 0
    return reshape_reward

# ---------------------------------------------------------------------------
# Plotting / checkpointing
# ---------------------------------------------------------------------------

def _smooth(values, window=20):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def save_plots(history, plot_dir, planner_tag):
    frames = history["frames"]
    if len(frames) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"LGRL PPO ({planner_tag}) -- Training Curves", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.3, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.5, label="Smoothed")
    ax.set(xlabel="Frames", ylabel="Average Return", title="Average Return per Episode")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(frames, history["avg_steps"], alpha=0.3, color="tab:orange", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_steps"]), color="tab:orange", linewidth=1.5, label="Smoothed")
    ax.set(xlabel="Frames", ylabel="Average Steps", title="Average Steps per Episode")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(frames, history["entropy"], color="tab:green", linewidth=1)
    ax.set(xlabel="Frames", ylabel="Entropy", title="Policy Entropy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(frames, _smooth(history["policy_loss"]), color="tab:red", linewidth=1.5, label="Policy Loss")
    ax.plot(frames, _smooth(history["value_loss"]), color="tab:purple", linewidth=1.5, label="Value Loss")
    ax.set(xlabel="Frames", ylabel="Loss", title="Policy & Value Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "lgrl_training_curves.png"), dpi=150)
    plt.close(fig)


def _save_checkpoint(path, model, algo, vocab, update, total_frames, planner_tag):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": algo.optimizer.state_dict(),
        "vocab": vocab.word2idx,
        "update": update, "total_frames": total_frames,
        "planner": planner_tag,
    }, path)


def _load_history_from_csv(csv_path, csv_fields):
    history = {k: [] for k in csv_fields}
    if not os.path.exists(csv_path):
        return history
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in csv_fields:
                history[k].append(float(row[k]))
    return history

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    planner_tag = args.planner

    csv_path = os.path.join(LOG_DIR, "lgrl_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "lgrl.pt")

    print("=" * 60)
    print("  LGRL PPO Training (stage-based) -- MiniGrid-DoorKey-5x5-v0")
    print("=" * 60)
    print(f"  Device         : {DEVICE}")
    print(f"  Envs           : {NUM_ENVS} parallel")
    print(f"  Frames         : {TOTAL_FRAMES:,}")
    print(f"  Planner        : {planner_tag}")
    print(f"  Stages         : {N_SUBGOALS}")
    print(f"  R_mission      : {R_MISSION}")
    print(f"  R_subgoal      : {R_SUBGOAL}")
    print("=" * 60)

    envs = [make_env(ENV_NAME, seed=i) for i in range(NUM_ENVS)]

    vocab = Vocabulary()
    sample_obs, _ = gym.make(ENV_NAME).reset()
    vocab.tokenize(f"{sample_obs['mission']} [SEP] search for the yellow key", max_len=32)

    obs_space = envs[0].observation_space
    act_space = envs[0].action_space
    model = LGRLAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Action space   : {act_space.n} actions")
    print(f"  Model params   : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    if planner_tag == "llm":
        planner = LLMPlanner()
    else:
        planner = RuleBasedPlanner()

    hierarchy_state = HierarchyState(NUM_ENVS, planner, envs)

    logger = (
        SubgoalLogger(LOG_DIR, "lgrl", NUM_ENVS)
        if args.subgoal_log else None
    )

    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, device=DEVICE)
    reshape_reward = make_reshape_reward(hierarchy_state, logger=logger)

    algo = torch_ac.PPOAlgo(
        envs=envs, acmodel=model, device=DEVICE,
        num_frames_per_proc=NUM_FRAMES_PER_PROC,
        discount=DISCOUNT, lr=LR, gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF,
        max_grad_norm=MAX_GRAD_NORM, recurrence=RECURRENCE,
        clip_eps=CLIP_EPS, epochs=EPOCHS, batch_size=BATCH_SIZE,
        preprocess_obss=preprocess_obss, reshape_reward=reshape_reward,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    update = 0
    total_frames = 0
    frames_per_update = NUM_ENVS * NUM_FRAMES_PER_PROC
    start_time = time.time()

    csv_fields = [
        "update", "frames", "avg_return", "avg_steps",
        "entropy", "policy_loss", "value_loss", "elapsed_sec",
    ]

    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            algo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        update = ckpt.get("update", 0)
        total_frames = ckpt.get("total_frames", 0)
        print(f"  Resumed: update={update}, frames={total_frames:,}")
        history = _load_history_from_csv(csv_path, csv_fields)
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    else:
        history = {k: [] for k in csv_fields}
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

    print(f"\n  Logging to : {csv_path}")
    print(f"  Plots      : {PLOT_DIR}")
    print(
        f"\n{'Update':>7} | {'Frames':>10} | {'Avg Return':>11} | {'Avg Steps':>10} | "
        f"{'Entropy':>8} | {'Policy Loss':>12} | {'Value Loss':>11}"
    )
    print("-" * 90)

    while total_frames < TOTAL_FRAMES:
        update += 1
        total_frames += frames_per_update

        reshape_reward._current_env_idx = 0
        exps, collect_logs = algo.collect_experiences()
        update_logs = algo.update_parameters(exps)

        avg_return  = np.mean(collect_logs["return_per_episode"])
        avg_steps   = np.mean(collect_logs["num_frames_per_episode"])
        entropy     = np.mean(update_logs["entropy"])
        policy_loss = np.mean(update_logs["policy_loss"])
        value_loss  = np.mean(update_logs["value_loss"])
        elapsed     = time.time() - start_time

        print(
            f"{update:>7} | {total_frames:>10,} | {avg_return:>11.3f} | "
            f"{avg_steps:>10.1f} | {entropy:>8.4f} | {policy_loss:>12.4f} | "
            f"{value_loss:>11.4f}"
        )

        row = {
            "update": update, "frames": total_frames,
            "avg_return": f"{avg_return:.6f}", "avg_steps": f"{avg_steps:.1f}",
            "entropy": f"{entropy:.6f}", "policy_loss": f"{policy_loss:.6f}",
            "value_loss": f"{value_loss:.6f}", "elapsed_sec": f"{elapsed:.1f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()
        for k, v in row.items():
            history[k].append(float(v))

        if update % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_path, model, algo, vocab,
                             update, total_frames, planner_tag)
        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR, planner_tag)

    _save_checkpoint(checkpoint_path, model, algo, vocab,
                     update, total_frames, planner_tag)
    csv_file.close()
    if logger is not None:
        logger.close()
    save_plots(history, PLOT_DIR, planner_tag)

    print(f"\nTraining complete.")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {csv_path}")
    print(f"  Plots      : {PLOT_DIR}")
    print(f"  Updates: {update}, Frames: {total_frames:,}")


if __name__ == "__main__":
    main()
