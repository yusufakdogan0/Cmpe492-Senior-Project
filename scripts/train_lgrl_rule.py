"""
LGRL PPO training with the rule-based oracle planner only.

This script is self-contained: it does not import ``train_lgrl.py``.
Use ``python scripts/train_lgrl.py`` for LLM-based training.

Outputs (under project ``logs/`` and ``checkpoints/``):
  - ``logs/lgrl_rule_metrics.csv``
  - ``checkpoints/lgrl_rule.pt``
  - ``logs/plots/lgrl_rule_training_curves.png``
  - ``logs/lgrl_rule_subgoal_log.jsonl`` (with ``--subgoal-log``)

Reward scaffolding matches the LGRL setup (mission + shaped subgoal rewards).

Usage:
    python scripts/train_lgrl_rule.py
    python scripts/train_lgrl_rule.py --resume
    python scripts/train_lgrl_rule.py --subgoal-log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import gymnasium as gym
import matplotlib
import minigrid  # noqa: F401 -- registers MiniGrid envs
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
from utils.rule_based_planner import RuleBasedPlanner
from utils.subgoal_tracker import SubgoalTracker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_NAME = "MiniGrid-DoorKey-5x5-v0"
NUM_ENVS = 16
NUM_FRAMES_PER_PROC = 128
TOTAL_FRAMES = 10_000_000

LR = 1e-4
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPOCHS = 4
BATCH_SIZE = 256
RECURRENCE = 4

R_MISSION = 0.5
R_SUBGOAL = 0.5
N_SUBGOALS_EST = 5
MAX_ENV_STEPS = 250

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY = 10

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
PLOT_EVERY = 50

# Isolated from LLM runs (train_lgrl.py uses its own filenames).
ARTIFACT_STEM = "lgrl_rule"
PLANNER_TAG = "rule_based"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LGRL with the deterministic rule-based subgoal planner."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/lgrl_rule.pt (appends to lgrl_rule_metrics.csv).",
    )
    parser.add_argument(
        "--subgoal-log",
        action="store_true",
        default=False,
        help="Append subgoal events to logs/lgrl_rule_subgoal_log.jsonl.",
    )
    return parser.parse_args()


class HierarchyState:
    """Tracks active subgoals, step budgets, and history for all parallel envs."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset_all()

    def reset_all(self):
        self.active_subgoals: list[str] = ["explore"] * self.num_envs
        self.step_counters: list[int] = [0] * self.num_envs
        self.subgoal_indices: list[int] = [1] * self.num_envs
        self.histories: list[list[dict]] = [[] for _ in range(self.num_envs)]
        self.trackers: list[SubgoalTracker] = [
            SubgoalTracker() for _ in range(self.num_envs)
        ]
        self.episode_steps: list[int] = [0] * self.num_envs

    def reset_env(self, env_idx: int):
        self.active_subgoals[env_idx] = "explore"
        self.step_counters[env_idx] = 0
        self.subgoal_indices[env_idx] = 1
        self.histories[env_idx] = []
        self.trackers[env_idx].reset()
        self.episode_steps[env_idx] = 0

    def subgoal_budget(self, env_idx: int) -> float:
        i = min(self.subgoal_indices[env_idx], N_SUBGOALS_EST)
        return (i / N_SUBGOALS_EST) * MAX_ENV_STEPS


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
                else "explore"
            )
            combined = f"{obs['mission']} [SEP] {subgoal}"
            token_ids.append(vocab.tokenize(combined, max_len=32))

        texts = torch.tensor(token_ids, dtype=torch.long, device=device)
        return torch_ac.DictList({"image": images, "text": texts})

    return preprocess_obss


def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def make_reshape_reward(hierarchy_state, planner, envs, subgoal_log=None):
    def reshape_reward(obs, action, reward, done):
        env_idx = reshape_reward._current_env_idx
        reshape_reward._current_env_idx = (env_idx + 1) % hierarchy_state.num_envs

        total_reward = 0.0
        hierarchy_state.step_counters[env_idx] += 1
        hierarchy_state.episode_steps[env_idx] += 1

        if done:
            if reward > 0:
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / MAX_ENV_STEPS, 1.0)
                total_reward += R_MISSION * (1.0 - 0.5 * ratio)
            hierarchy_state.reset_env(env_idx)
            return total_reward

        uw = envs[env_idx].unwrapped
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
            total_reward += r_i / N_SUBGOALS_EST

            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Success", "steps": t_used}
            )
            _log_subgoal_event(
                subgoal_log, env_idx, "completed",
                obs.get("mission", ""), subgoal,
                t_used, t_budget, r_i / N_SUBGOALS_EST,
            )
            _advance_subgoal(
                env_idx, uw, obs, hierarchy_state, planner, subgoal_log,
            )

        elif timed_out:
            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Failed", "steps": t_used}
            )
            _log_subgoal_event(
                subgoal_log, env_idx, "timed_out",
                obs.get("mission", ""), subgoal,
                t_used, t_budget, 0.0,
            )
            _advance_subgoal(
                env_idx, uw, obs, hierarchy_state, planner, subgoal_log,
            )

        return total_reward

    reshape_reward._current_env_idx = 0
    return reshape_reward


def _log_subgoal_event(
    log_file, env_idx, event, mission, subgoal,
    steps_used, budget, reward_given,
    env_state=None, raw_llm=None,
):
    if log_file is None:
        return
    entry = {
        "env": env_idx,
        "event": event,
        "mission": mission,
        "subgoal": subgoal,
        "valid": SubgoalTracker.is_recognized(subgoal),
        "steps_used": steps_used,
        "budget": round(budget, 1),
        "reward": round(reward_given, 6),
    }
    if env_state is not None:
        entry["env_state"] = env_state
    if raw_llm is not None:
        entry["raw_llm_response"] = raw_llm
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def _advance_subgoal(env_idx, uw, obs, hierarchy_state, planner, subgoal_log=None):
    env_json = parse_env_description(obs["image"], uw.carrying)
    direction = obs.get("direction", 0)
    hist = hierarchy_state.histories[env_idx]
    past = [h["subgoal"] for h in hist]

    new_subgoal = planner.get_subgoal(
        obs["mission"],
        env_json,
        direction,
        past,
    )
    raw = getattr(planner, "last_raw_response", None)

    hierarchy_state.active_subgoals[env_idx] = new_subgoal
    hierarchy_state.step_counters[env_idx] = 0
    finished = (hist[-1]["subgoal"] if hist else "").strip().lower()
    if finished != "explore":
        hierarchy_state.subgoal_indices[env_idx] += 1

    _log_subgoal_event(
        subgoal_log, env_idx, "new",
        obs.get("mission", ""), new_subgoal,
        0, hierarchy_state.subgoal_budget(env_idx), 0.0,
        env_state=env_json, raw_llm=raw,
    )


def _smooth(values, window=20):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def save_plots(history, plot_dir):
    frames = history["frames"]
    if len(frames) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "LGRL PPO (rule oracle) -- Training Curves",
        fontsize=14, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.3, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.5, label="Smoothed")
    ax.set(xlabel="Frames", ylabel="Average Return", title="Average Return per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(frames, history["avg_steps"], alpha=0.3, color="tab:orange", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_steps"]), color="tab:orange", linewidth=1.5, label="Smoothed")
    ax.set(xlabel="Frames", ylabel="Average Steps", title="Average Steps per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(frames, history["entropy"], color="tab:green", linewidth=1)
    ax.set(xlabel="Frames", ylabel="Entropy", title="Policy Entropy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(frames, _smooth(history["policy_loss"]), color="tab:red", linewidth=1.5, label="Policy Loss")
    ax.plot(frames, _smooth(history["value_loss"]), color="tab:purple", linewidth=1.5, label="Value Loss")
    ax.set(xlabel="Frames", ylabel="Loss", title="Policy & Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_name = f"{ARTIFACT_STEM}_training_curves.png"
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=150)
    plt.close(fig)


def _save_checkpoint(path, model, algo, vocab, update, total_frames):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": algo.optimizer.state_dict(),
            "vocab": vocab.word2idx,
            "update": update,
            "total_frames": total_frames,
            "planner": PLANNER_TAG,
        },
        path,
    )


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


def main():
    args = parse_args()

    csv_path = os.path.join(LOG_DIR, f"{ARTIFACT_STEM}_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{ARTIFACT_STEM}.pt")

    print("=" * 60)
    print("  LGRL PPO Training (rule oracle) -- MiniGrid-DoorKey-5x5-v0")
    print("=" * 60)
    print(f"  Device         : {DEVICE}")
    print(f"  Envs           : {NUM_ENVS} parallel")
    print(f"  Frames         : {TOTAL_FRAMES:,}")
    print(f"  Planner        : {PLANNER_TAG} (RuleBasedPlanner)")
    print(f"  R_mission      : {R_MISSION}")
    print(f"  R_subgoal      : {R_SUBGOAL}")
    print(f"  Subgoal budget : T_i = (i/{N_SUBGOALS_EST}) * {MAX_ENV_STEPS}")
    print(f"  Metrics file   : {ARTIFACT_STEM}_metrics.csv")
    print(f"  Checkpoint     : {ARTIFACT_STEM}.pt")
    print("=" * 60)

    envs = [make_env(ENV_NAME, seed=i) for i in range(NUM_ENVS)]

    vocab = Vocabulary()
    sample_obs, _ = gym.make(ENV_NAME).reset()
    vocab.tokenize(f"{sample_obs['mission']} [SEP] explore", max_len=32)

    obs_space = envs[0].observation_space
    act_space = envs[0].action_space
    model = LGRLAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Action space   : {act_space.n} actions")
    print(f"  Model params   : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    hierarchy_state = HierarchyState(NUM_ENVS)
    planner = RuleBasedPlanner()

    subgoal_log_path = os.path.join(LOG_DIR, f"{ARTIFACT_STEM}_subgoal_log.jsonl")
    subgoal_log = (
        open(subgoal_log_path, "a", encoding="utf-8") if args.subgoal_log else None
    )

    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, device=DEVICE)
    reshape_reward = make_reshape_reward(
        hierarchy_state, planner, envs, subgoal_log=subgoal_log,
    )

    algo = torch_ac.PPOAlgo(
        envs=envs,
        acmodel=model,
        device=DEVICE,
        num_frames_per_proc=NUM_FRAMES_PER_PROC,
        discount=DISCOUNT,
        lr=LR,
        gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF,
        value_loss_coef=VALUE_LOSS_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        recurrence=RECURRENCE,
        clip_eps=CLIP_EPS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        preprocess_obss=preprocess_obss,
        reshape_reward=reshape_reward,
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
        saved = ckpt.get("planner")
        if saved is not None and saved != PLANNER_TAG:
            raise SystemExit(
                f"Checkpoint planner mismatch: expected {PLANNER_TAG!r}, got {saved!r}."
            )
        if saved is None:
            print(
                "  Warning: checkpoint has no 'planner' field; resume only if this is a rule run.",
            )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            algo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        update = ckpt.get("update", 0)
        total_frames = ckpt.get("total_frames", 0)
        print(f"  Resumed from checkpoint: update={update}, frames={total_frames:,}")

        history = _load_history_from_csv(csv_path, csv_fields)
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    else:
        history = {k: [] for k in csv_fields}
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

    print(f"\n  Logging to : {csv_path}")
    if subgoal_log is not None:
        print(f"  Subgoal log: {subgoal_log_path}")
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

        avg_return = np.mean(collect_logs["return_per_episode"])
        avg_steps = np.mean(collect_logs["num_frames_per_episode"])
        entropy = np.mean(update_logs["entropy"])
        policy_loss = np.mean(update_logs["policy_loss"])
        value_loss = np.mean(update_logs["value_loss"])
        elapsed = time.time() - start_time

        print(
            f"{update:>7} | {total_frames:>10,} | {avg_return:>11.3f} | "
            f"{avg_steps:>10.1f} | {entropy:>8.4f} | {policy_loss:>12.4f} | "
            f"{value_loss:>11.4f}"
        )

        row = {
            "update": update,
            "frames": total_frames,
            "avg_return": f"{avg_return:.6f}",
            "avg_steps": f"{avg_steps:.1f}",
            "entropy": f"{entropy:.6f}",
            "policy_loss": f"{policy_loss:.6f}",
            "value_loss": f"{value_loss:.6f}",
            "elapsed_sec": f"{elapsed:.1f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()
        for k, v in row.items():
            history[k].append(float(v))

        if update % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_path, model, algo, vocab, update, total_frames)

        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR)

    _save_checkpoint(checkpoint_path, model, algo, vocab, update, total_frames)
    csv_file.close()
    if subgoal_log is not None:
        subgoal_log.close()
    save_plots(history, PLOT_DIR)

    print("\nTraining complete.")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {csv_path}")
    if subgoal_log is not None:
        print(f"  Subgoal log: {subgoal_log_path}")
    print(f"  Plots      : {PLOT_DIR}")
    print(f"  Updates: {update}, Frames: {total_frames:,}")


if __name__ == "__main__":
    main()
