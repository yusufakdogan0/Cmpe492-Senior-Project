"""
LGRL PPO training with the rule-based oracle planner only.

Stage-based hierarchy: the planner produces a fixed forward-only sequence
of subgoals matching the LGRL paper. No "explore" or "go to" subgoals.

Supports three environment families:
  - MiniGrid-DoorKey-5x5-v0              (5 stages)
  - MiniGrid-GoToDoor-{5x5,6x6,8x8}-v0   (1 stage)
  - MiniGrid-GoToObject-{6x6,8x8}-N2-v0  (1 stage)

Artifact naming:
  - DoorKey-5x5 (legacy default) keeps base names:
      checkpoints/lgrl_rule.pt, logs/lgrl_rule_metrics.csv, ...
  - Other envs are suffixed:
      checkpoints/lgrl_rule_gotodoor5x5.pt, etc.

Usage:
    python scripts/train_lgrl_rule.py
    python scripts/train_lgrl_rule.py --env MiniGrid-GoToDoor-5x5-v0
    python scripts/train_lgrl_rule.py --env MiniGrid-GoToObject-6x6-N2-v0
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
from utils.env_utils import (
    SUPPORTED_ENVS,
    LEGACY_DEFAULT_ENV,
    env_max_steps,
    resolve_artifact_stem,
)
from utils.rule_based_planner import RuleBasedPlanner
from utils.subgoal_tracker import SubgoalTracker
from utils.subgoal_logger import SubgoalLogger

# ---------------------------------------------------------------------------
# Static configuration (not env-dependent)
# ---------------------------------------------------------------------------

DEFAULT_ENV = LEGACY_DEFAULT_ENV
NUM_ENVS = 16
NUM_FRAMES_PER_PROC = 128
TOTAL_FRAMES = 20_000_000

# PPO hyperparameters from LGRL paper (Section 4.3)
LR = 1e-4
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
BATCH_SIZE = 256
# Not specified in the paper; torch-ac-style defaults
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPOCHS = 4
RECURRENCE = 4

# Reward scaffolding from LGRL paper (Eqs. 5–7)
R_MISSION = 0.5                  # Rm
R_SUBGOAL = 0.5                  # Rt
MISSION_TIME_COEF = 0.5          # 0.5 factor in Eq. 5
SUBGOAL_TIME_COEF = 0.5          # 0.5 factor in Eq. 6
SUBGOAL_TIMEOUT_MULT = 2.0       # Eq. 6 "if Tused > 2*Ti, ri = 0"

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY = 10

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
PLOT_EVERY = 50

BASE_ARTIFACT_STEM = "lgrl_rule"
PLANNER_TAG = "rule_based"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LGRL with the deterministic rule-based subgoal planner."
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=SUPPORTED_ENVS,
        help=f"MiniGrid environment id (default: {DEFAULT_ENV})",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subgoal-log", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-env hierarchy state (stage-based, forward-only)
# ---------------------------------------------------------------------------

class HierarchyState:
    """Tracks the current stage, active subgoal, and step budgets per env.

    The stage index only advances forward — once a stage is completed
    (or timed out), the planner is queried with the next stage index,
    preventing the agent from farming rewards on repeated subgoals.
    """

    def __init__(
        self,
        num_envs: int,
        planner: RuleBasedPlanner,
        envs: list,
        *,
        n_subgoals: int,
        t_max: int,
    ):
        self.num_envs = num_envs
        self.planner = planner
        self.envs = envs
        self.n_subgoals = n_subgoals
        self.t_max = t_max
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
        """Query planner for the initial subgoal at stage 0."""
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
        """T_i = (i/n) * T_max  (paper Eq. 6), using 1-indexed stage."""
        i = min(self.stage_indices[env_idx] + 1, self.n_subgoals)
        return (i / self.n_subgoals) * self.t_max

    def advance(self, env_idx: int, obs: dict):
        """Advance to the next stage and query the planner."""
        next_stage = self.stage_indices[env_idx] + 1
        if next_stage >= self.n_subgoals:
            # All stages done — no more subgoal rewards.
            # Keep the last active subgoal visible to the agent's text
            # stream (do not overwrite), so the policy still has a useful
            # conditioning signal while it finishes the mission.
            self.stage_indices[env_idx] = self.n_subgoals
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
                else ""
            )
            # Before the first planner query this frame, fall back to a
            # neutral "search" subgoal that covers all env families.
            if not subgoal:
                subgoal = "search for the target"
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
# Reward shaping (stage-based, forward-only)
# ---------------------------------------------------------------------------

def make_reshape_reward(hierarchy_state, logger=None):
    """Build the reward callback. Stage-based: each stage can only be
    completed once per episode."""

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
                # Paper Eq. 5: rm = Rm * (1 - 0.5 * Tused/Tmax)
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / hierarchy_state.t_max, 1.0)
                total_reward += R_MISSION * (1.0 - MISSION_TIME_COEF * ratio)
            if logger:
                logger.on_episode_end(
                    env_idx, mission, success,
                    hierarchy_state.episode_steps[env_idx],
                )
            hierarchy_state.reset_env(env_idx)
            return total_reward

        # If all stages exhausted, no subgoal checking
        if hierarchy_state.stage_indices[env_idx] >= hierarchy_state.n_subgoals:
            return total_reward

        # Initialize subgoal on first step if needed
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
        timed_out = t_used > SUBGOAL_TIMEOUT_MULT * t_budget

        if completed:
            # Paper Eq. 6: ri = Rt * (1 - 0.5 * Tused/Ti), clipped at 2*Ti
            ratio = min(t_used / max(t_budget, 1), SUBGOAL_TIMEOUT_MULT)
            r_i = max(R_SUBGOAL * (1.0 - SUBGOAL_TIME_COEF * ratio), 0.0)
            # Paper Eq. 7: normalize by n
            total_reward += r_i / hierarchy_state.n_subgoals

            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Success", "steps": t_used,
                 "stage": hierarchy_state.stage_indices[env_idx]}
            )
            if logger:
                logger.log(
                    env_idx, "completed", mission=mission, subgoal=subgoal,
                    stage=hierarchy_state.stage_indices[env_idx],
                    steps_used=t_used, budget=t_budget,
                    reward=r_i / hierarchy_state.n_subgoals,
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
# Plotting
# ---------------------------------------------------------------------------

def _smooth(values, window=20):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def save_plots(history, plot_dir, artifact_stem, env_name):
    frames = history["frames"]
    if len(frames) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"LGRL PPO (rule oracle, stage-based) -- {env_name}",
        fontsize=14, fontweight="bold",
    )

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
    plt.savefig(os.path.join(plot_dir, f"{artifact_stem}_training_curves.png"), dpi=150)
    plt.close(fig)


def _save_checkpoint(path, model, algo, vocab, update, total_frames, env_name):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": algo.optimizer.state_dict(),
            "vocab": vocab.word2idx,
            "update": update,
            "total_frames": total_frames,
            "planner": PLANNER_TAG,
            "env": env_name,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    env_name = args.env

    # Per-env derived config
    t_max = env_max_steps(env_name)
    artifact_stem = resolve_artifact_stem(BASE_ARTIFACT_STEM, env_name)

    # Probe a sample mission to determine stage count for this env family.
    # All episodes in a single env use the same mission family, so N is
    # determined once at startup and stays constant.
    sample_env = gym.make(env_name)
    sample_obs, _ = sample_env.reset()
    sample_env.close()
    n_subgoals = RuleBasedPlanner.num_stages(sample_obs["mission"])

    csv_path = os.path.join(LOG_DIR, f"{artifact_stem}_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{artifact_stem}.pt")

    print("=" * 70)
    print("  LGRL PPO Training (rule oracle, stage-based)")
    print("=" * 70)
    print(f"  Env            : {env_name}")
    print(f"  Artifact stem  : {artifact_stem}")
    print(f"  Device         : {DEVICE}")
    print(f"  Envs           : {NUM_ENVS} parallel")
    print(f"  Frames         : {TOTAL_FRAMES:,}")
    print(f"  Planner        : {PLANNER_TAG} (stage-based, {n_subgoals} stages)")
    print(f"  T_max          : {t_max}  (from env.unwrapped.max_steps)")
    print(f"  R_mission      : {R_MISSION}")
    print(f"  R_subgoal      : {R_SUBGOAL}")
    print(f"  Subgoal budget : T_i = ((stage+1)/{n_subgoals}) * {t_max}")
    print("=" * 70)

    envs = [make_env(env_name, seed=i) for i in range(NUM_ENVS)]

    vocab = Vocabulary()
    # Pre-seed the vocab with a representative "mission [SEP] subgoal"
    # from each env family so common tokens are present before training.
    vocab.tokenize(
        f"{sample_obs['mission']} [SEP] search for the target", max_len=32
    )

    obs_space = envs[0].observation_space
    act_space = envs[0].action_space
    model = LGRLAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Action space   : {act_space.n} actions")
    print(f"  Model params   : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    planner = RuleBasedPlanner()
    hierarchy_state = HierarchyState(
        NUM_ENVS, planner, envs,
        n_subgoals=n_subgoals, t_max=t_max,
    )

    logger = (
        SubgoalLogger(LOG_DIR, artifact_stem, NUM_ENVS)
        if args.subgoal_log else None
    )

    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, device=DEVICE)
    reshape_reward = make_reshape_reward(hierarchy_state, logger=logger)

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
        saved_env = ckpt.get("env")
        if saved_env is not None and saved_env != env_name:
            raise SystemExit(
                f"Checkpoint env mismatch: expected {env_name!r}, got {saved_env!r}."
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
    if logger is not None:
        print(f"  Subgoal log: {logger.dir}/ (per-env files)")
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
            _save_checkpoint(
                checkpoint_path, model, algo, vocab, update, total_frames, env_name
            )

        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR, artifact_stem, env_name)

    _save_checkpoint(
        checkpoint_path, model, algo, vocab, update, total_frames, env_name
    )
    csv_file.close()
    if logger is not None:
        logger.close()
    save_plots(history, PLOT_DIR, artifact_stem, env_name)

    print("\nTraining complete.")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {csv_path}")
    print(f"  Plots      : {PLOT_DIR}")
    print(f"  Updates: {update}, Frames: {total_frames:,}")


if __name__ == "__main__":
    main()
