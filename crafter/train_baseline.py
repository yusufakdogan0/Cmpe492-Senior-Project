"""
train_baseline.py — baseline PPO training on Crafter (control condition).

Crafter port of the MiniGrid project's ``scripts/train_baseline.py``. A
standard PPO agent conditioned ONLY on the mission string — no LLM
planner, no subgoals, no reward shaping. It learns from the sparse
environment reward (+1 the step the target achievement unlocks, 0
otherwise), so its average return equals its success rate.

This is the "Base" condition the LGRL agent is compared against (paper
Table 2): same network architecture (``CrafterACModel``) and same PPO
hyperparameters, differing only in that the text stream sees the mission
alone and there is no subgoal scaffolding.

Usage:
    python crafter/train_baseline.py --task collect_wood
    python crafter/train_baseline.py --task make_stone_pickaxe --frames 5000000
    python crafter/train_baseline.py --task collect_stone --resume
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from collections import deque

import numpy as np
import torch

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_ac
import torch_ac.algos.base
from seq_env import SequentialEnv

torch_ac.algos.base.ParallelEnv = SequentialEnv

from crafter_env import CrafterTaskEnv
from crafter_agent import CrafterACModel, Vocabulary
from crafter_tasks import (
    TASK_MISSION, SUPPORTED_TASKS, num_subgoals, task_max_steps,
    resolve_artifact_stem,
)

# ---------------------------------------------------------------------------
# Static configuration (identical PPO hyperparameters to the LGRL scripts)
# ---------------------------------------------------------------------------

NUM_ENVS = 16
NUM_FRAMES_PER_PROC = 128
TOTAL_FRAMES = 3_000_000

LR = 1e-4
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
BATCH_SIZE = 256
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPOCHS = 4
RECURRENCE = 4

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
CHECKPOINT_EVERY = 10
PLOT_EVERY = 25
SUCCESS_WINDOW = 200

BASE_ARTIFACT_STEM = "baseline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Minimal per-env episode tracker (success bookkeeping only; no subgoals)
# ---------------------------------------------------------------------------

class EpisodeTracker:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.episode_raw_return = [0.0] * num_envs
        self.completed = []  # list of bool (success)


def make_preprocess_obss(vocab, device=None):
    def preprocess_obss(obss, device=device):
        images = np.array([obs["image"] for obs in obss])
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images.permute(0, 3, 1, 2) / 255.0
        token_ids = [vocab.tokenize(obs["mission"], max_len=32) for obs in obss]
        texts = torch.tensor(token_ids, dtype=torch.long, device=device)
        return torch_ac.DictList({"image": images, "text": texts})
    return preprocess_obss


def make_reshape_reward(tracker):
    """Pass the raw (sparse) reward through unchanged; just record success
    on episode end so we can report a rolling success rate."""
    def reshape_reward(obs, action, reward, done):
        env_idx = reshape_reward._current_env_idx
        reshape_reward._current_env_idx = (env_idx + 1) % tracker.num_envs
        tracker.episode_raw_return[env_idx] += float(reward)
        if done:
            tracker.completed.append(tracker.episode_raw_return[env_idx] > 0)
            tracker.episode_raw_return[env_idx] = 0.0
        return float(reward)
    reshape_reward._current_env_idx = 0
    return reshape_reward


def make_env(task, seed):
    return CrafterTaskEnv(task, seed=seed)


# ---------------------------------------------------------------------------
# Plotting / checkpoint
# ---------------------------------------------------------------------------

def _smooth(values, window=20):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


def save_plots(history, plot_dir, artifact_stem, task):
    frames = history["frames"]
    if len(frames) < 2:
        return
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Baseline PPO (mission-only) -- Crafter:{task}",
                 fontsize=14, fontweight="bold")
    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.25, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.8)
    ax.set(xlabel="Frames", ylabel="Avg return", title="Average Return per Episode")
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.plot(frames, history["success_rate"], alpha=0.25, color="tab:green", linewidth=0.5)
    ax.plot(frames, _smooth(history["success_rate"]), color="tab:green", linewidth=1.8)
    ax.set(xlabel="Frames", ylabel="Success rate",
           title=f"Rolling Success Rate (last {SUCCESS_WINDOW})")
    ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(frames, _smooth(history["avg_steps"]), color="tab:orange", linewidth=1.5)
    ax.set(xlabel="Frames", ylabel="Avg steps", title="Average Steps per Episode")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(frames, _smooth(history["policy_loss"]), color="tab:red", linewidth=1.5, label="Policy Loss")
    ax.plot(frames, _smooth(history["value_loss"]), color="tab:purple", linewidth=1.5, label="Value Loss")
    ax.set(xlabel="Frames", ylabel="Loss", title="Policy & Value Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{artifact_stem}_training_curves.png"), dpi=150)
    plt.close(fig)


def _save_checkpoint(path, model, algo, vocab, update, total_frames, task):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": algo.optimizer.state_dict(),
        "vocab": vocab.word2idx,
        "update": update,
        "total_frames": total_frames,
        "planner": "none",
        "task": task,
    }, path)


def _load_history_from_csv(csv_path, csv_fields):
    history = {k: [] for k in csv_fields}
    if not os.path.exists(csv_path):
        return history
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            for k in csv_fields:
                history[k].append(float(row[k]))
    return history


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline PPO training on Crafter (mission-only control).")
    p.add_argument("--task", default="collect_wood", choices=SUPPORTED_TASKS)
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    task = args.task
    mission = TASK_MISSION[task]
    t_max = task_max_steps(task)

    artifact_stem = resolve_artifact_stem(BASE_ARTIFACT_STEM, task)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"{artifact_stem}_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{artifact_stem}.pt")
    csv_fields = ["update", "frames", "avg_return", "avg_steps",
                  "success_rate", "entropy", "policy_loss", "value_loss",
                  "elapsed_sec"]

    print("=" * 70)
    print("  Baseline PPO Training on Crafter (mission-only)")
    print("=" * 70)
    print(f"  Task           : {task}")
    print(f"  Mission        : {mission!r}")
    print(f"  T_max          : {t_max}")
    print(f"  Device         : {DEVICE}")
    print(f"  Frames         : {args.frames:,}")
    print(f"  Envs           : {NUM_ENVS}")
    print("=" * 70)

    envs = [make_env(task, seed=args.seed * 1000 + i) for i in range(NUM_ENVS)]
    tracker = EpisodeTracker(NUM_ENVS)

    vocab = Vocabulary()
    vocab.tokenize(mission)
    for s in ("collect", "place", "make", "a", "an", "table", "furnace",
              "wood", "stone", "coal", "iron", "diamond", "pickaxe", "sword"):
        _ = vocab[s]

    model = CrafterACModel(envs[0].observation_space, envs[0].action_space,
                           vocab).to(DEVICE)
    preprocess = make_preprocess_obss(vocab, device=DEVICE)
    reshape = make_reshape_reward(tracker)

    algo = torch_ac.PPOAlgo(
        envs, model, device=DEVICE, num_frames_per_proc=NUM_FRAMES_PER_PROC,
        discount=DISCOUNT, lr=LR, gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF,
        max_grad_norm=MAX_GRAD_NORM, recurrence=RECURRENCE, clip_eps=CLIP_EPS,
        epochs=EPOCHS, batch_size=BATCH_SIZE, preprocess_obss=preprocess,
        reshape_reward=reshape,
    )

    start_update, total_frames = 0, 0
    history = {k: [] for k in csv_fields}
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        algo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = ckpt["update"]
        total_frames = ckpt["total_frames"]
        history = _load_history_from_csv(csv_path, csv_fields)
        print(f"Resumed from update {start_update} ({total_frames:,} frames)")

    csv_mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if csv_mode == "w":
        csv_writer.writeheader()

    success_buf = deque(maxlen=SUCCESS_WINDOW)
    num_updates = args.frames // (NUM_ENVS * NUM_FRAMES_PER_PROC)
    print(f"{'upd':>6} | {'frames':>10} | {'return':>7} | {'steps':>6} | "
          f"{'succ%':>6} | {'ploss':>8} | {'vloss':>8}")
    print("-" * 70)

    start_time = time.time()
    update = start_update
    for update in range(start_update + 1, num_updates + 1):
        reshape._current_env_idx = 0
        exps, collect_logs = algo.collect_experiences()
        update_logs = algo.update_parameters(exps)
        total_frames += NUM_ENVS * NUM_FRAMES_PER_PROC

        for success in tracker.completed:
            success_buf.append(1 if success else 0)
        tracker.completed.clear()
        success_rate = float(np.mean(success_buf)) if success_buf else 0.0

        avg_return = float(np.mean(collect_logs["return_per_episode"]))
        avg_steps = float(np.mean(collect_logs["num_frames_per_episode"]))
        entropy = float(np.mean(update_logs["entropy"]))
        policy_loss = float(np.mean(update_logs["policy_loss"]))
        value_loss = float(np.mean(update_logs["value_loss"]))
        elapsed = time.time() - start_time

        row = {"update": update, "frames": total_frames,
               "avg_return": f"{avg_return:.6f}", "avg_steps": f"{avg_steps:.1f}",
               "success_rate": f"{success_rate:.4f}", "entropy": f"{entropy:.6f}",
               "policy_loss": f"{policy_loss:.6f}", "value_loss": f"{value_loss:.6f}",
               "elapsed_sec": f"{elapsed:.1f}"}
        csv_writer.writerow(row); csv_file.flush()
        for k, v in row.items():
            history[k].append(float(v))

        if update % 5 == 0 or update == 1:
            print(f"{update:>6} | {total_frames:>10,} | {avg_return:>7.3f} | "
                  f"{avg_steps:>6.1f} | {success_rate*100:>6.1f} | "
                  f"{policy_loss:>8.4f} | {value_loss:>8.4f}")

        if update % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_path, model, algo, vocab, update,
                             total_frames, task)
        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR, artifact_stem, task)

    _save_checkpoint(checkpoint_path, model, algo, vocab, update, total_frames, task)
    save_plots(history, PLOT_DIR, artifact_stem, task)
    csv_file.close()
    print("\nTraining complete.")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {csv_path}")


if __name__ == "__main__":
    main()
