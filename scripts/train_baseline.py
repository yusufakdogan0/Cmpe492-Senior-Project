"""
train_baseline.py — Baseline PPO (no subgoal guidance) for MiniGrid.

Standard recurrent PPO agent conditioned only on the mission string.
Used as the control condition for comparison with the LGRL agent.

Supported environments:
  - MiniGrid-DoorKey-5x5-v0              (legacy default)
  - MiniGrid-GoToDoor-{5x5,6x6,8x8}-v0
  - MiniGrid-GoToObject-{6x6,8x8}-N2-v0

Artifact naming:
  - DoorKey-5x5 keeps base names:  baseline.pt, baseline_metrics.csv, training_curves.png
  - Other envs are suffixed:       baseline_gotodoor5x5.pt, baseline_gotodoor5x5_metrics.csv, etc.

Hyperparams:
    lr=1e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, batch_size=256

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --env MiniGrid-GoToDoor-5x5-v0
    python scripts/train_baseline.py --env MiniGrid-GoToObject-6x6-N2-v0
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid envs with gymnasium)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# resolve project root so imports work from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch_ac

from models.baseline_agent import BaselineAgent, Vocabulary
from utils.env_utils import (
    SUPPORTED_ENVS,
    LEGACY_DEFAULT_ENV,
    resolve_artifact_stem,
)

# --- Static config -----------------------------------------------------

DEFAULT_ENV        = LEGACY_DEFAULT_ENV
NUM_ENVS           = 16          # parallel environments
NUM_FRAMES_PER_PROC = 128        # rollout length per env per update
TOTAL_FRAMES       = 20_000_000  # total training budget

# PPO hyperparameters
LR                 = 1e-4
DISCOUNT           = 0.99
GAE_LAMBDA         = 0.95
CLIP_EPS           = 0.2
BATCH_SIZE         = 256
# Not specified in the paper
ENTROPY_COEF       = 0.01
VALUE_LOSS_COEF    = 0.5
MAX_GRAD_NORM      = 0.5
EPOCHS             = 4
RECURRENCE         = 4

CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY   = 10

LOG_DIR            = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR           = os.path.join(LOG_DIR, "plots")
PLOT_EVERY         = 50

BASE_ARTIFACT_STEM = "baseline"
# DoorKey-5x5 retains the legacy plot filename "training_curves.png" for
# compatibility with existing project documentation.
LEGACY_PLOT_NAME   = "training_curves.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the baseline PPO agent (no subgoal guidance)."
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        choices=SUPPORTED_ENVS,
        help=f"MiniGrid environment id (default: {DEFAULT_ENV})",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


# --- Observation preprocessing ----------------------------------------

def make_preprocess_obss(vocab, device=None):
    """Returns a function that converts raw obs dicts into model-ready tensors."""

    def preprocess_obss(obss, device=device):
        # image: (B, 7, 7, 3) uint8 → (B, 3, 7, 7) float
        images = np.array([obs["image"] for obs in obss])
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images.permute(0, 3, 1, 2)       # NHWC → NCHW
        images = images / 255.0

        # mission text → tokenized ints
        missions = [obs["mission"] for obs in obss]
        token_ids = [vocab.tokenize(m) for m in missions]
        texts = torch.tensor(token_ids, dtype=torch.long, device=device)

        return torch_ac.DictList({"image": images, "text": texts})

    return preprocess_obss


# --- Env factory -------------------------------------------------------

def make_env(env_name, seed):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


# --- Plotting helpers --------------------------------------------------

def _smooth(values, window=20):
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
    return smoothed


def save_plots(history, plot_dir, plot_filename, env_name):
    frames = history["frames"]
    if len(frames) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Baseline PPO — {env_name}", fontsize=14, fontweight="bold"
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
    plt.savefig(os.path.join(plot_dir, plot_filename), dpi=150)
    plt.close(fig)


# --- Main training loop -----------------------------------------------

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
    env_name = args.env

    artifact_stem = resolve_artifact_stem(BASE_ARTIFACT_STEM, env_name)
    csv_path = os.path.join(LOG_DIR, f"{artifact_stem}_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{artifact_stem}.pt")
    # Preserve the legacy plot filename for DoorKey-5x5
    if env_name == LEGACY_DEFAULT_ENV:
        plot_filename = LEGACY_PLOT_NAME
    else:
        plot_filename = f"{artifact_stem}_training_curves.png"

    print("=" * 60)
    print(f"  Baseline PPO Training — {env_name}")
    print("=" * 60)
    print(f"  Artifact stem : {artifact_stem}")
    print(f"  Device        : {DEVICE}")
    print(f"  Envs          : {NUM_ENVS} parallel")
    print(f"  Frames        : {TOTAL_FRAMES:,}")
    print("=" * 60)

    # create parallel environments
    envs = [make_env(env_name, seed=i) for i in range(NUM_ENVS)]

    # build vocabulary (pre-seed with one mission to populate common words)
    vocab = Vocabulary()
    sample_obs, _ = gym.make(env_name).reset()
    vocab.tokenize(sample_obs["mission"])

    # build model
    obs_space  = envs[0].observation_space
    act_space  = envs[0].action_space
    model = BaselineAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Action space : {act_space.n} actions")
    print(f"  Model params : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    preprocess_obss = make_preprocess_obss(vocab, device=DEVICE)

    # set up PPO
    algo = torch_ac.PPOAlgo(
        envs              = envs,
        acmodel            = model,
        device             = DEVICE,
        num_frames_per_proc = NUM_FRAMES_PER_PROC,
        discount           = DISCOUNT,
        lr                 = LR,
        gae_lambda         = GAE_LAMBDA,
        entropy_coef       = ENTROPY_COEF,
        value_loss_coef    = VALUE_LOSS_COEF,
        max_grad_norm      = MAX_GRAD_NORM,
        recurrence         = RECURRENCE,
        clip_eps           = CLIP_EPS,
        epochs             = EPOCHS,
        batch_size         = BATCH_SIZE,
        preprocess_obss    = preprocess_obss,
    )

    # prepare directories
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

    # Resume or fresh start
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        saved_env = ckpt.get("env")
        if saved_env is not None and saved_env != env_name:
            raise SystemExit(
                f"Checkpoint env mismatch: expected {env_name!r}, got {saved_env!r}."
            )
        model.load_state_dict(ckpt["model_state_dict"])
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

    print(f"\n  Logging to  : {csv_path}")
    print(f"  Plots saved : {os.path.join(PLOT_DIR, plot_filename)}")
    print(
        f"\n{'Update':>7} | {'Frames':>10} | {'Avg Return':>11} | {'Avg Steps':>10} | "
        f"{'Entropy':>8} | {'Policy Loss':>12} | {'Value Loss':>11}"
    )
    print("-" * 90)

    while total_frames < TOTAL_FRAMES:
        update += 1
        total_frames += frames_per_update

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
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab.word2idx,
                "update": update,
                "total_frames": total_frames,
                "env": env_name,
            }, checkpoint_path)
            print(f"         → Checkpoint saved to {checkpoint_path}")

        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR, plot_filename, env_name)
            print(f"         → Plots updated in {PLOT_DIR}")

    # final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.word2idx,
        "update": update,
        "total_frames": total_frames,
        "env": env_name,
    }, checkpoint_path)
    csv_file.close()
    save_plots(history, PLOT_DIR, plot_filename, env_name)
    print(f"\n Training complete.")
    print(f" Final checkpoint : {checkpoint_path}")
    print(f" Metrics CSV      : {csv_path}")
    print(f" Plots            : {os.path.join(PLOT_DIR, plot_filename)}")
    print(f" Total updates: {update}, Total frames: {total_frames:,}")


if __name__ == "__main__":
    main()
