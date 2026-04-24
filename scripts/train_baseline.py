"""
train_baseline.py — PPO training on MiniGrid-DoorKey-5x5-v0.

This is the baseline (no LLM subgoals). Trains a standard recurrent
PPO agent as the control condition for comparison with the LGRL agent.

Usage:
    python scripts/train_baseline.py

Hyperparams follow the LGRL paper:
    lr=1e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
    entropy_coef=0.01, value_coef=0.5
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import gymnasium as gym
import minigrid                  # registers MiniGrid envs with gymnasium
import matplotlib
matplotlib.use("Agg")            # headless backend for saving plots
import matplotlib.pyplot as plt

# resolve project root so imports work from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch_ac

from models.baseline_agent import BaselineAgent, Vocabulary

# --- Config ---

ENV_NAME           = "MiniGrid-DoorKey-5x5-v0"
NUM_ENVS           = 16          # parallel environments
NUM_FRAMES_PER_PROC = 128        # rollout length per env per update
TOTAL_FRAMES       = 10_000_000  # total training budget

# PPO hyperparameters
LR                 = 1e-4
DISCOUNT           = 0.99
GAE_LAMBDA         = 0.95
CLIP_EPS           = 0.2
ENTROPY_COEF       = 0.01
VALUE_LOSS_COEF    = 0.5
MAX_GRAD_NORM      = 0.5
EPOCHS             = 4           # PPO update epochs
BATCH_SIZE         = 256
RECURRENCE         = 4           # BPTT window

CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY   = 10          # save every N updates
CHECKPOINT_PATH    = os.path.join(CHECKPOINT_DIR, "baseline.pt")

LOG_DIR            = os.path.join(PROJECT_ROOT, "logs")
CSV_PATH           = os.path.join(LOG_DIR, "baseline_metrics.csv")
PLOT_DIR           = os.path.join(LOG_DIR, "plots")
PLOT_EVERY         = 50          # regenerate plots every N updates

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Observation preprocessing ---
# MiniGrid gives us dicts with 'image', 'direction', 'mission'.
# torch-ac expects tensors, so we convert here.

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


# --- Env factory ---

def make_env(env_name, seed):
    """Create a single MiniGrid environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


# --- Plotting helpers ---

def _smooth(values, window=20):
    """Simple rolling average."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i+1]) / (i - start + 1))
    return smoothed


def save_plots(history, plot_dir):
    """Generate and save the four training curve subplots."""
    frames = history["frames"]
    if len(frames) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Baseline PPO — Training Curves", fontsize=14, fontweight="bold")

    # return
    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.3, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Average Return")
    ax.set_title("Average Return per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # steps
    ax = axes[0, 1]
    ax.plot(frames, history["avg_steps"], alpha=0.3, color="tab:orange", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_steps"]), color="tab:orange", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Average Steps")
    ax.set_title("Average Steps per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # entropy
    ax = axes[1, 0]
    ax.plot(frames, history["entropy"], color="tab:green", linewidth=1)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # losses
    ax = axes[1, 1]
    ax.plot(frames, _smooth(history["policy_loss"]), color="tab:red", linewidth=1.5, label="Policy Loss")
    ax.plot(frames, _smooth(history["value_loss"]), color="tab:purple", linewidth=1.5, label="Value Loss")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Loss")
    ax.set_title("Policy & Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


# --- Main training loop ---

def main():
    print("=" * 60)
    print("  Baseline PPO Training — MiniGrid-DoorKey-5x5-v0")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Envs   : {NUM_ENVS} parallel")
    print(f"  Frames : {TOTAL_FRAMES:,}")
    print("=" * 60)

    # create parallel environments
    envs = [make_env(ENV_NAME, seed=i) for i in range(NUM_ENVS)]

    # build vocabulary (pre-seed with one mission to populate common words)
    vocab = Vocabulary()
    sample_obs, _ = gym.make(ENV_NAME).reset()
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

    # CSV logger
    csv_fields = ["update", "frames", "avg_return", "avg_steps",
                  "entropy", "policy_loss", "value_loss", "elapsed_sec"]
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # keep history in memory for plotting
    history = {k: [] for k in csv_fields}

    print(f"\n  Logging to  : {CSV_PATH}")
    print(f"  Plots saved : {PLOT_DIR}")
    print(f"\n{'Update':>7} | {'Frames':>10} | {'Avg Return':>11} | {'Avg Steps':>10} | "
          f"{'Entropy':>8} | {'Policy Loss':>12} | {'Value Loss':>11}")
    print("-" * 90)

    while total_frames < TOTAL_FRAMES:
        update += 1
        total_frames += frames_per_update

        # collect rollouts
        exps, collect_logs = algo.collect_experiences()

        # PPO update
        update_logs = algo.update_parameters(exps)

        # stats
        avg_return = np.mean(collect_logs["return_per_episode"])
        avg_steps  = np.mean(collect_logs["num_frames_per_episode"])
        entropy    = np.mean(update_logs["entropy"])
        policy_loss = np.mean(update_logs["policy_loss"])
        value_loss = np.mean(update_logs["value_loss"])

        elapsed = time.time() - start_time

        # console output
        print(f"{update:>7} | {total_frames:>10,} | {avg_return:>11.3f} | "
              f"{avg_steps:>10.1f} | {entropy:>8.4f} | {policy_loss:>12.4f} | "
              f"{value_loss:>11.4f}")

        # write to CSV
        row = {"update": update, "frames": total_frames,
               "avg_return": f"{avg_return:.6f}", "avg_steps": f"{avg_steps:.1f}",
               "entropy": f"{entropy:.6f}", "policy_loss": f"{policy_loss:.6f}",
               "value_loss": f"{value_loss:.6f}", "elapsed_sec": f"{elapsed:.1f}"}
        csv_writer.writerow(row)
        csv_file.flush()

        for k, v in row.items():
            history[k].append(float(v))

        # periodic checkpoint
        if update % CHECKPOINT_EVERY == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab.word2idx,
                "update": update,
                "total_frames": total_frames,
            }, CHECKPOINT_PATH)
            print(f"         → Checkpoint saved to {CHECKPOINT_PATH}")

        # periodic plot update
        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR)
            print(f"         → Plots updated in {PLOT_DIR}")

    # final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.word2idx,
        "update": update,
        "total_frames": total_frames,
    }, CHECKPOINT_PATH)
    csv_file.close()
    save_plots(history, PLOT_DIR)
    print(f"\n Training complete.")
    print(f" Final checkpoint : {CHECKPOINT_PATH}")
    print(f" Metrics CSV      : {CSV_PATH}")
    print(f" Plots            : {PLOT_DIR}")
    print(f" Total updates: {update}, Total frames: {total_frames:,}")


if __name__ == "__main__":
    main()
