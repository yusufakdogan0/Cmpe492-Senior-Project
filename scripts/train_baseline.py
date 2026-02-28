"""
Training script for the Baseline PPO agent on MiniGrid-DoorKey-5x5-v0.

This is the "Base" method: a standard PPO agent that learns WITHOUT
LLM-generated subgoals.  It serves as the control condition against
which the LGRL agent will be compared.

Usage
-----
    python scripts/train_baseline.py

Hyperparameters follow the LGRL reference paper:
    lr=1e-4, discount=0.99, gae_lambda=0.95, clip_eps=0.2,
    entropy_coef=0.01, value_loss_coef=0.5
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
import minigrid                  # registers MiniGrid envs with gymnasium

# ── resolve project root (one level up from scripts/) ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch_ac

from models.baseline_agent import BaselineAgent, Vocabulary

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

ENV_NAME           = "MiniGrid-DoorKey-5x5-v0"
NUM_ENVS           = 16          # parallel environments
NUM_FRAMES_PER_PROC = 128        # rollout length per env per update
TOTAL_FRAMES       = 1_000_000   # total training frames

# Paper hyperparameters
LR                 = 1e-4
DISCOUNT           = 0.99
GAE_LAMBDA         = 0.95
CLIP_EPS           = 0.2
ENTROPY_COEF       = 0.01
VALUE_LOSS_COEF    = 0.5
MAX_GRAD_NORM      = 0.5
EPOCHS             = 4           # PPO epochs per update
BATCH_SIZE         = 256         # PPO mini-batch size
RECURRENCE         = 4           # BPTT window for recurrence

CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY   = 10          # save every N updates
CHECKPOINT_PATH    = os.path.join(CHECKPOINT_DIR, "baseline.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Observation Preprocessor
# ──────────────────────────────────────────────
# MiniGrid observations are dicts: {"image": ndarray(7,7,3),
# "direction": int, "mission": str}.  torch-ac's default
# preprocessor only handles plain tensors, so we need a custom one
# that converts images to float CHW tensors and tokenizes missions.

def make_preprocess_obss(vocab: Vocabulary, device=None):
    """
    Return a preprocess_obss function closed over `vocab`.

    The returned function accepts a list of observation dicts
    (one per env) and returns a DictList with `.image` and `.text`
    tensors, ready for the model's forward pass.
    """

    def preprocess_obss(obss, device=device):
        # obss is a list of dicts, each with keys: image, direction, mission

        # --- image: (B, 7, 7, 3) uint8  →  (B, 3, 7, 7) float32 ---
        images = np.array([obs["image"] for obs in obss])
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images.permute(0, 3, 1, 2)       # NHWC → NCHW
        images = images / 255.0                    # normalise to [0, 1]

        # --- mission text → tokenised integers ---
        missions = [obs["mission"] for obs in obss]
        token_ids = [vocab.tokenize(m) for m in missions]
        texts = torch.tensor(token_ids, dtype=torch.long, device=device)

        return torch_ac.DictList({"image": images, "text": texts})

    return preprocess_obss


# ──────────────────────────────────────────────
# Environment Factory
# ──────────────────────────────────────────────

def make_env(env_name: str, seed: int):
    """Create a single MiniGrid environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Baseline PPO Training — MiniGrid-DoorKey-5x5-v0")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Envs   : {NUM_ENVS} parallel")
    print(f"  Frames : {TOTAL_FRAMES:,}")
    print("=" * 60)

    # ---- create environments ----
    envs = [make_env(ENV_NAME, seed=i) for i in range(NUM_ENVS)]

    # ---- build vocabulary (pre-seed with a reset to capture mission words) ----
    vocab = Vocabulary()
    sample_obs, _ = gym.make(ENV_NAME).reset()
    vocab.tokenize(sample_obs["mission"])   # pre-seed vocab

    # ---- build model ----
    obs_space  = envs[0].observation_space
    act_space  = envs[0].action_space
    model = BaselineAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Action space : {act_space.n} actions")
    print(f"  Model params : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # ---- preprocessor ----
    preprocess_obss = make_preprocess_obss(vocab, device=DEVICE)

    # ---- PPO algorithm ----
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

    # ---- training loop ----
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    update = 0
    total_frames = 0
    frames_per_update = NUM_ENVS * NUM_FRAMES_PER_PROC

    print(f"\n{'Update':>7} | {'Frames':>10} | {'Avg Return':>11} | {'Avg Steps':>10} | "
          f"{'Entropy':>8} | {'Policy Loss':>12} | {'Value Loss':>11}")
    print("-" * 90)

    while total_frames < TOTAL_FRAMES:
        update += 1
        total_frames += frames_per_update

        # 1. Collect rollout experiences
        exps, collect_logs = algo.collect_experiences()

        # 2. Update policy with PPO
        update_logs = algo.update_parameters(exps)

        # 3. Compute summary statistics
        avg_return = np.mean(collect_logs["return_per_episode"])
        avg_steps  = np.mean(collect_logs["num_frames_per_episode"])
        entropy    = np.mean(update_logs["entropy"])
        policy_loss = np.mean(update_logs["policy_loss"])
        value_loss = np.mean(update_logs["value_loss"])

        # 4. Log
        print(f"{update:>7} | {total_frames:>10,} | {avg_return:>11.3f} | "
              f"{avg_steps:>10.1f} | {entropy:>8.4f} | {policy_loss:>12.4f} | "
              f"{value_loss:>11.4f}")

        # 5. Checkpoint
        if update % CHECKPOINT_EVERY == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab.word2idx,
                "update": update,
                "total_frames": total_frames,
            }, CHECKPOINT_PATH)
            print(f"         → Checkpoint saved to {CHECKPOINT_PATH}")

    # ---- final save ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.word2idx,
        "update": update,
        "total_frames": total_frames,
    }, CHECKPOINT_PATH)
    print(f"\n Training complete. Final checkpoint saved to {CHECKPOINT_PATH}")
    print(f" Total updates: {update}, Total frames: {total_frames:,}")


if __name__ == "__main__":
    main()
