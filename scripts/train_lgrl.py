"""
Training script for the LGRL agent on MiniGrid-DoorKey-5x5-v0.

Bi-level hierarchy:
  - High level: LLM (Qwen 2.5 7B) generates subgoals from JSON observations
  - Low level:  PPO agent executes subgoals, conditioned on "mission [SEP] subgoal"

Reward scaffolding:
  - R_m = 0.5 (mission completion, scaled by speed)
  - R_t = 0.5 (subgoal completion, scaled by speed)
  - Budget per subgoal: T_i = max_steps / n  (n=5 assumed)

Usage:
    python scripts/train_lgrl.py
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import gymnasium as gym
import minigrid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch_ac

from models.baseline_agent import Vocabulary
from models.lgrl_agent import LGRLAgent
from utils.env_parser import parse_env_description
from utils.llm_planner import LLMPlanner, IDX_TO_DIRECTION
from utils.subgoal_tracker import SubgoalTracker

# --- Config ---

ENV_NAME           = "MiniGrid-DoorKey-5x5-v0"
NUM_ENVS           = 16
NUM_FRAMES_PER_PROC = 128
TOTAL_FRAMES       = 10_000_000

# PPO hyperparameters
LR                 = 1e-4
DISCOUNT           = 0.99
GAE_LAMBDA         = 0.95
CLIP_EPS           = 0.2
ENTROPY_COEF       = 0.01
VALUE_LOSS_COEF    = 0.5
MAX_GRAD_NORM      = 0.5
EPOCHS             = 4
BATCH_SIZE         = 256
RECURRENCE         = 4

# Reward scaffolding
R_MISSION          = 0.5       # max reward for mission completion
R_SUBGOAL          = 0.5       # max reward per subgoal
N_SUBGOALS_EST     = 5         # estimated subgoals per episode
MAX_ENV_STEPS      = 100       # DoorKey-5x5 max steps (5*5*4 actually but ~100 typical)

# Subgoal step budget: T_i = MAX_ENV_STEPS / N_SUBGOALS_EST
SUBGOAL_BUDGET     = MAX_ENV_STEPS // N_SUBGOALS_EST  # 20 steps per subgoal

CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY   = 10
CHECKPOINT_PATH    = os.path.join(CHECKPOINT_DIR, "lgrl.pt")

LOG_DIR            = os.path.join(PROJECT_ROOT, "logs")
CSV_PATH           = os.path.join(LOG_DIR, "lgrl_metrics.csv")
PLOT_DIR           = os.path.join(LOG_DIR, "plots")
PLOT_EVERY         = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Per-env state for the bi-level hierarchy ---

class HierarchyState:
    """Tracks active subgoals, step counters, and history for all parallel envs."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.reset_all()

    def reset_all(self):
        self.active_subgoals = ["explore"] * self.num_envs
        self.step_counters = [0] * self.num_envs
        self.histories = [[] for _ in range(self.num_envs)]
        self.trackers = [SubgoalTracker() for _ in range(self.num_envs)]
        self.episode_steps = [0] * self.num_envs

    def reset_env(self, env_idx):
        """Reset state for a single env (on episode end)."""
        self.active_subgoals[env_idx] = "explore"
        self.step_counters[env_idx] = 0
        self.histories[env_idx] = []
        self.trackers[env_idx].reset()
        self.episode_steps[env_idx] = 0


# --- Observation preprocessing ---
# Tokenizes "mission [SEP] subgoal" as the text input

def make_preprocess_obss(vocab, hierarchy_state, device=None):
    """
    Returns preprocessor that concatenates mission + subgoal with [SEP].
    The hierarchy_state provides the current subgoal for each env.
    """

    def preprocess_obss(obss, device=device):
        # image
        images = np.array([obs["image"] for obs in obss])
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images.permute(0, 3, 1, 2)
        images = images / 255.0

        # text: "mission [SEP] subgoal"
        token_ids = []
        for i, obs in enumerate(obss):
            mission = obs["mission"]
            subgoal = hierarchy_state.active_subgoals[i] if i < hierarchy_state.num_envs else "explore"
            combined = f"{mission} [SEP] {subgoal}"
            token_ids.append(vocab.tokenize(combined, max_len=32))

        texts = torch.tensor(token_ids, dtype=torch.long, device=device)

        return torch_ac.DictList({"image": images, "text": texts})

    return preprocess_obss


# --- Env factory ---

def make_env(env_name: str, seed: int):
    """Create a single MiniGrid environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


# --- Reward shaping callback ---

def make_reshape_reward(hierarchy_state, planner, envs):
    """
    Returns a reshape_reward function for torch-ac.
    Called per (obs, action, reward, done) tuple for each env.

    Handles:
      1. Subgoal completion checking + intrinsic reward
      2. Subgoal timeout (T_used > 2*T_i) → mark failed, query LLM
      3. Mission completion → scale reward by speed
      4. Episode reset → reset hierarchy state
    """

    def reshape_reward(obs, action, reward, done):
        # figure out which env this is from the observation
        # torch-ac calls this per env, so we need env_idx
        # Unfortunately torch-ac doesn't pass env_idx, so we track
        # via the internal counter pattern
        env_idx = reshape_reward._current_env_idx
        reshape_reward._current_env_idx += 1
        if reshape_reward._current_env_idx >= hierarchy_state.num_envs:
            reshape_reward._current_env_idx = 0

        total_reward = 0.0
        hierarchy_state.step_counters[env_idx] += 1
        hierarchy_state.episode_steps[env_idx] += 1

        # check subgoal completion
        uw = envs[env_idx].unwrapped
        subgoal = hierarchy_state.active_subgoals[env_idx]
        completed = hierarchy_state.trackers[env_idx].check_completion(subgoal, uw, action)
        t_used = hierarchy_state.step_counters[env_idx]
        timed_out = t_used > 2 * SUBGOAL_BUDGET

        if completed:
            # intrinsic reward: r_i = R_t * (1 - 0.5 * T_used / T_i)
            ratio = min(t_used / SUBGOAL_BUDGET, 2.0)
            r_i = R_SUBGOAL * (1.0 - 0.5 * ratio)
            r_i = max(r_i, 0.0)
            total_reward += r_i

            # log and get new subgoal
            hierarchy_state.histories[env_idx].append({
                "subgoal": subgoal, "status": "Success", "steps": t_used
            })
            _query_new_subgoal(env_idx, uw, obs, hierarchy_state, planner)

        elif timed_out:
            # mark as failed, no reward
            hierarchy_state.histories[env_idx].append({
                "subgoal": subgoal, "status": "Failed", "steps": t_used
            })
            _query_new_subgoal(env_idx, uw, obs, hierarchy_state, planner)

        # mission completion
        if done:
            if reward > 0:
                # scale mission reward by speed
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / MAX_ENV_STEPS, 1.0)
                r_m = R_MISSION * (1.0 - 0.5 * ratio)
                total_reward += r_m
            # reset hierarchy state for this env
            hierarchy_state.reset_env(env_idx)
        else:
            # no mission reward if not done
            pass

        return total_reward

    # internal counter to track which env is being processed
    reshape_reward._current_env_idx = 0

    return reshape_reward


def _query_new_subgoal(env_idx, uw, obs, hierarchy_state, planner):
    """Query the LLM for a new subgoal and update hierarchy state."""
    env_json = parse_env_description(obs["image"], uw.carrying)
    direction = obs.get("direction", 0)
    past = [h["subgoal"] for h in hierarchy_state.histories[env_idx]]

    new_subgoal = planner.get_subgoal(
        obs["mission"], env_json, direction, past
    )

    hierarchy_state.active_subgoals[env_idx] = new_subgoal
    hierarchy_state.step_counters[env_idx] = 0


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
    """Generate and save training curves."""
    frames = history["frames"]
    if len(frames) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("LGRL PPO — Training Curves", fontsize=14, fontweight="bold")

    # 1. Average Return
    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.3, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Average Return")
    ax.set_title("Average Return per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Average Steps
    ax = axes[0, 1]
    ax.plot(frames, history["avg_steps"], alpha=0.3, color="tab:orange", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_steps"]), color="tab:orange", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Average Steps")
    ax.set_title("Average Steps per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Entropy
    ax = axes[1, 0]
    ax.plot(frames, history["entropy"], color="tab:green", linewidth=1)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # 4. Losses
    ax = axes[1, 1]
    ax.plot(frames, _smooth(history["policy_loss"]), color="tab:red", linewidth=1.5, label="Policy Loss")
    ax.plot(frames, _smooth(history["value_loss"]), color="tab:purple", linewidth=1.5, label="Value Loss")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Loss")
    ax.set_title("Policy & Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "lgrl_training_curves.png"), dpi=150)
    plt.close(fig)


# --- Main training loop ---

def main():
    print("=" * 60)
    print("  LGRL PPO Training — MiniGrid-DoorKey-5x5-v0")
    print("=" * 60)
    print(f"  Device       : {DEVICE}")
    print(f"  Envs         : {NUM_ENVS} parallel")
    print(f"  Frames       : {TOTAL_FRAMES:,}")
    print(f"  R_mission    : {R_MISSION}")
    print(f"  R_subgoal    : {R_SUBGOAL}")
    print(f"  Subgoal budget: {SUBGOAL_BUDGET} steps")
    print("=" * 60)

    # create parallel environments
    envs = [make_env(ENV_NAME, seed=i) for i in range(NUM_ENVS)]

    # build vocabulary — make sure [SEP] gets a token
    vocab = Vocabulary()
    sample_obs, _ = gym.make(ENV_NAME).reset()
    vocab.tokenize(f"{sample_obs['mission']} [SEP] explore", max_len=32)

    # build model
    obs_space  = envs[0].observation_space
    act_space  = envs[0].action_space
    model = LGRLAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Action space : {act_space.n} actions")
    print(f"  Model params : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # hierarchy state + LLM planner
    hierarchy_state = HierarchyState(NUM_ENVS)
    planner = LLMPlanner()

    # preprocessor (needs hierarchy_state for subgoal access)
    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, device=DEVICE)

    # reward shaping callback
    reshape_reward = make_reshape_reward(hierarchy_state, planner, envs)

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
        reshape_reward     = reshape_reward,
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

    history = {k: [] for k in csv_fields}

    print(f"\n  Logging to  : {CSV_PATH}")
    print(f"  Plots saved : {PLOT_DIR}")
    print(f"\n{'Update':>7} | {'Frames':>10} | {'Avg Return':>11} | {'Avg Steps':>10} | "
          f"{'Entropy':>8} | {'Policy Loss':>12} | {'Value Loss':>11}")
    print("-" * 90)

    while total_frames < TOTAL_FRAMES:
        update += 1
        total_frames += frames_per_update

        # reset the reshape_reward env counter at the start of each rollout
        reshape_reward._current_env_idx = 0

        # collect rollouts (reshape_reward handles subgoal checking + LLM queries)
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
