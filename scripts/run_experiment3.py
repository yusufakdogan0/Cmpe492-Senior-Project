"""
run_experiment3.py — Decaying subgoal budget experiment.

Hypothesis: large MAX_SUBGOAL_STEPS speeds up early convergence but caps
the final reward; a schedule that starts large and decays should combine
the benefits of both.

Conditions
----------
DoorKey-5x5  (1 run — fixed-budget baselines already in experiment 1):
  decay_5x5:  500 → 250 → 100 → 25  over 10M frames

DoorKey-8x8  (3 fixed + 1 decay = 4 runs):
  8x8_budget_3200:  fixed 3200  (640×5)
  8x8_budget_2560:  fixed 2560  (640×4)
  8x8_budget_1920:  fixed 1920  (640×3)
  decay_8x8:        3200 → 2560 → 1920 → 1280 → 640  over 10M frames

Output structure
----------------
  logs/experiment3/<condition>/metrics.csv
  logs/experiment3/<condition>/training_curves.png
  checkpoints/experiment3/<condition>.pt

Usage
-----
    python scripts/run_experiment3.py
    python scripts/run_experiment3.py --total-frames 2000000   # shorter run
"""

from __future__ import annotations

import argparse
import csv
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
from utils.rule_based_planner import RuleBasedPlanner
from utils.subgoal_tracker import SubgoalTracker

# ---------------------------------------------------------------------------
# Fixed hyperparameters (same across all conditions)
# ---------------------------------------------------------------------------

NUM_ENVS            = 16
NUM_FRAMES_PER_PROC = 128
LR                  = 1e-4
DISCOUNT            = 0.99
GAE_LAMBDA          = 0.95
CLIP_EPS            = 0.2
ENTROPY_COEF        = 0.01
VALUE_LOSS_COEF     = 0.5
MAX_GRAD_NORM       = 0.5
EPOCHS              = 4
BATCH_SIZE          = 256
RECURRENCE          = 4

# Reward parameters (fixed)
R_MISSION           = 0.5
R_SUBGOAL           = 0.5
MISSION_TIME_COEF   = 0.5
SUBGOAL_TIME_COEF   = 0.5
SUBGOAL_TIMEOUT_MULT = 2.0
N_SUBGOALS          = RuleBasedPlanner.NUM_STAGES  # 5

CHECKPOINT_EVERY    = 10
PLOT_EVERY          = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Register missing DoorKey sizes
# ---------------------------------------------------------------------------

def ensure_env_registered(size: int):
    env_name = f"MiniGrid-DoorKey-{size}x{size}-v0"
    if env_name not in gym.registry:
        gym.register(
            id=env_name,
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": size},
        )
    return env_name

# ---------------------------------------------------------------------------
# Budget schedule helpers
# ---------------------------------------------------------------------------

def make_fixed_schedule(budget: int):
    """Return a schedule function that always returns the same budget."""
    def schedule(frames_done: int, total_frames: int) -> int:
        return budget
    return schedule


def make_decay_schedule(steps: list[tuple[float, int]]):
    """Build a piecewise-constant schedule from (fraction, budget) pairs.

    ``steps`` is a list of (end_fraction, budget) sorted by end_fraction.
    The budget for a given frame count is the first entry whose
    end_fraction has not yet been exceeded.

    Example for 5x5:
        [(0.25, 500), (0.50, 250), (0.75, 100), (1.0, 25)]
    means 0–25% → 500, 25–50% → 250, etc.
    """
    def schedule(frames_done: int, total_frames: int) -> int:
        frac = frames_done / total_frames if total_frames > 0 else 0.0
        for end_frac, budget in steps:
            if frac < end_frac:
                return budget
        # Past the last boundary — use the final budget
        return steps[-1][1]
    return schedule

# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

def build_conditions():
    conditions: list[dict] = []

    # --- DoorKey-5x5: decay only (fixed baselines from experiment 1) ---
    env_5x5 = ensure_env_registered(5)
    conditions.append({
        "name": "5x5_decay_500_250_100_25",
        "env_name": env_5x5,
        "size": 5,
        "max_env_steps": 250,                          # 5*5*10
        "schedule": make_decay_schedule([
            (0.25, 500),   # 0   – 2.5M
            (0.50, 250),   # 2.5 – 5M
            (0.75, 100),   # 5   – 7.5M
            (1.00, 25),    # 7.5 – 10M
        ]),
        "schedule_desc": "500→250→100→25 (quarters)",
    })

    # --- DoorKey-8x8: 3 fixed baselines + 1 decay ---
    env_8x8 = ensure_env_registered(8)
    max_env_8 = 8 * 8 * 10                              # 640

    for mult in [5, 4, 3]:
        budget = 640 * mult
        conditions.append({
            "name": f"8x8_budget_{budget}",
            "env_name": env_8x8,
            "size": 8,
            "max_env_steps": max_env_8,
            "schedule": make_fixed_schedule(budget),
            "schedule_desc": f"fixed {budget}",
        })

    conditions.append({
        "name": "8x8_decay_3200_to_640",
        "env_name": env_8x8,
        "size": 8,
        "max_env_steps": max_env_8,
        "schedule": make_decay_schedule([
            (0.20, 3200),  # 0 – 2M
            (0.40, 2560),  # 2 – 4M
            (0.60, 1920),  # 4 – 6M
            (0.80, 1280),  # 6 – 8M
            (1.00, 640),   # 8 – 10M
        ]),
        "schedule_desc": "3200→2560→1920→1280→640 (fifths)",
    })

    return conditions

# ---------------------------------------------------------------------------
# HierarchyState (budget is mutable via .max_subgoal_steps)
# ---------------------------------------------------------------------------

class HierarchyState:
    def __init__(self, num_envs, planner, envs, initial_budget):
        self.num_envs = num_envs
        self.planner = planner
        self.envs = envs
        self.max_subgoal_steps = initial_budget      # updated each tick
        self._init_lists()

    def _init_lists(self):
        self.active_subgoals = [""] * self.num_envs
        self.stage_indices = [0] * self.num_envs
        self.step_counters = [0] * self.num_envs
        self.episode_steps = [0] * self.num_envs
        self.trackers = [SubgoalTracker() for _ in range(self.num_envs)]

    def init_env_subgoal(self, env_idx, obs):
        uw = self.envs[env_idx].unwrapped
        env_json = parse_env_description(obs["image"], uw.carrying)
        subgoal, new_stage = self.planner.get_subgoal(
            obs["mission"], env_json, obs.get("direction", 0), stage_index=0,
        )
        self.active_subgoals[env_idx] = subgoal
        self.stage_indices[env_idx] = new_stage

    def reset_env(self, env_idx):
        self.active_subgoals[env_idx] = ""
        self.stage_indices[env_idx] = 0
        self.step_counters[env_idx] = 0
        self.episode_steps[env_idx] = 0
        self.trackers[env_idx].reset()

    def subgoal_budget(self, env_idx):
        i = min(self.stage_indices[env_idx] + 1, N_SUBGOALS)
        return (i / N_SUBGOALS) * self.max_subgoal_steps

    def advance(self, env_idx, obs):
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
# Preprocessing
# ---------------------------------------------------------------------------

def make_preprocess_obss(vocab, hierarchy_state, device):
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

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def make_reshape_reward(hierarchy_state, max_env_steps):
    def reshape_reward(obs, action, reward, done):
        env_idx = reshape_reward._current_env_idx
        reshape_reward._current_env_idx = (env_idx + 1) % hierarchy_state.num_envs

        total_reward = 0.0
        hierarchy_state.step_counters[env_idx] += 1
        hierarchy_state.episode_steps[env_idx] += 1

        if done:
            success = reward > 0
            if success:
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / max_env_steps, 1.0)
                total_reward += R_MISSION * (1.0 - MISSION_TIME_COEF * ratio)
            hierarchy_state.reset_env(env_idx)
            return total_reward

        if hierarchy_state.stage_indices[env_idx] >= N_SUBGOALS:
            return total_reward

        if not hierarchy_state.active_subgoals[env_idx]:
            hierarchy_state.init_env_subgoal(env_idx, obs)

        uw = hierarchy_state.envs[env_idx].unwrapped
        subgoal = hierarchy_state.active_subgoals[env_idx]
        completed = hierarchy_state.trackers[env_idx].check_completion(
            subgoal, uw, action, obs_image=obs["image"],
        )
        t_used = hierarchy_state.step_counters[env_idx]
        t_budget = hierarchy_state.subgoal_budget(env_idx)
        timed_out = t_used > SUBGOAL_TIMEOUT_MULT * t_budget

        if completed:
            ratio = min(t_used / max(t_budget, 1), SUBGOAL_TIMEOUT_MULT)
            r_i = max(R_SUBGOAL * (1.0 - SUBGOAL_TIME_COEF * ratio), 0.0)
            total_reward += r_i / N_SUBGOALS
            hierarchy_state.advance(env_idx, obs)
        elif timed_out:
            hierarchy_state.advance(env_idx, obs)

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


def save_plots(history, plot_path, title):
    frames = history["frames"]
    if len(frames) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

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
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------

def run_condition(cond: dict, total_frames: int, log_root: str, ckpt_root: str):
    name = cond["name"]
    env_name = cond["env_name"]
    max_env_steps = cond["max_env_steps"]
    schedule_fn = cond["schedule"]
    schedule_desc = cond["schedule_desc"]

    # Directories
    log_dir = os.path.join(log_root, name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)
    csv_path = os.path.join(log_dir, "metrics.csv")
    plot_path = os.path.join(log_dir, "training_curves.png")
    ckpt_path = os.path.join(ckpt_root, f"{name}.pt")

    print("\n" + "=" * 70)
    print(f"  Condition: {name}")
    print(f"  Env: {env_name}  |  MAX_ENV_STEPS: {max_env_steps}")
    print(f"  Budget schedule: {schedule_desc}")
    print(f"  Total frames: {total_frames:,}")
    print("=" * 70)

    # Create envs
    envs = []
    for i in range(NUM_ENVS):
        env = gym.make(env_name)
        env.reset(seed=i)
        envs.append(env)

    # Vocab
    vocab = Vocabulary()
    sample_obs, _ = gym.make(env_name).reset()
    vocab.tokenize(
        f"{sample_obs['mission']} [SEP] search for the yellow key", max_len=32
    )

    # Model
    obs_space = envs[0].observation_space
    act_space = envs[0].action_space
    model = LGRLAgent(obs_space, act_space, vocab)
    model.to(DEVICE)

    print(f"  Model: {model.__class__.__name__}  |  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Planner & hierarchy
    planner = RuleBasedPlanner()
    initial_budget = schedule_fn(0, total_frames)
    hierarchy_state = HierarchyState(NUM_ENVS, planner, envs, initial_budget)

    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, DEVICE)
    reshape_reward = make_reshape_reward(hierarchy_state, max_env_steps)

    # PPO
    algo = torch_ac.PPOAlgo(
        envs=envs, acmodel=model, device=DEVICE,
        num_frames_per_proc=NUM_FRAMES_PER_PROC,
        discount=DISCOUNT, lr=LR, gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF,
        max_grad_norm=MAX_GRAD_NORM, recurrence=RECURRENCE,
        clip_eps=CLIP_EPS, epochs=EPOCHS, batch_size=BATCH_SIZE,
        preprocess_obss=preprocess_obss,
        reshape_reward=reshape_reward,
    )

    # Training loop
    csv_fields = ["update", "frames", "avg_return", "avg_steps",
                  "entropy", "policy_loss", "value_loss",
                  "max_subgoal_steps", "elapsed_sec"]
    history = {k: [] for k in csv_fields}

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    update = 0
    frames_done = 0
    frames_per_update = NUM_ENVS * NUM_FRAMES_PER_PROC
    start_time = time.time()
    prev_budget = initial_budget

    print(f"  Initial budget: {initial_budget}")
    print(f"\n{'Update':>7} | {'Frames':>10} | {'Avg Return':>11} | {'Avg Steps':>10} | "
          f"{'Entropy':>8} | {'P.Loss':>8} | {'V.Loss':>8} | {'Budget':>7}")
    print("-" * 90)

    while frames_done < total_frames:
        update += 1
        frames_done += frames_per_update

        # --- Update budget from schedule ---
        new_budget = schedule_fn(frames_done, total_frames)
        if new_budget != prev_budget:
            print(f"  >>> Budget changed: {prev_budget} -> {new_budget} "
                  f"at frame {frames_done:,}")
            prev_budget = new_budget
        hierarchy_state.max_subgoal_steps = new_budget

        reshape_reward._current_env_idx = 0
        exps, collect_logs = algo.collect_experiences()
        update_logs = algo.update_parameters(exps)

        avg_return  = np.mean(collect_logs["return_per_episode"])
        avg_steps   = np.mean(collect_logs["num_frames_per_episode"])
        entropy     = np.mean(update_logs["entropy"])
        policy_loss = np.mean(update_logs["policy_loss"])
        value_loss  = np.mean(update_logs["value_loss"])
        elapsed     = time.time() - start_time

        if update % 10 == 0 or update == 1:
            print(f"{update:>7} | {frames_done:>10,} | {avg_return:>11.3f} | "
                  f"{avg_steps:>10.1f} | {entropy:>8.4f} | {policy_loss:>8.4f} | "
                  f"{value_loss:>8.4f} | {new_budget:>7}")

        row = {
            "update": update, "frames": frames_done,
            "avg_return": f"{avg_return:.6f}", "avg_steps": f"{avg_steps:.1f}",
            "entropy": f"{entropy:.6f}", "policy_loss": f"{policy_loss:.6f}",
            "value_loss": f"{value_loss:.6f}",
            "max_subgoal_steps": new_budget,
            "elapsed_sec": f"{elapsed:.1f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()
        for k, v in row.items():
            history[k].append(float(v))

        if update % CHECKPOINT_EVERY == 0:
            torch.save({"model_state_dict": model.state_dict(),
                         "update": update, "total_frames": frames_done}, ckpt_path)

        if update % PLOT_EVERY == 0:
            save_plots(history, plot_path, f"Exp3: {name}")

    # Final save
    torch.save({"model_state_dict": model.state_dict(),
                 "update": update, "total_frames": frames_done}, ckpt_path)
    csv_file.close()
    save_plots(history, plot_path, f"Exp3: {name}")

    final_return = (np.mean(history["avg_return"][-20:])
                    if len(history["avg_return"]) >= 20 else avg_return)
    final_steps  = (np.mean(history["avg_steps"][-20:])
                    if len(history["avg_steps"]) >= 20 else avg_steps)
    elapsed = time.time() - start_time

    print(f"\n  Done: {name}  |  Final avg return: {final_return:.4f}  |  "
          f"Final avg steps: {final_steps:.1f}  |  Time: {elapsed:.0f}s")

    return {"name": name, "schedule": schedule_desc,
            "final_return": final_return, "final_steps": final_steps,
            "elapsed": elapsed}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Decaying Subgoal Budget Schedule"
    )
    parser.add_argument("--total-frames", type=int, default=10_000_000)
    args = parser.parse_args()

    log_root = os.path.join(PROJECT_ROOT, "logs", "experiment3")
    ckpt_root = os.path.join(PROJECT_ROOT, "checkpoints", "experiment3")

    conditions = build_conditions()

    print("=" * 70)
    print("  EXPERIMENT 3: Decaying Subgoal Budget Schedule")
    print(f"  Device: {DEVICE}")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Frames per condition: {args.total_frames:,}")
    print("=" * 70)
    for c in conditions:
        print(f"    {c['name']:35s}  schedule={c['schedule_desc']}")
    print("=" * 70)

    results = []
    for i, cond in enumerate(conditions):
        print(f"\n>>> Condition {i+1}/{len(conditions)}: {cond['name']}")
        result = run_condition(cond, args.total_frames, log_root, ckpt_root)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3 — SUMMARY")
    print("=" * 70)
    print(f"  {'Condition':35s} | {'Schedule':35s} | {'Avg Return':>11} | "
          f"{'Avg Steps':>10} | {'Time (s)':>9}")
    print("-" * 110)
    for r in results:
        print(f"  {r['name']:35s} | {r['schedule']:35s} | "
              f"{r['final_return']:>11.4f} | {r['final_steps']:>10.1f} | "
              f"{r['elapsed']:>9.0f}")
    print("=" * 70)

    # Save summary CSV
    summary_path = os.path.join(log_root, "summary.csv")
    os.makedirs(log_root, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "schedule", "final_return", "final_steps", "elapsed",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
