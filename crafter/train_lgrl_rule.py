"""
train_lgrl_rule.py — LGRL PPO training on Crafter with the rule-based
oracle planner.

Crafter port of the MiniGrid project's ``scripts/train_lgrl_rule.py``.
Keeps the LGRL "spine" identical:

    planner -> subgoal -> tracker verifies -> reward scaffold -> PPO

Only the env I/O is Crafter-specific. The reward scaffolding, PPO
hyperparameters, forward-only stage logic, and per-subgoal time budgets
are the same as the MiniGrid scripts.

Usage:
    python crafter/train_lgrl_rule.py --task collect_wood
    python crafter/train_lgrl_rule.py --task make_stone_pickaxe --frames 5000000
    python crafter/train_lgrl_rule.py --task collect_diamond --subgoal-log
    python crafter/train_lgrl_rule.py --task collect_stone --resume
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

# Replace torch-ac's multiprocess ParallelEnv with the in-process stepper so
# the loop can read each env's unwrapped Crafter state every step.
torch_ac.algos.base.ParallelEnv = SequentialEnv

from crafter_env import CrafterTaskEnv
from crafter_parser import parse_crafter_description
from crafter_planner import RuleBasedCrafterPlanner
from crafter_tracker import CrafterSubgoalTracker
from crafter_agent import CrafterACModel, Vocabulary
from crafter_logger import CrafterSubgoalLogger
from crafter_tasks import (
    TASK_MISSION, SUPPORTED_TASKS, num_subgoals, task_max_steps,
    resolve_artifact_stem,
)

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

NUM_ENVS = 16
NUM_FRAMES_PER_PROC = 128
TOTAL_FRAMES = 3_000_000

# PPO hyperparameters — identical to the MiniGrid scripts.
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

# Reward scaffolding — identical to the MiniGrid scripts.
R_MISSION = 0.5
R_SUBGOAL = 0.5
MISSION_TIME_COEF = 0.5
SUBGOAL_TIME_COEF = 0.5
SUBGOAL_TIMEOUT_MULT = 2.0

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
CHECKPOINT_EVERY = 10
PLOT_EVERY = 25
SUCCESS_WINDOW = 200

BASE_ARTIFACT_STEM = "lgrl_rule"
PLANNER_TAG = "rule_based"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hierarchy state (per-env subgoal/stage bookkeeping)
# ---------------------------------------------------------------------------

class HierarchyState:
    def __init__(self, envs, planner, n_subgoals_per_env, t_max_per_env,
                 family_per_env):
        self.envs = envs
        self.num_envs = len(envs)
        self.planner = planner
        self.n_subgoals_per_env = n_subgoals_per_env
        self.t_max_per_env = t_max_per_env
        self.family_per_env = family_per_env

        self.active_subgoals = [""] * self.num_envs
        self.stage_indices = [0] * self.num_envs
        self.step_counters = [0] * self.num_envs
        self.episode_steps = [0] * self.num_envs
        self.episode_raw_return = [0.0] * self.num_envs
        self.trackers = [CrafterSubgoalTracker() for _ in range(self.num_envs)]
        self.histories = [[] for _ in range(self.num_envs)]
        self.completed_episodes = []  # (family, raw_return, steps, success)

        # previous-step snapshots for achievement-diff completion checks.
        # Crafter achievement counters start at 0 and the tracker defaults
        # missing keys to 0, so empty snapshots are the correct baseline.
        # (Envs are reset by PPOAlgo *after* this object is built, so we
        # must not read env.last_state here.)
        self.prev_achievements = [{} for _ in range(self.num_envs)]
        self.prev_inventory = [{} for _ in range(self.num_envs)]

    def init_env_subgoal(self, env_idx, obs):
        uw = self.envs[env_idx].unwrapped
        env_json = parse_crafter_description(uw.last_state)
        subgoal, new_stage = self.planner.get_subgoal(
            obs["mission"], env_json, 0, stage_index=0)
        self.active_subgoals[env_idx] = subgoal
        self.stage_indices[env_idx] = new_stage

    def reset_env(self, env_idx):
        self.active_subgoals[env_idx] = ""
        self.stage_indices[env_idx] = 0
        self.step_counters[env_idx] = 0
        self.episode_steps[env_idx] = 0
        self.episode_raw_return[env_idx] = 0.0
        self.trackers[env_idx].reset()
        self.histories[env_idx] = []
        # snapshot the post-reset state of the (already auto-reset) env
        st = self.envs[env_idx].unwrapped.last_state
        self.prev_achievements[env_idx] = dict(st["achievements"])
        self.prev_inventory[env_idx] = dict(st["inventory"])

    def subgoal_budget(self, env_idx):
        n = self.n_subgoals_per_env[env_idx]
        t_max = self.t_max_per_env[env_idx]
        i = min(self.stage_indices[env_idx] + 1, n)
        return (i / n) * t_max

    def n_subgoals(self, env_idx):
        return self.n_subgoals_per_env[env_idx]

    def t_max(self, env_idx):
        return self.t_max_per_env[env_idx]

    def advance(self, env_idx, obs):
        next_stage = self.stage_indices[env_idx] + 1
        n = self.n_subgoals_per_env[env_idx]
        if next_stage >= n:
            self.stage_indices[env_idx] = n
            self.step_counters[env_idx] = 0
            return
        uw = self.envs[env_idx].unwrapped
        env_json = parse_crafter_description(uw.last_state)
        subgoal, new_stage = self.planner.get_subgoal(
            obs["mission"], env_json, 0, stage_index=next_stage)
        self.active_subgoals[env_idx] = subgoal
        self.stage_indices[env_idx] = new_stage
        self.step_counters[env_idx] = 0


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def make_preprocess_obss(vocab, hierarchy_state, use_subgoal=True, device=None):
    def preprocess_obss(obss, device=device):
        images = np.array([obs["image"] for obs in obss])
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images.permute(0, 3, 1, 2) / 255.0

        # During experience COLLECTION, torch_ac calls this with exactly
        # num_envs observations (one per env, in env order), so
        # active_subgoals[i] is the authoritative *current* subgoal for obs i.
        # We STAMP it onto the obs dict. torch_ac stores that very dict
        # (self.obss[i] = self.obs), so when it later re-preprocesses the
        # flattened P*T rollout batch for the PPO update, each stored obs
        # still carries the subgoal that was active when it was collected.
        #
        # Reading active_subgoals[i] positionally during the update is WRONG:
        # there i ranges over P*T (=2048), not over envs, so every entry with
        # i >= num_envs fell back to "collect wood" — i.e. ~99% of the update
        # batch was conditioned on a constant subgoal, nullifying LGRL's text
        # conditioning during every gradient step. Stamping fixes that:
        # the per-timestep subgoal now flows into the PPO loss correctly.
        collection_call = use_subgoal and (len(obss) == hierarchy_state.num_envs)

        token_ids = []
        for i, obs in enumerate(obss):
            if use_subgoal:
                if collection_call:
                    subgoal = hierarchy_state.active_subgoals[i] or "collect wood"
                    obs["subgoal"] = subgoal          # persists into stored obs
                else:
                    subgoal = obs.get("subgoal") or "collect wood"
                combined = f"{obs['mission']} [SEP] {subgoal}"
            else:
                combined = obs["mission"]
            token_ids.append(vocab.tokenize(combined, max_len=32))

        texts = torch.tensor(token_ids, dtype=torch.long, device=device)
        return torch_ac.DictList({"image": images, "text": texts})

    return preprocess_obss


def make_env(task, seed):
    return CrafterTaskEnv(task, seed=seed)


# ---------------------------------------------------------------------------
# Reward shaping — Crafter completion via achievement diffs
# ---------------------------------------------------------------------------

def make_reshape_reward(hierarchy_state, logger=None):
    def reshape_reward(obs, action, reward, done):
        env_idx = reshape_reward._current_env_idx
        reshape_reward._current_env_idx = (env_idx + 1) % hierarchy_state.num_envs

        total_reward = 0.0
        hierarchy_state.step_counters[env_idx] += 1
        hierarchy_state.episode_steps[env_idx] += 1
        hierarchy_state.episode_raw_return[env_idx] += float(reward)
        mission = obs.get("mission", "")

        n_subgoals = hierarchy_state.n_subgoals(env_idx)
        t_max = hierarchy_state.t_max(env_idx)

        if done:
            success = reward > 0
            if success:
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / t_max, 1.0)
                total_reward += R_MISSION * (1.0 - MISSION_TIME_COEF * ratio)
            if logger:
                logger.on_episode_end(env_idx, mission, success,
                                      hierarchy_state.episode_steps[env_idx])
            hierarchy_state.completed_episodes.append((
                hierarchy_state.family_per_env[env_idx],
                hierarchy_state.episode_raw_return[env_idx],
                hierarchy_state.episode_steps[env_idx],
                success,
            ))
            hierarchy_state.reset_env(env_idx)
            return total_reward

        if hierarchy_state.stage_indices[env_idx] >= n_subgoals:
            return total_reward

        uw = hierarchy_state.envs[env_idx].unwrapped
        cur = uw.last_state
        cur_inv = cur["inventory"]
        cur_ach = cur["achievements"]

        if not hierarchy_state.active_subgoals[env_idx]:
            hierarchy_state.init_env_subgoal(env_idx, obs)
            if logger:
                logger.log(env_idx, "init", mission=mission,
                           subgoal=hierarchy_state.active_subgoals[env_idx],
                           stage=hierarchy_state.stage_indices[env_idx],
                           budget=hierarchy_state.subgoal_budget(env_idx),
                           env_state=parse_crafter_description(cur))

        subgoal = hierarchy_state.active_subgoals[env_idx]
        completed = hierarchy_state.trackers[env_idx].check_completion(
            subgoal,
            inventory=cur_inv,
            achievements=cur_ach,
            prev_achievements=hierarchy_state.prev_achievements[env_idx],
            prev_inventory=hierarchy_state.prev_inventory[env_idx],
        )

        t_used = hierarchy_state.step_counters[env_idx]
        t_budget = hierarchy_state.subgoal_budget(env_idx)
        timed_out = t_used > SUBGOAL_TIMEOUT_MULT * t_budget

        if completed:
            ratio = min(t_used / max(t_budget, 1), SUBGOAL_TIMEOUT_MULT)
            r_i = max(R_SUBGOAL * (1.0 - SUBGOAL_TIME_COEF * ratio), 0.0)
            total_reward += r_i / n_subgoals
            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Success", "steps": t_used,
                 "stage": hierarchy_state.stage_indices[env_idx]})
            if logger:
                logger.log(env_idx, "completed", mission=mission, subgoal=subgoal,
                           stage=hierarchy_state.stage_indices[env_idx],
                           steps_used=t_used, budget=t_budget,
                           reward=r_i / n_subgoals)
            hierarchy_state.advance(env_idx, obs)
            if logger:
                logger.log(env_idx, "new", mission=mission,
                           subgoal=hierarchy_state.active_subgoals[env_idx],
                           stage=hierarchy_state.stage_indices[env_idx],
                           budget=hierarchy_state.subgoal_budget(env_idx),
                           env_state=parse_crafter_description(cur),
                           raw_llm=getattr(hierarchy_state.planner,
                                           "last_raw_response", None))
        elif timed_out:
            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Failed", "steps": t_used,
                 "stage": hierarchy_state.stage_indices[env_idx]})
            if logger:
                logger.log(env_idx, "timed_out", mission=mission, subgoal=subgoal,
                           stage=hierarchy_state.stage_indices[env_idx],
                           steps_used=t_used, budget=t_budget)
            hierarchy_state.advance(env_idx, obs)
            if logger:
                logger.log(env_idx, "new", mission=mission,
                           subgoal=hierarchy_state.active_subgoals[env_idx],
                           stage=hierarchy_state.stage_indices[env_idx],
                           budget=hierarchy_state.subgoal_budget(env_idx),
                           env_state=parse_crafter_description(cur),
                           raw_llm=getattr(hierarchy_state.planner,
                                           "last_raw_response", None))

        # update previous-step snapshots for the next achievement-diff check
        hierarchy_state.prev_achievements[env_idx] = dict(cur_ach)
        hierarchy_state.prev_inventory[env_idx] = dict(cur_inv)
        return total_reward

    reshape_reward._current_env_idx = 0
    return reshape_reward


# ---------------------------------------------------------------------------
# Plotting / checkpoint helpers
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
    fig.suptitle(f"LGRL PPO (rule oracle) -- Crafter:{task}", fontsize=14,
                 fontweight="bold")

    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.25, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.8)
    ax.set(xlabel="Frames", ylabel="Avg return", title="Average Return per Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(frames, history["success_rate"], alpha=0.25, color="tab:green", linewidth=0.5)
    ax.plot(frames, _smooth(history["success_rate"]), color="tab:green", linewidth=1.8)
    ax.set(xlabel="Frames", ylabel="Success rate", title=f"Rolling Success Rate (last {SUCCESS_WINDOW})")
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
        "planner": PLANNER_TAG,
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


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LGRL PPO training on Crafter (rule-based oracle planner).")
    p.add_argument("--task", default="collect_wood", choices=SUPPORTED_TASKS,
                   help="Target Crafter achievement to train on.")
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES,
                   help="Total training frames.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the matching checkpoint.")
    p.add_argument("--subgoal-log", action="store_true",
                   help="Write per-env subgoal JSONL logs.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    task = args.task
    mission = TASK_MISSION[task]
    n = num_subgoals(task)
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
    print("  LGRL PPO Training on Crafter (rule oracle)")
    print("=" * 70)
    print(f"  Task           : {task}")
    print(f"  Mission        : {mission!r}")
    print(f"  Subgoals (n)   : {n}")
    print(f"  T_max          : {t_max}")
    print(f"  Device         : {DEVICE}")
    print(f"  Frames         : {args.frames:,}")
    print(f"  Envs           : {NUM_ENVS}")
    print(f"  R_mission/sub  : {R_MISSION} / {R_SUBGOAL}")
    print("=" * 70)

    envs = [make_env(task, seed=args.seed * 1000 + i) for i in range(NUM_ENVS)]
    n_subgoals_per_env = [n] * NUM_ENVS
    t_max_per_env = [t_max] * NUM_ENVS
    family_per_env = [task] * NUM_ENVS

    planner = RuleBasedCrafterPlanner()
    hierarchy_state = HierarchyState(envs, planner, n_subgoals_per_env,
                                     t_max_per_env, family_per_env)

    vocab = Vocabulary()
    # pre-seed vocab so token ids are stable across runs/resumes
    vocab.tokenize(f"{mission} [SEP] collect wood")
    for s in ("collect", "place", "make", "table", "furnace", "wood", "stone",
              "coal", "iron", "diamond", "pickaxe", "sword", "a", "an",
              "1", "2", "4", "[SEP]"):
        _ = vocab[s]

    logger = (CrafterSubgoalLogger(LOG_DIR, artifact_stem, NUM_ENVS)
              if args.subgoal_log else None)

    obs_space = envs[0].observation_space
    model = CrafterACModel(obs_space, envs[0].action_space, vocab).to(DEVICE)

    preprocess = make_preprocess_obss(vocab, hierarchy_state, use_subgoal=True,
                                      device=DEVICE)
    reshape = make_reshape_reward(hierarchy_state, logger)

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

        for (_fam, _ret, _steps, success) in hierarchy_state.completed_episodes:
            success_buf.append(1 if success else 0)
        hierarchy_state.completed_episodes.clear()
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
    if logger:
        logger.close()
    print("\nTraining complete.")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metrics    : {csv_path}")
    print(f"  Plot       : {os.path.join(PLOT_DIR, f'{artifact_stem}_training_curves.png')}")


if __name__ == "__main__":
    main()
