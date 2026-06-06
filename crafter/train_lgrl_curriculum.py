"""
train_lgrl_curriculum.py — curriculum / transfer-learning PPO training for
LGRL on Crafter (rule-based oracle).

This is the Crafter analogue of the MiniGrid project's
``scripts/train_lgrl_curriculum.py`` and implements the transfer-learning
idea you asked for: train on a sequence of Crafter targets from EASIEST to
HARDEST, carrying the agent's weights + optimizer state forward across
stage boundaries so each harder target inherits everything learned on the
shallower ones.

This mirrors the paper's curriculum (§4.1 / §4.4): "first trained on the
simpler ... tasks to acquire basic navigation abilities. The curriculum
then progresses to increasingly ... complex configurations." In Crafter
the tech tree gives a natural difficulty ordering, because each later
target literally requires the skills of the earlier ones:

    collect_wood -> place_table -> make_wood_pickaxe -> collect_stone
    -> make_stone_pickaxe -> place_furnace -> collect_coal -> collect_iron
    -> make_iron_pickaxe -> collect_diamond

(The default curriculum is ``crafter_tasks.DEFAULT_CURRICULUM``.)

Each stage advances to the next when EITHER:
  - the rolling success rate (over the last ``--success-window`` episodes)
    has stayed >= ``--success-threshold`` for ``--success-stability``
    consecutive PPO updates (a single dip resets the counter — the
    "constantly above threshold, not once" rule), OR
  - frames spent on the current stage reach ``--max-frames-per-stage``.

Model parameters, vocabulary, and optimizer state are preserved across
stage boundaries; only the env workers and the per-stage config
(``n_subgoals``, ``T_max``, target) are rebuilt. Same PPO loop and reward
shaping (Eqs. 5-7) as ``train_lgrl_rule.py``.

Usage:
    # Full tech-tree curriculum (default), up to 1M frames per stage
    python crafter/train_lgrl_curriculum.py

    # Custom curriculum
    python crafter/train_lgrl_curriculum.py --curriculum collect_wood,place_table,make_wood_pickaxe

    # Tighter advancement
    python crafter/train_lgrl_curriculum.py --success-threshold 0.85 --success-stability 15

    # Cap a stage so wallclock stays predictable
    python crafter/train_lgrl_curriculum.py --max-frames-per-stage 500000

    # Resume (curriculum spec must match)
    python crafter/train_lgrl_curriculum.py --resume
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
from crafter_parser import parse_crafter_description
from crafter_planner import RuleBasedCrafterPlanner
from crafter_logger import CrafterSubgoalLogger
from crafter_agent import CrafterACModel, Vocabulary
from crafter_tasks import (
    TASK_MISSION, SUPPORTED_TASKS, DEFAULT_CURRICULUM,
    num_subgoals, task_max_steps, curriculum_artifact_stem,
)

# Reuse the tested LGRL machinery from the single-task script.
from train_lgrl_rule import (
    HierarchyState, make_preprocess_obss, make_reshape_reward,
    NUM_ENVS, NUM_FRAMES_PER_PROC, LR, DISCOUNT, GAE_LAMBDA, CLIP_EPS,
    BATCH_SIZE, ENTROPY_COEF, VALUE_LOSS_COEF, MAX_GRAD_NORM, EPOCHS,
    RECURRENCE, R_MISSION, R_SUBGOAL,
)

# ---------------------------------------------------------------------------
# Curriculum advancement defaults
# ---------------------------------------------------------------------------

DEFAULT_SUCCESS_THRESHOLD = 0.98
DEFAULT_SUCCESS_WINDOW = 200            # episodes
DEFAULT_SUCCESS_STABILITY = 10          # consecutive PPO updates above threshold
DEFAULT_MAX_FRAMES_PER_STAGE = 5_000_000

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
CHECKPOINT_EVERY = 1
PLOT_EVERY = 25

BASE_ARTIFACT_STEM = "lgrl_rule"
PLANNER_TAG = "rule_based"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Per-stage algo (re)builder — keeps the model, swaps the envs
# ---------------------------------------------------------------------------

def _build_algo_for_stage(model, vocab, planner, task, seed,
                          optimizer_state=None, logger=None):
    """Build envs + HierarchyState + PPOAlgo for one curriculum stage,
    reusing the existing model. Returns (algo, envs, hierarchy_state,
    reshape_reward, n, t_max)."""
    n = num_subgoals(task)
    t_max = task_max_steps(task)
    envs = [CrafterTaskEnv(task, seed=seed * 1000 + i) for i in range(NUM_ENVS)]

    hierarchy_state = HierarchyState(
        envs, planner, [n] * NUM_ENVS, [t_max] * NUM_ENVS, [task] * NUM_ENVS)

    preprocess = make_preprocess_obss(vocab, hierarchy_state,
                                      use_subgoal=True, device=DEVICE)
    reshape = make_reshape_reward(hierarchy_state, logger)

    algo = torch_ac.PPOAlgo(
        envs, model, device=DEVICE, num_frames_per_proc=NUM_FRAMES_PER_PROC,
        discount=DISCOUNT, lr=LR, gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF,
        max_grad_norm=MAX_GRAD_NORM, recurrence=RECURRENCE, clip_eps=CLIP_EPS,
        epochs=EPOCHS, batch_size=BATCH_SIZE, preprocess_obss=preprocess,
        reshape_reward=reshape,
    )
    if optimizer_state is not None:
        algo.optimizer.load_state_dict(optimizer_state)
    return algo, envs, hierarchy_state, reshape, n, t_max


# ---------------------------------------------------------------------------
# Plotting (curriculum-aware: vertical lines at stage boundaries)
# ---------------------------------------------------------------------------

def _smooth(values, window=20):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


def save_plots(history, plot_dir, artifact_stem, curriculum, stage_boundaries):
    frames = history["frames"]
    if len(frames) < 2:
        return
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"LGRL Curriculum (rule oracle) -- Crafter tech tree",
                 fontsize=14, fontweight="bold")

    def mark_boundaries(ax):
        for (_si, _task, fr) in stage_boundaries:
            ax.axvline(fr, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.25, color="tab:blue", linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue", linewidth=1.8)
    mark_boundaries(ax)
    ax.set(xlabel="Frames", ylabel="Avg return", title="Average Return per Episode")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(frames, history["rolling_success_rate"], alpha=0.25, color="tab:green", linewidth=0.5)
    ax.plot(frames, _smooth(history["rolling_success_rate"]), color="tab:green", linewidth=1.8)
    mark_boundaries(ax)
    ax.set(xlabel="Frames", ylabel="Success rate", title="Rolling Success Rate")
    ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(frames, history["stage_idx"], color="tab:purple", linewidth=1.5, drawstyle="steps-post")
    ax.set(xlabel="Frames", ylabel="Curriculum stage", title="Curriculum Progress")
    ax.set_yticks(range(len(curriculum)))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(frames, _smooth(history["policy_loss"]), color="tab:red", linewidth=1.5, label="Policy Loss")
    ax.plot(frames, _smooth(history["value_loss"]), color="tab:purple", linewidth=1.5, label="Value Loss")
    mark_boundaries(ax)
    ax.set(xlabel="Frames", ylabel="Loss", title="Policy & Value Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{artifact_stem}_training_curves.png"), dpi=150)
    plt.close(fig)


def _save_checkpoint(path, model, algo, vocab, update, total_frames,
                     curriculum, stage_idx, frames_in_stage, stage_boundaries):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": algo.optimizer.state_dict(),
        "vocab": vocab.word2idx,
        "update": update,
        "total_frames": total_frames,
        "planner": PLANNER_TAG,
        "curriculum": curriculum,
        "stage_idx": stage_idx,
        "frames_in_stage": frames_in_stage,
        "stage_boundaries": stage_boundaries,
    }, path)


def _load_history_from_csv(csv_path, csv_fields):
    history = {k: [] for k in csv_fields}
    if not os.path.exists(csv_path):
        return history
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            for k in csv_fields:
                if k == "stage_env":
                    history[k].append(row[k])
                else:
                    history[k].append(float(row[k]))
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=("Curriculum / transfer-learning PPO training for LGRL on "
                     "Crafter (rule-based oracle). Trains sequentially through "
                     "tech-tree targets easy->hard, carrying model + optimizer "
                     "state across stage boundaries."))
    p.add_argument("--curriculum", default=",".join(DEFAULT_CURRICULUM), type=str,
                   help=("Comma-separated Crafter task names, easiest first. "
                         f"Default: {','.join(DEFAULT_CURRICULUM)}"))
    p.add_argument("--success-threshold", default=DEFAULT_SUCCESS_THRESHOLD, type=float)
    p.add_argument("--success-window", default=DEFAULT_SUCCESS_WINDOW, type=int)
    p.add_argument("--success-stability", default=DEFAULT_SUCCESS_STABILITY, type=int)
    p.add_argument("--max-frames-per-stage", default=DEFAULT_MAX_FRAMES_PER_STAGE, type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--subgoal-log", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    curriculum = [t.strip() for t in args.curriculum.split(",") if t.strip()]
    for t in curriculum:
        if t not in SUPPORTED_TASKS:
            raise SystemExit(f"Unknown task in curriculum: {t!r}")

    artifact_stem = curriculum_artifact_stem(BASE_ARTIFACT_STEM, curriculum)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"{artifact_stem}_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{artifact_stem}.pt")

    csv_fields = ["update", "frames", "stage_idx", "stage_env",
                  "frames_in_stage", "avg_return", "avg_steps",
                  "rolling_success_rate", "stable_count", "entropy",
                  "policy_loss", "value_loss", "elapsed_sec"]

    print("=" * 108)
    print("  LGRL Curriculum / Transfer Training on Crafter (rule oracle)")
    print("=" * 108)
    print(f"  Device              : {DEVICE}")
    print(f"  Curriculum ({len(curriculum)} stages):")
    for i, t in enumerate(curriculum):
        print(f"      {i:2d}. {t:20s} mission={TASK_MISSION[t]!r:26s} "
              f"n={num_subgoals(t)} T_max={task_max_steps(t)}")
    print(f"  Success threshold   : {args.success_threshold}")
    print(f"  Success window      : {args.success_window} episodes")
    print(f"  Success stability   : {args.success_stability} consecutive updates")
    print(f"  Max frames / stage  : {args.max_frames_per_stage:,}")
    print(f"  Artifact stem       : {artifact_stem}")
    print("=" * 108)

    # ---- Shared model + vocab (persist across all stages) --------------
    vocab = Vocabulary()
    for t in curriculum:
        vocab.tokenize(f"{TASK_MISSION[t]} [SEP] collect wood")
    for s in ("collect", "place", "make", "a", "an", "table", "furnace",
              "wood", "stone", "coal", "iron", "diamond", "pickaxe", "sword",
              "1", "2", "4", "[SEP]"):
        _ = vocab[s]

    planner = RuleBasedCrafterPlanner()
    logger = (CrafterSubgoalLogger(LOG_DIR, artifact_stem, NUM_ENVS)
              if args.subgoal_log else None)

    # Build a probe env just to get the obs/action spaces for the model.
    probe = CrafterTaskEnv(curriculum[0], seed=0)
    model = CrafterACModel(probe.observation_space, probe.action_space,
                           vocab).to(DEVICE)

    # ---- Resume bookkeeping -------------------------------------------
    stage_idx = 0
    frames_in_stage = 0
    total_frames = 0
    start_update = 0
    stage_boundaries = []
    history = {k: [] for k in csv_fields}
    optimizer_state = None

    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("curriculum") != curriculum:
            raise SystemExit("Resume failed: saved curriculum does not match --curriculum.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer_state = ckpt["optimizer_state_dict"]
        start_update = ckpt["update"]
        total_frames = ckpt["total_frames"]
        stage_idx = ckpt["stage_idx"]
        frames_in_stage = ckpt["frames_in_stage"]
        stage_boundaries = ckpt.get("stage_boundaries", [])
        history = _load_history_from_csv(csv_path, csv_fields)
        print(f"Resumed at stage {stage_idx} ({curriculum[stage_idx]}), "
              f"update {start_update}, {total_frames:,} frames.")

    # ---- Build the first (or resumed) stage ---------------------------
    algo, envs, hierarchy_state, reshape, n, t_max = _build_algo_for_stage(
        model, vocab, planner, curriculum[stage_idx], args.seed,
        optimizer_state=optimizer_state, logger=logger)

    csv_mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if csv_mode == "w":
        csv_writer.writeheader()

    success_buf = deque(maxlen=args.success_window)
    stable_updates = 0

    print(f"\n{'upd':>6} | {'frames':>11} | {'st':>2} | {'st_frames':>10} | "
          f"{'ret':>6} | {'steps':>6} | {'succ%':>6} | {'stab':>6} | "
          f"{'ploss':>8} | {'vloss':>8}  [stage]")
    print("-" * 108)

    start_time = time.time()
    update = start_update
    while True:
        update += 1
        reshape._current_env_idx = 0
        exps, collect_logs = algo.collect_experiences()
        update_logs = algo.update_parameters(exps)
        total_frames += NUM_ENVS * NUM_FRAMES_PER_PROC
        frames_in_stage += NUM_ENVS * NUM_FRAMES_PER_PROC

        for (_fam, _ret, _steps, success) in hierarchy_state.completed_episodes:
            success_buf.append(1 if success else 0)
        hierarchy_state.completed_episodes.clear()
        rolling_success = float(np.mean(success_buf)) if success_buf else 0.0

        buffer_full = len(success_buf) >= args.success_window
        if buffer_full and rolling_success >= args.success_threshold:
            stable_updates += 1
        else:
            stable_updates = 0

        avg_return = float(np.mean(collect_logs["return_per_episode"]))
        avg_steps = float(np.mean(collect_logs["num_frames_per_episode"]))
        entropy = float(np.mean(update_logs["entropy"]))
        policy_loss = float(np.mean(update_logs["policy_loss"]))
        value_loss = float(np.mean(update_logs["value_loss"]))
        elapsed = time.time() - start_time

        row = {"update": update, "frames": total_frames, "stage_idx": stage_idx,
               "stage_env": curriculum[stage_idx], "frames_in_stage": frames_in_stage,
               "avg_return": f"{avg_return:.6f}", "avg_steps": f"{avg_steps:.1f}",
               "rolling_success_rate": f"{rolling_success:.4f}",
               "stable_count": stable_updates, "entropy": f"{entropy:.6f}",
               "policy_loss": f"{policy_loss:.6f}", "value_loss": f"{value_loss:.6f}",
               "elapsed_sec": f"{elapsed:.1f}"}
        csv_writer.writerow(row); csv_file.flush()
        for k, v in row.items():
            history[k].append(v if k == "stage_env" else float(v))

        if update % 5 == 0 or update == start_update + 1:
            print(f"{update:>6} | {total_frames:>11,} | {stage_idx:>2} | "
                  f"{frames_in_stage:>10,} | {avg_return:>6.3f} | {avg_steps:>6.1f} | "
                  f"{rolling_success*100:>6.1f} | {stable_updates:>3}/{args.success_stability:<2} | "
                  f"{policy_loss:>8.4f} | {value_loss:>8.4f}  [{curriculum[stage_idx]}]")

        if update % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_path, model, algo, vocab, update,
                             total_frames, curriculum, stage_idx,
                             frames_in_stage, stage_boundaries)
        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR, artifact_stem, curriculum, stage_boundaries)

        # ---- Advancement / termination ------------------------------
        success_stable = stable_updates >= args.success_stability
        frames_exhausted = frames_in_stage >= args.max_frames_per_stage
        if not (success_stable or frames_exhausted):
            continue

        reason = (f"success {rolling_success:.2f} >= {args.success_threshold:.2f} "
                  f"for {stable_updates} updates" if success_stable
                  else f"frame cap {args.max_frames_per_stage:,} reached")
        is_last = (stage_idx == len(curriculum) - 1)
        print("\n" + "-" * 108)
        if is_last:
            print(f"  >>> FINAL STAGE COMPLETE — stage {stage_idx} ({curriculum[stage_idx]})")
            print(f"       reason: {reason} | frames in stage: {frames_in_stage:,} | total: {total_frames:,}")
            print("-" * 108 + "\n")
            break
        next_idx = stage_idx + 1
        print(f"  >>> ADVANCING stage {stage_idx} ({curriculum[stage_idx]}) -> "
              f"stage {next_idx} ({curriculum[next_idx]})")
        print(f"       reason: {reason} | frames in stage: {frames_in_stage:,} | total: {total_frames:,}")
        print("-" * 108 + "\n")

        stage_boundaries.append((stage_idx, curriculum[stage_idx], total_frames))

        # Carry the model + optimizer forward into the next, harder stage.
        optimizer_state = algo.optimizer.state_dict()
        algo, envs, hierarchy_state, reshape, n, t_max = _build_algo_for_stage(
            model, vocab, planner, curriculum[next_idx], args.seed,
            optimizer_state=optimizer_state, logger=logger)
        stage_idx = next_idx
        frames_in_stage = 0
        success_buf.clear()
        stable_updates = 0

        _save_checkpoint(checkpoint_path, model, algo, vocab, update,
                         total_frames, curriculum, stage_idx, 0, stage_boundaries)

    # ---- Final save + plot ------------------------------------------
    _save_checkpoint(checkpoint_path, model, algo, vocab, update, total_frames,
                     curriculum, stage_idx, frames_in_stage, stage_boundaries)
    save_plots(history, PLOT_DIR, artifact_stem, curriculum, stage_boundaries)
    csv_file.close()
    if logger:
        logger.close()
    print("Curriculum training complete.")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Metrics     : {csv_path}")
    print(f"  Plot        : {os.path.join(PLOT_DIR, f'{artifact_stem}_training_curves.png')}")
    print(f"  Final stage : {stage_idx} ({curriculum[stage_idx]})")
    print(f"  Transitions : {len(stage_boundaries)} / {len(curriculum) - 1}")


if __name__ == "__main__":
    main()
