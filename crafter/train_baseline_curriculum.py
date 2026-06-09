"""
train_baseline_curriculum.py — baseline (mission-only) curriculum / transfer
training on Crafter.

The CONTROL twin of ``train_lgrl_curriculum.py``: identical curriculum,
identical transfer mechanism (carry the model + optimizer forward across
stages, easy -> hard), identical advancement rule (rolling-success stability
or per-stage frame cap), and identical PPO hyperparameters and network — but
with NO LGRL scaffolding: no planner, no subgoals, no reward shaping. The
agent sees only the mission string and learns from the sparse environment
reward (+1 the step the target unlocks, 0 otherwise).

Why this script exists
----------------------
``train_lgrl_curriculum.py`` bundles two ideas: (1) the curriculum / transfer
schedule and (2) the LGRL subgoal-reward scaffolding. Comparing it only
against the single-task baseline confounds the two. Running this script lets
you separate them cleanly:

    baseline single-task   vs  baseline curriculum   -> effect of the curriculum alone
    baseline curriculum    vs  LGRL curriculum       -> effect of LGRL on top of the curriculum
    LGRL single-task       vs  LGRL curriculum       -> effect of the curriculum under LGRL

Everything except the per-stage learning machinery is identical to
``train_lgrl_curriculum.py`` (same stage walk, same model + optimizer
carry-forward, same advancement rule, same plotting / checkpoint / resume),
so the two runs are directly comparable. The advancement defaults are kept
the same as the LGRL curriculum so both halts use the same criteria.

Usage:
    # Full tech-tree curriculum (default)
    python crafter/train_baseline_curriculum.py

    # Custom curriculum
    python crafter/train_baseline_curriculum.py --curriculum collect_wood,place_table,make_wood_pickaxe

    # Tighter / looser advancement
    python crafter/train_baseline_curriculum.py --success-threshold 0.9 --success-stability 15

    # Cap a stage so wallclock stays predictable
    python crafter/train_baseline_curriculum.py --max-frames-per-stage 1000000

    # Resume (curriculum spec must match)
    python crafter/train_baseline_curriculum.py --resume
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
    TASK_MISSION, SUPPORTED_TASKS, DEFAULT_CURRICULUM,
    task_max_steps, curriculum_artifact_stem,
)

# Reuse the tested baseline machinery from the single-task baseline script
# (mission-only preprocess, raw-reward passthrough + success bookkeeping).
from train_baseline import (
    EpisodeTracker, make_preprocess_obss, make_reshape_reward,
    NUM_ENVS, NUM_FRAMES_PER_PROC, LR, DISCOUNT, GAE_LAMBDA, CLIP_EPS,
    BATCH_SIZE, ENTROPY_COEF, VALUE_LOSS_COEF, MAX_GRAD_NORM, EPOCHS,
    RECURRENCE,
)

# ---------------------------------------------------------------------------
# Curriculum advancement defaults (kept identical to train_lgrl_curriculum.py
# so the baseline and LGRL curricula halt under the same criteria)
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

BASE_ARTIFACT_STEM = "baseline"
PLANNER_TAG = "none"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Per-stage algo (re)builder — keeps the model, swaps the envs
# ---------------------------------------------------------------------------

def _build_algo_for_stage(model, vocab, task, seed, optimizer_state=None):
    """Build envs + EpisodeTracker + PPOAlgo for one curriculum stage,
    reusing the existing model. Returns (algo, envs, tracker, reshape, t_max).

    Baseline analogue of the LGRL curriculum's builder: an EpisodeTracker
    (success bookkeeping only) replaces the HierarchyState, the text stream
    sees the mission alone (no subgoal), and the reward is passed through raw
    (no subgoal/mission shaping)."""
    t_max = task_max_steps(task)
    envs = [CrafterTaskEnv(task, seed=seed * 1000 + i) for i in range(NUM_ENVS)]

    tracker = EpisodeTracker(NUM_ENVS)
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
    if optimizer_state is not None:
        algo.optimizer.load_state_dict(optimizer_state)
    return algo, envs, tracker, reshape, t_max


# ---------------------------------------------------------------------------
# Plotting (curriculum-aware: vertical lines at stage boundaries + staircase)
# Mirrors train_lgrl_curriculum.py.save_plots exactly.
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
    fig.suptitle(f"Baseline (mission-only) Curriculum -- Crafter tech tree",
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
        description=("Baseline (mission-only) curriculum / transfer PPO on "
                     "Crafter. Control twin of train_lgrl_curriculum.py: same "
                     "curriculum + transfer + advancement rule, no subgoals."))
    p.add_argument("--curriculum", default=",".join(DEFAULT_CURRICULUM), type=str,
                   help=("Comma-separated Crafter task names, easiest first. "
                         f"Default: {','.join(DEFAULT_CURRICULUM)}"))
    p.add_argument("--success-threshold", default=DEFAULT_SUCCESS_THRESHOLD, type=float)
    p.add_argument("--success-window", default=DEFAULT_SUCCESS_WINDOW, type=int)
    p.add_argument("--success-stability", default=DEFAULT_SUCCESS_STABILITY, type=int)
    p.add_argument("--max-frames-per-stage", default=DEFAULT_MAX_FRAMES_PER_STAGE, type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None, type=str,
                   help=("Warm-start the model + vocab + optimizer from this "
                         "checkpoint, then run --curriculum from stage 0 under a "
                         "NEW artifact stem. Unlike --resume, the saved curriculum "
                         "need not match: use it to CONTINUE a trained agent into a "
                         "deeper curriculum. Pair a baseline checkpoint with this "
                         "(baseline) script and an LGRL checkpoint with the LGRL one."))
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

    # Same columns as the LGRL curriculum (the baseline has no subgoals, so
    # there is no avg_subgoals column — neither does the LGRL version here).
    csv_fields = ["update", "frames", "stage_idx", "stage_env",
                  "frames_in_stage", "avg_return", "avg_steps",
                  "rolling_success_rate", "stable_count", "entropy",
                  "policy_loss", "value_loss", "elapsed_sec"]

    print("=" * 108)
    print("  Baseline (mission-only) Curriculum / Transfer Training on Crafter")
    print("=" * 108)
    print(f"  Device              : {DEVICE}")
    print(f"  Curriculum ({len(curriculum)} stages):")
    for i, t in enumerate(curriculum):
        print(f"      {i:2d}. {t:20s} mission={TASK_MISSION[t]!r:26s} "
              f"T_max={task_max_steps(t)}")
    print(f"  Success threshold   : {args.success_threshold}")
    print(f"  Success window      : {args.success_window} episodes")
    print(f"  Success stability   : {args.success_stability} consecutive updates")
    print(f"  Max frames / stage  : {args.max_frames_per_stage:,}")
    print(f"  Planner             : none (mission-only baseline)")
    print(f"  Artifact stem       : {artifact_stem}")
    print("=" * 108)

    # ---- Shared model + vocab (persist across all stages) --------------
    # Mission-only vocab: no subgoal tokens, but seed the same tech-tree
    # words so token ids stay stable across runs / resumes.
    vocab = Vocabulary()
    for t in curriculum:
        vocab.tokenize(TASK_MISSION[t])
    for s in ("collect", "place", "make", "a", "an", "table", "furnace",
              "wood", "stone", "coal", "iron", "diamond", "pickaxe", "sword"):
        _ = vocab[s]

    # ---- Optional warm-start: load checkpoint + restore its vocab ------
    # Restoring the trained vocab before building the model keeps embedding
    # token-ids aligned with the trained weights.
    init_state = None
    if args.init_from and not (args.resume and os.path.exists(checkpoint_path)):
        if not os.path.exists(args.init_from):
            raise SystemExit(f"--init-from checkpoint not found: {args.init_from}")
        init_state = torch.load(args.init_from, map_location=DEVICE, weights_only=False)
        if "vocab" in init_state:
            vocab.word2idx = dict(init_state["vocab"])
            vocab.idx2word = [None] * len(vocab.word2idx)
            for w, idx in vocab.word2idx.items():
                vocab.idx2word[idx] = w

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
    elif init_state is not None:
        # Warm-start: carry trained weights + optimizer into a FRESH run of
        # the given curriculum (counters/CSV/plots start clean under the new
        # stem).
        model.load_state_dict(init_state["model_state_dict"])
        optimizer_state = init_state.get("optimizer_state_dict")
        trained_on = init_state.get("curriculum", init_state.get("task", "?"))
        print(f"Warm-started model + optimizer from {args.init_from}")
        print(f"  (that checkpoint was trained on: {trained_on})")
        print(f"  Starting curriculum {curriculum} from stage 0 under stem "
              f"'{artifact_stem}'.")

    # ---- Build the first (or resumed) stage ---------------------------
    algo, envs, tracker, reshape, t_max = _build_algo_for_stage(
        model, vocab, curriculum[stage_idx], args.seed,
        optimizer_state=optimizer_state)

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

        for success in tracker.completed:
            success_buf.append(1 if success else 0)
        tracker.completed.clear()
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
        algo, envs, tracker, reshape, t_max = _build_algo_for_stage(
            model, vocab, curriculum[next_idx], args.seed,
            optimizer_state=optimizer_state)
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
    print("Baseline curriculum training complete.")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Metrics     : {csv_path}")
    print(f"  Plot        : {os.path.join(PLOT_DIR, f'{artifact_stem}_training_curves.png')}")
    print(f"  Final stage : {stage_idx} ({curriculum[stage_idx]})")
    print(f"  Transitions : {len(stage_boundaries)} / {len(curriculum) - 1}")


if __name__ == "__main__":
    main()
