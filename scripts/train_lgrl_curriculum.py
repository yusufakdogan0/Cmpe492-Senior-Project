"""
train_lgrl_curriculum.py — Curriculum PPO training for LGRL (rule-based oracle).

Curriculum-learning setup: the agent is first trained on the simpler
GoToDoor and GoToObject tasks to acquire basic navigation abilities, then
the curriculum progresses to increasingly larger instances of the
KeyCorridor environment.

The default curriculum has 11 stages: GoToDoor 5x5/6x6/8x8 -> GoToObject
6x6/8x8 -> KeyCorridor S3R1/S3R2/S3R3/S4R3/S5R3/S6R3.

UnlockPickup is intentionally NOT in the curriculum. We treat it as a
zero-shot transfer test environment: evaluate the final checkpoint of
this run on UnlockPickup without further training.

Each stage advances to the next when EITHER:

  - the rolling success rate (over the last ``--success-window``
    episodes) has stayed at or above ``--success-threshold`` for
    ``--success-stability`` consecutive PPO updates. A single update
    that dips below the threshold resets the counter to zero — this
    prevents premature advancement on a transient lucky window. OR
  - frames spent on the current stage reach ``--max-frames-per-stage``
    (default 50M).

There is no global frame budget. Every stage — including the last —
gets the same ``--max-frames-per-stage`` cap independently. Training
stops once the last stage either meets the stability criterion or hits
its frame cap.

Model parameters, vocabulary, and optimizer state are preserved across
stage boundaries. Only the env workers and HierarchyState's per-env
config (``n_subgoals``, ``T_max``, family) are rebuilt for the new
stage. The agent therefore inherits everything it learned on earlier
stages into the harder ones.

Same PPO loop and reward shaping as train_lgrl_rule.py. The
KeyCorridor stages reuse the existing 10-stage UnlockPickup machine in
``RuleBasedPlanner`` (same mission template + same structure: find key
-> open locked door -> drop key -> pickup target).

Artifact naming:
  - The default curriculum gets a compact stem:
        checkpoints/lgrl_rule_curriculum_default.pt
  - A KeyCorridor-only sweep gets:
        checkpoints/lgrl_rule_curriculum_keycorridor_s3r1_to_s6r3.pt
  - Other custom curricula list every env tag.

Usage:
    # Full default curriculum: 11 stages, up to 50M frames each
    python scripts/train_lgrl_curriculum.py

    # Custom curriculum
    python scripts/train_lgrl_curriculum.py --curriculum \
        "MiniGrid-KeyCorridorS3R1-v0,MiniGrid-KeyCorridorS3R3-v0"

    # Tighter advancement (raise the success bar and demand more stability)
    python scripts/train_lgrl_curriculum.py \
        --success-threshold 0.90 --success-stability 20

    # Hard cap a stage so wallclock stays predictable
    python scripts/train_lgrl_curriculum.py --max-frames-per-stage 10000000

    # Resume from the latest checkpoint (curriculum spec must match)
    python scripts/train_lgrl_curriculum.py --resume

    # Per-env subgoal JSONL logging
    python scripts/train_lgrl_curriculum.py --subgoal-log
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque

import gymnasium as gym
import matplotlib
import minigrid  # noqa: F401  (registers MiniGrid envs)
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch_ac
import torch_ac.algos.base
from utils.sequential_env import SequentialEnv

# Same monkey-patch as the rest of the project: torch-ac uses a
# multiprocess ParallelEnv by default; we substitute a sequential
# stepper so all envs live in the training process.
torch_ac.algos.base.ParallelEnv = SequentialEnv

from models.baseline_agent import Vocabulary
from models.lgrl_agent import LGRLAgent
from utils.env_parser import parse_env_description
from utils.env_utils import (
    SUPPORTED_ENVS,
    DEFAULT_CURRICULUM,
    env_max_steps,
    parse_curriculum_spec,
    curriculum_artifact_stem,
)
from utils.rule_based_planner import RuleBasedPlanner
from utils.subgoal_tracker import SubgoalTracker
from utils.subgoal_logger import SubgoalLogger

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

NUM_ENVS = 16
NUM_FRAMES_PER_PROC = 128

# PPO hyperparameters — same as train_lgrl_rule.py
LR = 1e-4
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
BATCH_SIZE = 256
# torch-ac-style defaults
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPOCHS = 4
RECURRENCE = 4

# Reward scaffolding — same as train_lgrl_rule.py
R_MISSION = 0.5
R_SUBGOAL = 0.5
MISSION_TIME_COEF = 0.5
SUBGOAL_TIME_COEF = 0.5
SUBGOAL_TIMEOUT_MULT = 2.0

# Curriculum advancement defaults.
#
# Advancement requires the rolling success rate to stay >= threshold for
# SUCCESS_STABILITY consecutive PPO updates (a single dip resets the
# counter). This is the "constantly above threshold, not once" rule —
# 10 consecutive updates at 16*128 = 2,048 frames each ~= 20K frames of
# sustained performance before advancing, which is short enough to be
# responsive but long enough to filter out a single lucky window.
#
# Every stage — including the last — has the same max-frames-per-stage
# cap. There is no global frame budget; training stops once the last
# stage either meets the stability criterion or hits its frame cap.
DEFAULT_SUCCESS_THRESHOLD = 0.80
DEFAULT_SUCCESS_WINDOW = 200            # episodes
DEFAULT_SUCCESS_STABILITY = 10          # consecutive PPO updates above threshold
DEFAULT_MAX_FRAMES_PER_STAGE = 50_000_000

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_EVERY = 10                   # updates

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
PLOT_EVERY = 50

BASE_ARTIFACT_STEM = "lgrl_rule"
PLANNER_TAG = "rule_based"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Curriculum PPO training for LGRL (rule-based oracle planner). "
            "Trains sequentially through a list of MiniGrid envs, "
            "preserving model + optimizer state across stage boundaries."
        )
    )
    parser.add_argument(
        "--curriculum",
        default=",".join(DEFAULT_CURRICULUM),
        type=str,
        help=(
            "Comma-separated list of env names from easiest to hardest. "
            "Default: the 11-stage curriculum "
            "(GoToDoor 5x5/6x6/8x8 -> GoToObject 6x6/8x8 -> "
            "KeyCorridor S3R1/S3R2/S3R3/S4R3/S5R3/S6R3)."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        default=DEFAULT_SUCCESS_THRESHOLD,
        type=float,
        help=(
            f"Rolling success rate required to advance to the next stage. "
            f"Default: {DEFAULT_SUCCESS_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--success-window",
        default=DEFAULT_SUCCESS_WINDOW,
        type=int,
        help=(
            f"Number of recent completed episodes used to compute the "
            f"rolling success rate. Default: {DEFAULT_SUCCESS_WINDOW}."
        ),
    )
    parser.add_argument(
        "--success-stability",
        default=DEFAULT_SUCCESS_STABILITY,
        type=int,
        help=(
            f"Number of CONSECUTIVE PPO updates the rolling success rate "
            f"must stay at or above --success-threshold before advancing "
            f"(or stopping, on the final stage). A single update below "
            f"threshold resets the counter to zero. This prevents "
            f"premature advancement on a transient lucky window. "
            f"Default: {DEFAULT_SUCCESS_STABILITY}."
        ),
    )
    parser.add_argument(
        "--max-frames-per-stage",
        default=DEFAULT_MAX_FRAMES_PER_STAGE,
        type=int,
        help=(
            f"Hard upper bound on frames spent on a single stage before "
            f"force-advancing. Applied to every stage including the last. "
            f"There is no global frame budget. "
            f"Default: {DEFAULT_MAX_FRAMES_PER_STAGE:,}."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the matching curriculum checkpoint (curriculum spec must match).",
    )
    parser.add_argument(
        "--subgoal-log",
        action="store_true",
        default=False,
        help="Write per-env subgoal lifecycle JSONL files to logs/<stem>_subgoal_log/.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-env hierarchy state (stage-based, forward-only)
# ---------------------------------------------------------------------------
# This is the SAME shape as the HierarchyState in train_lgrl_rule.py — it
# tracks per-env subgoal lifecycle and supports per-env n_subgoals and
# T_max. The difference is that in curriculum mode we update the envs +
# per-env config in place each time we advance to a new stage.
# ---------------------------------------------------------------------------

class HierarchyState:
    def __init__(
        self,
        num_envs: int,
        planner: RuleBasedPlanner,
        envs: list,
        *,
        n_subgoals_per_env: list[int],
        t_max_per_env: list[int],
        family_per_env: list[str],
    ):
        assert len(n_subgoals_per_env) == num_envs
        assert len(t_max_per_env) == num_envs
        assert len(family_per_env) == num_envs
        self.num_envs = num_envs
        self.planner = planner
        self.envs = envs
        self.n_subgoals_per_env = n_subgoals_per_env
        self.t_max_per_env = t_max_per_env
        self.family_per_env = family_per_env
        self._init_lists()

    def _init_lists(self):
        self.active_subgoals: list[str] = [""] * self.num_envs
        self.stage_indices: list[int] = [0] * self.num_envs
        self.step_counters: list[int] = [0] * self.num_envs
        self.episode_steps: list[int] = [0] * self.num_envs
        self.episode_raw_return: list[float] = [0.0] * self.num_envs
        self.trackers: list[SubgoalTracker] = [
            SubgoalTracker() for _ in range(self.num_envs)
        ]
        self.histories: list[list[dict]] = [[] for _ in range(self.num_envs)]
        # Each completed episode is pushed here as (success: bool,
        # steps: int). Drained by the training loop after each rollout
        # to update the rolling success window.
        self.completed_episodes: list[tuple[bool, int]] = []

    def switch_to_stage(
        self,
        new_envs: list,
        *,
        n_subgoals_per_env: list[int],
        t_max_per_env: list[int],
        family_per_env: list[str],
    ):
        """Replace the env workers and per-env config for a new curriculum stage.

        All per-env lifecycle tracking (active subgoal, stage index, step
        counter, episode steps, etc.) is reset because the new envs are
        in fresh initial states.
        """
        assert len(n_subgoals_per_env) == self.num_envs
        assert len(t_max_per_env) == self.num_envs
        assert len(family_per_env) == self.num_envs
        self.envs = new_envs
        self.n_subgoals_per_env = n_subgoals_per_env
        self.t_max_per_env = t_max_per_env
        self.family_per_env = family_per_env
        self._init_lists()

    def init_env_subgoal(self, env_idx: int, obs: dict):
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
        self.episode_raw_return[env_idx] = 0.0
        self.trackers[env_idx].reset()
        self.histories[env_idx] = []

    def subgoal_budget(self, env_idx: int) -> float:
        """Ti = (i / n) * Tmax."""
        n = self.n_subgoals_per_env[env_idx]
        t_max = self.t_max_per_env[env_idx]
        i = min(self.stage_indices[env_idx] + 1, n)
        return (i / n) * t_max

    def n_subgoals(self, env_idx: int) -> int:
        return self.n_subgoals_per_env[env_idx]

    def t_max(self, env_idx: int) -> int:
        return self.t_max_per_env[env_idx]

    def advance(self, env_idx: int, obs: dict):
        next_stage = self.stage_indices[env_idx] + 1
        n = self.n_subgoals_per_env[env_idx]
        if next_stage >= n:
            # All stages done — freeze the last subgoal as the agent's
            # text conditioning while it finishes the mission.
            self.stage_indices[env_idx] = n
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
# Observation preprocessing — "mission [SEP] subgoal"
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
# Reward shaping (stage-based, forward-only) — identical to train_lgrl_rule.py
# except that completed-episode records only push (success, steps), since
# we don't need per-family aggregation in single-env-per-stage curriculum
# mode.
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
                # Mission reward: rm = Rm * (1 - 0.5 * Tused/Tmax)
                t_total = hierarchy_state.episode_steps[env_idx]
                ratio = min(t_total / t_max, 1.0)
                total_reward += R_MISSION * (1.0 - MISSION_TIME_COEF * ratio)
            if logger:
                logger.on_episode_end(
                    env_idx, mission, success,
                    hierarchy_state.episode_steps[env_idx],
                )
            hierarchy_state.completed_episodes.append(
                (success, hierarchy_state.episode_steps[env_idx])
            )
            hierarchy_state.reset_env(env_idx)
            return total_reward

        # If all stages exhausted, no subgoal checking
        if hierarchy_state.stage_indices[env_idx] >= n_subgoals:
            return total_reward

        # First step of the episode — initialise the subgoal from stage 0
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
            # Subgoal reward: ri = Rt * (1 - 0.5 * Tused/Ti), clipped at 2*Ti
            ratio = min(t_used / max(t_budget, 1), SUBGOAL_TIMEOUT_MULT)
            r_i = max(R_SUBGOAL * (1.0 - SUBGOAL_TIME_COEF * ratio), 0.0)
            # Normalise by the number of stages
            total_reward += r_i / n_subgoals

            hierarchy_state.histories[env_idx].append(
                {"subgoal": subgoal, "status": "Success", "steps": t_used,
                 "stage": hierarchy_state.stage_indices[env_idx]}
            )
            if logger:
                logger.log(
                    env_idx, "completed",
                    subgoal=subgoal,
                    stage=hierarchy_state.stage_indices[env_idx],
                    steps_used=t_used, budget=t_budget,
                    reward=r_i / n_subgoals,
                )
            hierarchy_state.advance(env_idx, obs)
            if logger and hierarchy_state.active_subgoals[env_idx]:
                env_json = parse_env_description(obs["image"], uw.carrying)
                logger.log(
                    env_idx, "new",
                    subgoal=hierarchy_state.active_subgoals[env_idx],
                    stage=hierarchy_state.stage_indices[env_idx],
                    budget=hierarchy_state.subgoal_budget(env_idx),
                    env_state=env_json,
                    raw_llm=hierarchy_state.planner.last_raw_response,
                )
        elif timed_out:
            if logger:
                logger.log(
                    env_idx, "timed_out",
                    subgoal=subgoal,
                    stage=hierarchy_state.stage_indices[env_idx],
                    steps_used=t_used, budget=t_budget,
                )
            hierarchy_state.advance(env_idx, obs)
            if logger and hierarchy_state.active_subgoals[env_idx]:
                env_json = parse_env_description(obs["image"], uw.carrying)
                logger.log(
                    env_idx, "new",
                    subgoal=hierarchy_state.active_subgoals[env_idx],
                    stage=hierarchy_state.stage_indices[env_idx],
                    budget=hierarchy_state.subgoal_budget(env_idx),
                    env_state=env_json,
                    raw_llm=hierarchy_state.planner.last_raw_response,
                )

        return total_reward

    reshape_reward._current_env_idx = 0
    return reshape_reward


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _smooth(values, window=20):
    if len(values) < window:
        return values
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def save_plots(history, plot_dir, artifact_stem, curriculum, stage_boundaries):
    """Plot training curves with stage boundary shading.

    ``stage_boundaries`` is a list of (stage_idx, env_name, frames_at_end)
    — the frame count at the END of each completed stage. Boundaries are
    drawn as vertical dotted lines on every panel.

    Panels:
        [0,0] Average Return    [0,1] Average Steps
        [1,0] Rolling Success   [1,1] Policy Entropy
    """
    if not history.get("update"):
        return
    frames = history["frames"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"LGRL PPO (rule oracle, curriculum) — "
        f"{len(curriculum)} stages: "
        f"{curriculum[0].replace('MiniGrid-', '').replace('-v0', '')} → "
        f"{curriculum[-1].replace('MiniGrid-', '').replace('-v0', '')}",
        fontsize=14, fontweight="bold",
    )

    # Panel 0,0 — Average return
    ax = axes[0, 0]
    ax.plot(frames, history["avg_return"], alpha=0.25, color="tab:blue",
            linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_return"]), color="tab:blue",
            linewidth=1.8, label="Avg return (smoothed)")
    ax.set(xlabel="Frames", ylabel="Average Return",
           title="Average Return per Episode")
    _draw_stage_lines(ax, stage_boundaries, curriculum)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 0,1 — Average steps
    ax = axes[0, 1]
    ax.plot(frames, history["avg_steps"], alpha=0.25, color="tab:orange",
            linewidth=0.5)
    ax.plot(frames, _smooth(history["avg_steps"]), color="tab:orange",
            linewidth=1.8, label="Avg steps (smoothed)")
    ax.set(xlabel="Frames", ylabel="Average Steps",
           title="Average Steps per Episode")
    _draw_stage_lines(ax, stage_boundaries, curriculum)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 1,0 — Rolling success rate (the advancement signal)
    ax = axes[1, 0]
    ax.plot(frames, history["rolling_success_rate"], color="tab:green",
            linewidth=1.5, label="Rolling success rate")
    ax.set(xlabel="Frames", ylabel="Success rate",
           title="Rolling Success Rate (curriculum advancement signal)",
           ylim=(-0.05, 1.05))
    _draw_stage_lines(ax, stage_boundaries, curriculum)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 1,1 — Policy Entropy
    ax = axes[1, 1]
    ax.plot(frames, history["entropy"], alpha=0.25, color="tab:green",
            linewidth=0.5)
    ax.plot(frames, _smooth(history["entropy"]), color="tab:green",
            linewidth=1.8, label="Entropy (smoothed)")
    ax.set(xlabel="Frames", ylabel="Entropy", title="Policy Entropy")
    _draw_stage_lines(ax, stage_boundaries, curriculum)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"{artifact_stem}_training_curves.png"),
        dpi=150,
    )
    plt.close(fig)


def save_per_stage_plots(history, plot_dir, artifact_stem, curriculum,
                         stage_boundaries):
    """Generate one 2×2 plot per curriculum stage.

    Each plot shows only the frames that belong to that stage, providing
    a clear view of how the agent learns within each individual
    environment without cross-stage noise.

    Files are named ``{artifact_stem}_stage_{i}_{env_tag}.png``.

    Panels per stage (same layout as the global plot):
        [0,0] Average Return    [0,1] Average Steps
        [1,0] Rolling Success   [1,1] Policy Entropy
    """
    if not history.get("update"):
        return

    all_stage_idxs = history["stage_idx"]
    all_frames = history["frames"]
    n_points = len(all_frames)

    # Determine which data-points belong to each stage.
    # Build a dict:  stage_i -> list of point indices
    stage_point_map: dict[int, list[int]] = {}
    for pt_idx in range(n_points):
        si = int(all_stage_idxs[pt_idx])
        stage_point_map.setdefault(si, []).append(pt_idx)

    from utils.env_utils import env_stem as _env_stem

    for si, pt_indices in sorted(stage_point_map.items()):
        if si >= len(curriculum):
            continue  # guard against out-of-range
        if len(pt_indices) < 2:
            continue  # not enough data for a meaningful plot

        env_name = curriculum[si]
        env_tag = _env_stem(env_name)

        # Slice history arrays for this stage
        s_frames = [all_frames[j] for j in pt_indices]
        s_return = [history["avg_return"][j] for j in pt_indices]
        s_steps = [history["avg_steps"][j] for j in pt_indices]
        s_success = [history["rolling_success_rate"][j] for j in pt_indices]
        s_entropy = [history["entropy"][j] for j in pt_indices]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        short_name = env_name.replace("MiniGrid-", "").replace("-v0", "")
        fig.suptitle(
            f"LGRL PPO (rule oracle, curriculum) — "
            f"Stage {si}: {short_name}",
            fontsize=14, fontweight="bold",
        )

        # Panel 0,0 — Average return
        ax = axes[0, 0]
        ax.plot(s_frames, s_return, alpha=0.25, color="tab:blue",
                linewidth=0.5)
        ax.plot(s_frames, _smooth(s_return), color="tab:blue",
                linewidth=1.8, label="Avg return (smoothed)")
        ax.set(xlabel="Frames", ylabel="Average Return",
               title="Average Return per Episode")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 0,1 — Average steps
        ax = axes[0, 1]
        ax.plot(s_frames, s_steps, alpha=0.25, color="tab:orange",
                linewidth=0.5)
        ax.plot(s_frames, _smooth(s_steps), color="tab:orange",
                linewidth=1.8, label="Avg steps (smoothed)")
        ax.set(xlabel="Frames", ylabel="Average Steps",
               title="Average Steps per Episode")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 1,0 — Rolling success rate
        ax = axes[1, 0]
        ax.plot(s_frames, s_success, color="tab:green",
                linewidth=1.5, label="Rolling success rate")
        ax.set(xlabel="Frames", ylabel="Success rate",
               title="Rolling Success Rate",
               ylim=(-0.05, 1.05))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 1,1 — Policy Entropy
        ax = axes[1, 1]
        ax.plot(s_frames, s_entropy, alpha=0.25, color="tab:green",
                linewidth=0.5)
        ax.plot(s_frames, _smooth(s_entropy), color="tab:green",
                linewidth=1.8, label="Entropy (smoothed)")
        ax.set(xlabel="Frames", ylabel="Entropy", title="Policy Entropy")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{artifact_stem}_stage_{si}_{env_tag}.png",
            ),
            dpi=150,
        )
        plt.close(fig)


def _draw_stage_lines(ax, stage_boundaries, curriculum):
    """Vertical dotted lines + short labels marking each completed stage."""
    if not stage_boundaries:
        return
    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.92 * (ymax - ymin)
    for stage_idx, _env_name, frames_end in stage_boundaries:
        ax.axvline(frames_end, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        # Compact label, e.g. "S3R1"
        label = (
            curriculum[stage_idx]
            .replace("MiniGrid-KeyCorridor", "")
            .replace("MiniGrid-", "")
            .replace("-v0", "")
        )
        ax.text(frames_end, label_y, f" {label}", fontsize=7,
                color="gray", verticalalignment="top")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path, model, algo, vocab, update, total_frames,
    curriculum, stage_idx, frames_in_stage, stage_boundaries,
    success_window_buffer, stable_updates,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": algo.optimizer.state_dict(),
            "vocab": vocab.word2idx,
            "update": update,
            "total_frames": total_frames,
            "planner": PLANNER_TAG,
            # Curriculum-specific state:
            "mode": "curriculum",
            "curriculum": list(curriculum),
            "stage_idx": stage_idx,
            "frames_in_stage": frames_in_stage,
            "stage_boundaries": list(stage_boundaries),
            "success_window_buffer": list(success_window_buffer),
            "stable_updates": stable_updates,
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
                v = row.get(k, "")
                # ``stage_env`` is a string column — keep it as-is.
                if k == "stage_env":
                    history[k].append(v)
                else:
                    try:
                        history[k].append(float(v))
                    except (TypeError, ValueError):
                        history[k].append(0.0)
    return history


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _probe_env_config(env_name: str) -> tuple[str, int, int, str]:
    """Return (sample_mission, n_subgoals, t_max, family) for an env name."""
    env = gym.make(env_name)
    obs, _ = env.reset()
    mission = obs["mission"]
    family = RuleBasedPlanner.classify_mission(mission)
    n = RuleBasedPlanner.num_stages(mission)
    env.close()
    t_max = env_max_steps(env_name)
    return mission, n, t_max, family


def _build_algo_for_stage(
    *, model, hierarchy_state, vocab, env_list, optimizer_state,
    logger=None,
):
    """Construct a fresh PPOAlgo for a curriculum stage.

    Builds fresh env workers, rebinds HierarchyState to them, and
    transfers the previous optimizer state (if any). The model itself is
    shared by reference — no weight reset.
    """
    new_envs = [
        make_env(env_list[i], seed=i + 100 * hash(env_list[i]) % 10000)
        for i in range(NUM_ENVS)
    ]

    # Probe per-env config (always uniform across NUM_ENVS in
    # curriculum mode, but we still build the per-env lists to match
    # HierarchyState's interface).
    _mission, n, t_max, family = _probe_env_config(env_list[0])
    hierarchy_state.switch_to_stage(
        new_envs,
        n_subgoals_per_env=[n] * NUM_ENVS,
        t_max_per_env=[t_max] * NUM_ENVS,
        family_per_env=[family] * NUM_ENVS,
    )

    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, device=DEVICE)
    reshape_reward = make_reshape_reward(hierarchy_state, logger=logger)

    algo = torch_ac.PPOAlgo(
        envs=new_envs,
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

    if optimizer_state is not None:
        algo.optimizer.load_state_dict(optimizer_state)

    return algo, new_envs, reshape_reward, (n, t_max, family)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    curriculum = parse_curriculum_spec(args.curriculum)
    artifact_stem = curriculum_artifact_stem(BASE_ARTIFACT_STEM, curriculum)
    csv_path = os.path.join(LOG_DIR, f"{artifact_stem}_metrics.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{artifact_stem}.pt")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---- Probe every curriculum env once for vocab seeding + banner ---
    stage_configs: list[tuple[str, int, int, str]] = []  # (mission, n, t_max, family)
    for env_name in curriculum:
        cfg = _probe_env_config(env_name)
        stage_configs.append(cfg)

    # ---- Banner ------------------------------------------------------
    print("=" * 70)
    print("  LGRL PPO Training (rule oracle, CURRICULUM)")
    print("=" * 70)
    print(f"  Artifact stem  : {artifact_stem}")
    print(f"  Device         : {DEVICE}")
    print(f"  Planner        : {PLANNER_TAG}")
    print(f"  Advancement    : success >= {args.success_threshold} over "
          f"last {args.success_window} episodes, sustained for")
    print(f"                   {args.success_stability} consecutive PPO updates "
          f"(any dip resets the counter),")
    print(f"                   OR {args.max_frames_per_stage:,} "
          f"frames/stage cap (uniform; last stage included)")
    print(f"  Global budget  : none — training stops when the final stage's "
          f"conditions are met")
    print(f"  Stages ({len(curriculum)}):")
    for i, env_name in enumerate(curriculum):
        mission, n, t_max, fam = stage_configs[i]
        print(
            f"    [{i:>2}] {env_name:35s} family={fam:13s} "
            f"stages={n}  Tmax={t_max}"
        )
    print(f"  R_mission      : {R_MISSION}")
    print(f"  R_subgoal      : {R_SUBGOAL}")
    print("=" * 70)

    # ---- Vocabulary --------------------------------------------------
    # Seed vocab with one sample subgoal-augmented mission per stage so
    # that all colour/object/door tokens are in-vocab from update 0.
    # Without this, transitions can introduce OOV tokens that map to UNK
    # and lose the planner's signal.
    vocab = Vocabulary()
    for mission, _, _, _ in stage_configs:
        vocab.tokenize(f"{mission} [SEP] search for the target", max_len=32)
        vocab.tokenize(f"{mission} [SEP] go near the target", max_len=32)
        vocab.tokenize(f"{mission} [SEP] pickup the target", max_len=32)
        vocab.tokenize(f"{mission} [SEP] open the locked door", max_len=32)
        vocab.tokenize(f"{mission} [SEP] drop the key", max_len=32)

    # ---- Bootstrap envs & model for stage 0 --------------------------
    initial_env_list = [curriculum[0]] * NUM_ENVS
    initial_envs = [make_env(initial_env_list[i], seed=i) for i in range(NUM_ENVS)]

    obs_space = initial_envs[0].observation_space
    act_space = initial_envs[0].action_space
    model = LGRLAgent(obs_space, act_space, vocab)
    model.to(DEVICE)
    print(f"  Action space   : {act_space.n} actions")
    print(f"  Model params   : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    planner = RuleBasedPlanner()
    mission0, n0, t_max0, fam0 = stage_configs[0]
    hierarchy_state = HierarchyState(
        NUM_ENVS, planner, initial_envs,
        n_subgoals_per_env=[n0] * NUM_ENVS,
        t_max_per_env=[t_max0] * NUM_ENVS,
        family_per_env=[fam0] * NUM_ENVS,
    )

    logger = (
        SubgoalLogger(LOG_DIR, artifact_stem, NUM_ENVS)
        if args.subgoal_log else None
    )
    preprocess_obss = make_preprocess_obss(vocab, hierarchy_state, device=DEVICE)
    reshape_reward = make_reshape_reward(hierarchy_state, logger=logger)

    algo = torch_ac.PPOAlgo(
        envs=initial_envs,
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

    # ---- Curriculum state --------------------------------------------
    stage_idx = 0
    frames_in_stage = 0
    update = 0
    total_frames = 0
    stage_boundaries: list[tuple[int, str, int]] = []  # (idx, env, frames_at_end)
    success_buf: deque[int] = deque(maxlen=args.success_window)
    # Consecutive-update counter for "constantly above threshold" rule:
    # incremented when (buffer is full AND rolling_rate >= threshold),
    # reset to 0 the moment either condition fails. Advancement (or
    # termination on the last stage) fires when this reaches
    # args.success_stability.
    stable_updates = 0

    # ---- CSV fields --------------------------------------------------
    csv_fields = [
        "update", "frames", "stage_idx", "stage_env", "frames_in_stage",
        "avg_return", "avg_steps",
        "rolling_success_rate", "rolling_window", "stable_count",
        "entropy", "policy_loss", "value_loss", "elapsed_sec",
    ]

    # ---- Resume ------------------------------------------------------
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("planner") != PLANNER_TAG:
            raise SystemExit(
                f"Checkpoint planner mismatch: expected {PLANNER_TAG!r}, "
                f"got {ckpt.get('planner')!r}."
            )
        saved_curr = ckpt.get("curriculum")
        if saved_curr is not None and list(saved_curr) != list(curriculum):
            raise SystemExit(
                "Checkpoint curriculum mismatch:\n"
                f"  expected: {list(curriculum)}\n"
                f"  got:      {list(saved_curr)}"
            )
        model.load_state_dict(ckpt["model_state_dict"])
        saved_optimizer_state = ckpt.get("optimizer_state_dict")
        update = ckpt.get("update", 0)
        total_frames = ckpt.get("total_frames", 0)
        stage_idx = ckpt.get("stage_idx", 0)
        frames_in_stage = ckpt.get("frames_in_stage", 0)
        stage_boundaries = [
            tuple(b) for b in ckpt.get("stage_boundaries", [])
        ]
        success_buf.extend(ckpt.get("success_window_buffer", []))
        stable_updates = int(ckpt.get("stable_updates", 0))
        print(
            f"  Resumed: update={update}, frames={total_frames:,}, "
            f"stage_idx={stage_idx}, frames_in_stage={frames_in_stage:,}, "
            f"stable={stable_updates}"
        )

        # Rebuild the algo on the resumed stage's envs (always — even if
        # stage_idx==0 the seed offset differs from the bootstrap envs).
        algo, _envs, reshape_reward, (n_i, t_max_i, fam_i) = _build_algo_for_stage(
            model=model, hierarchy_state=hierarchy_state, vocab=vocab,
            env_list=[curriculum[stage_idx]] * NUM_ENVS,
            optimizer_state=saved_optimizer_state,
            logger=logger,
        )
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
        f"\n{'Update':>7} | {'Frames':>10} | {'Stage':>5} | "
        f"{'StageFr':>10} | {'AvgRet':>7} | {'AvgSt':>6} | "
        f"{'Success':>7} | {'Stable':>6} | {'PLoss':>7} | {'VLoss':>7}"
    )
    print("-" * 108)

    frames_per_update = NUM_ENVS * NUM_FRAMES_PER_PROC
    start_time = time.time()

    # =============================================================
    # Training loop — advance through the curriculum as we go.
    # No global frame budget; loop terminates when the final stage
    # meets EITHER advancement condition (stability or frame cap).
    # =============================================================
    while True:
        update += 1
        total_frames += frames_per_update
        frames_in_stage += frames_per_update

        reshape_reward._current_env_idx = 0
        exps, collect_logs = algo.collect_experiences()
        update_logs = algo.update_parameters(exps)

        avg_return = float(np.mean(collect_logs["return_per_episode"]))
        avg_steps = float(np.mean(collect_logs["num_frames_per_episode"]))
        entropy = float(np.mean(update_logs["entropy"]))
        policy_loss = float(np.mean(update_logs["policy_loss"]))
        value_loss = float(np.mean(update_logs["value_loss"]))
        elapsed = time.time() - start_time

        # Drain completed-episode records from the rollout and feed
        # them to the rolling success buffer.
        for (success, _ep_steps) in hierarchy_state.completed_episodes:
            success_buf.append(1 if success else 0)
        hierarchy_state.completed_episodes.clear()

        rolling_success_rate = (
            float(np.mean(success_buf)) if len(success_buf) > 0 else 0.0
        )

        # ---- Stability counter -------------------------------------
        # The "constantly above threshold, not once" rule: only count
        # this update toward stability if the buffer is full AND the
        # rolling rate is >= threshold. Any failure resets to 0.
        buffer_full = len(success_buf) >= args.success_window
        if buffer_full and rolling_success_rate >= args.success_threshold:
            stable_updates += 1
        else:
            stable_updates = 0

        # ---- stdout line --------------------------------------------
        stage_label = (
            curriculum[stage_idx]
            .replace("MiniGrid-KeyCorridor", "KC-")
            .replace("MiniGrid-", "")
            .replace("-v0", "")
        )
        print(
            f"{update:>7} | {total_frames:>10,} | {stage_idx:>5} | "
            f"{frames_in_stage:>10,} | {avg_return:>7.3f} | "
            f"{avg_steps:>6.1f} | {rolling_success_rate*100:>6.1f}% | "
            f"{stable_updates:>3}/{args.success_stability:<2} | "
            f"{policy_loss:>7.4f} | {value_loss:>7.4f}  "
            f"[{stage_label}]"
        )

        # ---- CSV row ------------------------------------------------
        row = {
            "update": update,
            "frames": total_frames,
            "stage_idx": stage_idx,
            "stage_env": curriculum[stage_idx],
            "frames_in_stage": frames_in_stage,
            "avg_return": f"{avg_return:.6f}",
            "avg_steps": f"{avg_steps:.1f}",
            "rolling_success_rate": f"{rolling_success_rate:.4f}",
            "rolling_window": len(success_buf),
            "stable_count": stable_updates,
            "entropy": f"{entropy:.6f}",
            "policy_loss": f"{policy_loss:.6f}",
            "value_loss": f"{value_loss:.6f}",
            "elapsed_sec": f"{elapsed:.1f}",
        }
        csv_writer.writerow(row)
        csv_file.flush()
        for k, v in row.items():
            if k == "stage_env":
                history[k].append(v)
            else:
                history[k].append(float(v))

        # ---- Periodic checkpoint / plot ----------------------------
        if update % CHECKPOINT_EVERY == 0:
            _save_checkpoint(
                checkpoint_path, model, algo, vocab, update, total_frames,
                curriculum=curriculum, stage_idx=stage_idx,
                frames_in_stage=frames_in_stage,
                stage_boundaries=stage_boundaries,
                success_window_buffer=success_buf,
                stable_updates=stable_updates,
            )

        if update % PLOT_EVERY == 0:
            save_plots(history, PLOT_DIR, artifact_stem, curriculum,
                       stage_boundaries)
            save_per_stage_plots(history, PLOT_DIR, artifact_stem,
                                curriculum, stage_boundaries)

        # ---- Advancement / termination check ------------------------
        # Two advancement conditions:
        #   (a) stability counter reached --success-stability (rolling
        #       rate has stayed >= threshold for that many consecutive
        #       updates), OR
        #   (b) frames_in_stage reached --max-frames-per-stage cap.
        # Both conditions apply uniformly to every stage including the
        # last; on the last stage they cause termination instead of
        # advancement.
        success_stable = stable_updates >= args.success_stability
        frames_exhausted = frames_in_stage >= args.max_frames_per_stage

        if not (success_stable or frames_exhausted):
            continue

        reason = (
            f"success {rolling_success_rate:.2f} >= "
            f"{args.success_threshold:.2f} for "
            f"{stable_updates} consecutive updates"
            if success_stable
            else f"frame cap {args.max_frames_per_stage:,} reached"
        )
        is_last_stage = (stage_idx == len(curriculum) - 1)

        if is_last_stage:
            # No boundary recorded — boundaries mark transitions, and
            # there is no transition out of the last stage. The plot
            # x-axis naturally ends at total_frames anyway.
            print()
            print("-" * 108)
            print(
                f"  >>> FINAL STAGE COMPLETE — stage {stage_idx} "
                f"({curriculum[stage_idx]})"
            )
            print(f"       reason         : {reason}")
            print(f"       frames in stage: {frames_in_stage:,}")
            print(f"       total frames   : {total_frames:,}")
            print("-" * 108)
            print()
            break

        # Inter-stage transition: record the boundary so the plot draws
        # a vertical line marking the end of this stage on the x-axis.
        stage_boundaries.append(
            (stage_idx, curriculum[stage_idx], total_frames)
        )
        next_idx = stage_idx + 1
        print()
        print("-" * 108)
        print(
            f"  >>> ADVANCING stage {stage_idx} ({curriculum[stage_idx]}) "
            f"-> stage {next_idx} ({curriculum[next_idx]})"
        )
        print(f"       reason         : {reason}")
        print(f"       frames in stage: {frames_in_stage:,}")
        print(f"       total frames   : {total_frames:,}")
        print("-" * 108)
        print()

        # Save a stage-transition checkpoint BEFORE rebuilding so
        # resume can pick up at the new stage cleanly. The saved
        # stable_updates is 0 — the new stage's advancement decision
        # must be based on its OWN episode outcomes.
        _save_checkpoint(
            checkpoint_path, model, algo, vocab, update, total_frames,
            curriculum=curriculum, stage_idx=next_idx,
            frames_in_stage=0,
            stage_boundaries=stage_boundaries,
            success_window_buffer=deque(maxlen=args.success_window),
            stable_updates=0,
        )

        # Snapshot optimizer state, then rebuild PPOAlgo on next-stage envs.
        optimizer_state = algo.optimizer.state_dict()
        algo, _envs, reshape_reward, _cfg = _build_algo_for_stage(
            model=model, hierarchy_state=hierarchy_state, vocab=vocab,
            env_list=[curriculum[next_idx]] * NUM_ENVS,
            optimizer_state=optimizer_state,
            logger=logger,
        )
        stage_idx = next_idx
        frames_in_stage = 0
        # Reset rolling window and stability counter so the next stage's
        # advancement decision is based on its OWN episode outcomes.
        success_buf.clear()
        stable_updates = 0

    # ---- Final save + plot ------------------------------------------
    _save_checkpoint(
        checkpoint_path, model, algo, vocab, update, total_frames,
        curriculum=curriculum, stage_idx=stage_idx,
        frames_in_stage=frames_in_stage,
        stage_boundaries=stage_boundaries,
        success_window_buffer=success_buf,
        stable_updates=stable_updates,
    )
    csv_file.close()
    if logger is not None:
        logger.close()
    save_plots(history, PLOT_DIR, artifact_stem, curriculum, stage_boundaries)
    save_per_stage_plots(history, PLOT_DIR, artifact_stem, curriculum,
                         stage_boundaries)

    print("\nCurriculum training complete.")
    print(f"  Checkpoint    : {checkpoint_path}")
    print(f"  Metrics       : {csv_path}")
    print(f"  Plot          : {os.path.join(PLOT_DIR, f'{artifact_stem}_training_curves.png')}")
    print(f"  Updates       : {update}")
    print(f"  Frames        : {total_frames:,}")
    print(f"  Final stage   : {stage_idx} ({curriculum[stage_idx]})")
    print(f"  Transitions   : {len(stage_boundaries)} / {len(curriculum) - 1}")


if __name__ == "__main__":
    main()
