"""
Evaluation suite configuration for LGRL checkpoint benchmarking.

Default suite mirrors the LGRL paper environments used for transfer
testing (GoToDoor, GoToObject, KeyCorridor, UnlockPickup) plus the
project's DoorKey-5x5 training default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.env_utils import env_max_steps
from utils.rule_based_planner import DOORKEY_STAGES, RuleBasedPlanner

# Paper Table 2 style episode count
DEFAULT_NUM_EPISODES = 1000
DEFAULT_SEED_START = 0
DEFAULT_SUBGOAL_TIMEOUT_MULT = 2.0

# Subgoal advance matches training (train_lgrl.py)
SUBGOAL_TIMEOUT_MULT = DEFAULT_SUBGOAL_TIMEOUT_MULT


@dataclass(frozen=True)
class BenchmarkEnvSpec:
    """One environment in the evaluation suite."""

    key: str
    display_name: str
    env_id: str
    n_subgoals_override: Optional[int] = None
    description: str = ""

    def resolve_n_subgoals(self, mission: str) -> int:
        if self.n_subgoals_override is not None:
            return self.n_subgoals_override
        if "keycorridor" in self.env_id.lower():
            # Mission is often "pick up the ball" — misclassified as unlockpickup.
            return DOORKEY_STAGES
        return RuleBasedPlanner.num_stages(mission)

    def resolve_t_max(self) -> int:
        return env_max_steps(self.env_id)


# Keys accepted by --envs (comma-separated subset).
DEFAULT_EVAL_SUITE: tuple[BenchmarkEnvSpec, ...] = (
    BenchmarkEnvSpec(
        key="gotodoor",
        display_name="GoToDoor",
        env_id="MiniGrid-GoToDoor-5x5-v0",
        description="Navigate to the mission-specified door and issue done.",
    ),
    BenchmarkEnvSpec(
        key="gotoobject",
        display_name="GoToObject",
        env_id="MiniGrid-GoToObject-6x6-N2-v0",
        description="Navigate to the mission-specified object and issue done.",
    ),
    BenchmarkEnvSpec(
        key="doorkey5x5",
        display_name="DoorKey 5x5",
        env_id="MiniGrid-DoorKey-5x5-v0",
        description="Key–door–goal sequence (project default training env).",
    ),
    BenchmarkEnvSpec(
        key="unlockpickup",
        display_name="UnlockPickup",
        env_id="MiniGrid-UnlockPickup-v0",
        description="Find key, unlock door, pick up target in locked room.",
    ),
    BenchmarkEnvSpec(
        key="keycorridor",
        display_name="KeyCorridor",
        env_id="MiniGrid-KeyCorridor-S3R3-v0",
        n_subgoals_override=DOORKEY_STAGES,
        description=(
            "Multi-room key corridor (3×3 rooms). Uses doorkey stage budget "
            "for hierarchy; LLM supplies subgoals at eval time."
        ),
    ),
)

EVAL_SUITE_BY_KEY: dict[str, BenchmarkEnvSpec] = {
    spec.key: spec for spec in DEFAULT_EVAL_SUITE
}


def get_eval_suite(env_keys: Optional[list[str]] = None) -> list[BenchmarkEnvSpec]:
    """Return benchmark specs, optionally filtered by key."""
    if not env_keys:
        return list(DEFAULT_EVAL_SUITE)
    unknown = [k for k in env_keys if k not in EVAL_SUITE_BY_KEY]
    if unknown:
        valid = ", ".join(sorted(EVAL_SUITE_BY_KEY))
        raise ValueError(
            f"Unknown eval env key(s): {unknown}. Valid keys: {valid}"
        )
    return [EVAL_SUITE_BY_KEY[k] for k in env_keys]
