"""
Shared helpers for env-aware training scripts.

- SUPPORTED_ENVS: the MiniGrid environments covered by the LGRL paper
  that this project currently implements.
- env_stem(): short filesystem-safe label derived from the env name.
- resolve_artifact_stem(): joins a base stem with an env stem. When the
  env is the legacy default (DoorKey-5x5) the base is returned unchanged
  so existing checkpoints/CSVs keep working.
- env_max_steps(): reads ``env.unwrapped.max_steps`` — the paper's Tmax.
"""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid envs)

# Environment families the rule-based planner currently supports.
# Other MiniGrid envs with the same mission templates will also work
# but are not listed here as "officially supported".
SUPPORTED_ENVS: tuple[str, ...] = (
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-GoToDoor-8x8-v0",
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
)

# DoorKey-5x5 keeps legacy artifact names ("baseline.pt", "lgrl.pt",
# "lgrl_rule.pt") so existing runs remain resumable. Any other env gets
# a tagged stem like "lgrl_rule_gotodoor5x5".
LEGACY_DEFAULT_ENV = "MiniGrid-DoorKey-5x5-v0"


def env_stem(env_name: str) -> str:
    """Return a lowercase filesystem-safe tag for an env name.

    Examples:
        MiniGrid-DoorKey-5x5-v0          -> doorkey5x5
        MiniGrid-GoToDoor-5x5-v0         -> gotodoor5x5
        MiniGrid-GoToObject-6x6-N2-v0    -> gotoobject6x6n2
    """
    name = env_name
    if name.startswith("MiniGrid-"):
        name = name[len("MiniGrid-"):]
    if name.endswith("-v0"):
        name = name[: -len("-v0")]
    return name.lower().replace("-", "")


def resolve_artifact_stem(base: str, env_name: str) -> str:
    """Return the artifact stem to use for a run.

    - If env is the legacy default, return ``base`` unchanged.
    - Otherwise return ``f"{base}_{env_stem(env_name)}"``.
    """
    if env_name == LEGACY_DEFAULT_ENV:
        return base
    return f"{base}_{env_stem(env_name)}"


def env_max_steps(env_name: str) -> int:
    """Read the env's max_steps (paper Tmax) by instantiating it once."""
    env = gym.make(env_name)
    try:
        return int(env.unwrapped.max_steps)
    finally:
        env.close()
