"""
Shared helpers for env-aware training scripts.

- SUPPORTED_ENVS: the MiniGrid environments this project implements.
- env_stem(): short filesystem-safe label derived from the env name.
- resolve_artifact_stem(): joins a base stem with an env stem. When the
  env is the legacy default (DoorKey-5x5) the base is returned unchanged
  so existing checkpoints/CSVs keep working.
- env_max_steps(): reads ``env.unwrapped.max_steps`` — the env's Tmax.
"""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid envs)

# Environment families the rule-based planner currently supports.
# Other MiniGrid envs with the same mission templates will also work
# but are not listed here as "officially supported".
#
# KeyCorridor uses the same mission template as UnlockPickup
# ("pick up the <color> <obj>") and is handled by the same 10-stage
# machine in RuleBasedPlanner. The six variants are kept ordered from
# smallest to largest for use as the default curriculum.
SUPPORTED_ENVS: tuple[str, ...] = (
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-GoToDoor-8x8-v0",
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-KeyCorridorS4R3-v0",
    "MiniGrid-KeyCorridorS5R3-v0",
    "MiniGrid-KeyCorridorS6R3-v0",
)

# Default curriculum for train_lgrl_curriculum.py.
#
# The agent is first trained on the simpler GoToDoor and GoToObject tasks
# to acquire basic navigation abilities, then the curriculum progresses to
# increasingly larger instances of the KeyCorridor environment.
#
# Ordering: 3 GoToDoor sizes -> 2 GoToObject sizes -> 6 KeyCorridor
# variants (S3R1 -> S6R3). UnlockPickup is intentionally NOT part of the
# curriculum — we treat it as a zero-shot transfer test environment. The
# user can override via --curriculum.
DEFAULT_CURRICULUM: tuple[str, ...] = (
    # Stage 1-3: Basic navigation - go to a named door
    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-GoToDoor-8x8-v0",
    # Stage 4-5: Basic navigation - go to a named object
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
    # Stage 6-11: KeyCorridor, smallest to largest
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-KeyCorridorS4R3-v0",
    "MiniGrid-KeyCorridorS5R3-v0",
    "MiniGrid-KeyCorridorS6R3-v0",
)

# Sub-curriculum: KeyCorridor only. Kept as a convenience for ablation
# runs where the GoTo bootstrap is skipped.
DEFAULT_KEYCORRIDOR_CURRICULUM: tuple[str, ...] = (
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-KeyCorridorS4R3-v0",
    "MiniGrid-KeyCorridorS5R3-v0",
    "MiniGrid-KeyCorridorS6R3-v0",
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
        MiniGrid-UnlockPickup-v0         -> unlockpickup
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
    """Read the env's max_steps (Tmax) by instantiating it once."""
    env = gym.make(env_name)
    try:
        return int(env.unwrapped.max_steps)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Mixed-task training helpers (e.g. UnlockPickup + GoToObject 1:3)
# ---------------------------------------------------------------------------

def parse_mix_spec(spec: str) -> list[tuple[str, int]]:
    """Parse a ``--mix env1:r1,env2:r2`` string into ``[(env, ratio), ...]``.

    Example:
        "MiniGrid-UnlockPickup-v0:1,MiniGrid-GoToObject-6x6-N2-v0:3"
        -> [('MiniGrid-UnlockPickup-v0', 1), ('MiniGrid-GoToObject-6x6-N2-v0', 3)]

    Validates that env names are in SUPPORTED_ENVS and ratios are positive
    integers. At least 2 envs must be specified.
    """
    chunks = [c.strip() for c in spec.split(",") if c.strip()]
    if len(chunks) < 2:
        raise SystemExit(
            f"--mix needs at least two 'env:ratio' pairs separated by ','; "
            f"got {spec!r}"
        )
    items: list[tuple[str, int]] = []
    seen: set[str] = set()
    for chunk in chunks:
        if ":" not in chunk:
            raise SystemExit(
                f"--mix entry {chunk!r} missing ':ratio' suffix"
            )
        env_name, ratio_str = chunk.rsplit(":", 1)
        env_name = env_name.strip()
        if env_name not in SUPPORTED_ENVS:
            raise SystemExit(
                f"--mix env {env_name!r} not in SUPPORTED_ENVS"
            )
        if env_name in seen:
            raise SystemExit(
                f"--mix env {env_name!r} repeated"
            )
        try:
            ratio = int(ratio_str.strip())
        except ValueError:
            raise SystemExit(
                f"--mix ratio for {env_name!r} must be an integer; got {ratio_str!r}"
            )
        if ratio <= 0:
            raise SystemExit(
                f"--mix ratio for {env_name!r} must be positive; got {ratio}"
            )
        items.append((env_name, ratio))
        seen.add(env_name)
    return items


def build_env_list(mix: list[tuple[str, int]], num_envs: int) -> list[str]:
    """Build a length-``num_envs`` list of env names from a mix spec.

    Each block of (sum-of-ratios) slots contains ``ratio`` copies of each
    env in the order given. With ``mix=[(UP,1),(GO,3)]`` and
    ``num_envs=16``, the result is::

        [UP, UP, UP, UP, GO, GO, GO, GO, GO, GO, GO, GO, GO, GO, GO, GO]

    (4 + 12 = 16). The total ratio must divide num_envs evenly.
    """
    total_ratio = sum(r for _, r in mix)
    if num_envs % total_ratio != 0:
        raise SystemExit(
            f"NUM_ENVS={num_envs} must be divisible by total mix ratio "
            f"{total_ratio} (got mix {mix})"
        )
    block_count = num_envs // total_ratio
    env_list: list[str] = []
    for env_name, ratio in mix:
        env_list.extend([env_name] * (ratio * block_count))
    return env_list


def mix_artifact_stem(base: str, mix: list[tuple[str, int]]) -> str:
    """Artifact stem for a mixed-task run.

    Format: ``{base}_mix_{env1stem}_{env2stem}_..._{r1}to{r2}to...``

    Example::

        mix_artifact_stem("lgrl_rule",
                          [("MiniGrid-UnlockPickup-v0", 1),
                           ("MiniGrid-GoToObject-6x6-N2-v0", 3)])
        -> "lgrl_rule_mix_unlockpickup_gotoobject6x6n2_1to3"
    """
    env_tags = "_".join(env_stem(e) for e, _ in mix)
    ratios = "to".join(str(r) for _, r in mix)
    return f"{base}_mix_{env_tags}_{ratios}"


# ---------------------------------------------------------------------------
# Curriculum-task helpers (KeyCorridor progressive training)
# ---------------------------------------------------------------------------

def parse_curriculum_spec(spec: str) -> list[str]:
    """Parse ``--curriculum env1,env2,env3`` into a list of env names.

    Validates that each env is in ``SUPPORTED_ENVS`` and that at least two
    envs are listed (a one-env curriculum is just a normal single-env
    run).
    """
    envs = [e.strip() for e in spec.split(",") if e.strip()]
    if len(envs) < 2:
        raise SystemExit(
            f"--curriculum needs at least two env names separated by ',', "
            f"got {spec!r}"
        )
    seen: set[str] = set()
    for e in envs:
        if e not in SUPPORTED_ENVS:
            raise SystemExit(f"--curriculum env {e!r} not in SUPPORTED_ENVS")
        if e in seen:
            raise SystemExit(f"--curriculum env {e!r} repeated")
        seen.add(e)
    return envs


def curriculum_artifact_stem(base: str, envs: list[str]) -> str:
    """Artifact stem for a curriculum run.

    Short forms for the two canonical curricula keep filenames readable:
      - The full default curriculum (DEFAULT_CURRICULUM) -> ``..._default``
      - A contiguous KeyCorridor-only sweep -> ``..._keycorridor_<first>_to_<last>``

    Anything else falls back to listing every env stem joined by ``_``.

    Examples::

        # Default curriculum (GoToDoor x3 -> GoToObject x2 -> KC x6)
        curriculum_artifact_stem("lgrl_rule", list(DEFAULT_CURRICULUM))
        -> "lgrl_rule_curriculum_default"

        # KeyCorridor-only ablation
        curriculum_artifact_stem("lgrl_rule",
            ["MiniGrid-KeyCorridorS3R1-v0",
             "MiniGrid-KeyCorridorS3R2-v0",
             "MiniGrid-KeyCorridorS3R3-v0"])
        -> "lgrl_rule_curriculum_keycorridor_s3r1_to_s3r3"

        # Arbitrary custom curriculum
        curriculum_artifact_stem("lgrl_rule",
            ["MiniGrid-DoorKey-5x5-v0",
             "MiniGrid-DoorKey-8x8-v0"])
        -> "lgrl_rule_curriculum_doorkey5x5_doorkey8x8"
    """
    if tuple(envs) == DEFAULT_CURRICULUM:
        return f"{base}_curriculum_default"

    is_keycorridor_only = all("KeyCorridor" in e for e in envs)
    if is_keycorridor_only and len(envs) >= 2:
        # KeyCorridorS3R1 -> "s3r1"
        def short(name: str) -> str:
            tag = env_stem(name)
            return tag[len("keycorridor"):]
        return (
            f"{base}_curriculum_keycorridor_"
            f"{short(envs[0])}_to_{short(envs[-1])}"
        )
    env_tags = "_".join(env_stem(e) for e in envs)
    return f"{base}_curriculum_{env_tags}"
