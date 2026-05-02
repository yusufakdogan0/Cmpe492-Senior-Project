"""
Deterministic rule-based subgoal generator for MiniGrid environments.

Implements forward-only stage machines that produce exactly the subgoal
types from the LGRL paper: search for, pickup, open, close, drop.
No "explore" or "go to" subgoals.

Supported environments:

  DoorKey-5x5 (5 stages)
    Stage 0 — search for the [color] key   (skipped if key already visible)
    Stage 1 — pickup the [color] key
    Stage 2 — search for the [color] door  (skipped if door already visible)
    Stage 3 — open the locked [color] door
    Stage 4 — search for the goal

  GoToDoor-*  (2 stages)
    Stage 0 — search for the [color] door  (object enters FOV)
    Stage 1 — go near the [color] door     (agent adjacent to target)

  GoToObject-*  (2 stages)
    Stage 0 — search for the [color] [object]  (object enters FOV)
    Stage 1 — go near the [color] [object]     (agent adjacent to target)
              where [object] is one of {key, ball, box}

In all cases the stage index only advances forward, preventing the agent
from farming rewards by repeating earlier subgoals.
"""

from __future__ import annotations

import json

KNOWN_COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}
KNOWN_OBJECTS = {"key", "ball", "box", "door", "goal"}

# Per-mission-family stage counts (paper Eq. 6 n)
DOORKEY_STAGES = 5
GOTO_STAGES = 2


class RuleBasedPlanner:
    """
    Stage-based deterministic planner for LGRL-style missions.

    Interface:
        get_subgoal(mission, env_json_str, direction, stage_index)
        -> (subgoal_string, new_stage_index)

        num_stages(mission) -> int

    ``last_raw_response`` is set on each call for JSONL logging parity
    with ``LLMPlanner``.
    """

    # Default stage count, used if mission family cannot be identified.
    # Training scripts should prefer ``num_stages(mission)``.
    NUM_STAGES = DOORKEY_STAGES

    def __init__(self) -> None:
        self.last_raw_response: str | None = None

    # -- mission-family dispatch ---------------------------------------

    @staticmethod
    def classify_mission(mission: str) -> str:
        """Return one of 'doorkey' | 'gotodoor' | 'gotoobject'.

        Detection is purely from the mission string, not the environment
        name, so custom envs with the same mission template work too.
        """
        m = mission.lower().strip()

        # GoToDoor format: "go to the <color> door"
        if m.startswith("go to the") and "door" in m:
            return "gotodoor"

        # GoToObject format: "go to the <color> <key|ball|box>"
        if m.startswith("go to the") and any(t in m for t in ("key", "ball", "box")):
            return "gotoobject"

        # Default: DoorKey ("use the key to open the door and then get to the goal")
        return "doorkey"

    @classmethod
    def num_stages(cls, mission: str) -> int:
        """Number of reward-bearing stages for this mission."""
        family = cls.classify_mission(mission)
        if family in ("gotodoor", "gotoobject"):
            return GOTO_STAGES
        return DOORKEY_STAGES

    # -- main entry point ----------------------------------------------

    def get_subgoal(
        self,
        mission: str,
        env_json_str: str,
        direction: int | str,
        stage_index: int,
    ) -> tuple[str, int]:
        """Return (subgoal_string, new_stage_index).

        The returned stage_index is always >= the input stage_index,
        guaranteeing forward-only progress.
        """
        state = json.loads(env_json_str)
        inventory = state.get("inventory", "empty")
        entities = state.get("entities", [])

        family = self.classify_mission(mission)
        if family == "gotodoor":
            subgoal, new_stage = self._gotodoor_stages(
                stage_index, mission, entities
            )
        elif family == "gotoobject":
            subgoal, new_stage = self._gotoobject_stages(
                stage_index, mission, entities
            )
        else:
            subgoal, new_stage = self._doorkey_stages(
                stage_index, inventory, entities
            )

        self.last_raw_response = subgoal
        return subgoal, new_stage

    # -- DoorKey stage machine -----------------------------------------

    def _doorkey_stages(
        self, stage: int, inventory: str, entities: list[dict]
    ) -> tuple[str, int]:
        """Walk the DoorKey stage machine, skipping stages whose
        preconditions are already met."""

        inv_words = inventory.lower().split()
        carrying_key = "key" in inv_words

        keys = _find_entities(entities, obj_type="key")
        locked_doors = _find_entities(entities, obj_type="door", status="locked")
        all_doors = _find_entities(entities, obj_type="door")

        # Infer key color from visible key or visible door
        key_color = ""
        if keys:
            key_color = _entity_color(keys[0])
        elif all_doors:
            key_color = _entity_color(all_doors[0])
        elif carrying_key:
            key_color = _extract_color(inv_words)

        # Infer door color (same as key color in DoorKey envs)
        door_color = key_color
        if all_doors:
            door_color = _entity_color(all_doors[0])

        # --- Stage 0: search for key ---
        if stage <= 0:
            if carrying_key:
                return self._doorkey_stages(2, inventory, entities)
            if keys:
                return self._doorkey_stages(1, inventory, entities)
            color = key_color or door_color
            label = f"search for the {color} key" if color else "search for the key"
            return label, 0

        # --- Stage 1: pickup the key ---
        if stage <= 1:
            if carrying_key:
                return self._doorkey_stages(2, inventory, entities)
            color = key_color or door_color
            label = f"pickup the {color} key" if color else "pickup the key"
            return label, 1

        # --- Stage 2: search for locked door ---
        if stage <= 2:
            if locked_doors:
                return self._doorkey_stages(3, inventory, entities)
            open_doors = _find_entities(entities, obj_type="door", status="open")
            if open_doors:
                return self._doorkey_stages(4, inventory, entities)
            color = door_color or key_color
            label = f"search for the {color} door" if color else "search for the door"
            return label, 2

        # --- Stage 3: open the locked door ---
        if stage <= 3:
            open_doors = _find_entities(entities, obj_type="door", status="open")
            if open_doors:
                return self._doorkey_stages(4, inventory, entities)
            color = door_color or key_color
            label = f"open the locked {color} door" if color else "open the locked door"
            return label, 3

        # --- Stage 4: search for goal ---
        if stage <= 4:
            return "search for the goal", 4

        # All stages exhausted
        return "search for the goal", DOORKEY_STAGES

    # -- GoToDoor stage machine ----------------------------------------

    def _gotodoor_stages(
        self, stage: int, mission: str, entities: list[dict]
    ) -> tuple[str, int]:
        """Two-stage planner for GoToDoor.

        Mission: "go to the <color> door"
        Stage 0 — search for the <color> door  (door enters FOV)
        Stage 1 — go near the <color> door     (agent adjacent)

        The mission is solved by executing `done` next to the target door;
        the mission-level reward (Eq. 5) handles that terminal action.
        """
        color = _mission_color(mission)
        search_label = (
            f"search for the {color} door" if color else "search for the door"
        )
        near_label = (
            f"go near the {color} door" if color else "go near the door"
        )

        # --- Stage 0: search for door ---
        if stage <= 0:
            # Skip if target door is already visible
            target_doors = _find_entities(entities, obj_type="door", color=color)
            if target_doors:
                return self._gotodoor_stages(1, mission, entities)
            return search_label, 0

        # --- Stage 1: go near door ---
        if stage <= 1:
            return near_label, 1

        # All stages exhausted — keep the near subgoal visible
        return near_label, GOTO_STAGES

    # -- GoToObject stage machine --------------------------------------

    def _gotoobject_stages(
        self, stage: int, mission: str, entities: list[dict]
    ) -> tuple[str, int]:
        """Two-stage planner for GoToObject.

        Mission: "go to the <color> <key|ball|box>"
        Stage 0 — search for the <color> <object>  (object enters FOV)
        Stage 1 — go near the <color> <object>     (agent adjacent)
        """
        color = _mission_color(mission)
        obj_type = _mission_object(mission)

        if color and obj_type:
            search_label = f"search for the {color} {obj_type}"
            near_label = f"go near the {color} {obj_type}"
        elif obj_type:
            search_label = f"search for the {obj_type}"
            near_label = f"go near the {obj_type}"
        else:
            search_label = "search for the object"
            near_label = "go near the object"

        # --- Stage 0: search for object ---
        if stage <= 0:
            # Skip if target is already visible
            target = _find_entities(entities, obj_type=obj_type, color=color)
            if target:
                return self._gotoobject_stages(1, mission, entities)
            return search_label, 0

        # --- Stage 1: go near object ---
        if stage <= 1:
            return near_label, 1

        # All stages exhausted — keep the near subgoal visible
        return near_label, GOTO_STAGES


# -- helper functions (module-level) -----------------------------------

def _find_entities(
    entities: list[dict],
    obj_type: str,
    status: str | None = None,
    color: str | None = None,
) -> list[dict]:
    results = []
    for e in entities:
        words = e.get("entity", "").lower().split()
        if obj_type and obj_type not in words:
            continue
        if status and status not in words:
            continue
        if color and color not in words:
            continue
        results.append(e)
    return results


def _entity_color(entity: dict) -> str:
    for word in entity.get("entity", "").lower().split():
        if word in KNOWN_COLORS:
            return word
    return ""


def _extract_color(words: list[str]) -> str:
    for word in words:
        if word in KNOWN_COLORS:
            return word
    return ""


def _mission_color(mission: str) -> str:
    """Extract color token from a mission string."""
    for word in mission.lower().split():
        if word in KNOWN_COLORS:
            return word
    return ""


def _mission_object(mission: str) -> str:
    """Extract object type from a 'go to the ...' mission string.

    Returns one of {'key', 'ball', 'box', 'door'} or '' if none found.
    """
    for word in mission.lower().split():
        if word in KNOWN_OBJECTS:
            return word
    return ""


# -- self-test ---------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import minigrid  # noqa: F401
    from utils.env_parser import parse_env_description

    planner = RuleBasedPlanner()

    test_envs = [
        ("MiniGrid-DoorKey-5x5-v0", 5),
        ("MiniGrid-GoToDoor-5x5-v0", 2),
        ("MiniGrid-GoToObject-6x6-N2-v0", 2),
    ]

    print("=" * 70)
    print("  RuleBasedPlanner (multi-env stage-based) -- Self-Test")
    print("=" * 70)

    for env_name, expected_stages in test_envs:
        print(f"\n--- {env_name} ---")
        env = gym.make(env_name)

        for seed in range(3):
            obs, _ = env.reset(seed=seed)
            uw = env.unwrapped
            env_json = parse_env_description(obs["image"], uw.carrying)

            mission = obs["mission"]
            family = planner.classify_mission(mission)
            n_stages = planner.num_stages(mission)
            assert n_stages == expected_stages, (
                f"stage count mismatch for {env_name}: "
                f"expected {expected_stages}, got {n_stages}"
            )

            subgoal, new_stage = planner.get_subgoal(
                mission, env_json, obs["direction"], stage_index=0
            )
            print(
                f"  seed={seed} family={family:<10} "
                f"mission={mission!r} -> stage={new_stage} '{subgoal}'"
            )

            # Invariants: no forbidden prefixes
            assert "explore" not in subgoal.lower()
            assert not subgoal.lower().startswith("go to")

        # Stage advancement test — drive through every stage
        print(f"  Stage advancement (seed=0):")
        obs, _ = env.reset(seed=0)
        uw = env.unwrapped
        for stage in range(n_stages + 1):
            env_json = parse_env_description(obs["image"], uw.carrying)
            subgoal, new_stage = planner.get_subgoal(
                obs["mission"], env_json, obs["direction"],
                stage_index=stage,
            )
            print(f"    input stage={stage} -> '{subgoal}' (output stage={new_stage})")
            # Forward-only invariant: within valid range, stage never regresses
            assert new_stage >= min(stage, n_stages), \
                f"Stage must not regress: input={stage}, output={new_stage}"

    print("\n" + "=" * 70)
    print("  All assertions passed.")
    print("=" * 70)
