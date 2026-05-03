"""
Deterministic rule-based subgoal generator for MiniGrid environments.

Implements forward-only stage machines that produce the subgoal types
from the LGRL paper plus "go near": search for, go near, pickup, open,
close, drop. No "explore" or "go to" subgoals.

Supported environments:

  DoorKey-5x5 (5 stages)
    Stage 0 — search for the [color] key   (skipped if key already visible)
    Stage 1 — pickup the [color] key
    Stage 2 — search for the [color] door  (skipped if door already visible)
    Stage 3 — open the locked [color] door
    Stage 4 — search for the goal

  GoToDoor-*  (2 stages)
    Stage 0 — search for the [color] door  (skipped if door visible)
    Stage 1 — go near the [color] door     (agent adjacent to target)

  GoToObject-*  (2 stages)
    Stage 0 — search for the [color] [object]  (skipped if object visible)
    Stage 1 — go near the [color] [object]     (agent adjacent to target)
              where [object] is one of {key, ball, box}

  UnlockPickup-* (9 stages)
    Stage 0 — search for the [key_color] key       (skipped if visible/carried)
    Stage 1 — go near the [key_color] key          (skipped if already carried)
    Stage 2 — pickup the [key_color] key           (skipped if already carried)
    Stage 3 — search for the [door_color] door     (skipped if door visible)
    Stage 4 — go near the [door_color] door        (skipped if door already open)
    Stage 5 — open the locked [door_color] door    (skipped if door open)
    Stage 6 — search for the [target_color] [obj]  (skipped if target visible)
    Stage 7 — go near the [target_color] [obj]     (skipped if target carried)
    Stage 8 — pickup the [target_color] [obj]

    Note: the key/door color is inferred from the observation (it is NOT in
    the mission text). The target color and object type come from the
    mission "pick up the [color] [object]". The two colors typically differ.

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
UNLOCKPICKUP_STAGES = 9


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
        """Return one of 'doorkey' | 'gotodoor' | 'gotoobject' | 'unlockpickup'.

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

        # UnlockPickup format: "pick up the <color> <box|ball|key>"
        # Also accept "pickup the ..." for robustness.
        if m.startswith("pick up the") or m.startswith("pickup the"):
            return "unlockpickup"

        # Default: DoorKey ("use the key to open the door and then get to the goal")
        return "doorkey"

    @classmethod
    def num_stages(cls, mission: str) -> int:
        """Number of reward-bearing stages for this mission."""
        family = cls.classify_mission(mission)
        if family in ("gotodoor", "gotoobject"):
            return GOTO_STAGES
        if family == "unlockpickup":
            return UNLOCKPICKUP_STAGES
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
        elif family == "unlockpickup":
            subgoal, new_stage = self._unlockpickup_stages(
                stage_index, mission, inventory, entities
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
            target_doors = _find_entities(entities, obj_type="door", color=color or None)
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
            target = _find_entities(
                entities, obj_type=obj_type, color=color or None
            )
            if target:
                return self._gotoobject_stages(1, mission, entities)
            return search_label, 0

        # --- Stage 1: go near object ---
        if stage <= 1:
            return near_label, 1

        # All stages exhausted — keep the near subgoal visible
        return near_label, GOTO_STAGES

    # -- UnlockPickup stage machine ------------------------------------

    def _unlockpickup_stages(
        self, stage: int, mission: str, inventory: str, entities: list[dict]
    ) -> tuple[str, int]:
        """Nine-stage planner for UnlockPickup.

        Mission: "pick up the <target_color> <target_object>"
        (target_object is typically "box" in MiniGrid-UnlockPickup-v0).

        Stages:
          0 — search for the <key_color> key       (skip if key visible/carried)
          1 — go near the <key_color> key          (skip if already carried)
          2 — pickup the <key_color> key           (skip if already carried)
          3 — search for the <door_color> door     (skip if door visible)
          4 — go near the <door_color> door        (skip if door already open)
          5 — open the locked <door_color> door    (skip if door already open)
          6 — search for the <target_color> <target_object>
                                                   (skip if target visible/carried)
          7 — go near the <target_color> <target_object>
                                                   (skip if target carried)
          8 — pickup the <target_color> <target_object>

        Color handling: the key/door color is inferred from the observation
        (a visible key, or a visible door) — it is NOT in the mission text.
        The target color and object type come from the mission. The two
        colors typically differ (e.g. yellow key + purple box).

        Forward-only invariant: returned stage >= input stage.
        """
        inv_words = inventory.lower().split()
        carrying_key = "key" in inv_words

        # Mission-derived target details
        target_color = _mission_color(mission)
        target_type = _mission_object(mission)

        # Are we already carrying the target object?
        carrying_target = bool(
            target_type
            and target_type in inv_words
            and (not target_color or target_color in inv_words)
        )

        keys = _find_entities(entities, obj_type="key")
        locked_doors = _find_entities(entities, obj_type="door", status="locked")
        open_doors = _find_entities(entities, obj_type="door", status="open")
        all_doors = _find_entities(entities, obj_type="door")

        # Has the target object come into view?
        target_visible = bool(_find_entities(
            entities, obj_type=target_type, color=target_color or None
        )) if target_type else False

        # Infer the key color (key matches door in UnlockPickup).
        # If a key is visible we use that; else if a door is visible
        # we copy its color; else if we're carrying the key, parse the
        # inventory string.
        key_color = ""
        if keys:
            key_color = _entity_color(keys[0])
        elif all_doors:
            key_color = _entity_color(all_doors[0])
        elif carrying_key:
            key_color = _extract_color(inv_words)

        # Door color usually equals key color; prefer an actually-visible
        # door's color when available.
        door_color = key_color
        if all_doors:
            door_color = _entity_color(all_doors[0])

        # --- Stage 0: search for the key ---
        if stage <= 0:
            if carrying_key:
                return self._unlockpickup_stages(3, mission, inventory, entities)
            if keys:
                return self._unlockpickup_stages(1, mission, inventory, entities)
            color = key_color or door_color
            label = f"search for the {color} key" if color else "search for the key"
            return label, 0

        # --- Stage 1: go near the key ---
        if stage <= 1:
            if carrying_key:
                return self._unlockpickup_stages(3, mission, inventory, entities)
            color = key_color or door_color
            label = f"go near the {color} key" if color else "go near the key"
            return label, 1

        # --- Stage 2: pickup the key ---
        if stage <= 2:
            if carrying_key:
                return self._unlockpickup_stages(3, mission, inventory, entities)
            color = key_color or door_color
            label = f"pickup the {color} key" if color else "pickup the key"
            return label, 2

        # --- Stage 3: search for the door ---
        if stage <= 3:
            if locked_doors:
                return self._unlockpickup_stages(4, mission, inventory, entities)
            if open_doors:
                # Door already open (e.g. opened earlier this episode) —
                # skip past the go-near-door and open-door stages.
                return self._unlockpickup_stages(6, mission, inventory, entities)
            color = door_color or key_color
            label = (
                f"search for the {color} door" if color else "search for the door"
            )
            return label, 3

        # --- Stage 4: go near the door ---
        if stage <= 4:
            if open_doors:
                return self._unlockpickup_stages(6, mission, inventory, entities)
            color = door_color or key_color
            label = (
                f"go near the {color} door" if color else "go near the door"
            )
            return label, 4

        # --- Stage 5: open the locked door ---
        if stage <= 5:
            if open_doors:
                return self._unlockpickup_stages(6, mission, inventory, entities)
            color = door_color or key_color
            label = (
                f"open the locked {color} door"
                if color else "open the locked door"
            )
            return label, 5

        # --- Stage 6: search for the target object ---
        if stage <= 6:
            if carrying_target:
                return self._unlockpickup_stages(8, mission, inventory, entities)
            if target_visible:
                return self._unlockpickup_stages(7, mission, inventory, entities)
            if target_color and target_type:
                label = f"search for the {target_color} {target_type}"
            elif target_type:
                label = f"search for the {target_type}"
            else:
                label = "search for the target"
            return label, 6

        # --- Stage 7: go near the target object ---
        if stage <= 7:
            if carrying_target:
                return self._unlockpickup_stages(8, mission, inventory, entities)
            if target_color and target_type:
                label = f"go near the {target_color} {target_type}"
            elif target_type:
                label = f"go near the {target_type}"
            else:
                label = "go near the target"
            return label, 7

        # --- Stage 8: pickup the target object ---
        if stage <= 8:
            if target_color and target_type:
                label = f"pickup the {target_color} {target_type}"
            elif target_type:
                label = f"pickup the {target_type}"
            else:
                label = "pickup the target"
            return label, 8

        # All stages exhausted — keep the final subgoal visible to the
        # agent's text stream while the training loop's
        # stage_index >= n_subgoals check prevents further reward.
        if target_color and target_type:
            label = f"pickup the {target_color} {target_type}"
        elif target_type:
            label = f"pickup the {target_type}"
        else:
            label = "pickup the target"
        return label, UNLOCKPICKUP_STAGES


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
    """Extract object type from a mission string.

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
        ("MiniGrid-UnlockPickup-v0", 9),
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
                f"  seed={seed} family={family:<13} "
                f"mission={mission!r} -> stage={new_stage} '{subgoal}'"
            )

            # Invariants: no forbidden prefixes ("go near" is allowed,
            # "go to " is not).
            assert "explore" not in subgoal.lower()
            assert not subgoal.lower().startswith("go to ")

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
