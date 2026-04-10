"""
Deterministic rule-based subgoal generator for MiniGrid DoorKey environments.

Implements a forward-only stage machine that produces exactly the subgoal
types from the LGRL paper: search for, pickup, open, close, drop.
No "explore" or "go to" subgoals.

For DoorKey-5x5, the canonical stage sequence is:
  Stage 0 — search for the [color] key   (skipped if key already visible)
  Stage 1 — pickup the [color] key
  Stage 2 — search for the [color] door  (skipped if door already visible)
  Stage 3 — open the locked [color] door
  Stage 4 — search for the goal

The stage index only advances forward, preventing the agent from farming
rewards by repeating earlier subgoals.
"""

from __future__ import annotations

import json

KNOWN_COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}


class RuleBasedPlanner:
    """
    Stage-based deterministic planner for DoorKey missions.

    Interface:
        get_subgoal(mission, env_json_str, direction, stage_index)
        -> (subgoal_string, new_stage_index)

    ``last_raw_response`` is set on each call for JSONL logging parity
    with ``LLMPlanner``.
    """

    # Total number of rewardable stages
    NUM_STAGES = 5

    def __init__(self) -> None:
        self.last_raw_response: str | None = None

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

        subgoal, new_stage = self._doorkey_stages(
            stage_index, inventory, entities
        )
        self.last_raw_response = subgoal
        return subgoal, new_stage

    # ------------------------------------------------------------------

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
        goals = _find_entities(entities, obj_type="goal")

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
                # Already have key, skip to door phase
                return self._doorkey_stages(2, inventory, entities)
            if keys:
                # Key visible, skip search, go to pickup
                return self._doorkey_stages(1, inventory, entities)
            color = key_color or door_color
            label = f"search for the {color} key" if color else "search for the key"
            return label, 0

        # --- Stage 1: pickup the key ---
        if stage <= 1:
            if carrying_key:
                # Already picked up, advance to door phase
                return self._doorkey_stages(2, inventory, entities)
            color = key_color or door_color
            label = f"pickup the {color} key" if color else "pickup the key"
            return label, 1

        # --- Stage 2: search for locked door ---
        if stage <= 2:
            if locked_doors:
                # Door visible, skip search, go to open
                return self._doorkey_stages(3, inventory, entities)
            # Check if door is already open (no locked door visible, but an open door is)
            open_doors = _find_entities(entities, obj_type="door", status="open")
            if open_doors:
                # Door already open, skip to goal phase
                return self._doorkey_stages(4, inventory, entities)
            color = door_color or key_color
            label = f"search for the {color} door" if color else "search for the door"
            return label, 2

        # --- Stage 3: open the locked door ---
        if stage <= 3:
            # Check if door is already open
            open_doors = _find_entities(entities, obj_type="door", status="open")
            if open_doors:
                return self._doorkey_stages(4, inventory, entities)
            color = door_color or key_color
            label = f"open the locked {color} door" if color else "open the locked door"
            return label, 3

        # --- Stage 4: search for goal ---
        if stage <= 4:
            return "search for the goal", 4

        # All stages exhausted — keep last subgoal (no more rewards)
        return "search for the goal", 5


# -- helper functions (module-level) ------------------------------------

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


# -- self-test -----------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import minigrid  # noqa: F401
    from utils.env_parser import parse_env_description

    planner = RuleBasedPlanner()
    env = gym.make("MiniGrid-DoorKey-5x5-v0")

    print("=" * 60)
    print("  RuleBasedPlanner (stage-based) -- Self-Test")
    print("=" * 60)

    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        env_json = parse_env_description(obs["image"], uw.carrying)

        subgoal, new_stage = planner.get_subgoal(
            obs["mission"], env_json, obs["direction"], stage_index=0
        )
        print(f"Seed {seed} | mission={obs['mission']!r} -> stage={new_stage} '{subgoal}'")

        # Verify no "explore" or "go to"
        assert "explore" not in subgoal.lower()
        assert not subgoal.lower().startswith("go to")

    # Test stage advancement
    print("\n--- Stage advancement test ---")
    obs, _ = env.reset(seed=0)
    uw = env.unwrapped
    for stage in range(6):
        env_json = parse_env_description(obs["image"], uw.carrying)
        subgoal, new_stage = planner.get_subgoal(
            obs["mission"], env_json, obs["direction"], stage_index=stage
        )
        print(f"  Input stage={stage} -> '{subgoal}' (output stage={new_stage})")
        assert new_stage >= stage, "Stage must not go backwards"

    print("\n" + "=" * 60)
    print("  All assertions passed.")
    print("=" * 60)
