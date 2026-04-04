"""
Deterministic rule-based subgoal generator for MiniGrid environments.

Serves as an oracle ablation baseline: it produces the same subgoal
format as ``LLMPlanner`` but uses hand-coded heuristics instead of an
LLM query.  This lets us isolate the contribution of LLM guidance from
the reward scaffolding when comparing results.

Supports:
    - DoorKey variants  ("use the key to open the door and then get to the goal")
    - UnlockPickup      ("pick up the <color> <object>")
    - GoToObject / GoToDoor  (mapped to ``search for`` / ``open`` / ``pickup`` / ``close`` — no ``go to`` subgoals)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

KNOWN_COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}
KNOWN_TYPES = {"key", "door", "ball", "box", "goal"}


class RuleBasedPlanner:
    """
    Drop-in replacement for ``LLMPlanner`` that uses deterministic rules.

    Shares the same ``get_subgoal`` interface so either planner can be
    injected into the training loop without changes.

    ``last_raw_response`` is set to the returned subgoal string on each call
    (parity with ``LLMPlanner`` for JSONL logging).
    """

    def __init__(self) -> None:
        self.last_raw_response: str | None = None

    def get_subgoal(
        self,
        mission: str,
        env_json_str: str,
        direction: int | str,
        past_subgoals: list[str],
    ) -> str:
        result = self._get_subgoal_impl(
            mission, env_json_str, direction, past_subgoals
        )
        self.last_raw_response = result
        return result

    def _get_subgoal_impl(
        self,
        mission: str,
        env_json_str: str,
        direction: int | str,
        past_subgoals: list[str],
    ) -> str:
        state = json.loads(env_json_str)
        inventory = state.get("inventory", "empty")
        entities = state.get("entities", [])
        mission_lower = mission.lower()

        if "use the key" in mission_lower or "open the door" in mission_lower:
            return self._doorkey_subgoal(mission_lower, inventory, entities)

        if "pick up" in mission_lower or "pickup" in mission_lower:
            return self._pickup_subgoal(mission_lower, inventory, entities)

        if "go to" in mission_lower:
            return self._goto_subgoal(mission_lower, entities)

        # Unrecognized mission — best-effort: parse color+type from mission
        color, obj_type = _parse_color_and_type(mission_lower)
        if color and obj_type:
            return f"search for the {color} {obj_type}"
        if obj_type:
            return f"search for the {obj_type}"
        return "search for the key"

    # -- mission-specific strategies ------------------------------------

    def _doorkey_subgoal(
        self, mission_lower: str, inventory: str, entities: list[dict]
    ) -> str:
        """DoorKey: get key -> open locked door -> goal via pickup (visible) or search (not)."""
        keys = self._find_entities(entities, obj_type="key")
        locked_doors = self._find_entities(entities, obj_type="door", status="locked")
        all_doors = self._find_entities(entities, obj_type="door")
        goals = self._find_entities(entities, obj_type="goal")

        inv_words = inventory.lower().split()
        carrying_key = "key" in inv_words

        if not carrying_key:
            if keys:
                color = self._entity_color(keys[0])
                return f"pickup the {color} key"
            # Key not visible — infer color from visible door if possible
            if all_doors:
                door_color = self._entity_color(all_doors[0])
                return f"search for the {door_color} key"
            return "search for the key"

        # Carrying key — look for locked door
        if locked_doors:
            color = self._entity_color(locked_doors[0])
            return f"open the locked {color} door"

        # Door open (no locked door in FOV): goal visible -> pickup; else search goal / door
        if goals:
            color = self._entity_color(goals[0])
            return f"pickup the {color} goal"

        gc = _goal_color_hint(mission_lower, goals)
        if all_doors:
            return f"search for the {gc} goal"

        inv_color = _extract_color_from_words(inv_words)
        if inv_color:
            return f"search for the locked {inv_color} door"
        return "search for the locked door"

    def _pickup_subgoal(
        self, mission: str, inventory: str, entities: list[dict]
    ) -> str:
        """UnlockPickup / generic pickup: acquire prerequisites then pick up target."""
        target_color, target_type = _parse_color_and_type(mission)

        # Already carrying the target — mission should complete on its own.
        inv_words = inventory.lower().split()
        if target_type and target_type in inv_words and target_color in inv_words:
            # Dead code in practice (env terminates on pickup), but safe fallback
            return f"pickup the {target_color} {target_type}"

        # Check if a locked door blocks the way; if so, get the key first.
        locked_doors = self._find_entities(entities, obj_type="door", status="locked")
        if locked_doors and "key" not in inv_words:
            keys = self._find_entities(entities, obj_type="key")
            if keys:
                color = self._entity_color(keys[0])
                return f"pickup the {color} key"
            # No key visible — infer color from visible locked door
            door_color = self._entity_color(locked_doors[0])
            if door_color:
                return f"search for the {door_color} key"
            return "search for the key"

        if locked_doors and "key" in inv_words:
            color = self._entity_color(locked_doors[0])
            return f"open the locked {color} door"

        # No door blocking — go for the target directly.
        targets = self._find_entities(
            entities, obj_type=target_type, color=target_color
        )
        if targets:
            color = self._entity_color(targets[0])
            return f"pickup the {color} {target_type}"

        # Target not visible — search for it
        if target_color and target_type:
            return f"search for the {target_color} {target_type}"
        if target_type:
            return f"search for the {target_type}"
        return "search for the key"

    def _goto_subgoal(self, mission: str, entities: list[dict]) -> str:
        """GoTo* missions without ``go to``: doors/keys/balls as before; goal = pickup if seen else search."""
        target_color, target_type = _parse_color_and_type(mission)

        targets = self._find_entities(
            entities, obj_type=target_type, color=target_color
        )

        if target_type == "door" and targets:
            door = targets[0]
            color = self._entity_color(door)
            status = self._entity_status(door)
            if status == "locked":
                return f"open the locked {color} door"
            if status == "open":
                return f"close the open {color} door"
            # closed (unlocked): toggling opens it — same tracker branch as locked wording
            return f"open the locked {color} door"

        if target_type == "key" and targets:
            color = self._entity_color(targets[0])
            return f"pickup the {color} key"

        if target_type in ("ball", "box") and targets:
            color = self._entity_color(targets[0])
            return f"pickup the {color} {target_type}"

        if target_type == "goal":
            if targets:
                color = self._entity_color(targets[0])
                return f"pickup the {color} goal"
            gc = _goal_color_hint(mission.lower(), targets)
            return f"search for the {gc} goal"

        # Target not visible — search
        if target_color and target_type:
            return f"search for the {target_color} {target_type}"
        if target_type:
            return f"search for the {target_type}"
        return "search for the key"

    # -- entity helpers --------------------------------------------------

    @staticmethod
    def _find_entities(
        entities: list[dict],
        obj_type: str,
        status: str | None = None,
        color: str | None = None,
    ) -> list[dict]:
        """Filter visible entities by object type, optional status, and optional color.

        Uses word-level matching instead of substring to avoid false positives
        (e.g. 'key' would not accidentally match hypothetical 'monkey').
        """
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

    @staticmethod
    def _entity_color(entity: dict) -> str:
        for word in entity.get("entity", "").lower().split():
            if word in KNOWN_COLORS:
                return word
        return ""

    @staticmethod
    def _entity_status(entity: dict) -> str:
        name = entity.get("entity", "").lower()
        for s in ("locked", "open", "closed"):
            if s in name:
                return s
        return ""


def _parse_color_and_type(text: str) -> tuple[str, str]:
    color, obj_type = "", ""
    for word in text.split():
        if word in KNOWN_COLORS:
            color = word
        if word in KNOWN_TYPES:
            obj_type = word
    return color, obj_type


def _extract_color_from_words(words: list[str]) -> str:
    """Extract the first known color from a list of words."""
    for word in words:
        if word in KNOWN_COLORS:
            return word
    return ""


def _goal_color_hint(mission_lower: str, goals: list[dict]) -> str:
    """Color for ``search for the <c> goal`` when the goal is not in the entity list."""
    for g in goals:
        for word in g.get("entity", "").lower().split():
            if word in KNOWN_COLORS:
                return word
    c, _ = _parse_color_and_type(mission_lower)
    if c:
        return c
    return "green"


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
    print("  RuleBasedPlanner -- Self-Test")
    print("=" * 60)

    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        uw = env.unwrapped
        env_json = parse_env_description(obs["image"], uw.carrying)

        subgoal = planner.get_subgoal(
            obs["mission"], env_json, obs["direction"], past_subgoals=[]
        )
        print(f"Seed {seed} | mission={obs['mission']!r} -> {subgoal}")

        # Verify subgoal is NOT "explore"
        assert subgoal != "explore", f"Planner returned 'explore' for seed {seed}"
        # Verify subgoal is recognized by SubgoalTracker
        from utils.subgoal_tracker import SubgoalTracker
        assert SubgoalTracker.is_recognized(subgoal), (
            f"Subgoal not recognized by tracker: {subgoal!r}"
        )

    print("=" * 60)
    print("  All assertions passed.")
    print("=" * 60)
