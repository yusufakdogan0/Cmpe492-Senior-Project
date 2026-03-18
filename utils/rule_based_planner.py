"""
Deterministic rule-based subgoal generator for MiniGrid environments.

Serves as an oracle ablation baseline: it produces the same subgoal
format as ``LLMPlanner`` but uses hand-coded heuristics instead of an
LLM query.  This lets us isolate the contribution of LLM guidance from
the reward scaffolding when comparing results.

Supports:
    - DoorKey variants  ("use the key to open the door and then get to the goal")
    - UnlockPickup      ("pick up the <color> <object>")
    - GoToObject / GoToDoor  (single navigation objective)
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
    """

    def get_subgoal(
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
            return self._doorkey_subgoal(inventory, entities)

        if "pick up" in mission_lower or "pickup" in mission_lower:
            return self._pickup_subgoal(mission_lower, inventory, entities)

        if "go to" in mission_lower:
            return self._goto_subgoal(mission_lower, entities)

        return "explore"

    # -- mission-specific strategies ------------------------------------

    def _doorkey_subgoal(self, inventory: str, entities: list[dict]) -> str:
        """DoorKey: get key -> open locked door -> reach goal."""
        keys = self._find_entities(entities, obj_type="key")
        locked_doors = self._find_entities(entities, obj_type="door", status="locked")
        goals = self._find_entities(entities, obj_type="goal")

        carrying_key = "key" in inventory

        if not carrying_key:
            if keys:
                color = self._entity_color(keys[0])
                return f"pickup the {color} key"
            return "explore"

        if locked_doors:
            color = self._entity_color(locked_doors[0])
            return f"open the locked {color} door"

        if goals:
            color = self._entity_color(goals[0])
            return f"go to the {color} goal"

        return "explore"

    def _pickup_subgoal(
        self, mission: str, inventory: str, entities: list[dict]
    ) -> str:
        """UnlockPickup / generic pickup: acquire prerequisites then pick up target."""
        target_color, target_type = _parse_color_and_type(mission)

        # Already carrying the target -- mission should complete on its own.
        if target_type and target_type in inventory and target_color in inventory:
            return "explore"

        # Check if a locked door blocks the way; if so, get the key first.
        locked_doors = self._find_entities(entities, obj_type="door", status="locked")
        if locked_doors and "key" not in inventory:
            keys = self._find_entities(entities, obj_type="key")
            if keys:
                color = self._entity_color(keys[0])
                return f"pickup the {color} key"
            return "explore"

        if locked_doors and "key" in inventory:
            color = self._entity_color(locked_doors[0])
            return f"open the locked {color} door"

        # No door blocking -- go for the target directly.
        targets = self._find_entities(entities, obj_type=target_type)
        if targets:
            color = self._entity_color(targets[0])
            return f"pickup the {color} {target_type}"

        return "explore"

    def _goto_subgoal(self, mission: str, entities: list[dict]) -> str:
        """GoToObject / GoToDoor: navigate to the named target."""
        target_color, target_type = _parse_color_and_type(mission)
        targets = self._find_entities(entities, obj_type=target_type)
        if targets:
            color = self._entity_color(targets[0])
            label = f"the {color} {target_type}"
            if target_type == "door":
                status = self._entity_status(targets[0])
                if status:
                    label = f"the {status} {color} {target_type}"
            return f"go to {label}"
        return "explore"

    # -- entity helpers --------------------------------------------------

    @staticmethod
    def _find_entities(
        entities: list[dict],
        obj_type: str,
        status: str | None = None,
    ) -> list[dict]:
        """Filter visible entities by object type and optional door status."""
        results = []
        for e in entities:
            name = e.get("entity", "").lower()
            if obj_type and obj_type not in name:
                continue
            if status and status not in name:
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

    print("=" * 60)
