"""
subgoal_tracker.py — Verifies whether an LLM-generated subgoal has been achieved.

Action-aware checking: instead of scanning the entire grid, we check
what just happened at the agent's position after each step.

- open/close: only triggers if agent just toggled (action 5) the matching door
- pickup/drop: checks env.carrying
- go to: checks if target is at front_pos
- search/explore: checks if target entity appears in current 7x7 view
"""

import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

# tiles that aren't real objects (skip during observation scans)
NON_ACTIONABLE_IDS = {0, 1, 2, 3}  # unseen, empty, wall, floor

# MiniGrid action constants
ACTION_TOGGLE = 5


class SubgoalTracker:
    """
    Checks if the agent has completed its current subgoal.
    Requires the action taken at each step so we can distinguish
    "the agent opened this door" from "a door was already open".
    """

    def check_completion(self, subgoal, env, action) -> bool:
        """
        Check if the given subgoal was achieved by the last action.

        Args:
            subgoal: e.g. "pickup the yellow key", "open the locked red door"
            env: the unwrapped MiniGrid environment (after the step)
            action: the integer action the agent just took (0-6)

        Returns True if the subgoal is completed.
        """
        subgoal = subgoal.lower().strip()

        if subgoal.startswith("pickup"):
            return self._check_pickup(subgoal, env)
        elif subgoal.startswith("open"):
            return self._check_open(subgoal, env, action)
        elif subgoal.startswith("close"):
            return self._check_close(subgoal, env, action)
        elif subgoal.startswith("go to"):
            return self._check_go_to(subgoal, env)
        elif subgoal.startswith("drop"):
            return self._check_drop(subgoal, env)
        elif subgoal in ("explore",) or subgoal.startswith("search for"):
            return self._check_search(subgoal, env)
        else:
            return False

    # --- Individual checkers ---

    def _check_pickup(self, subgoal, env):
        """'pickup the yellow key' → True if agent is now carrying a yellow key."""
        color, obj_type = self._extract_color_and_type(subgoal)
        if env.carrying is None:
            return False
        return env.carrying.color == color and env.carrying.type == obj_type

    def _check_open(self, subgoal, env, action):
        """
        'open the locked red door' → True if agent just toggled (action 5)
        and the door in front of it is red and now open.

        This prevents false positives from doors that were already open
        elsewhere on the map.
        """
        if action != ACTION_TOGGLE:
            return False

        color, _ = self._extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)

        if fwd_cell is None or fwd_cell.type != "door":
            return False

        return fwd_cell.color == color and fwd_cell.is_open

    def _check_close(self, subgoal, env, action):
        """'close the open red door' → True if agent just toggled and door is now closed."""
        if action != ACTION_TOGGLE:
            return False

        color, _ = self._extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)

        if fwd_cell is None or fwd_cell.type != "door":
            return False

        return fwd_cell.color == color and not fwd_cell.is_open

    def _check_go_to(self, subgoal, env):
        """
        'go to the yellow key' → True if the target object is directly
        in front of the agent (1 step forward, 0 lateral).
        """
        color, obj_type = self._extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)

        if fwd_cell is None:
            return False

        return fwd_cell.color == color and fwd_cell.type == obj_type

    def _check_drop(self, subgoal, env):
        """'drop the yellow key' → True if agent is no longer carrying it."""
        color, obj_type = self._extract_color_and_type(subgoal)
        if env.carrying is None:
            return True
        # still carrying the same thing → not dropped
        if env.carrying.color == color and env.carrying.type == obj_type:
            return False
        return True

    def _check_search(self, subgoal, env):
        """
        'explore' → True if any actionable entity is visible in the 7x7 view.
        'search for the purple key' → True if that specific entity is visible.

        Scans the current observation directly — no memory of past views needed
        since this is checked every step. The subgoal stays active until
        the target is found.
        """
        image = env.gen_obs()["image"]
        visible_entities = set()

        for x in range(7):
            for y in range(7):
                obj_id = int(image[x, y, 0])
                if obj_id in NON_ACTIONABLE_IDS:
                    continue
                color_id = int(image[x, y, 1])
                obj_name = IDX_TO_OBJECT.get(obj_id, "unknown")
                color_name = IDX_TO_COLOR.get(color_id, "unknown")
                visible_entities.add(f"{color_name} {obj_name}")

        if subgoal.startswith("search for"):
            # only succeed if the specific target is visible
            target_color, target_type = self._extract_color_and_type(subgoal)
            return f"{target_color} {target_type}" in visible_entities

        # plain "explore" — any actionable entity in view counts
        return len(visible_entities) > 0

    # --- String parsing helpers ---

    @staticmethod
    def _extract_color(subgoal):
        """Pull the color word from a subgoal string."""
        known_colors = {"red", "green", "blue", "purple", "yellow", "grey"}
        for word in subgoal.split():
            if word in known_colors:
                return word
        return ""

    @staticmethod
    def _extract_color_and_type(subgoal):
        """
        Pull color and object type from a subgoal string.
        e.g. 'pickup the yellow key' → ('yellow', 'key')
        e.g. 'open the locked red door' → ('red', 'door')
        """
        known_colors = {"red", "green", "blue", "purple", "yellow", "grey"}
        known_types = {"key", "door", "ball", "box", "goal"}

        color = ""
        obj_type = ""
        for word in subgoal.split():
            if word in known_colors:
                color = word
            if word in known_types:
                obj_type = word

        return color, obj_type


# --- Quick self-test ---

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import minigrid

    tracker = SubgoalTracker()
    env = gym.make("MiniGrid-DoorKey-5x5-v0")

    print("=" * 60)
    print("  SubgoalTracker — Self-Test (Action-Aware)")
    print("=" * 60)

    obs, _ = env.reset(seed=0)
    uw = env.unwrapped

    print(f"\nMission: {obs['mission']}")
    print(f"Carrying: {uw.carrying}")

    # search for key — should be True if key is visible
    result = tracker.check_completion("search for the yellow key", uw, action=0)
    print(f"\n'search for the yellow key': {result}")

    # explore — should be True (key visible)
    result = tracker.check_completion("explore", uw, action=0)
    print(f"'explore': {result}")

    # pickup before picking up
    result = tracker.check_completion("pickup the yellow key", uw, action=0)
    print(f"'pickup the yellow key' (not carrying): {result}")

    # open without toggling (action=2 is forward, not toggle)
    result = tracker.check_completion("open the locked yellow door", uw, action=2)
    print(f"'open the locked yellow door' (action=forward): {result}")

    # open with toggle but no door in front
    result = tracker.check_completion("open the locked yellow door", uw, action=5)
    print(f"'open the locked yellow door' (toggle, no door ahead): {result}")

    # go to
    result = tracker.check_completion("go to the yellow key", uw, action=0)
    print(f"'go to the yellow key': {result}")

    # string parsing
    c, t = SubgoalTracker._extract_color_and_type("open the locked purple door")
    print(f"\nParsing 'open the locked purple door': color={c}, type={t}")

    c, t = SubgoalTracker._extract_color_and_type("search for the green ball")
    print(f"Parsing 'search for the green ball': color={c}, type={t}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
