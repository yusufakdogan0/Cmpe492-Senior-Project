"""
Subgoal completion verification for MiniGrid environments.

Uses action-aware checking to distinguish agent-caused state changes
from pre-existing states (e.g. "agent opened this door" vs "door was
already open"). Each checker inspects the environment after the step
and the action that was just taken.
"""

import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

NON_ACTIONABLE_IDS = {0, 1, 2, 3}  # unseen, empty, wall, floor
ACTION_TOGGLE = 5
KNOWN_COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}
KNOWN_TYPES = {"key", "door", "ball", "box", "goal"}


class SubgoalTracker:
    """
    Stateless verifier for LLM-generated subgoals.

    Each call to ``check_completion`` examines the unwrapped MiniGrid env
    and the action just taken to determine whether a subgoal string
    (e.g. "pickup the yellow key") has been satisfied.
    """

    RECOGNIZED_PREFIXES = ("pickup", "open", "close", "go to", "drop", "search for")

    def reset(self):
        """Reset any per-episode state. Currently a no-op (tracker is stateless)."""
        pass

    @classmethod
    def is_recognized(cls, subgoal: str) -> bool:
        """Return True if *subgoal* matches a format that ``check_completion`` can evaluate."""
        s = subgoal.lower().strip()
        return s == "explore" or any(s.startswith(p) for p in cls.RECOGNIZED_PREFIXES)

    def check_completion(
        self, subgoal: str, env, action: int, *, obs_image=None,
    ) -> bool:
        """Return True if *subgoal* was achieved by the agent's last *action*.

        Args:
            obs_image: optional (7,7,3) ndarray from the step observation.
                       If provided, ``_check_search`` uses it directly
                       instead of regenerating the observation via
                       ``env.gen_obs()``.
        """
        subgoal = subgoal.lower().strip()

        if subgoal.startswith("pickup"):
            return self._check_pickup(subgoal, env)
        if subgoal.startswith("open"):
            return self._check_open(subgoal, env, action)
        if subgoal.startswith("close"):
            return self._check_close(subgoal, env, action)
        if subgoal.startswith("go to"):
            return self._check_go_to(subgoal, env)
        if subgoal.startswith("drop"):
            return self._check_drop(subgoal, env)
        if subgoal == "explore" or subgoal.startswith("search for"):
            return self._check_search(subgoal, env, obs_image=obs_image)
        return False

    # -- checkers --------------------------------------------------------

    def _check_pickup(self, subgoal, env):
        color, obj_type = self._extract_color_and_type(subgoal)
        if env.carrying is None:
            return False
        return env.carrying.color == color and env.carrying.type == obj_type

    def _check_open(self, subgoal, env, action):
        """Only fires when the agent just toggled (action 5) the matching door."""
        if action != ACTION_TOGGLE:
            return False
        color, _ = self._extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)
        if fwd_cell is None or fwd_cell.type != "door":
            return False
        return fwd_cell.color == color and fwd_cell.is_open

    def _check_close(self, subgoal, env, action):
        if action != ACTION_TOGGLE:
            return False
        color, _ = self._extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)
        if fwd_cell is None or fwd_cell.type != "door":
            return False
        return fwd_cell.color == color and not fwd_cell.is_open

    def _check_go_to(self, subgoal, env):
        """True when the target object is directly in front of the agent."""
        color, obj_type = self._extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)
        if fwd_cell is None:
            return False
        return fwd_cell.color == color and fwd_cell.type == obj_type

    def _check_drop(self, subgoal, env):
        color, obj_type = self._extract_color_and_type(subgoal)
        if env.carrying is None:
            return True
        if env.carrying.color == color and env.carrying.type == obj_type:
            return False
        return True

    def _check_search(self, subgoal, env, *, obs_image=None):
        """'explore' succeeds on any visible entity; 'search for X' requires X."""
        image = obs_image if obs_image is not None else env.gen_obs()["image"]
        visible = set()
        for x in range(7):
            for y in range(7):
                obj_id = int(image[x, y, 0])
                if obj_id in NON_ACTIONABLE_IDS:
                    continue
                color_id = int(image[x, y, 1])
                obj_name = IDX_TO_OBJECT.get(obj_id, "unknown")
                color_name = IDX_TO_COLOR.get(color_id, "unknown")
                visible.add(f"{color_name} {obj_name}")

        if subgoal.startswith("search for"):
            tc, tt = self._extract_color_and_type(subgoal)
            if tc:
                return f"{tc} {tt}" in visible
            # Color-less search: match any entity of the right type
            return any(item.endswith(f" {tt}") for item in visible)
        return len(visible) > 0

    # -- parsing helpers -------------------------------------------------

    @staticmethod
    def _extract_color(text):
        for word in text.split():
            if word in KNOWN_COLORS:
                return word
        return ""

    @staticmethod
    def _extract_color_and_type(text):
        color, obj_type = "", ""
        for word in text.split():
            if word in KNOWN_COLORS:
                color = word
            if word in KNOWN_TYPES:
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
