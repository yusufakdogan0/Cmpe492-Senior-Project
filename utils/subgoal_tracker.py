"""
Subgoal completion verification for MiniGrid environments.

Supports only the subgoal types from the LGRL paper:
  - search for the [color] [object]
  - pickup the [color] [object]
  - open the [status] [color] door
  - close the [status] [color] door
  - drop the [color] [object]

No "explore" or "go to" subgoals.
"""

import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

NON_ACTIONABLE_IDS = {0, 1, 2, 3}  # unseen, empty, wall, floor
ACTION_TOGGLE = 5
KNOWN_COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}
KNOWN_TYPES = {"key", "door", "ball", "box", "goal"}


class SubgoalTracker:
    """
    Stateless verifier for subgoals.

    Each call to ``check_completion`` examines the unwrapped MiniGrid env
    and the action just taken to determine whether a subgoal string has
    been satisfied.
    """

    RECOGNIZED_PREFIXES = ("pickup", "open", "close", "drop", "search for")

    def reset(self):
        pass

    @classmethod
    def is_recognized(cls, subgoal: str) -> bool:
        s = subgoal.lower().strip()
        return any(s.startswith(p) for p in cls.RECOGNIZED_PREFIXES)

    def check_completion(
        self, subgoal: str, env, action: int, *, obs_image=None,
    ) -> bool:
        subgoal = subgoal.lower().strip()

        if subgoal.startswith("pickup"):
            return self._check_pickup(subgoal, env)
        if subgoal.startswith("open"):
            return self._check_open(subgoal, env, action)
        if subgoal.startswith("close"):
            return self._check_close(subgoal, env, action)
        if subgoal.startswith("drop"):
            return self._check_drop(subgoal, env)
        if subgoal.startswith("search for"):
            return self._check_search(subgoal, env, obs_image=obs_image)
        return False

    # -- checkers --------------------------------------------------------

    def _check_pickup(self, subgoal, env):
        color, obj_type = _extract_color_and_type(subgoal)
        if env.carrying is None:
            return False
        return env.carrying.color == color and env.carrying.type == obj_type

    def _check_open(self, subgoal, env, action):
        if action != ACTION_TOGGLE:
            return False
        color, _ = _extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)
        if fwd_cell is None or fwd_cell.type != "door":
            return False
        return fwd_cell.color == color and fwd_cell.is_open

    def _check_close(self, subgoal, env, action):
        if action != ACTION_TOGGLE:
            return False
        color, _ = _extract_color_and_type(subgoal)
        fwd_cell = env.grid.get(*env.front_pos)
        if fwd_cell is None or fwd_cell.type != "door":
            return False
        return fwd_cell.color == color and not fwd_cell.is_open

    def _check_drop(self, subgoal, env):
        color, obj_type = _extract_color_and_type(subgoal)
        # Completed when we are no longer carrying the target
        if env.carrying is None:
            return True
        if env.carrying.color == color and env.carrying.type == obj_type:
            return False
        return True

    def _check_search(self, subgoal, env, *, obs_image=None):
        """'search for the [color] [object]' succeeds when the specific
        target entity is visible in the agent's field of view."""
        image = obs_image if obs_image is not None else env.gen_obs()["image"]

        tc, tt = _extract_color_and_type(subgoal)

        for x in range(7):
            for y in range(7):
                obj_id = int(image[x, y, 0])
                if obj_id in NON_ACTIONABLE_IDS:
                    continue
                color_id = int(image[x, y, 1])
                obj_name = IDX_TO_OBJECT.get(obj_id, "unknown")
                color_name = IDX_TO_COLOR.get(color_id, "unknown")

                if tt and obj_name == tt:
                    if tc and color_name == tc:
                        return True
                    if not tc:
                        return True
        return False


# -- parsing helpers (module-level) --------------------------------------

def _extract_color_and_type(text: str) -> tuple[str, str]:
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
    print("  SubgoalTracker — Self-Test (Paper Subgoals Only)")
    print("=" * 60)

    obs, _ = env.reset(seed=0)
    uw = env.unwrapped

    print(f"\nMission: {obs['mission']}")
    print(f"Carrying: {uw.carrying}")

    result = tracker.check_completion("search for the yellow key", uw, action=0)
    print(f"\n'search for the yellow key': {result}")

    result = tracker.check_completion("pickup the yellow key", uw, action=0)
    print(f"'pickup the yellow key' (not carrying): {result}")

    result = tracker.check_completion("open the locked yellow door", uw, action=2)
    print(f"'open the locked yellow door' (action=forward): {result}")

    result = tracker.check_completion("open the locked yellow door", uw, action=5)
    print(f"'open the locked yellow door' (toggle, no door ahead): {result}")

    # Verify "explore" is NOT recognized
    assert not SubgoalTracker.is_recognized("explore"), "explore should not be recognized"
    # Verify "go to" is NOT recognized
    assert not SubgoalTracker.is_recognized("go to the yellow key"), "go to should not be recognized"
    # Verify paper subgoals ARE recognized
    assert SubgoalTracker.is_recognized("search for the yellow key")
    assert SubgoalTracker.is_recognized("pickup the yellow key")
    assert SubgoalTracker.is_recognized("open the locked yellow door")

    c, t = _extract_color_and_type("open the locked purple door")
    print(f"\nParsing 'open the locked purple door': color={c}, type={t}")

    print("\n" + "=" * 60)
    print("  All assertions passed.")
    print("=" * 60)
