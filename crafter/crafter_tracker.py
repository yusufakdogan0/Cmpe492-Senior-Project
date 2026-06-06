"""
crafter_tracker.py — subgoal completion verification for Crafter.

The Crafter analogue of the MiniGrid project's ``subgoal_tracker.py``.
Where the MiniGrid tracker had to poke engine internals (env.carrying,
env.grid, front_pos), Crafter hands us ground truth directly:

  * "collect N <mat>"   completes when inventory[<mat>] >= N
                        (threshold sub-goals for crafting inputs)
  * "collect <mat>"     completes when the achievement collect_<mat>
                        increments this step (terminal collect goals)
  * "place a <station>" completes when achievement place_<station> increments
  * "make a/an <tool>"  completes when achievement make_<tool> increments
  * "eat a cow"         completes when achievement eat_cow increments
  * "defeat a <foe>"    completes when achievement defeat_<foe> increments

Achievement counters are one-time-per-episode and reset on env.reset(),
so the increment test is robust. The tracker is given the post-step and
previous-step snapshots by the training loop.
"""

from __future__ import annotations

import re

from crafter_tasks import ITEM_TO_COLLECT_ACH

RECOGNIZED_PREFIXES = ("collect", "place", "make", "eat", "defeat")

_RE_COLLECT_N = re.compile(r"^collect (\d+) (\w+)$")
_RE_COLLECT = re.compile(r"^collect (?:a |an )?(\w+)$")
_RE_PLACE = re.compile(r"^place (?:a |an )?(\w+)$")
_RE_MAKE = re.compile(r"^make (?:a |an )?(.+)$")
_RE_EAT = re.compile(r"^eat (?:a |an )?(\w+)$")
_RE_DEFEAT = re.compile(r"^defeat (?:a |an )?(\w+)$")


class CrafterSubgoalTracker:
    """Stateless-per-call subgoal verifier (mirrors SubgoalTracker API)."""

    def __init__(self):
        pass

    def reset(self) -> None:
        # No per-episode internal state needed; snapshots are passed in.
        pass

    @staticmethod
    def is_recognized(subgoal: str) -> bool:
        s = (subgoal or "").strip().lower()
        return s.startswith(RECOGNIZED_PREFIXES)

    @staticmethod
    def _inc(achievements: dict, prev: dict, key: str) -> bool:
        return int(achievements.get(key, 0)) > int(prev.get(key, 0))

    def check_completion(
        self,
        subgoal: str,
        *,
        inventory: dict,
        achievements: dict,
        prev_achievements: dict,
        prev_inventory: dict | None = None,
        nearby: dict | None = None,
    ) -> bool:
        """Return True if ``subgoal`` was completed on this step."""
        s = (subgoal or "").strip().lower()
        if not s:
            return False

        m = _RE_COLLECT_N.match(s)
        if m:
            count = int(m.group(1))
            item = m.group(2)
            return inventory.get(item, 0) >= count

        m = _RE_PLACE.match(s)
        if m:
            station = m.group(1)
            return self._inc(achievements, prev_achievements, f"place_{station}")

        m = _RE_MAKE.match(s)
        if m:
            tool = m.group(1).strip().replace(" ", "_")
            return self._inc(achievements, prev_achievements, f"make_{tool}")

        m = _RE_EAT.match(s)
        if m:
            foe = m.group(1)
            return self._inc(achievements, prev_achievements, f"eat_{foe}")

        m = _RE_DEFEAT.match(s)
        if m:
            foe = m.group(1)
            return self._inc(achievements, prev_achievements, f"defeat_{foe}")

        # plain "collect <item>" — terminal achievement-based collect
        m = _RE_COLLECT.match(s)
        if m:
            item = m.group(1)
            ach = ITEM_TO_COLLECT_ACH.get(item, f"collect_{item}")
            return self._inc(achievements, prev_achievements, ach)

        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tr = CrafterSubgoalTracker()

    # recognition
    for sg in ["collect 2 wood", "place a table", "make an iron pickaxe",
               "collect diamond", "eat a cow", "defeat a zombie", "wander around"]:
        print(f"  is_recognized({sg!r}) = {tr.is_recognized(sg)}")

    print("\nCompletion checks:")
    # threshold sub-goal
    print("  collect 2 wood  | inv wood=2 ->",
          tr.check_completion("collect 2 wood", inventory={"wood": 2},
                              achievements={}, prev_achievements={}))
    print("  collect 2 wood  | inv wood=1 ->",
          tr.check_completion("collect 2 wood", inventory={"wood": 1},
                              achievements={}, prev_achievements={}))
    # achievement increment
    print("  place a table   | place_table 0->1 ->",
          tr.check_completion("place a table", inventory={},
                              achievements={"place_table": 1},
                              prev_achievements={"place_table": 0}))
    print("  make a wood pickaxe | make_wood_pickaxe 1->1 (no inc) ->",
          tr.check_completion("make a wood pickaxe", inventory={},
                              achievements={"make_wood_pickaxe": 1},
                              prev_achievements={"make_wood_pickaxe": 1}))
    print("  collect diamond | collect_diamond 0->1 ->",
          tr.check_completion("collect diamond", inventory={},
                              achievements={"collect_diamond": 1},
                              prev_achievements={"collect_diamond": 0}))
    print("crafter_tracker self-test OK")
