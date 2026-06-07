"""
crafter_planner.py — deterministic rule-based oracle planner for Crafter.

The Crafter analogue of the MiniGrid project's ``rule_based_planner.py``.
Instead of per-family hand-written stage machines, it walks the
quantity-correct stage plan precomputed in ``crafter_tasks.STAGE_PLANS``
for the mission's target achievement.

Behaviour mirrors the MiniGrid planner exactly:
  * forward-only: the returned stage index never decreases;
  * skip-if-satisfied: stages whose precondition is already met (judged
    purely from the current OBSERVATION — inventory + nearby stations)
    are skipped, so the agent is always pointed at the next *unmet*
    subgoal;
  * the same call signature as the MiniGrid planner and the LLM planner,
    so the training loop is agnostic to which planner it holds.

Completion / reward is NOT decided here — that is the tracker's job (it
uses crafter achievement diffs). The planner only decides which subgoal
string to show the agent next, exactly as in the MiniGrid split.
"""

from __future__ import annotations

import json

from crafter_tasks import STAGE_PLANS, MISSION_TO_TASK, num_subgoals


class RuleBasedCrafterPlanner:
    """Deterministic tech-tree planner over Crafter stage plans."""

    def __init__(self):
        self.last_raw_response: str = ""

    # -- static dispatch helpers (match MiniGrid RuleBasedPlanner) ------

    @staticmethod
    def classify_mission(mission: str) -> str:
        """Map a mission string to its target achievement (task) key."""
        if mission not in MISSION_TO_TASK:
            raise ValueError(f"Unrecognized Crafter mission: {mission!r}")
        return MISSION_TO_TASK[mission]

    @staticmethod
    def num_stages(mission: str) -> int:
        return num_subgoals(RuleBasedCrafterPlanner.classify_mission(mission))

    # -- skip predicate -------------------------------------------------

    @staticmethod
    def _satisfied(stage: dict, inventory: dict, nearby: dict) -> bool:
        """True if this stage's precondition is already met by the current
        observation, so the planner should skip past it."""
        t = stage["type"]
        if t == "collect_n":
            return inventory.get(stage["item"], 0) >= stage["n"]
        if t == "collect":
            return inventory.get(stage["item"], 0) >= 1
        if t == "place":
            station = stage["station"]
            if station in ("table", "furnace"):
                return bool(nearby.get(station, False))
            return False                      # place_stone / place_plant: always attempt
        if t == "make":
            return inventory.get(stage["tool"], 0) >= 1
        return False                          # eat / defeat: never skip

    # -- main interface -------------------------------------------------

    def get_subgoal(self, mission: str, env_json_str: str,
                    direction=0, stage_index: int = 0) -> tuple[str, int]:
        """Return (subgoal_string, new_stage_index).

        ``direction`` is accepted for signature parity with the MiniGrid /
        LLM planners; Crafter skip logic is driven by inventory + nearby
        stations parsed from ``env_json_str``.
        """
        task = self.classify_mission(mission)
        plan = STAGE_PLANS[task]
        n = len(plan)

        try:
            desc = json.loads(env_json_str)
            inventory = desc.get("inventory", {})
            nearby = desc.get("nearby_stations", {})
        except (ValueError, TypeError):
            inventory, nearby = {}, {}

        i = max(0, int(stage_index))
        # Forward-only skip: advance past already-satisfied stages.
        while i < n and self._satisfied(plan[i], inventory, nearby):
            i += 1

        if i >= n:
            # Everything satisfied — keep the final subgoal visible.
            text = plan[-1]["text"]
            self.last_raw_response = f"[rule] all stages satisfied -> {text}"
            return text, n

        text = plan[i]["text"]
        self.last_raw_response = f"[rule] stage {i}/{n}: {text}"
        return text, i


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from crafter_env import CrafterTaskEnv
    from crafter_parser import parse_crafter_description

    planner = RuleBasedCrafterPlanner()

    # 1) Static walk-through with synthetic observations for make_stone_pickaxe.
    mission = "make a stone pickaxe"
    print(f"Mission: {mission!r}  ({planner.num_stages(mission)} stages)")
    fake_states = [
        {"inventory": {}, "nearby_stations": {}},                                 # nothing
        {"inventory": {"wood": 2}, "nearby_stations": {}},                        # have wood
        {"inventory": {"wood": 2}, "nearby_stations": {"table": True}},           # table down
        {"inventory": {"wood": 1, "wood_pickaxe": 1}, "nearby_stations": {"table": True}},
        {"inventory": {"wood": 1, "stone": 1, "wood_pickaxe": 1}, "nearby_stations": {"table": True}},
    ]
    stage = 0
    for st in fake_states:
        sub, stage = planner.get_subgoal(mission, json.dumps(st), stage_index=stage)
        print(f"  inv={st['inventory']} near={st['nearby_stations']} -> stage {stage}: {sub}")

    # 2) Live walk against a real Crafter rollout (random actions).
    print("\nLive (random policy, collect_stone):")
    env = CrafterTaskEnv("collect_stone", seed=2)
    obs, info = env.reset(seed=2)
    import numpy as np
    rng = np.random.default_rng(2)
    stage = 0
    seen = set()
    for _ in range(60):
        env_json = parse_crafter_description(env.last_state)
        sub, stage = planner.get_subgoal(obs["mission"], env_json, stage_index=stage)
        if (stage, sub) not in seen:
            print(f"  step stage {stage}: {sub}")
            seen.add((stage, sub))
        obs, r, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            break
    print("crafter_planner self-test OK")
