"""
crafter_tasks.py — Crafter task registry, recipe graph, and the
quantity-correct, dependency-ordered stage-plan builder.

This is the Crafter analogue of the MiniGrid project's ``env_utils.py``
*and* the per-family stage tables baked into ``rule_based_planner.py``.

Crafter (https://github.com/danijar/crafter) is a 2D survival game with a
22-achievement tech tree. We treat a single target achievement as the
"mission" and decompose it into an ordered, forward-only sequence of
subgoal *stages* — exactly the structure the LGRL reward scaffold
expects.

Everything in here is pure data + logic (no torch, no crafter runtime),
so it can be unit-tested in isolation. The recipe numbers are copied
verbatim from Crafter's bundled ``data.yaml`` (collect/place/make
sections) and re-validated at import time by ``_validate_all_plans``.

Stage dict schema (the contract consumed by the planner and tracker):
    {
      "text":        str,    # subgoal string shown to the agent / parsed by tracker
      "type":        "collect_n" | "collect" | "place" | "make",
      "item":        str,    # for collect_n / collect  (inventory item)
      "station":     str,    # for place                (table/furnace/stone/plant)
      "tool":        str,    # for make                 (wood_pickaxe, ...)
      "n":           int,    # for collect_n            (inventory threshold)
      "achievement": str,    # crafter achievement key used for completion
    }
"""

from __future__ import annotations

from collections import defaultdict

# ---------------------------------------------------------------------------
# Crafter semantic-grid id -> name (from crafter.env._sem_view; see README).
# Materials 0-12, then object classes 13-18.
# ---------------------------------------------------------------------------

SEMANTIC_ID_TO_NAME: dict[int, str] = {
    0: "void",
    1: "water",
    2: "grass",
    3: "stone",
    4: "path",
    5: "sand",
    6: "tree",
    7: "lava",
    8: "coal",
    9: "iron",
    10: "diamond",
    11: "table",
    12: "furnace",
    13: "player",
    14: "cow",
    15: "zombie",
    16: "skeleton",
    17: "arrow",
    18: "plant",
}

# Tiles the agent can stand on (from data.yaml "walkable").
WALKABLE = {"grass", "path", "sand"}

# All 16 inventory keys (data.yaml "items").
INVENTORY_KEYS = (
    "health", "food", "drink", "energy",
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)
VITALS = ("health", "food", "drink", "energy")
TOOLS = (
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)

# ---------------------------------------------------------------------------
# Recipe graph (verbatim from crafter/data.yaml).
# ---------------------------------------------------------------------------

# Collectibles, keyed by the INVENTORY ITEM they produce.
#   source      — the world material the agent faces + `do`s on
#   achievement — the crafter achievement that fires on first unit
#   requires    — tools that must be held to harvest it
COLLECTIBLES: dict[str, dict] = {
    "wood":    {"source": "tree",    "achievement": "collect_wood",    "requires": {}},
    "stone":   {"source": "stone",   "achievement": "collect_stone",   "requires": {"wood_pickaxe": 1}},
    "coal":    {"source": "coal",    "achievement": "collect_coal",    "requires": {"wood_pickaxe": 1}},
    "iron":    {"source": "iron",    "achievement": "collect_iron",    "requires": {"stone_pickaxe": 1}},
    "diamond": {"source": "diamond", "achievement": "collect_diamond", "requires": {"iron_pickaxe": 1}},
    "drink":   {"source": "water",   "achievement": "collect_drink",   "requires": {}},
    "sapling": {"source": "grass",   "achievement": "collect_sapling", "requires": {}},
}

# Placeables (data.yaml "place").
PLACEABLES: dict[str, dict] = {
    "stone":   {"uses": {"stone": 1},   "achievement": "place_stone"},
    "table":   {"uses": {"wood": 2},    "achievement": "place_table"},
    "furnace": {"uses": {"stone": 4},   "achievement": "place_furnace"},
    "plant":   {"uses": {"sapling": 1}, "achievement": "place_plant"},
}

# Makeables (data.yaml "make").
MAKEABLES: dict[str, dict] = {
    "wood_pickaxe":  {"uses": {"wood": 1},                       "nearby": ["table"],            "achievement": "make_wood_pickaxe"},
    "stone_pickaxe": {"uses": {"wood": 1, "stone": 1},           "nearby": ["table"],            "achievement": "make_stone_pickaxe"},
    "iron_pickaxe":  {"uses": {"wood": 1, "coal": 1, "iron": 1}, "nearby": ["table", "furnace"], "achievement": "make_iron_pickaxe"},
    "wood_sword":    {"uses": {"wood": 1},                       "nearby": ["table"],            "achievement": "make_wood_sword"},
    "stone_sword":   {"uses": {"wood": 1, "stone": 1},           "nearby": ["table"],            "achievement": "make_stone_sword"},
    "iron_sword":    {"uses": {"wood": 1, "coal": 1, "iron": 1}, "nearby": ["table", "furnace"], "achievement": "make_iron_sword"},
}

ITEM_TO_COLLECT_ACH = {item: spec["achievement"] for item, spec in COLLECTIBLES.items()}

# Standalone (no-prerequisite) achievements we expose as single-stage tasks.
STANDALONE = {
    "collect_drink":   ("collect", "drink"),
    "collect_sapling": ("collect", "sapling"),
    "eat_cow":         ("eat_cow", None),
    "defeat_zombie":   ("defeat_zombie", None),
    "defeat_skeleton": ("defeat_skeleton", None),
}

# ---------------------------------------------------------------------------
# Natural-language mission strings (used as obs["mission"] and for dispatch).
# Must be invertible.
# ---------------------------------------------------------------------------

TASK_MISSION: dict[str, str] = {
    "collect_wood":      "collect wood",
    "place_table":       "place a table",
    "make_wood_pickaxe": "make a wood pickaxe",
    "collect_stone":     "collect stone",
    "make_stone_pickaxe": "make a stone pickaxe",
    "place_furnace":     "place a furnace",
    "collect_coal":      "collect coal",
    "collect_iron":      "collect iron",
    "make_iron_pickaxe": "make an iron pickaxe",
    "collect_diamond":   "collect a diamond",
    # additional supported targets (not in the default curriculum)
    "make_wood_sword":   "make a wood sword",
    "make_stone_sword":  "make a stone sword",
    "make_iron_sword":   "make an iron sword",
    "place_stone":       "place a stone",
    "place_plant":       "place a plant",
    "collect_sapling":   "collect a sapling",
    "collect_drink":     "drink water",
    "eat_cow":           "eat a cow",
    "defeat_zombie":     "defeat a zombie",
    "defeat_skeleton":   "defeat a skeleton",
}
MISSION_TO_TASK: dict[str, str] = {v: k for k, v in TASK_MISSION.items()}

# Tech-tree spine, easiest -> hardest. Each later task subsumes the skills
# of the earlier ones, so carrying the agent forward across stages
# (train_lgrl_curriculum.py) gives compounding transfer — the Crafter
# analogue of the MiniGrid GoToDoor -> GoToObject -> KeyCorridor curriculum.
DEFAULT_CURRICULUM: tuple[str, ...] = (
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "make_iron_pickaxe",
    "collect_diamond",
)

SUPPORTED_TASKS: tuple[str, ...] = tuple(TASK_MISSION.keys())


# ---------------------------------------------------------------------------
# Subgoal-string helpers (planner emits these; tracker parses them).
# ---------------------------------------------------------------------------

def _make_phrase(tool: str) -> str:
    adj, kind = tool.split("_")          # wood_pickaxe -> wood, pickaxe
    article = "an" if adj == "iron" else "a"
    return f"make {article} {adj} {kind}"


def _place_phrase(station: str) -> str:
    return f"place a {station}"


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def _collect_n_stage(item: str, n: int) -> dict:
    return {"text": f"collect {n} {item}", "type": "collect_n",
            "item": item, "n": n, "achievement": ITEM_TO_COLLECT_ACH[item]}


def _collect_stage(item: str) -> dict:
    return {"text": f"collect {item}", "type": "collect",
            "item": item, "achievement": ITEM_TO_COLLECT_ACH[item]}


def _place_stage(station: str) -> dict:
    return {"text": _place_phrase(station), "type": "place",
            "station": station, "achievement": PLACEABLES[station]["achievement"]}


def _make_stage(tool: str) -> dict:
    return {"text": _make_phrase(tool), "type": "make",
            "tool": tool, "achievement": MAKEABLES[tool]["achievement"]}


def build_stage_plan(target: str) -> list[dict]:
    """Return the ordered, quantity-correct list of subgoal stages for a
    target achievement.

    Ordering invariant (the key to runtime correctness): the gather
    stages for any single make/place are emitted *immediately* before
    that make/place, with no other consuming action in between. Tools
    and stations needed either as a `nearby` requirement OR to harvest a
    material are produced *before* that material is gathered. This makes
    each collect threshold an absolute, contiguous "top up to N" with no
    cross-stage interference, so the planner's `inv >= N` skip check can
    never misfire.
    """
    stages: list[dict] = []
    vinv: dict[str, int] = defaultdict(int)   # plan-time virtual inventory
    placed: set[str] = set()
    made: set[str] = set()

    def gather(mat: str, count: int) -> None:
        # tools needed to harvest `mat` must already exist (caller ensures),
        # but double-check for safety.
        for tool in COLLECTIBLES[mat]["requires"]:
            ensure_made(tool)
        if vinv[mat] < count:
            stages.append(_collect_n_stage(mat, count))
            vinv[mat] = count

    def ensure_placed(station: str) -> None:
        if station in placed:
            return
        recipe = PLACEABLES[station]
        # tools to harvest this station's input materials
        for mat in recipe["uses"]:
            for tool in COLLECTIBLES.get(mat, {}).get("requires", {}):
                ensure_made(tool)
        # gather just-in-time, then consume
        for mat, c in recipe["uses"].items():
            gather(mat, c)
        for mat, c in recipe["uses"].items():
            vinv[mat] -= c
        stages.append(_place_stage(station))
        placed.add(station)

    def ensure_made(tool: str) -> None:
        if tool in made:
            return
        recipe = MAKEABLES[tool]
        # 1. stations this make must be standing next to
        for st in recipe["nearby"]:
            ensure_placed(st)
        # 2. tools needed to harvest this make's input materials
        for mat in recipe["uses"]:
            for t in COLLECTIBLES.get(mat, {}).get("requires", {}):
                ensure_made(t)
        # 3. gather just-in-time, then consume + make
        for mat, c in recipe["uses"].items():
            gather(mat, c)
        for mat, c in recipe["uses"].items():
            vinv[mat] -= c
        stages.append(_make_stage(tool))
        made.add(tool)

    # ---- dispatch on target type --------------------------------------
    # Targets are achievement keys; map them to a station/tool/item.
    if target in STANDALONE:
        kind, item = STANDALONE[target]
        if kind == "collect":
            for tool in COLLECTIBLES[item]["requires"]:
                ensure_made(tool)
            stages.append(_collect_stage(item))
        else:  # eat_cow / defeat_zombie / defeat_skeleton
            stages.append({"text": TASK_MISSION[target], "type": kind,
                           "achievement": target})
        return stages

    if target.startswith("make_"):
        ensure_made(target[len("make_"):])
        return stages

    if target.startswith("place_"):
        ensure_placed(target[len("place_"):])
        return stages

    if target.startswith("collect_"):
        item = next(i for i, a in ITEM_TO_COLLECT_ACH.items() if a == target)
        for tool in COLLECTIBLES[item]["requires"]:
            ensure_made(tool)
        stages.append(_collect_stage(item))
        return stages

    raise ValueError(f"Unsupported Crafter target: {target!r}")


# Precompute once.
STAGE_PLANS: dict[str, list[dict]] = {t: build_stage_plan(t) for t in SUPPORTED_TASKS}


def num_subgoals(task: str) -> int:
    """Number of reward-bearing stages ('n')."""
    return len(STAGE_PLANS[task])


def task_max_steps(task: str) -> int:
    """Per-task step budget (T_max). Scaled by plan depth; deeper
    tech-tree targets get a longer horizon. Crafter's native 10k cap is
    far too long for per-subgoal time budgeting."""
    n = num_subgoals(task)
    return max(150, 120 + 60 * n)


# ---------------------------------------------------------------------------
# Artifact naming (mirrors env_utils.resolve_artifact_stem semantics).
# ---------------------------------------------------------------------------

def task_stem(task: str) -> str:
    return task.lower()


def resolve_artifact_stem(base: str, task: str) -> str:
    return f"{base}_{task_stem(task)}"


def curriculum_artifact_stem(base: str, tasks: list[str]) -> str:
    if tuple(tasks) == DEFAULT_CURRICULUM:
        return f"{base}_curriculum_techtree"
    return f"{base}_curriculum_" + "_".join(task_stem(t) for t in tasks)


# ---------------------------------------------------------------------------
# Import-time validation: simulate every plan against the recipes and make
# sure each action's preconditions hold. Catches any ordering/quantity bug.
# ---------------------------------------------------------------------------

def _validate_plan(target: str) -> None:
    plan = STAGE_PLANS[target]
    inv: dict[str, int] = defaultdict(int)
    placed: set[str] = set()
    nearby_ok = lambda st: st in placed  # once placed we assume reachable

    for k, stage in enumerate(plan):
        t = stage["type"]
        if t == "collect_n":
            item = stage["item"]
            for tool in COLLECTIBLES[item]["requires"]:
                assert inv[tool] >= 1, (
                    f"{target} stage {k} ({stage['text']}): missing tool "
                    f"{tool} to harvest {item}")
            inv[item] = max(inv[item], stage["n"])    # top up to threshold
        elif t == "collect":
            item = stage["item"]
            for tool in COLLECTIBLES[item]["requires"]:
                assert inv[tool] >= 1, (
                    f"{target} stage {k} ({stage['text']}): missing tool "
                    f"{tool} to harvest {item}")
            inv[item] += 1
        elif t == "place":
            station = stage["station"]
            for mat, c in PLACEABLES[station]["uses"].items():
                assert inv[mat] >= c, (
                    f"{target} stage {k} ({stage['text']}): need {c} {mat}, "
                    f"have {inv[mat]}")
                inv[mat] -= c
            placed.add(station)
        elif t == "make":
            if "tool" in stage:
                tool = stage["tool"]
                recipe = MAKEABLES[tool]
                for st in recipe["nearby"]:
                    assert nearby_ok(st), (
                        f"{target} stage {k} ({stage['text']}): not near {st}")
                for mat, c in recipe["uses"].items():
                    assert inv[mat] >= c, (
                        f"{target} stage {k} ({stage['text']}): need {c} {mat}, "
                        f"have {inv[mat]}")
                    inv[mat] -= c
                inv[tool] += 1
        # standalone eat/defeat stages: nothing to validate


def _validate_all_plans() -> None:
    for target in SUPPORTED_TASKS:
        _validate_plan(target)


_validate_all_plans()


# ---------------------------------------------------------------------------
# Self-test / inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Crafter task registry — stage plans (all validated)")
    print("=" * 70)
    for task in DEFAULT_CURRICULUM:
        plan = STAGE_PLANS[task]
        print(f"\n--- {task}  (mission={TASK_MISSION[task]!r}, "
              f"n={len(plan)}, T_max={task_max_steps(task)}) ---")
        for i, s in enumerate(plan):
            print(f"  {i:2d}. [{s['type']:9s}] {s['text']:24s} -> {s['achievement']}")
    print("\nAll", len(SUPPORTED_TASKS), "task plans validated successfully.")
