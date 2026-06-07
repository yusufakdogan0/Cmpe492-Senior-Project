"""
crafter_parser.py — turn a Crafter symbolic state into a compact JSON
description for the planner and the LLM.

This is the Crafter analogue of the MiniGrid project's ``env_parser.py``
(``parse_env_description``). Where MiniGrid parsed a 7x7x3 egocentric
image, here we crop a local window of Crafter's 64x64 full-world
``semantic`` grid around the player.

Output JSON shape:
    {
      "inventory": {"health":9,"food":9,"drink":9,"energy":9,
                    "wood":2,"stone":1, "wood_pickaxe":1, ...},   # vitals + nonzero
      "facing": "south",
      "faced_tile": "tree",                 # what `do` would act on
      "nearby_stations": {"table": true, "furnace": false},       # within radius 1
      "entities": [ {"entity":"tree","location":"2 steps east, 1 step north"}, ... ]
    }

The planner only needs `inventory` + `nearby_stations` + `faced_tile`; the
`entities` list is informational context that helps the LLM planner.
"""

from __future__ import annotations

import json

import numpy as np

from crafter_tasks import (
    SEMANTIC_ID_TO_NAME, WALKABLE, VITALS, TOOLS,
)

# Tiles worth reporting as "entities" (skip terrain + player + void).
_TERRAIN = {"void", "grass", "path", "sand"}

_FACING_NAME = {
    (1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north",
}

# Materials shown in inventory only when non-zero.
_MATERIALS = ("sapling", "wood", "stone", "coal", "iron", "diamond")


def _facing_name(facing: tuple[int, int]) -> str:
    return _FACING_NAME.get(tuple(facing), "unknown")


def _rel_phrase(rx: int, ry: int) -> str:
    """Player-centric, axis-aligned (Crafter's view is north-up, never
    rotated). rx>0 east, ry>0 south."""
    parts = []
    if ry < 0:
        parts.append(f"{-ry} step{'s' if -ry > 1 else ''} north")
    elif ry > 0:
        parts.append(f"{ry} step{'s' if ry > 1 else ''} south")
    if rx < 0:
        parts.append(f"{-rx} step{'s' if -rx > 1 else ''} west")
    elif rx > 0:
        parts.append(f"{rx} step{'s' if rx > 1 else ''} east")
    return ", ".join(parts) if parts else "here"


def _inventory_block(inv: dict[str, int]) -> dict[str, int]:
    block: dict[str, int] = {k: int(inv.get(k, 0)) for k in VITALS}
    for m in _MATERIALS:
        if inv.get(m, 0) > 0:
            block[m] = int(inv[m])
    for t in TOOLS:
        if inv.get(t, 0) > 0:
            block[t] = int(inv[t])
    return block


def parse_crafter_description(state: dict, view_radius: int = 4,
                             max_entities: int = 12) -> str:
    """Return a compact JSON string describing the local Crafter view.

    ``state`` is the dict cached by CrafterTaskEnv.last_state:
    semantic (HxW), player_pos (x,y), facing (dx,dy), inventory (dict).
    """
    sem = state["semantic"]
    px, py = state["player_pos"]
    facing = state["facing"]
    inv = state["inventory"]
    W, H = sem.shape  # sem indexed [x, y]

    # nearby stations within crafting radius 1 (the 3x3 around the player)
    nearby = {"table": False, "furnace": False}
    for x in range(max(0, px - 1), min(W, px + 2)):
        for y in range(max(0, py - 1), min(H, py + 2)):
            name = SEMANTIC_ID_TO_NAME.get(int(sem[x, y]))
            if name in nearby:
                nearby[name] = True

    # faced tile
    fx, fy = px + facing[0], py + facing[1]
    if 0 <= fx < W and 0 <= fy < H:
        faced_tile = SEMANTIC_ID_TO_NAME.get(int(sem[fx, fy]), "void")
    else:
        faced_tile = "void"

    # entities in the local window, nearest first
    entities = []
    for x in range(max(0, px - view_radius), min(W, px + view_radius + 1)):
        for y in range(max(0, py - view_radius), min(H, py + view_radius + 1)):
            rx, ry = x - px, y - py
            if rx == 0 and ry == 0:
                continue
            name = SEMANTIC_ID_TO_NAME.get(int(sem[x, y]), "void")
            if name in _TERRAIN or name == "player":
                continue
            entities.append((abs(rx) + abs(ry), name, _rel_phrase(rx, ry)))
    entities.sort(key=lambda e: e[0])
    entity_list = [{"entity": n, "location": loc} for _, n, loc in entities[:max_entities]]

    desc = {
        "inventory": _inventory_block(inv),
        "facing": _facing_name(facing),
        "faced_tile": faced_tile,
        "nearby_stations": nearby,
        "entities": entity_list,
    }
    return json.dumps(desc)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from crafter_env import CrafterTaskEnv

    env = CrafterTaskEnv("collect_stone", seed=1)
    obs, info = env.reset(seed=1)
    print("Initial view:")
    print(parse_crafter_description(env.last_state))

    rng = np.random.default_rng(1)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            break
    print("\nAfter 20 random steps:")
    print(parse_crafter_description(env.last_state))
    print("\ncrafter_parser self-test OK")
