"""
Environment Parser for MiniGrid.

Translates a MiniGrid egocentric (7, 7, 3) visual observation into a
strict JSON string for an LLM planner.

This module does NOT include the mission string. Its sole job is
translating the visual matrix. The LLMPlanner will combine this
output with the mission text later.

Usage
-----
    from utils.env_parser import parse_env_description

    obs, _ = env.reset()
    json_str = parse_env_description(obs["image"], env.unwrapped.carrying)
"""

import json
import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Object IDs to ignore
IGNORE_IDS = {0, 1, 2, 3}   # 0: unseen, 1: empty, 2: wall, 3: floor
WALL_ID = 2

# Door state mapping
DOOR_STATES = {0: "open", 1: "closed", 2: "locked"}

# Agent's fixed position in the egocentric 7×7 grid
AGENT_X = 3
AGENT_Y = 6


# ──────────────────────────────────────────────
# Main Parser
# ──────────────────────────────────────────────

def parse_env_description(image_array: np.ndarray, carrying_obj) -> str:
    """
    Convert a MiniGrid egocentric observation into a JSON string.

    Parameters
    ----------
    image_array : np.ndarray, shape (7, 7, 3)
        The agent's partial view. Indexed as [x, y, channel].
    carrying_obj : minigrid WorldObj or None
        The object the agent is currently carrying.

    Returns
    -------
    str
        A JSON string with keys: inventory, boundaries, entities.
    """

    # ── Step 1: Parse Inventory ──
    if carrying_obj is not None:
        inv_str = f"{carrying_obj.color} {carrying_obj.type}"
    else:
        inv_str = "empty"

    # ── Step 2: Detect Room Boundaries (Ray Casting) ──

    # Forward: from agent (y=6), cast ray upward along x=3
    bound_fwd = "unknown"
    for y in range(5, -1, -1):
        if int(image_array[3, y, 0]) == WALL_ID:
            steps = 6 - y
            bound_fwd = f"{steps} step{'s' if steps != 1 else ''}"
            break

    # Left: from agent (x=3), cast ray leftward along y=6
    bound_left = "unknown"
    for x in range(2, -1, -1):
        if int(image_array[x, 6, 0]) == WALL_ID:
            steps = 3 - x
            bound_left = f"{steps} step{'s' if steps != 1 else ''}"
            break

    # Right: from agent (x=3), cast ray rightward along y=6
    bound_right = "unknown"
    for x in range(4, 7):
        if int(image_array[x, 6, 0]) == WALL_ID:
            steps = x - 3
            bound_right = f"{steps} step{'s' if steps != 1 else ''}"
            break

    # ── Step 3: Extract Actionable Entities ──
    entities_list = []

    for x in range(7):
        for y in range(7):
            obj_id   = int(image_array[x, y, 0])
            color_id = int(image_array[x, y, 1])
            state_id = int(image_array[x, y, 2])

            # Skip non-actionable tiles
            if obj_id in IGNORE_IDS:
                continue

            # Skip agent's own tile
            if x == AGENT_X and y == AGENT_Y:
                continue

            # Object semantics
            obj_name = IDX_TO_OBJECT.get(obj_id, f"unknown({obj_id})")
            color_name = IDX_TO_COLOR.get(color_id, f"unknown({color_id})")

            if obj_name == "door":
                state_str = DOOR_STATES.get(state_id, "unknown")
                entity_string = f"{state_str} {color_name} {obj_name}"
            else:
                entity_string = f"{color_name} {obj_name}"

            # Spatial distances
            forward_steps = AGENT_Y - y
            lateral_steps = x - AGENT_X

            location_string = _build_location_string(forward_steps, lateral_steps)

            entities_list.append({
                "entity": entity_string,
                "location": location_string,
            })

    # ── Step 4: JSON Assembly ──
    result = {
        "inventory": inv_str,
        "boundaries": {
            "forward": bound_fwd,
            "left": bound_left,
            "right": bound_right,
        },
        "entities": entities_list,
    }

    return json.dumps(result)


def _build_location_string(forward_steps: int, lateral_steps: int) -> str:
    """Build a human-readable relative location string."""
    parts = []

    if forward_steps > 0:
        s = "s" if forward_steps != 1 else ""
        parts.append(f"{forward_steps} step{s} forward")
    elif forward_steps < 0:
        steps = abs(forward_steps)
        s = "s" if steps != 1 else ""
        parts.append(f"{steps} step{s} behind")

    if lateral_steps < 0:
        steps = abs(lateral_steps)
        s = "s" if steps != 1 else ""
        parts.append(f"{steps} step{s} left")
    elif lateral_steps > 0:
        s = "s" if lateral_steps != 1 else ""
        parts.append(f"{lateral_steps} step{s} right")

    if not parts:
        return "at agent location"

    return ", ".join(parts)


# ──────────────────────────────────────────────
# Self-Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import gymnasium as gym
    import minigrid  # noqa: F401

    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")

    print("=" * 60)
    print("  EnvParser — Self-Test (JSON Output)")
    print("=" * 60)

    for episode in range(3):
        obs, _ = env.reset(seed=episode)
        raw_json = parse_env_description(obs["image"], env.unwrapped.carrying)
        pretty = json.dumps(json.loads(raw_json), indent=2)

        print(f"\n--- Episode {episode} (seed={episode}) ---")
        print(f"Mission: {obs['mission']}")
        print(f"\nParser output:")
        print(pretty)
        print(f"\nRaw grid (object IDs):")
        print(obs["image"][:, :, 0])

        if episode < 2:
            input("\nPress Enter for next episode...")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
