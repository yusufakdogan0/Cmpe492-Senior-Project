"""
env_parser.py — Converts MiniGrid observations to JSON for the LLM.

Takes the raw (7,7,3) egocentric grid and produces a compact JSON with:
  - inventory (what the agent is carrying)
  - boundaries (wall distances via ray casting)
  - entities  (visible doors, keys, goals, etc. with relative positions)

Does NOT include mission text — that's handled by the planner.
"""

import json
import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

# --- Constants ---

IGNORE_IDS = {0, 1, 2, 3}   # 0=unseen, 1=empty, 2=wall, 3=floor
WALL_ID = 2

DOOR_STATES = {0: "open", 1: "closed", 2: "locked"}

# Agent is always at bottom-center of the 7x7 egocentric grid
AGENT_X = 3
AGENT_Y = 6


def parse_env_description(image_array: np.ndarray, carrying_obj) -> str:
    """
    Parse the 7x7x3 observation grid into a JSON string.

    image_array: shape (7,7,3), indexed as [x, y, channel]
    carrying_obj: env.carrying object or None
    Returns: JSON string with inventory, boundaries, and entities.
    """

    # -- Inventory --
    if carrying_obj is not None:
        inv_str = f"{carrying_obj.color} {carrying_obj.type}"
    else:
        inv_str = "empty"

    # -- Boundary detection via ray casting --
    # Cast rays from agent position to find nearest wall in each direction

    # Forward: scan upward from agent along x=3
    bound_fwd = "unknown"
    for y in range(5, -1, -1):
        if int(image_array[3, y, 0]) == WALL_ID:
            steps = 6 - y
            bound_fwd = f"{steps} step{'s' if steps != 1 else ''}"
            break

    # Left: scan leftward from agent along y=6
    bound_left = "unknown"
    for x in range(2, -1, -1):
        if int(image_array[x, 6, 0]) == WALL_ID:
            steps = 3 - x
            bound_left = f"{steps} step{'s' if steps != 1 else ''}"
            break

    # Right: scan rightward from agent along y=6
    bound_right = "unknown"
    for x in range(4, 7):
        if int(image_array[x, 6, 0]) == WALL_ID:
            steps = x - 3
            bound_right = f"{steps} step{'s' if steps != 1 else ''}"
            break

    # -- Actionable entities --
    # Scan grid for anything that isn't wall/floor/empty/unseen
    entities_list = []

    for x in range(7):
        for y in range(7):
            obj_id   = int(image_array[x, y, 0])
            color_id = int(image_array[x, y, 1])
            state_id = int(image_array[x, y, 2])

            if obj_id in IGNORE_IDS:
                continue
            if x == AGENT_X and y == AGENT_Y:
                continue

            obj_name = IDX_TO_OBJECT.get(obj_id, f"unknown({obj_id})")
            color_name = IDX_TO_COLOR.get(color_id, f"unknown({color_id})")

            if obj_name == "door":
                state_str = DOOR_STATES.get(state_id, "unknown")
                entity_string = f"{state_str} {color_name} {obj_name}"
            else:
                entity_string = f"{color_name} {obj_name}"

            # relative position from agent
            forward_steps = AGENT_Y - y
            lateral_steps = x - AGENT_X

            location_string = _build_location_string(forward_steps, lateral_steps)

            entities_list.append({
                "entity": entity_string,
                "location": location_string,
            })

    # -- Assemble JSON --
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


def _build_location_string(forward_steps, lateral_steps):
    """Turn (forward, lateral) distances into a readable string like '2 steps forward, 1 step left'."""
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


# --- Quick self-test ---

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
