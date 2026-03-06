"""
llm_planner.py — Subgoal generation via local LLM (Ollama + Qwen 2.5 7B).

Sends the parsed environment JSON and mission string to Qwen through
Ollama's HTTP API. Uses few-shot examples to keep the model on track
and regex to extract the subgoal from the response.

Usage:
    from utils.llm_planner import LLMPlanner

    planner = LLMPlanner()
    subgoal = planner.get_subgoal(
        mission="open the purple door",
        env_json_str=env_json,
        direction=obs["direction"],
        past_subgoals=["pickup the purple key"],
    )
"""

import re
import json
import requests


# --- Few-shot prompt ---
# This is the full prompt sent to the LLM. The few-shot examples
# teach it to output in the exact format we need.

SYSTEM_PROMPT = """\
## Task
You are an agent in a Minigrid game, and your goal is to complete the Mission. \
The game world consists of doors, keys, balls, and boxes in various colors. \
You receive a JSON observation of your egocentric view and must generate the next subgoal.

## Allowed Subgoals
Your output must be one of these action forms:
- Subgoal: explore<end>
- Subgoal: go to the [color] [object/door/goal]<end>
- Subgoal: pickup the [color] [object]<end>
- Subgoal: drop the [color] [object]<end>
- Subgoal: open the [color] door<end>
- Subgoal: close the [color] door<end>
- Subgoal: search for the [color] [object]<end>

## System Rules
1. You can only interact with entities listed in the JSON. Do not hallucinate objects.
2. If your path is blocked by a locked door, you must hold a matching key before you can open it.
3. If no actionable entities are visible to achieve the mission, your subgoal must be to explore.
4. Output strictly in this format: Subgoal: <action><end>

--- Example 1: Empty View & No Past Subgoals ---
Mission: open the purple door
Direction: east
Environment: {"inventory": "empty", "boundaries": {"forward": "1 step", "left": "1 step", "right": "3 steps"}, "entities": []}
Past Subgoals: None
Output: Subgoal: explore<end>

--- Example 2: Deducing Prerequisites ---
Mission: pick up the green ball
Direction: south
Environment: {"inventory": "empty", "boundaries": {"forward": "3 steps", "left": "unknown", "right": "unknown"}, "entities": [{"entity": "locked purple door", "location": "3 steps forward"}, {"entity": "purple key", "location": "1 step left"}]}
Past Subgoals: None
Output: Subgoal: pickup the purple key<end>

--- Example 3: Executing the Sequence ---
Mission: pick up the green ball
Direction: south
Environment: {"inventory": "purple key", "boundaries": {"forward": "1 step", "left": "unknown", "right": "unknown"}, "entities": [{"entity": "locked purple door", "location": "1 step forward"}]}
Past Subgoals: pickup the purple key
Output: Subgoal: open the purple door<end>"""

# Maps MiniGrid direction indices to readable names
IDX_TO_DIRECTION = {0: "east", 1: "south", 2: "west", 3: "north"}

# Regex to pull out the subgoal text between "Subgoal:" and "<end>"
SUBGOAL_PATTERN = re.compile(r"Subgoal:\s*(.*?)<end>", re.IGNORECASE | re.DOTALL)


class LLMPlanner:
    """
    Queries a local Ollama server to get the next subgoal for the RL agent.

    By default uses Qwen 2.5 7B (4-bit). The model runs locally.
    """

    def __init__(self, model_name="qwen2.5:7b", host="http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/generate"

    def get_subgoal(
        self,
        mission: str,
        env_json_str: str,
        direction: int | str,
        past_subgoals: list[str],
    ) -> str:
        """
        Generate the next subgoal given the current state.

        Args:
            mission: environment mission string
            env_json_str: JSON from parse_env_description()
            direction: agent facing direction (int 0-3 or string)
            past_subgoals: list of previously completed subgoals

        Returns:
            Cleaned subgoal string, e.g. "pickup the yellow key"
        """

        # convert direction index to name if needed
        if isinstance(direction, int):
            dir_str = IDX_TO_DIRECTION.get(direction, "unknown")
        else:
            dir_str = direction

        # Format past subgoals
        if past_subgoals:
            past_str = ", ".join(past_subgoals)
        else:
            past_str = "None"

        # Construct final prompt
        final_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Current Task ---\n"
            f"Mission: {mission}\n"
            f"Direction: {dir_str}\n"
            f"Environment: {env_json_str}\n"
            f"Past Subgoals: {past_str}\n"
            f"Output: "
        )

        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   # no randomness — we want deterministic plans
                "num_predict": 30,    # short cap since we only need "Subgoal: ...<end>"
            },
        }

        try:
            response = requests.post(self.url, json=payload, timeout=10)
            response.raise_for_status()
            raw_text = response.json()["response"]
        except requests.Timeout:
            print("[LLMPlanner] Ollama timed out (>10s), defaulting to 'explore'")
            return "explore"
        except (requests.RequestException, KeyError) as e:
            print(f"[LLMPlanner] request failed: {e}")
            return "explore"

        # Extract subgoal with regex
        return self._parse_subgoal(raw_text)

    @staticmethod
    def _parse_subgoal(raw_text):
        """Try to extract the subgoal via regex. If it fails, log the raw output and fall back."""
        match = SUBGOAL_PATTERN.search(raw_text)
        if match:
            return match.group(1).strip()

        # regex didn't match — print what the model actually said so we can debug
        print(f"[LLMPlanner] regex miss, raw output:")
        print(f"  >>> {repr(raw_text[:200])}")
        cleaned = raw_text.strip().split("\n")[-1].strip()
        if cleaned:
            return cleaned

        return "explore"


# --- Quick self-test ---

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import minigrid
    from utils.env_parser import parse_env_description

    planner = LLMPlanner()
    env = gym.make("MiniGrid-DoorKey-5x5-v0")

    print("=" * 60)
    print("  LLMPlanner — Self-Test")
    print("=" * 60)

    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        env_json = parse_env_description(obs["image"], env.unwrapped.carrying)
        mission = obs["mission"]
        direction = obs["direction"]

        print(f"\n--- Seed {seed} ---")
        print(f"Mission  : {mission}")
        print(f"Direction: {IDX_TO_DIRECTION[direction]}")
        print(f"Env JSON : {env_json}")

        start = time.time()
        subgoal = planner.get_subgoal(mission, env_json, direction, past_subgoals=[])
        elapsed = time.time() - start

        print(f"Subgoal  : {subgoal}  ({elapsed:.1f}s)")

    # test with past subgoals
    print(f"\n--- With past subgoals ---")
    obs, _ = env.reset(seed=0)
    env_json = parse_env_description(obs["image"], env.unwrapped.carrying)
    start = time.time()
    subgoal = planner.get_subgoal(
        obs["mission"], env_json, obs["direction"],
        past_subgoals=["pickup the yellow key"],
    )
    elapsed = time.time() - start
    print(f"Subgoal  : {subgoal}  ({elapsed:.1f}s)")

    # edge case: nothing visible, should return "explore"
    print(f"\n--- Empty entities (explore test) ---")
    empty_json = json.dumps({
        "inventory": "empty",
        "boundaries": {"forward": "1 step", "left": "1 step", "right": "1 step"},
        "entities": [],
    })
    start = time.time()
    subgoal = planner.get_subgoal(
        "use the key to open the door and then get to the goal",
        empty_json, "east", past_subgoals=[],
    )
    elapsed = time.time() - start
    print(f"Subgoal  : {subgoal}  ({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
