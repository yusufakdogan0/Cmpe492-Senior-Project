"""
Subgoal generation via a local LLM (Ollama + Qwen 2.5 7B).

Sends the parsed environment JSON, mission, direction, and subgoal
history to Qwen through Ollama's HTTP API.  Few-shot examples anchor
the model's output format; a regex extracts the subgoal string.

This is the primary planner used during both training and evaluation.
Falls back to ``"explore"`` on network errors or parse failures so
training can continue even if the LLM is temporarily unreachable.
"""

from __future__ import annotations

import re
import json
import requests

SYSTEM_PROMPT = """\
## Task
You are an agent in a Minigrid game, and your goal is to complete the Mission. \
The game world consists of doors, keys, balls, and boxes in various colors. \
You receive a JSON observation of your egocentric view and must generate the next subgoal.

## Allowed Subgoals
Your output must be one of these action forms:
- Subgoal: explore<end>
- Subgoal: go to the [status?] [color] [object/door]<end>
- Subgoal: pickup/drop the [color] [object]<end>
- Subgoal: open/close the [status] [color] door<end>
- Subgoal: search for the [status?] [color] [object/door]<end>

## System Rules
1. You can only interact with entities listed in the JSON. Do not hallucinate objects.
2. If your path is blocked by a locked door, you must hold a matching key before you can open it.
3. If no actionable entities are visible to achieve the mission, your subgoal must be to explore.
4. Output strictly in this format: Subgoal: <action><end>
5. [status?] means status is optional (keys, balls have no state). [status] means required (doors are always open, closed, or locked).

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
Output: Subgoal: open the locked purple door<end>"""

IDX_TO_DIRECTION = {0: "east", 1: "south", 2: "west", 3: "north"}

SUBGOAL_PATTERN = re.compile(r"Subgoal:\s*(.*?)<end>", re.IGNORECASE | re.DOTALL)


class LLMPlanner:
    """
    Queries a local Ollama server to generate the next subgoal.

    Default model: Qwen 2.5 7B (4-bit quantised via Ollama).
    Falls back to ``"explore"`` on network errors or parse failures.
    """

    def __init__(self, model_name: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/generate"

    def get_subgoal(
        self,
        mission: str,
        env_json_str: str,
        direction: int | str,
        past_subgoals: list[str],
    ) -> str:
        if isinstance(direction, int):
            dir_str = IDX_TO_DIRECTION.get(direction, "unknown")
        else:
            dir_str = direction

        past_str = ", ".join(past_subgoals) if past_subgoals else "None"

        prompt = (
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
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 30},
        }

        try:
            response = requests.post(self.url, json=payload, timeout=10)
            response.raise_for_status()
            raw_text = response.json()["response"]
        except requests.Timeout:
            print("[LLMPlanner] Ollama timed out, defaulting to 'explore'")
            return "explore"
        except (requests.RequestException, KeyError) as e:
            print(f"[LLMPlanner] request failed: {e}")
            return "explore"

        return self._parse_subgoal(raw_text)

    @staticmethod
    def _parse_subgoal(raw_text: str) -> str:
        match = SUBGOAL_PATTERN.search(raw_text)
        if match:
            return match.group(1).strip()

        print(f"[LLMPlanner] regex miss: {repr(raw_text[:200])}")
        cleaned = raw_text.strip().split("\n")[-1].strip()
        return cleaned if cleaned else "explore"


# -- self-test -----------------------------------------------------------

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import minigrid  # noqa: F401
    from utils.env_parser import parse_env_description

    planner = LLMPlanner()
    env = gym.make("MiniGrid-DoorKey-5x5-v0")

    print("=" * 60)
    print("  LLMPlanner -- Self-Test")
    print("=" * 60)

    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        env_json = parse_env_description(obs["image"], env.unwrapped.carrying)

        start = time.time()
        subgoal = planner.get_subgoal(
            obs["mission"], env_json, obs["direction"], past_subgoals=[]
        )
        elapsed = time.time() - start
        print(f"Seed {seed} | {subgoal}  ({elapsed:.1f}s)")

    obs, _ = env.reset(seed=0)
    env_json = parse_env_description(obs["image"], env.unwrapped.carrying)
    start = time.time()
    subgoal = planner.get_subgoal(
        obs["mission"], env_json, obs["direction"],
        past_subgoals=["pickup the yellow key"],
    )
    elapsed = time.time() - start
    print(f"With history | {subgoal}  ({elapsed:.1f}s)")

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
    print(f"Empty view   | {subgoal}  ({elapsed:.1f}s)")

    print("=" * 60)
