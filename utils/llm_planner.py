"""
Subgoal generation via a local LLM (Ollama + Qwen 2.5 7B).

Produces only subgoals from the LGRL paper:
  search for, pickup, open, close, drop.
No "explore" or "go to" subgoals.

Falls back to "search for the key" on errors or parse failures.
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
- Subgoal: search for the [status?] [color] [object/door]<end>
- Subgoal: pickup the [color] [object]<end>
- Subgoal: drop the [color] [object]<end>
- Subgoal: open the [status] [color] door<end>
- Subgoal: close the [status] [color] door<end>

## System Rules
1. You can only interact with entities listed in the JSON. Do not hallucinate objects.
2. If your path is blocked by a locked door, you must hold a matching key before you can open it.
3. If no actionable entities are visible, use "search for" to find the next needed object.
4. Output strictly in this format: Subgoal: <action><end>
5. [status?] means status is optional (keys, balls have no state). [status] means required (doors are always open, closed, or locked).
6. Never output "explore" or "go to" — use "search for" instead.

--- Example 1: Empty View & No Past Subgoals ---
Mission: open the purple door
Direction: east
Environment: {"inventory": "empty", "boundaries": {"forward": "1 step", "left": "1 step", "right": "3 steps"}, "entities": []}
Past Subgoals: None
Output: Subgoal: search for the purple door<end>

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

# Valid prefixes from the paper
VALID_PREFIXES = ("search for", "pickup", "open", "close", "drop")


class LLMPlanner:
    """
    Queries a local Ollama server to generate the next subgoal.

    Falls back to "search for the key" on errors or invalid output.
    """

    def __init__(self, model_name: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/generate"
        self.last_raw_response: str = ""

    def get_subgoal(
        self,
        mission: str,
        env_json_str: str,
        direction: int | str,
        stage_index: int = 0,
    ) -> tuple[str, int]:
        """Return (subgoal_string, stage_index).

        stage_index is passed through unchanged — the LLM planner does not
        use stages internally. The training loop manages stage advancement.
        """
        if isinstance(direction, int):
            dir_str = IDX_TO_DIRECTION.get(direction, "unknown")
        else:
            dir_str = direction

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Current Task ---\n"
            f"Mission: {mission}\n"
            f"Direction: {dir_str}\n"
            f"Environment: {env_json_str}\n"
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
            print("[LLMPlanner] Ollama timed out, defaulting to 'search for the key'")
            self.last_raw_response = "[timeout]"
            return "search for the key", stage_index
        except (requests.RequestException, KeyError) as e:
            print(f"[LLMPlanner] request failed: {e}")
            self.last_raw_response = f"[error: {e}]"
            return "search for the key", stage_index

        self.last_raw_response = raw_text
        subgoal = self._parse_subgoal(raw_text)
        return subgoal, stage_index

    @staticmethod
    def _parse_subgoal(raw_text: str) -> str:
        match = SUBGOAL_PATTERN.search(raw_text)
        if match:
            candidate = match.group(1).strip().lower()
            # Validate it starts with an allowed prefix
            if any(candidate.startswith(p) for p in VALID_PREFIXES):
                return candidate
            print(f"[LLMPlanner] invalid prefix in: {candidate!r}")

        # Try last line
        cleaned = raw_text.strip().split("\n")[-1].strip().lower()
        if cleaned and any(cleaned.startswith(p) for p in VALID_PREFIXES):
            return cleaned

        print(f"[LLMPlanner] regex miss: {repr(raw_text[:200])}")
        return "search for the key"


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
    print("  LLMPlanner -- Self-Test (No Explore)")
    print("=" * 60)

    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        env_json = parse_env_description(obs["image"], env.unwrapped.carrying)

        start = time.time()
        subgoal, _ = planner.get_subgoal(
            obs["mission"], env_json, obs["direction"], stage_index=0
        )
        elapsed = time.time() - start
        print(f"Seed {seed} | {subgoal}  ({elapsed:.1f}s)")
        assert "explore" not in subgoal.lower()

    print("=" * 60)
