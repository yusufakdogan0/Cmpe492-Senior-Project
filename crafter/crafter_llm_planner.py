"""
crafter_llm_planner.py — subgoal generation via a local LLM (Ollama +
Qwen 2.5 7B) for Crafter.

Crafter analogue of the MiniGrid project's ``llm_planner.py``. Same
interface (``get_subgoal(mission, env_json_str, direction, stage_index)
-> (subgoal, stage_index)``) and the same Ollama call, but the system
prompt teaches Crafter's subgoal grammar and tech-tree rules instead of
MiniGrid's door/key vocabulary.

In our evaluation the rule-based planner is the training oracle; this
LLM planner is what you swap in to test the *full* framework (LLM
decomposition + PPO low-level control).

Falls back to "collect wood" (the universal first step in Crafter's tech
tree) on errors or parse failures.
"""

from __future__ import annotations

import re
import requests

SYSTEM_PROMPT = """\
## Task
You are an agent in the survival game Crafter. Your goal is to complete the \
Mission by issuing ONE next subgoal. You see a JSON summary of your local \
view: your inventory, the tile you are facing, which crafting stations are \
within reach, and nearby objects/materials.

## Allowed Subgoals
Output exactly one subgoal in one of these forms:
- Subgoal: collect <N> <material><end>      (material: wood, stone, coal, iron, sapling; N is a number)
- Subgoal: collect <material><end>          (material: wood, stone, coal, iron, diamond, drink, sapling)
- Subgoal: place a <station><end>           (station: table, furnace, stone, plant)
- Subgoal: make a <tool><end>               (tool: wood pickaxe, stone pickaxe, wood sword, stone sword)
- Subgoal: make an <tool><end>              (tool: iron pickaxe, iron sword)
- Subgoal: eat a cow<end>
- Subgoal: defeat a zombie<end>  /  Subgoal: defeat a skeleton<end>

## Crafter Tech-Tree Rules
1. Collecting wood needs nothing. You must `place a table` (needs 2 wood) before making any tool.
2. A wood pickaxe (needs 1 wood, near table) is required to mine stone and coal.
3. A stone pickaxe (needs 1 wood + 1 stone, near table) is required to mine iron.
4. A furnace (needs 4 stone) plus a table are required to make iron tools.
5. An iron pickaxe (needs 1 wood + 1 coal + 1 iron, near table and furnace) is required to mine diamond.
6. Only request a subgoal whose prerequisites you already have; otherwise request the missing prerequisite first.
7. Output strictly: Subgoal: <action><end>

--- Example 1: Empty hands ---
Mission: make a wood pickaxe
Environment: {"inventory": {"health":9,"food":9,"drink":9,"energy":9}, "facing":"south", "faced_tile":"grass", "nearby_stations":{"table":false,"furnace":false}, "entities":[{"entity":"tree","location":"2 steps east"}]}
Output: Subgoal: collect 2 wood<end>

--- Example 2: Have wood, no table ---
Mission: make a wood pickaxe
Environment: {"inventory": {"health":9,"food":9,"drink":9,"energy":9,"wood":2}, "facing":"south", "faced_tile":"grass", "nearby_stations":{"table":false,"furnace":false}, "entities":[]}
Output: Subgoal: place a table<end>

--- Example 3: Table ready, have wood ---
Mission: make a wood pickaxe
Environment: {"inventory": {"health":9,"food":9,"drink":9,"energy":9,"wood":1}, "facing":"north", "faced_tile":"table", "nearby_stations":{"table":true,"furnace":false}, "entities":[]}
Output: Subgoal: make a wood pickaxe<end>

--- Example 4: Need stone, have pickaxe ---
Mission: make a stone pickaxe
Environment: {"inventory": {"health":9,"food":9,"drink":9,"energy":9,"wood":1,"wood_pickaxe":1}, "facing":"east", "faced_tile":"stone", "nearby_stations":{"table":true,"furnace":false}, "entities":[{"entity":"stone","location":"1 step east"}]}
Output: Subgoal: collect 1 stone<end>"""

SUBGOAL_PATTERN = re.compile(r"Subgoal:\s*(.*?)<end>", re.IGNORECASE | re.DOTALL)

VALID_PREFIXES = ("collect", "place", "make", "eat", "defeat")


class CrafterLLMPlanner:
    """Queries a local Ollama server for the next Crafter subgoal."""

    def __init__(self, model_name: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{host}/api/generate"
        self.last_raw_response: str = ""

    def get_subgoal(self, mission: str, env_json_str: str,
                    direction=0, stage_index: int = 0) -> tuple[str, int]:
        """Return (subgoal_string, stage_index). stage_index is passed
        through unchanged — the training loop manages stage advancement."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Current Task ---\n"
            f"Mission: {mission}\n"
            f"Environment: {env_json_str}\n"
            f"Output: "
        )
        payload = {"model": self.model_name, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.0, "num_predict": 30}}
        try:
            response = requests.post(self.url, json=payload, timeout=10)
            response.raise_for_status()
            raw_text = response.json()["response"]
        except requests.Timeout:
            self.last_raw_response = "[timeout]"
            return "collect wood", stage_index
        except (requests.RequestException, KeyError) as e:
            self.last_raw_response = f"[error: {e}]"
            return "collect wood", stage_index

        self.last_raw_response = raw_text
        return self._parse_subgoal(raw_text), stage_index

    @staticmethod
    def _parse_subgoal(raw_text: str) -> str:
        match = SUBGOAL_PATTERN.search(raw_text)
        if match:
            candidate = match.group(1).strip().lower()
            if candidate.startswith(VALID_PREFIXES):
                return candidate
        cleaned = raw_text.strip().split("\n")[-1].strip().lower()
        if cleaned and cleaned.startswith(VALID_PREFIXES):
            return cleaned
        return "collect wood"


if __name__ == "__main__":
    # Offline parse test (no Ollama needed).
    p = CrafterLLMPlanner()
    samples = [
        "Subgoal: collect 2 wood<end>",
        "blah blah\nSubgoal: place a table<end>\nextra",
        "Subgoal: make an iron pickaxe<end>",
        "garbage with no marker",
    ]
    for s in samples:
        print(f"  {s!r:55s} -> {p._parse_subgoal(s)!r}")
    print("crafter_llm_planner parse self-test OK")
