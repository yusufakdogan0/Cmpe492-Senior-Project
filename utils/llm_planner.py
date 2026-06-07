"""
Subgoal generation via a local LLM (Ollama + Qwen 2.5 7B).

Produces only subgoals from the LGRL paper plus "go near":
  search for, go near, pickup, open, close, drop.
No "explore" or "go to" subgoals.

Falls back to "search for the key" on errors or parse failures.

CPU-only Ollama installs need a generous HTTP timeout (model load ~10s,
then prefill on long prompts). Use ``compact_prompt=True`` (default)
and call ``warmup()`` before evaluation.
"""

from __future__ import annotations

import re
import time

import requests

# Few-shot prompt (higher quality, slow on CPU due to long prefill).
SYSTEM_PROMPT_FEWSHOT = """\
## Task
You are an agent in a Minigrid game, and your goal is to complete the Mission. \
The game world consists of doors, keys, balls, and boxes in various colors. \
You receive a JSON observation of your egocentric view and must generate the next subgoal.

## Allowed Subgoals
Your output must be one of these action forms:
- Subgoal: search for the [status?] [color] [object/door]<end>
- Subgoal: go near the [color] [object/door]<end>
- Subgoal: pickup the [color] [object]<end>
- Subgoal: drop the [color] [object]<end>
- Subgoal: open the [status] [color] door<end>
- Subgoal: close the [status] [color] door<end>

## System Rules
1. You can only interact with entities listed in the JSON. Do not hallucinate objects.
2. If your path is blocked by a locked door, you must hold a matching key before you can open it.
3. If no actionable entities are visible, use "search for" to find the next needed object.
4. Output strictly one line: Subgoal: <action>  (do not add <end>, [end], or :end)
5. [status?] means status is optional (keys, balls have no state). [status] means required (doors are always open, closed, or locked).
6. Use "search for" to find things, "go near" to approach them. Never output "explore" or "go to".

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
Output: Subgoal: open the locked purple door<end>

--- Example 4: Drop Key Before Pickup ---
Mission: pick up the green ball
Direction: south
Environment: {"inventory": "purple key", "boundaries": {"forward": "2 steps", "left": "1 step", "right": "unknown"}, "entities": [{"entity": "open purple door", "location": "1 step behind"}, {"entity": "green ball", "location": "2 steps forward"}]}
Past Subgoals: pickup the purple key, open the locked purple door
Output: Subgoal: drop the purple key<end>"""

# Shorter prompt — much faster on CPU; default for training/eval.
SYSTEM_PROMPT_COMPACT = """\
You are a Minigrid agent. Given Mission, Direction, and Environment JSON, \
output exactly one line starting with Subgoal: (no other text).

Format: Subgoal: <action>
Example: Subgoal: pickup the red key

Subgoal types:
- search for the [color] [key/ball/box/door] — target not in entities.
- pickup the [color] [key/ball/box] — target is in entities; use directly (not go near first).
- open the locked [color] door — locked door in entities.
- drop the [color] [key/ball/box] — free inventory.
- go near the [color] [door/key/ball/box] — only when Mission says "go to the ...".
- close the [color] door — rare.

Use key, ball, or box (never the word "object"). Only entities from the JSON. \
Never output explore, go to, or an end tag."""

DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_KEEP_ALIVE = "30m"

IDX_TO_DIRECTION = {0: "east", 1: "south", 2: "west", 3: "north"}

# Qwen often garbles the literal "<end>" from the prompt as [end], :end, " end", =end, etc.
_END_SUFFIX = re.compile(
    r"\s*(?:"
    r"<\s*end\s*>|"           # <end>
    r"\[\s*end\s*\]|"         # [end]
    r"[:=]\s*end\b|"          # :end  =end
    r"<\s*/?\s*end\s*>|"      # </end>
    r"\bend\s*$"              # trailing word "end"
    r")\s*",
    re.IGNORECASE,
)
_SUBGOAL_PREFIX = re.compile(r"^\s*Subgoal\s*:\s*", re.IGNORECASE)

VALID_PREFIXES = ("search for", "go near", "pickup", "open", "close", "drop")


class LLMPlanner:
    """
    Queries a local Ollama server to generate the next subgoal.

    Falls back to "search for the key" on errors or invalid output.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        host: str = "http://127.0.0.1:11434",
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        compact_prompt: bool = True,
    ):
        self.model_name = model_name
        host = host.rstrip("/")
        self.url = f"{host}/api/generate"
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.system_prompt = (
            SYSTEM_PROMPT_COMPACT if compact_prompt else SYSTEM_PROMPT_FEWSHOT
        )
        self.last_raw_response: str = ""
        self._timeout_count = 0
        self._error_count = 0
        self._parse_miss_count = 0

    def warmup(self) -> float:
        """Load the model into Ollama memory with a minimal request.

        On CPU the first call often takes 30–90s (weights load + inference).
        Returns elapsed seconds.
        """
        payload = self._build_payload(
            prompt=(
                f"{self.system_prompt}\n\n"
                "Mission: go to the red door\n"
                "Direction: east\n"
                'Environment: {"inventory": "empty", "entities": []}\n'
                "Output: "
            ),
            num_predict=24,
        )
        t0 = time.time()
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        _ = response.json().get("response", "")
        return time.time() - t0

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
            f"{self.system_prompt}\n\n"
            f"--- Current Task ---\n"
            f"Mission: {mission}\n"
            f"Direction: {dir_str}\n"
            f"Environment: {env_json_str}\n"
            f"Output: "
        )

        try:
            response = requests.post(
                self.url,
                json=self._build_payload(prompt),
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_text = response.json()["response"]
        except requests.Timeout:
            self._timeout_count += 1
            if self._timeout_count <= 3 or self._timeout_count % 25 == 0:
                print(
                    f"[LLMPlanner] Ollama timed out after {self.timeout}s "
                    f"({self._timeout_count} times) — using fallback subgoal"
                )
            self.last_raw_response = "[timeout]"
            return "search for the key", stage_index
        except (requests.RequestException, KeyError) as e:
            self._error_count += 1
            if self._error_count <= 3 or self._error_count % 25 == 0:
                print(f"[LLMPlanner] request failed: {e}")
            self.last_raw_response = f"[error: {e}]"
            return "search for the key", stage_index

        self.last_raw_response = raw_text
        return self._parse_subgoal(raw_text), stage_index

    def _build_payload(self, prompt: str, num_predict: int = 30) -> dict:
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": num_predict,
                # Smaller context → faster prefill on CPU.
                "num_ctx": 2048,
            },
        }

    @staticmethod
    def _clean_subgoal_body(body: str) -> str:
        """Strip end markers and junk; return lowercased action phrase."""
        body = body.strip().split("\n")[0].strip()
        body = _END_SUFFIX.sub("", body)
        return body.strip(" \t.:;=<>[]").lower()

    def _parse_subgoal(self, raw_text: str) -> str:
        text = raw_text.strip()
        if not text:
            return self._fallback_subgoal(text, "empty response")

        # Prefer a line that starts with "Subgoal:"
        for line in text.splitlines():
            if not _SUBGOAL_PREFIX.match(line):
                continue
            body = _SUBGOAL_PREFIX.sub("", line, count=1)
            candidate = self._clean_subgoal_body(body)
            if self._is_valid_subgoal(candidate):
                return candidate
            return self._fallback_subgoal(
                text, f"invalid prefix in {candidate!r}"
            )

        # Whole-string fallback (single-line completions without newline)
        m = re.search(r"Subgoal\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            candidate = self._clean_subgoal_body(m.group(1))
            if self._is_valid_subgoal(candidate):
                return candidate

        # Last line might be just "pickup the red key" with no prefix
        last = self._clean_subgoal_body(text.splitlines()[-1])
        if self._is_valid_subgoal(last):
            return last

        return self._fallback_subgoal(text, "could not parse Subgoal line")

    @staticmethod
    def _is_valid_subgoal(candidate: str) -> bool:
        return bool(candidate) and any(
            candidate.startswith(p) for p in VALID_PREFIXES
        )

    def _fallback_subgoal(self, raw_text: str, reason: str) -> str:
        self._parse_miss_count += 1
        if self._parse_miss_count <= 3 or self._parse_miss_count % 25 == 0:
            print(
                f"[LLMPlanner] parse fallback ({reason}): "
                f"{repr(raw_text[:120])} → 'search for the key'"
            )
        return "search for the key"


# -- self-test -----------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import minigrid  # noqa: F401
    from utils.env_parser import parse_env_description

    planner = LLMPlanner()
    print("Warming up…")
    print(f"  warmup took {planner.warmup():.1f}s")

    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    print("=" * 60)
    print("  LLMPlanner -- Self-Test")
    print("=" * 60)

    parse_cases = [
        "Subgoal: pickup the grey key<end>",
        "Subgoal: search for the purple box end",
        "Subgoal: go near the green key:end",
        "Subgoal: search for the purple object[end]",
        "pickup the yellow key",
    ]
    print("Parse self-check:")
    for sample in parse_cases:
        print(f"  {sample!r} -> {planner._parse_subgoal(sample)!r}")

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
