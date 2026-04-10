"""
Per-environment subgoal logger for LGRL training.

Creates one JSONL file per environment under a run-specific directory:
    logs/<stem>_subgoal_log/
        env_00.jsonl
        env_01.jsonl
        ...

Each line is a JSON object with:
    - timestamp, env, episode, event, stage, mission, subgoal, valid,
      steps_used, budget, reward, env_state (for "new" events)

Events logged:
    - "init"      — first subgoal assigned at episode start
    - "completed" — subgoal verified as done
    - "timed_out" — subgoal budget exceeded
    - "new"       — next subgoal issued after completed/timed_out
    - "episode_end" — episode terminated (success or failure)
"""

from __future__ import annotations

import json
import os
import time

from utils.subgoal_tracker import SubgoalTracker


class SubgoalLogger:
    """Writes per-env JSONL logs with episode grouping."""

    def __init__(self, log_dir: str, stem: str, num_envs: int):
        self.num_envs = num_envs
        self.dir = os.path.join(log_dir, f"{stem}_subgoal_log")
        os.makedirs(self.dir, exist_ok=True)

        self.files: list = []
        for i in range(num_envs):
            path = os.path.join(self.dir, f"env_{i:02d}.jsonl")
            self.files.append(open(path, "a", encoding="utf-8"))

        self.episode_counters: list[int] = [0] * num_envs
        self._start_time = time.time()

    def log(
        self,
        env_idx: int,
        event: str,
        mission: str = "",
        subgoal: str = "",
        stage: int | None = None,
        steps_used: int = 0,
        budget: float = 0.0,
        reward: float = 0.0,
        env_state: str | None = None,
        raw_llm: str | None = None,
        episode_success: bool | None = None,
        episode_steps: int | None = None,
    ):
        entry: dict = {
            "t": round(time.time() - self._start_time, 2),
            "episode": self.episode_counters[env_idx],
            "event": event,
        }

        if stage is not None:
            entry["stage"] = stage
        if mission:
            entry["mission"] = mission
        if subgoal:
            entry["subgoal"] = subgoal
            entry["valid"] = SubgoalTracker.is_recognized(subgoal)
        if steps_used:
            entry["steps_used"] = steps_used
        if budget:
            entry["budget"] = round(budget, 1)
        if reward:
            entry["reward"] = round(reward, 6)
        if env_state is not None:
            entry["env_state"] = env_state
        if raw_llm is not None:
            entry["raw_llm"] = raw_llm
        if episode_success is not None:
            entry["success"] = episode_success
        if episode_steps is not None:
            entry["episode_steps"] = episode_steps

        self.files[env_idx].write(json.dumps(entry) + "\n")
        self.files[env_idx].flush()

    def on_episode_end(self, env_idx: int, mission: str, success: bool,
                       episode_steps: int):
        """Log episode termination and increment the episode counter."""
        self.log(
            env_idx, "episode_end",
            mission=mission,
            episode_success=success,
            episode_steps=episode_steps,
        )
        self.episode_counters[env_idx] += 1

    def close(self):
        for f in self.files:
            f.close()
        self.files = []
