"""
crafter_logger.py — per-environment subgoal JSONL logger.

Crafter analogue of the MiniGrid project's ``subgoal_logger.py``; identical
event schema (init / completed / timed_out / new / episode_end), but the
"valid" flag is computed with the Crafter tracker's recognizer.
"""

from __future__ import annotations

import json
import os
import time

from crafter_tracker import CrafterSubgoalTracker


class CrafterSubgoalLogger:
    """Writes one JSONL file per env under logs/<stem>_subgoal_log/."""

    def __init__(self, log_dir: str, stem: str, num_envs: int):
        self.num_envs = num_envs
        self.dir = os.path.join(log_dir, f"{stem}_subgoal_log")
        os.makedirs(self.dir, exist_ok=True)
        self.files = [open(os.path.join(self.dir, f"env_{i:02d}.jsonl"),
                           "a", encoding="utf-8") for i in range(num_envs)]
        self.episode_counters = [0] * num_envs
        self._start_time = time.time()

    def log(self, env_idx, event, mission="", subgoal="", stage=None,
            steps_used=0, budget=0.0, reward=0.0, env_state=None,
            raw_llm=None, episode_success=None, episode_steps=None):
        entry = {"t": round(time.time() - self._start_time, 2),
                 "episode": self.episode_counters[env_idx], "event": event}
        if stage is not None:
            entry["stage"] = stage
        if mission:
            entry["mission"] = mission
        if subgoal:
            entry["subgoal"] = subgoal
            entry["valid"] = CrafterSubgoalTracker.is_recognized(subgoal)
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

    def on_episode_end(self, env_idx, mission, success, episode_steps):
        self.log(env_idx, "episode_end", mission=mission,
                 episode_success=success, episode_steps=episode_steps)
        self.episode_counters[env_idx] += 1

    def close(self):
        for f in self.files:
            f.close()
        self.files = []
