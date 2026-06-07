"""
crafter_env.py — Gymnasium wrapper around crafter.Env that turns a single
target achievement into an episodic goal.

Why this exists
---------------
Crafter ships the legacy 4-tuple `gym` API and an open-ended survival
episode (length 10k, ends only on death). The LGRL training loop expects
the Gymnasium 5-tuple AND a MiniGrid-style episodic mission that ends with
a positive reward the moment the goal is reached. This wrapper bridges
both:

  * obs is a dict ``{"image": 64x64x3 uint8, "mission": <nl string>}``
  * the episode TERMINATES with reward +1 the step the target
    achievement unlocks  -> ``success = reward > 0`` (paper Eq. 5 transfers
    unchanged)
  * the episode TERMINATES with reward 0 on death
  * the episode TRUNCATES with reward 0 at ``max_steps`` (per-task T_max)

The wrapped (unwrapped) env also exposes everything the Crafter parser and
tracker need, snapshotted every step:

    env.task              target achievement key (e.g. "collect_diamond")
    env.mission           natural-language mission string
    env.last_state        dict(semantic, player_pos, facing, inventory, achievements)
    env.state()           -> last_state

Crafter API facts verified against the installed package:
    obs (agent view)      64x64x3 uint8 RGB
    action space          Discrete(17)
    info keys             inventory, achievements, semantic, player_pos, ...
    info["semantic"]      64x64 uint8 full-world grid, indexed [x, y]
    player.facing         (dx, dy) tuple; `do` acts on player_pos + facing
"""

from __future__ import annotations

import crafter
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crafter_tasks import TASK_MISSION, task_max_steps, INVENTORY_KEYS


class CrafterTaskEnv(gym.Env):
    """Single-target, episodic Crafter task with a MiniGrid-style mission."""

    metadata = {"render_modes": []}

    def __init__(self, task: str, max_steps: int | None = None,
                 seed: int | None = None):
        super().__init__()
        if task not in TASK_MISSION:
            raise ValueError(f"Unknown Crafter task: {task!r}")
        self.task = task
        self.mission = TASK_MISSION[task]
        self.max_steps = max_steps if max_steps is not None else task_max_steps(task)
        self._seed = seed

        # Underlying Crafter env. Keep its internal length at the default so
        # its own `done` flag means death (we cap episodes ourselves below).
        self._env = crafter.Env(seed=seed)

        self.action_space = spaces.Discrete(len(self._env.action_names))
        # The agent reads obs["image"]; the text stream is handled by the
        # custom preprocess. We still publish a Dict space for parity.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        })

        self._elapsed = 0
        self._prev_target_count = 0
        self.last_state: dict = {}

    # -- helpers --------------------------------------------------------

    def _snapshot(self, info: dict) -> None:
        """Cache the per-step symbolic state for the parser/tracker."""
        px, py = (int(v) for v in info["player_pos"])
        facing = tuple(int(v) for v in self._env._player.facing)
        self.last_state = {
            "semantic": info["semantic"],
            "player_pos": (px, py),
            "facing": facing,
            "inventory": {k: int(info["inventory"].get(k, 0)) for k in INVENTORY_KEYS},
            "achievements": {k: int(v) for k, v in info["achievements"].items()},
        }

    def state(self) -> dict:
        return self.last_state

    def _obs(self, image) -> dict:
        return {"image": np.asarray(image, dtype=np.uint8), "mission": self.mission}

    # -- gymnasium API --------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            # Rebuild with the new seed (crafter seeds at construction).
            self._env = crafter.Env(seed=seed)
            self._seed = seed
        image = self._env.reset()
        # Prime a state snapshot with a no-op-free initial step-less read:
        # crafter exposes player/world right after reset.
        info0 = {
            "semantic": self._env._sem_view(),
            "player_pos": np.array(self._env._player.pos),
            "inventory": dict(self._env._player.inventory),
            "achievements": dict(self._env._player.achievements),
        }
        self._snapshot(info0)
        self._elapsed = 0
        self._prev_target_count = self.last_state["achievements"].get(self.task, 0)
        return self._obs(image), {"task": self.task, "state": self.last_state}

    def step(self, action):
        image, _crafter_reward, crafter_done, info = self._env.step(int(action))
        self._snapshot(info)
        self._elapsed += 1

        target_count = self.last_state["achievements"].get(self.task, 0)
        target_unlocked = target_count > self._prev_target_count
        self._prev_target_count = target_count

        terminated = False
        truncated = False
        reward = 0.0

        if target_unlocked:
            terminated = True
            reward = 1.0                      # mission success (Eq. 5 trigger)
        elif crafter_done:
            terminated = True                 # death before reaching the goal
            reward = 0.0
        elif self._elapsed >= self.max_steps:
            truncated = True                  # ran out of time
            reward = 0.0

        info_out = {
            "task": self.task,
            "state": self.last_state,
            "achievements": self.last_state["achievements"],
            "is_success": bool(target_unlocked),
        }
        return self._obs(image), reward, terminated, truncated, info_out

    def render(self):
        return self._env.render()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    env = CrafterTaskEnv("collect_wood", seed=0)
    obs, info = env.reset(seed=0)
    print("obs keys:", list(obs.keys()), "| image", obs["image"].shape, obs["image"].dtype)
    print("mission:", obs["mission"], "| task:", env.task, "| T_max:", env.max_steps)
    print("state keys:", list(env.last_state.keys()))
    print("facing:", env.last_state["facing"], "| player_pos:", env.last_state["player_pos"])

    total_term = total_trunc = 0
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        done = False
        steps = 0
        while not done:
            a = int(rng.integers(0, env.action_space.n))
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
        total_term += int(term)
        total_trunc += int(trunc)
        print(f"  ep{ep}: {steps:3d} steps | reward {r} | term={term} trunc={trunc} "
              f"| success={info['is_success']}")
    print(f"random policy over 3 eps: {total_term} terminated, {total_trunc} truncated")
    print("crafter_env self-test OK")
