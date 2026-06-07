"""
seq_env.py — single-process environment stepper that replaces torch_ac's
ParallelEnv (self-contained copy of the MiniGrid project's
``sequential_env.py``).

torch_ac's ParallelEnv spawns worker processes, which makes the original
env objects in the main process stale and their unwrapped state
(``last_state``: semantic, inventory, achievements) unreachable. The LGRL
loop needs that state every step for subgoal completion checking, so we
step all envs sequentially in the training process and auto-reset on done.
"""

import gymnasium as gym


class SequentialEnv(gym.Env):
    """Drop-in replacement for ``torch_ac.utils.ParallelEnv``."""

    def __init__(self, envs):
        assert len(envs) >= 1
        self.envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(self):
        return [env.reset()[0] for env in self.envs]

    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            results.append((obs, reward, terminated, truncated, info))
        return zip(*results)

    def render(self):
        raise NotImplementedError
