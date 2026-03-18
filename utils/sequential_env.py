"""
Single-process environment stepper that replaces torch_ac's ParallelEnv.

ParallelEnv spawns worker processes for envs 1..N, which makes the
original env objects in the main process stale and inaccessible.  The
LGRL training loop needs direct access to each env's unwrapped state
(carrying, grid, front_pos) for subgoal completion checking, so we
step all envs sequentially in the main process instead.

For small environments like MiniGrid 5x5 the parallelism loss is
negligible; the LLM planner calls dominate wall-clock time.
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
