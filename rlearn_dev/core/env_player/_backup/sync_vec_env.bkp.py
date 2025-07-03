import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from .base import BaseVecEnvPlayer

class SyncVecEnvPlayer(BaseVecEnvPlayer):
    """
    SyncVecEnvPlayer is a vectorized environment player that uses a single process to step multiple environments.
    Reference:
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py
    """

    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)

    def reset(self, **kwargs):
        if self._num_envs == 1:
            ob, info = self.envs[0].reset(**kwargs)
            return np.expand_dims(ob, axis=0), [info]

        results = []
        infos = []
        for env in self.envs:
            ob, info = env.reset(**kwargs)
            results.append(ob)
            infos.append(info)
        return np.stack(results), infos

    def step(self, actions):
        if self._num_envs == 1:
            ob, reward, terminated, truncated, info = self.envs[0].step(actions[0])
            if terminated or truncated:
                ob, info = self.envs[0].reset()
            return np.expand_dims(ob, axis=0), np.array([reward]), np.array([terminated]), np.array([truncated]), [info]

        obs = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        # FIX TODO 
        # https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._step_env, env, action) for env, action in zip(self.envs, actions)]
            for future, env in zip(futures, self.envs): 
                ob, reward, terminated, truncated, info = future.result()
                if terminated or truncated:
                    ob, info = env.reset()
                
                obs.append(ob)
                rewards.append(reward)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)

        return np.stack(obs), np.stack(rewards), np.stack(terminateds), np.stack(truncateds), infos

    def _step_env(self, env, action):
        return env.step(action)

    def render(self):
        for env in self.envs:
            env.render()

    def close(self, **kwargs):
        for env in self.envs:
            env.close()