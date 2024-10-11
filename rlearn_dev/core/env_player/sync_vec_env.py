import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from .base import BaseVecEnvPlayer

class SyncVecEnvPlayer(BaseVecEnvPlayer):

    def __init__(self, env_fn, num_envs=1, max_workers=None):
        assert num_envs > 0, "num_envs must be greater than 0"
        self.envs = [env_fn() for _ in range(num_envs)]
        self._num_envs = num_envs
        self._single_action_space = self.envs[0].action_space
        self._single_observation_space = self.envs[0].observation_space
        self.max_workers = max_workers if max_workers is not None else min(multiprocessing.cpu_count(), num_envs)

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

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def num_envs(self):
        return self._num_envs

    @property   
    def single_action_space(self):
        return self._single_action_space

    @property
    def single_observation_space(self):
        return self._single_observation_space
