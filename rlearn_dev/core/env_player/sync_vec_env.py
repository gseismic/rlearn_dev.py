import numpy as np
from copy import deepcopy
from .base import BaseVecEnvPlayer

# Gymnasium-like SyncVecEnvPlayer
class SyncVecEnvPlayer(BaseVecEnvPlayer):
    """
    SyncVecEnvPlayer is a vectorized environment player that uses a single process to step multiple environments.
    Reference:
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py
    """

    def __init__(self, env_fns, **kwargs):
        """
        Args:
            copy (bool): Whether to copy the observation space.
        """
        super().__init__(env_fns, **kwargs)
        self._rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._terminateds = np.zeros(self.num_envs, dtype=np.bool_)
        self._truncateds = np.zeros(self.num_envs, dtype=np.bool_)
        self._observations = np.zeros((self.num_envs,) + self.single_observation_space.shape,
                                      dtype=self.single_observation_space.dtype)
        self._should_reset = np.zeros(self.num_envs, dtype=np.bool_)
        self._infos = {'infos': []}

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        elif isinstance(seed, list):
            assert len(seed) == self.num_envs, f"The length of seed ({len(seed)}) must be equal to the number of environments ({self.num_envs})."

        obs, infos = [], {'infos': []}
        for i, (env, my_seed) in enumerate(zip(self.envs, seed)):
            ob, info = env.reset(seed=my_seed, options=options)
            obs.append(ob)
            infos['infos'].append(info)
        return np.stack(obs), infos

    def step(self, actions):
        obs, infos = [], {'infos': []}
        for i, action in enumerate(actions):
            if self._should_reset[i]:
                ob, info = self.envs[i].reset()
                
                self._rewards[i] = 0.0
                self._terminateds[i] = False
                self._truncateds[i] = False
            else:
                (
                    ob, 
                    self._rewards[i], 
                    self._terminateds[i], 
                    self._truncateds[i], 
                    info
                ) = self.envs[i].step(action)
            
            obs.append(ob)
            infos['infos'].append(info)
        
        self._should_reset = np.logical_or(self._terminateds, self._truncateds)
        return np.stack(obs), np.copy(self._rewards), np.copy(self._terminateds), np.copy(self._truncateds), infos

    def do_close(self, **kwargs):
        for env in self.envs:
            env.close()
    
    def render(self, mode='human'):
        for env in self.envs:
            env.render(mode)
