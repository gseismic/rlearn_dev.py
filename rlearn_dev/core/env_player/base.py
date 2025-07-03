from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseEnvPlayer(ABC):

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment.
        Returns:
            observation: The initial observation of the environment.
            info: The info of the environment.
        """
        pass

    @abstractmethod
    def step(self, action_or_actions: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.
        Args:
            action_or_actions: The action or actions to step the environment.
        Returns:
            observation: The observation of the environment.
            reward: The reward of the environment.
            terminated: Whether the environment is terminated.
            truncated: Whether the environment is truncated.
            info: The info of the environment.
        """
        pass

    @abstractmethod
    def close(self, **kwargs):
        """
        Close the environment.
        """
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass


class BaseVecEnvPlayer(ABC):
    
    def __init__(self, env_fns, **kwargs):
        self.envs = [env_fn() for env_fn in env_fns]
        self._single_action_space = self.envs[0].action_space
        self._single_observation_space = self.envs[0].observation_space
        self._num_envs = len(self.envs)
        self._is_closed = False
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def do_close(self, **kwargs):
        pass
    
    def close(self, **kwargs):
        if self._is_closed:
            return
        self.do_close(**kwargs)
        self._is_closed = True
    
    @property
    def is_closed(self):
        return self._is_closed

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def single_observation_space(self):
        return self._single_observation_space
    
    @property
    def action_space(self):
        return self.single_action_space

    @property
    def observation_space(self):
        return self.single_observation_space
    