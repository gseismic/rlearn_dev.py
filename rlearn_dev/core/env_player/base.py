from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseEnvPlayer(ABC):

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, action_or_actions: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def close(self):
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
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def num_envs(self) -> int:
        pass

    @property
    @abstractmethod
    def single_action_space(self):
        pass

    @property
    @abstractmethod
    def single_observation_space(self):
        pass
    
    @property
    def action_space(self):
        return self.single_action_space

    @property
    def observation_space(self):
        return self.single_observation_space
    