from abc import ABC, abstractmethod
import numpy as np
import warnings
import torch
import gymnasium as gym
from typing import NamedTuple
from ...env import get_obs_shape, get_action_dim
from ...cuda import get_device

# extras用于存储任何额外的信息，例如动作的对数概率 | extras is used to store any additional information, e.g. the log probabilities of the actions
class ReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    extras: dict

# NOTE:
# 1. 没有考虑TimeLimit.truncated | Not considering TimeLimit.truncated
class BaseReplayBuffer(ABC):
    
    def __init__(self,
                 max_samples: int,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 num_envs: int, 
                 device: str = 'auto',
                 # handle_timeout_termination: bool = True,
                 state_dtype: np.dtype = np.float32,
                 action_dtype: np.dtype = np.float32,
                 reward_dtype: np.dtype = np.float32):
        self.state_space = state_space
        self.action_space = action_space
        self.max_samples = max_samples
        if max_samples % num_envs != 0:
            warnings.warn(f'max_samples={max_samples} is not a multiple of num_envs={num_envs}.')
        self.buffer_size = max_samples // num_envs
        # self.handle_timeout_termination = handle_timeout_termination
        self.device = get_device(device)
        self.num_envs = num_envs
        self.state_shape = get_obs_shape(self.state_space)
        self.action_dim = get_action_dim(self.action_space)
        self.states = np.zeros((self.buffer_size, self.num_envs, *self.state_shape), dtype=state_dtype)
        self.next_states = np.zeros((self.buffer_size, self.num_envs, *self.state_shape), dtype=state_dtype)
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=reward_dtype)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=reward_dtype)
        # self.timeouts = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.extras = {}
        # 为了防止不正确的继承，不应该在此处使用reset() | To prevent incorrect inheritance, reset() should not be used here
        self._pos = 0
        self._full = False
    
    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
    
    def reset(self):
        self._pos = 0
        self._full = False
    
    @abstractmethod
    def add(self, *args):
        pass
    
    @abstractmethod
    def sample(self, batch_size):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @property
    def full(self):
        return self._full

    def __len__(self):
        if self._full:
            return self.buffer_size
        return self._pos
    
    def size(self):
        return len(self)

