import numpy as np
import gymnasium as gym
from ...env import is_discrete_space
from .base import BaseReplayBuffer, ReplayBufferSamples

class RandomReplayBuffer(BaseReplayBuffer):
        
    def add(self, states, actions, rewards, next_states, dones, infos=None):
        # order: sarsa-like
        actions = actions.reshape((self.num_envs, self.action_dim))
        if is_discrete_space(self.state_space):
            states = states.reshape((self.num_envs, *self.state_shape))
            next_states = next_states.reshape((self.num_envs, *self.state_shape))
        
        self.states[self._pos] = np.array(states)
        self.actions[self._pos] = np.array(actions)
        self.rewards[self._pos] = np.array(rewards)
        self.next_states[self._pos] = np.array(next_states)
        self.dones[self._pos] = np.array(dones)
        self._pos += 1  
        if self._pos == self.buffer_size:
            self._pos = 0
            self._full = True
            
    def sample(self, batch_size, copy: bool = True):
        # Note: batch_size 采样的实际样本数是 batch_size 而不是batch_size * num_envs | the actual number of samples is batch_size, not batch_size * num_envs
        buf_end = self.buffer_size if self._full else self._pos
        indices = np.random.randint(0, buf_end, size=batch_size)
        env_indices = np.random.randint(0, high=self.num_envs, size=(len(indices),))
        # states.shape: (batch_size, *state_shape)
        samples = ReplayBufferSamples(
            states=self.to_torch(self.states[indices, env_indices], copy=copy),
            actions=self.to_torch(self.actions[indices, env_indices], copy=copy),
            rewards=self.to_torch(self.rewards[indices, env_indices], copy=copy),
            next_states=self.to_torch(self.next_states[indices, env_indices], copy=copy),
            dones=self.to_torch(self.dones[indices, env_indices], copy=copy),
            extras=self.extras
        )
        return samples
    
    def state_dict(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'dones': self.dones,
            'extras': self.extras,
            '_pos': self._pos,
            '_full': self._full
        }
    
    def load_state_dict(self, state_dict):
        self.states = state_dict['states']
        self.actions = state_dict['actions']
        self.rewards = state_dict['rewards']
        self.next_states = state_dict['next_states']
        self.dones = state_dict['dones']
        self.extras = state_dict['extras']
        self._pos = state_dict['_pos']
        self._full = state_dict['_full']
