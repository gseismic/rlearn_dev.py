import numpy as np
from typing import Optional
from .actor_critic import Actor, Critic
from .....utils.env import is_box_space

def get_actor(state_space, action_space, 
              log_std_min: Optional[float] = None,
              log_std_max: Optional[float] = None,
              hidden_dim: Optional[int] = None):
    assert is_box_space(state_space), 'SAC only supports continuous state space'
    assert is_box_space(action_space), 'SAC only supports continuous action space'
    return Actor(np.prod(state_space.shape), 
                 np.prod(action_space.shape), action_space.high, action_space.low,
                 log_std_min=log_std_min, log_std_max=log_std_max,
                 hidden_dim=hidden_dim)

def get_critic(state_space, action_space):
    return Critic(np.prod(state_space.shape), np.prod(action_space.shape))
