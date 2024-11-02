import numpy as np
from .actor_critic import Actor, Critic

def get_actor(state_space, action_space):
    return Actor(np.prod(state_space.shape), np.prod(action_space.shape), action_space.high, action_space.low)

def get_critic(state_space, action_space):
    return Critic(np.prod(state_space.shape), np.prod(action_space.shape))
