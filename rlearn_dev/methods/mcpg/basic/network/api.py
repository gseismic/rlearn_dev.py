import numpy as np
from .policy import PolicyNetwork
from .....utils.env import is_box_space, is_discrete_space #, get_action_dim

def get_policy_network(state_space, action_space, hidden_dim=None):
    assert is_box_space(state_space), 'MCPG only supports continuous state space'
    assert is_discrete_space(action_space), 'MCPG only supports discrete action space'
    # action_dim = get_action_dim(action_space)
    action_dim = action_space.n
    return PolicyNetwork(np.prod(state_space.shape), action_dim, hidden_dim=hidden_dim)

__all__ = ['PolicyNetwork', 'get_policy_network']