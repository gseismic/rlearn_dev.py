import torch.nn as nn
from .....utils.activation import get_activation_class

class ActorCriticMLP(nn.Module):
    
    default_hidden_sizes = [64, 64]
    default_activation = 'tanh'

    def __init__(self, state_dim, action_dim, hidden_sizes=None, activation=None):
        super(ActorCriticMLP, self).__init__()
        hidden_sizes = hidden_sizes or self.default_hidden_sizes
        activation = get_activation_class(activation, default=self.default_activation)
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            activation(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation()
        )
        
        # Actor 头部
        self.actor = nn.Sequential(
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 头部
        self.critic = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
