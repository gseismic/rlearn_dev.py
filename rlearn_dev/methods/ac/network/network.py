import torch.nn as nn
from ....utils.activation import get_activation_class

class IndependentActorCriticMLP(nn.Module):
    
    default_hidden_sizes = [128]
    default_activation = 'relu'
    def __init__(self, state_dim, action_dim, hidden_sizes=None, activation=None):
        super(IndependentActorCriticMLP, self).__init__()
        hidden_sizes = hidden_sizes or self.default_hidden_sizes
        activation = get_activation_class(activation, default=self.default_activation)
        # 构建 actor 网络
        actor_layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            actor_layers.append(nn.Linear(input_dim, hidden_size))
            actor_layers.append(activation())
            input_dim = hidden_size
        actor_layers.append(nn.Linear(input_dim, action_dim))
        actor_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor_layers)
        
        # 构建 critic 网络
        critic_layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            critic_layers.append(nn.Linear(input_dim, hidden_size))
            critic_layers.append(activation())
            input_dim = hidden_size
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value