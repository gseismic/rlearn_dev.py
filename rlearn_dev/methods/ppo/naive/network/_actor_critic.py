import torch
import torch.nn as nn
import numpy as np
# from rlearn_dev.utils.nn.mlp import mlp

def mlp(sizes, activation, output_activation=nn.Identity, init_std=np.sqrt(2), bias_const=0.0):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPActorCritic(nn.Module):
    default_hidden_dims = [64, 64]
    default_activation = nn.Tanh
    default_actor_init_std = 0.01
    default_critic_init_std = 1.0
    def __init__(self, state_dim, action_dim, 
                 hidden_dims=None, activation=None,
                 actor_init_std=None, critic_init_std=None):
        super(MLPActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else self.default_hidden_dims
        self.activation = activation if activation is not None else self.default_activation
        actor_layers = []
        actor_sizes = [state_dim] + list(hidden_dims) + [action_dim]
        critic_sizes = [state_dim] + list(hidden_dims) + [1]
        for i in range(len(actor_sizes)-1):
            actor_layers.append(layer_init(nn.Linear(actor_sizes[i], actor_sizes[i+1])))
            actor_layers.append(self.activation())
        self.actor = nn.Sequential(*actor_layers)
        self.critic = mlp(critic_sizes, self.activation)
    
    def init_parameters(self):
        # 初始化 actor 的所有层
        for i, layer in enumerate(self.actor):
            if isinstance(layer, nn.Linear):
                if i == len(self.actor) - 2:  # 最后一层
                    layer_init(layer, std=self.default_actor_init_std)
                else:  # 其他层
                    layer_init(layer)
        
        # 初始化 critic 的所有层
        for i, layer in enumerate(self.critic):
            if isinstance(layer, nn.Linear):
                if i == len(self.critic) - 2:  # 最后一层
                    layer_init(layer, std=self.default_critic_init_std)
                else:  # 其他层
                    layer_init(layer)
    
    def forward(self, x):
        return self.actor(x), self.critic(x)
    
# TODO: MLP/CNN: discrete/continuous
class MLPActor(nn.Module):
    def __init__(self, env):
        super(MLPActor, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(env.observation_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),   
        )
    
    def forward(self, x):
        return self.actor(x)
    
class MLPCritic(nn.Module):
    def __init__(self, env):
        super(MLPCritic, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(env.observation_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  
        )
    
    def forward(self, x):
        return self.critic(x)