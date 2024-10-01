import torch.nn as nn
from .....utils.activation import get_activation_class

class C51MLP(nn.Module):
    default_hidden_layers = [128, 128]
    default_activation = 'relu'
    def __init__(self, state_dim, action_dim, num_atoms, 
                 hidden_layers=None, activation=None):
        super(C51MLP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        hidden_layers = hidden_layers or self.default_hidden_layers
        activation = get_activation_class(activation, default=self.default_activation)
        
        # build network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim * num_atoms))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, state_dim)
        distribution = self.network(x) # (batch_size, action_dim * num_atoms)
        distribution = distribution.view(-1, self.action_dim, self.num_atoms)
        distribution = nn.functional.softmax(distribution, dim=-1)
        return distribution