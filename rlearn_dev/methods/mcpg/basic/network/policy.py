import torch.nn as nn

class PolicyNetwork(nn.Module):
    default_hidden_dim = 64
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(PolicyNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = self.default_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)