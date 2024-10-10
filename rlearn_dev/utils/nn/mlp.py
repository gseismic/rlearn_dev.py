import torch.nn as nn

# from:
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
