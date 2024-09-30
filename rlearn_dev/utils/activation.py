import torch.nn as nn

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU
}

def get_activation_class(activation, default='relu'):
    if activation is None:
        if isinstance(default, str):
            return ACTIVATIONS[default]
        elif issubclass(default, nn.Module):
            return default
        else:
            raise ValueError(f"Not supported default activation: {default}")
    elif isinstance(activation, str):
        return ACTIVATIONS[activation]
    elif issubclass(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Not supported activation: {activation}")
    
