import torch.optim as optim

OPTIMIZERS = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}

def get_optimizer_class(optimizer, default='adam'):
    if optimizer is None:
        if isinstance(default, str):
            return OPTIMIZERS[default]
        elif issubclass(default, optim.Optimizer):
            return default
        else:
            raise ValueError(f"Not supported default optimizer: {default}")
    elif isinstance(optimizer, str):    
        return OPTIMIZERS[optimizer]
    elif issubclass(optimizer, optim.Optimizer):
        return optimizer
    else:
        raise ValueError(f"Not supported optimizer: {optimizer}")