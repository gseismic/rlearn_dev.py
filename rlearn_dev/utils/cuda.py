import torch

def get_device(device):
    device = device or 'auto'
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)