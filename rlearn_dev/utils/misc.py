import torch
from typing import Iterable
from itertools import zip_longest


def zip_strict(*iterables: Iterable) -> Iterable:
    # from stable_baselines3.common.utils import zip_strict
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(params: Iterable[torch.Tensor], target_params: Iterable[torch.Tensor], tau: float) -> None:
    # from stable_baselines3.common.utils import polyak_update
    # polyak update: target_params = (1 - tau) * target_params + tau * params
    assert isinstance(tau, float) and 0 <= tau <= 1, "tau must be a float between 0 and 1"
    with torch.no_grad():
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
