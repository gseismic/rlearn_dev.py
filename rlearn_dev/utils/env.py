from typing import Union, Tuple, Dict
import numpy as np
import gymnasium.spaces as spaces

def is_box_space(space) -> bool:
    return isinstance(space, spaces.Box)

def is_multi_binary_space(space) -> bool:
    return isinstance(space, spaces.MultiBinary)

def is_multi_discrete_space(space) -> bool:
    return isinstance(space, spaces.MultiDiscrete)

def is_discrete_space(space) -> bool:
    return isinstance(space, spaces.Discrete)

def is_dict_space(space) -> bool:
    return isinstance(space, spaces.Dict)

# from stable_baselines3.common.preprocessing
def get_obs_shape(
    obs_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    if is_box_space(obs_space):
        return obs_space.shape
    elif is_discrete_space(obs_space):
        return (1,)
    elif is_multi_discrete_space(obs_space):
        return (int(len(obs_space.nvec)),)
    elif is_multi_binary_space(obs_space):
        return obs_space.shape
    elif is_dict_space(obs_space):
        return {key: get_obs_shape(subspace) for (key, subspace) in obs_space.spaces.items()}
    else:
        raise NotImplementedError(f"{obs_space} observation space is not supported")

# from stable_baselines3.common.preprocessing
def get_action_dim(action_space: spaces.Space) -> int:
    if is_box_space(action_space):
        return int(np.prod(action_space.shape)) # different from get_obs_shape
    elif is_discrete_space(action_space):
        return 1
    elif is_multi_discrete_space(action_space):
        return int(len(action_space.nvec))
    elif is_multi_binary_space(action_space):
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
