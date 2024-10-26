from rlearn_dev.core.env_player import SyncVecEnvPlayer
import gymnasium as gym
import numpy as np
import pytest


def test_sync_vec_env_player():
    num_envs = 4
    env = SyncVecEnvPlayer(lambda: gym.make('CartPole-v1'), num_envs=num_envs)
    print(f'{env.action_space=}')
    print(f'{env.observation_space=}')
    print(f'{env.num_envs=}')
    
    obs, infos = env.reset()
    assert obs.shape == (num_envs, 4)
    assert isinstance(infos, list)
    
    actions = np.array([env.action_space.sample() for _ in range(num_envs)])
    next_obs, rewards, dones, truncated, infos = env.step(actions)
    
    print(f'{next_obs=}')
    
    assert next_obs.shape == (num_envs, 4)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(infos, list)
    
    assert env.action_space.shape == ()
    assert env.action_space.n == 2 
    assert env.observation_space.shape == (4,)
    
    env.close()


if __name__ == '__main__':
    if 1:
        test_sync_vec_env_player()
