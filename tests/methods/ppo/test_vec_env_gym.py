import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

def make_env():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def test_vector_env():
    num_envs = 6
    env = SyncVectorEnv([make_env for _ in range(num_envs)])

    obs, infos = env.reset()
    assert len(obs) == num_envs
    assert isinstance(infos, dict)

    actions = env.action_space.sample()
    assert len(actions) == num_envs
    obs, rewards, terminated, truncated, infos = env.step(actions)
    assert len(obs) == num_envs
    assert len(rewards) == num_envs
    assert len(terminated) == num_envs
    assert len(truncated) == num_envs
    assert isinstance(infos, dict)
    print("Observations:", obs)
    print("Rewards:", rewards)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Infos:", infos)
    env.close()
    

if __name__ == "__main__":
    if 1:
        test_vector_env()