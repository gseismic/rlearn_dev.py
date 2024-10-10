from rlearn_dev.methods.ppo.naive import PPOAgent
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

# 创建多个环境实例
def make_env():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def test_vector_env():
    num_envs = 6
    env = SyncVectorEnv([make_env for _ in range(num_envs)])

    obs, info = env.reset()
    assert len(obs) == num_envs
    assert isinstance(info, dict)

    actions = env.action_space.sample()
    assert len(actions) == num_envs
    obs, rewards, terminated, truncated, infos = env.step(actions)
    assert len(obs) == num_envs
    assert len(rewards) == num_envs
    assert len(terminated) == num_envs
    assert len(truncated) == num_envs
    assert isinstance(info, dict)
    print("Observations:", obs)
    print("Rewards:", rewards)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Infos:", infos)
    env.close()
    

if __name__ == "__main__":
    if 1:
        test_vector_env()