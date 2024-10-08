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
    # print("Observations:", obs)
    # print("Rewards:", rewards)
    # print("Terminated:", terminated)
    # print("Truncated:", truncated)
    # print("Infos:", infos)
    env.close()

def test_ppo_naive():
    # 创建SyncVectorEnv
    env = SyncVectorEnv([make_env for _ in range(4)])
    pass

    # 配置
    class Config:
        def __init__(self):
            self.num_envs = 4
            self.num_steps = 128
            self.total_timesteps = 500000
            self.learning_rate = 2.5e-4
            self.num_iterations = self.total_timesteps // (self.num_envs * self.num_steps)
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.update_epochs = 4
            self.norm_adv = True
            self.clip_coef = 0.2
            self.ent_coef = 0.01
            self.vf_coef = 0.5
            self.max_grad_norm = 0.5
            self.batch_size = int(self.num_envs * self.num_steps)
            self.minibatch_size = int(self.batch_size // 4)
            self.cuda = True

    # 创建环境
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # 创建配置和日志记录器
    config = Config()

    # 创建并训练 PPO 代理
    agent = PPOAgent(env, config)
    agent.learn()
    
    
if __name__ == "__main__":
    if 1:
        test_vector_env()
    if 0:   
        test_ppo_naive()