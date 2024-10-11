import numpy as np
import gymnasium as gym
from rlearn_dev.core.env_player import SyncVecEnvPlayer
from rlearn_dev.methods.ppo.naive import PPOAgent
from rlearn_dev.utils.eval_agent import eval_agent_performance


def test_ppo_naive():
    # 创建SyncVectorEnv
    num_envs = 5
    env = SyncVecEnvPlayer(lambda: gym.make('CartPole-v1'), num_envs=num_envs)
    config = {
        # 'learning_rate': 2.5e-4,
        # 'anneal_lr': True,
        # 'gamma': 0.99,
        # 'gae_lambda': 0.95,
        # 'num_minibatches': 4,
        # 'update_epochs': 4,
        # 'norm_adv': True,
        # 'clip_coef': 0.2,
        # 'clip_vloss': True,
        # 'ent_coef': 0.01,
        # 'vf_coef': 0.5,
        # 'max_grad_norm': 0.5,
        # 'target_kl': None,
    }
    
    max_epochs = 100
    steps_per_epoch = 5
    # 创建并训练 PPO 代理
    agent = PPOAgent(env, config=config)
    info = agent.learn(max_epochs, 
                       steps_per_epoch=steps_per_epoch,
                       reward_window_size=5,
                       verbose_freq=1)
    print(info)
    print('eval:')
    
    single_env = gym.make('CartPole-v1')
    performance_stats = eval_agent_performance(agent, single_env, num_episodes=10)
    print(f'{performance_stats=}')
    for key, value in performance_stats.items():
        print(f"{key}: {value}")

    env.close()
    
if __name__ == "__main__":
    if 1:
        test_ppo_naive()