import gymnasium as gym
from rlearn_dev.methods.mcpg.basic.agent import MCPGAgent

def test_mcpg_basic():
    seed = 36
    env = gym.make('CartPole-v1')
    config = {
        'learning_rate': 0.01,
        'gamma': 0.99,
        'normalize_return': True,
    }
    agent = MCPGAgent(env, config, seed=seed)
    learn_config = {
        'max_episodes': 1000,
        'max_episode_steps': 2000,
        'max_total_steps': 100_000,
        'verbose_freq': 10,
    }
    info = agent.learn(**learn_config)
    env.close()
    # print(info)
    
if __name__ == '__main__':
    test_mcpg_basic()