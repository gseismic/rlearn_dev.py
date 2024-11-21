import gymnasium as gym
from rlearn_dev.methods.tabular.basic import TabularNStepSarsaAgent as Agent

def test_tabular_nstep_sarsa_basic():
    seed = 36

    # env_id = 'FrozenLake-v1'
    env_id = 'CliffWalking-v0'
    # env = gym.make(env_id, is_slippery=False)
    env = gym.make(env_id)

    print(f"Observation Space: {env.observation_space}")  # Discrete(16)
    print(f"Action Space: {env.action_space}") 

    config = {
        'learning_rate': 0.005,
        'gamma': 0.90,
        'epsilon_start': 0.5,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.99,
    }
    agent = Agent(env, config, seed=seed)
    learn_config = {
        'max_episodes': 2000_000,
        'max_episode_steps': 2000,
        'max_total_steps': 10000_000,
        'verbose_freq': 100,
    }
    info = agent.learn(**learn_config)
    env.close()
    # print(info)
    
if __name__ == '__main__':
    test_tabular_nstep_sarsa_basic()
