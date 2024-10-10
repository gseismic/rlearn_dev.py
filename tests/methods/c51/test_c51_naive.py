import gymnasium as gym
import matplotlib.pyplot as plt
from rlearn_dev.methods.c51.naive import C51Agent
from rlearn_dev.utils.eval_agent import eval_agent_performance
from rlearn_dev.utils.i18n import Translator
from pathlib import Path
import numpy as np

def test_c51_naive():
    env = gym.make("CartPole-v1")
    
    # 策略可能失败
    # Key params:
    # buffer_size, target_hard_update_freq
    # 可能原因:
    #   (1) 过早放弃探索没有优先级，可能产生恶性循环
    #   (2) 过小的replay, buffer_size大小
    #   (3) 过快的target_hard_update_freq
    # half_life 
    #
    max_episodes = 300
    # 使用混合类型配置
    # gamma = 0.9 # bad, becasue 0.9**500 = 1.322070819480823e-23
    # gamma = 0.99 # 
    # 确保env_episode_max_steps内，单步reward不衰减到机器精度
    machine_eps = 1e-3
    env_episode_max_steps = 500
    gamma = np.exp(np.log(machine_eps)/env_episode_max_steps)
    print(f"**gamma**: {gamma}")
    v_max = 1/(1-gamma)*2
    v_min = -v_max
    replay_buffer_size = 10000
    batch_size = 32 # replay_buffer_size // 100
    # episode freq
    target_hard_update = True
    target_hard_update_freq = 10 #  larger is more stable
    target_soft_update_tau = 0.2
    
    epsilon_start = 0.5
    epsilon_end = 0.01
    
    if 1:
        half_life = max_episodes//6 # decay (0.5)^(3)
        epsilon_scheduler_type = 'half_life'
        epsilon_scheduler_kwargs = {
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'half_life': half_life,
        }
    if 0:
        half_life = max_episodes//6 # decay (0.5)^(3)
        cycle_steps = half_life//2
        epsilon_scheduler_type = 'half_life_cycle'
        epsilon_scheduler_kwargs = {
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'half_life': half_life,
            'cycle_epsilon': 0.05,
            'cycle_steps': cycle_steps,
        }
    
    config = {
        'model_type': 'C51MLP',
        'model_kwargs': {
            'hidden_layers': [64, 64],
            'activation': 'relu'
        },
        'num_atoms': 51,
        'optimizer': 'adam',
        'optimizer_kwargs': {'lr': 0.001},
        'gamma': gamma, # 明确指定gamma值
        'v_min': v_min,
        'v_max': v_max,
        'buffer_size': replay_buffer_size,
        'batch_size': batch_size,
        'target_hard_update': target_hard_update,
        'target_soft_update_tau': target_soft_update_tau,
        'target_hard_update_freq': target_hard_update_freq,
        'epsilon_scheduler_type': epsilon_scheduler_type,
        'epsilon_scheduler_kwargs': epsilon_scheduler_kwargs
    }
    
    agent = C51Agent(env, config=config)

    learn_params = {
        'max_episodes': max_episodes,
        'max_total_steps': 2000,
        'max_episode_steps': None,
        'max_runtime': None,
        'reward_threshold': None,
        'reward_window_size': 100,
        'min_reward_threshold': None,
        'max_reward_threshold': None,
        'reward_check_freq': 10,
        'verbose_freq': 10,
        'no_improvement_threshold': 50,
        'improvement_threshold': None,
        'improvement_ratio_threshold': None,
        'checkpoint_freq': 100,
        'checkpoint_path': Path('checkpoints'),
        'final_model_path': Path('final_models') / 'final_model.pth',
    }

    training_info = agent.learn(**learn_params)

    print(f"Final model saved at: {training_info['final_model_path']}")

    plt.figure(figsize=(10, 5))
    plt.plot(training_info['rewards_history'])
    plt.title('Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_rewards.png')

    tr = Translator(agent.lang)
    performance_stats = eval_agent_performance(agent, env, num_episodes=10)
    for key, value in performance_stats.items():
        print(f"{tr(key)}: {value}")

    env.close()

    load_model(env, training_info['final_model_path'])

def load_model(env, model_path, num_episodes=10):
    agent = C51Agent.load(model_path, env)
    
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if truncated:
                break
        total_rewards.append(episode_reward)
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    test_c51_naive()
    plt.show()
