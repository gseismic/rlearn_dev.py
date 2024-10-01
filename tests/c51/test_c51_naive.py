import gymnasium as gym
import matplotlib.pyplot as plt
from rlearn_dev.methods.c51.naive import C51Agent
from rlearn_dev.utils.eval_agent import eval_agent_performance
from rlearn_dev.utils.i18n import Translator
from pathlib import Path

def test_c51_naive():
    env = gym.make("CartPole-v1")
    
    # 使用混合类型配置
    config = {
        'model_type': 'C51MLP',
        'model_kwargs': {
            'hidden_layers': [64, 64],
            'activation': 'relu'
        },
        'num_atoms': 51,
        'optimizer': 'adam',
        'optimizer_kwargs': {'lr': 0.0003},
        'gamma': 0.99, # 明确指定gamma值
        'v_min': -10,
        'v_max': 10,
        'buffer_size': 100,
        'batch_size': 32,
        'target_update_freq': 10,
    }
    
    agent = C51Agent(env, config=config)

    # 创建学习参数字典
    learn_params = {
        'max_episodes': 500,
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

    # 绘制训练过程中的奖励
    plt.figure(figsize=(10, 5))
    plt.plot(training_info['rewards_history'])
    plt.title('Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_rewards.png')

    # 使用新的测试函数
    tr = Translator(agent.lang)
    performance_stats = eval_agent_performance(agent, env, num_episodes=10)
    for key, value in performance_stats.items():
        print(f"{tr(key)}: {value}")

    env.close()

    # 测试加载的模型
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