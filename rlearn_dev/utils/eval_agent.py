import numpy as np
import time
from .i18n import Translator


def eval_agent_performance(agent, 
                           env, 
                           num_episodes=10, 
                           max_steps=1000, 
                           deterministic=True):
    """
    测试agent的性能 | Test agent performance
    
    Args:
    - deterministic: 是否使用确定性策略 | Whether to use deterministic policy
    
    Returns:
    - 包含性能统计信息的字典 | Dictionary containing performance statistics
    """
    total_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=deterministic)
            # print('action', action, type(action))
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    # 计算统计信息
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    max_reward = np.max(total_rewards)
    min_reward = np.min(total_rewards)
    median_reward = np.median(total_rewards)
    avg_episode_length = np.mean(episode_lengths)
    
    # print(avg_reward)
    # 计算成功率（假设奖励大于某个阈值为成功）
    # 'CliffWalking-v0' no reward_threshold
    # success_threshold = env.spec.reward_threshold if hasattr(env.spec, 'reward_threshold') else avg_reward
    # print(f'{total_rewards[0], success_threshold=}')
    # success_rate = sum(r >= success_threshold for r in total_rewards) / num_episodes
    
    info = {
        'rewards': total_rewards,
        'average_reward': avg_reward,
        'median_reward': median_reward,
        'reward_std': std_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'average_episode_length': avg_episode_length,
        # 'success_rate': success_rate,
        'test_episodes': num_episodes,
        'test_duration': test_duration
    }
    return info
