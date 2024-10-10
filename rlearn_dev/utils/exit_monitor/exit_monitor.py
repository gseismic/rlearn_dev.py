import time
from collections import deque
import numpy as np

class ExitMonitor:
    def __init__(self, config):
        self.max_episodes = config.get('max_episodes', float('inf'))
        self.max_total_steps = config.get('max_total_steps', float('inf'))
        self.max_runtime = config.get('max_runtime', float('inf'))  # 将默认值设为 float('inf')
        self.reward_threshold = config.get('reward_threshold')
        self.reward_window_size = config.get('reward_window_size', 100)
        self.min_reward_threshold = config.get('min_reward_threshold')
        self.max_reward_threshold = config.get('max_reward_threshold')
        self.reward_check_freq = config.get('reward_check_freq', 10)
        self.no_improvement_threshold = config.get('no_improvement_threshold', 50)
        self.improvement_threshold = config.get('improvement_threshold')
        self.improvement_ratio_threshold = config.get('improvement_ratio_threshold')
        
        self.start_time = time.time()
        self.recent_rewards = deque(maxlen=self.reward_window_size)
        self.total_steps = 0
        self.episode_count = 0
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0

    def should_exit(self, episode_reward):
        self.episode_count += 1
        self.recent_rewards.append(episode_reward)
        self.total_steps += 1

        if self.max_episodes is not None and self.episode_count >= self.max_episodes:
            return True, "maximum_episodes_reached"

        if self.max_total_steps is not None and self.total_steps >= self.max_total_steps:
            return True, "maximum_total_steps_reached"

        if self.max_runtime is not None and time.time() - self.start_time > self.max_runtime:
            return True, "maximum_runtime_reached"

        if len(self.recent_rewards) >= self.reward_window_size and self.episode_count % self.reward_check_freq == 0:
            avg_reward = np.mean(self.recent_rewards)
            min_reward = np.min(self.recent_rewards)
            
            if self.reward_threshold is not None and avg_reward >= self.reward_threshold:
                if self.min_reward_threshold is None or min_reward >= self.min_reward_threshold:
                    return True, "reward_threshold_reached"
            
            if self.max_reward_threshold is not None and avg_reward > self.max_reward_threshold:
                return True, "exceeded_maximum_reward_threshold"
            
            if self.improvement_threshold is not None:
                if avg_reward > self.best_avg_reward + self.improvement_threshold:
                    self.best_avg_reward = avg_reward
                    self.episodes_without_improvement = 0
                else:
                    self.episodes_without_improvement += self.reward_check_freq
                    if self.episodes_without_improvement >= self.no_improvement_threshold:
                        return True, "no_performance_improvement_absolute"
            
            if self.improvement_ratio_threshold is not None:
                if self.best_avg_reward > 0 and avg_reward / self.best_avg_reward > 1 + self.improvement_ratio_threshold:
                    self.best_avg_reward = avg_reward
                    self.episodes_without_improvement = 0
                else:
                    self.episodes_without_improvement += self.reward_check_freq
                    if self.episodes_without_improvement >= self.no_improvement_threshold:
                        return True, "no_performance_improvement_ratio"

        return False, ""