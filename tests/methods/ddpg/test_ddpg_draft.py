import gymnasium as gym
from rlearn_dev.methods.ddpg.draft.agent import DDPGAgent


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

"""
Pendulum-v1
    状态空间：Pendulum-v1 的状态空间是一个二维向量 
    [
    theta
    theta_dot
    ]
    动作空间：动作空间是一维，表示施加的扭矩，因此动作维度为 1。
"""
def test_ddpg_draft():
    capture_video = False # True
    run_name = 'test'
    num_envs = 5
    env_id = 'Hopper-v4'
    env = gym.vector.SyncVectorEnv([make_env(env_id, 36, i, capture_video, run_name)
                                     for i in range(num_envs)])
    print(env.num_envs)
    g_seed = 36
    config = {
        'gamma': 0.99,
        'batch_size': 64,
        'tau': 0.005,
        'actor_lr': 0.0003,
        'critic_lr': 0.0003,
        'buffer_size': 100_000,
        'exploration_noise': 0.1,
    }
    agent = DDPGAgent(env, config=config, seed=g_seed)
    learn_config = {
        'max_episodes': 1000_000//256,
        'max_episode_steps': 256,
        'max_total_steps': 1000_000,
        'verbose_freq': 1,
    }
    agent.learn(**learn_config)
    env.close() 
    
    
if __name__ == '__main__':
    if 1:
        test_ddpg_draft()