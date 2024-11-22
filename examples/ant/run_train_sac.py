import gymnasium as gym
from rlearn_dev.methods.sac.naive import SACAgent as Agent


def make_env(env_id, seed, idx, capture_video, run_name):
    def _make_env():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                name_prefix=f"sac_{env_id}",
                episode_trigger=lambda episode_idx: episode_idx % 100 == 0,
                video_length=0, # 0 means infinite
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return _make_env

def main():
    capture_video = True
    run_name = 'sac_ant_autotune'
    num_envs = 128
    env_id = 'Ant-v5'
    env = gym.vector.SyncVectorEnv([make_env(env_id, 36, i, capture_video, run_name)
                                     for i in range(num_envs)])
    g_seed = 36
    config = {
        'gamma': 0.99,
        'batch_size': 64,
        'tau': 0.005,
        'actor_lr': 0.0003,
        'critic_lr': 0.0003,
        'buffer_size': 100_000,
        'exploration_noise': 0.1,
        'autotune': True, # True
    }
    agent = Agent(env, config=config, seed=g_seed)
    learn_config = {
        'max_episodes': 2, # 100_000//2048,
        'max_episode_steps': 2048,
        'max_total_steps': 1000_000,
        'verbose_freq': 1,
        'final_model_name': 'ant_sac.pth',
        'final_model_dir': 'final_models'
    }
    agent.learn(**learn_config)
    env.close() 
    
if __name__ == '__main__':
    main()
