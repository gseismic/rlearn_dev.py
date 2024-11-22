import gymnasium as gym
import time
import numpy as np
from rlearn_dev.methods.ppo.naive import PPOAgent
from rlearn_dev.utils.eval_agent import eval_agent_performance
from rlearn_dev.utils.seed import seed_all


# make reproducible
g_seed = None
seed_all(g_seed) # do NOT forget PPOAgent(.., seed=g_seed)


# same with cleanrl
def make_env(env_id, idx, capture_video, run_name, gamma):
    def _make_env():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, 
                                         f"videos/{run_name}",
                                         episode_trigger=lambda episode: episode % 10 == 0)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.ClipAction(env)  # 确保动作在合法范围内
        env = gym.wrappers.NormalizeObservation(env)  # 标准化观察空间
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)  # 标准化奖励
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return _make_env

def test_ppo_draft_continous():
    # env setup
    num_envs = 16
    env_id = 'Ant-v5'
    capture_video = False
    run_name = "ant_ppo"
    gamma = 0.99

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name, gamma) for i in range(num_envs)],
    )

    gae_lambda = 0.95
    config = {
        'learning_rate': 3e-4,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'rpo_alpha': 0.5,
        'ent_coef': 0.01,
        'clip_coef': 0.2,
        'vf_coef': 0.5,
        'clip_vloss': True,
        'clip_coef_v': 0.2,
        'update_epochs': 10,
        'num_minibatches': 32,
    }
    max_epochs = 500
    steps_per_epoch = 4096
    agent = PPOAgent(envs, config=config, seed=g_seed)
    info = agent.learn(max_epochs, 
                       steps_per_epoch=steps_per_epoch,
                       reward_window_size=5,
                       verbose_freq=1)
    print(info)
    
    max_steps = 1000
    single_env = gym.make(env_id)
    performance_stats = eval_agent_performance(agent, single_env, 
                                               num_episodes=10, 
                                               max_steps=max_steps,
                                               deterministic=True)
    for key, value in performance_stats.items():
        print(f"{key}: {value}")
        
    single_env.close()
    envs.close()
    
if __name__ == '__main__':
    test_ppo_draft_continous()