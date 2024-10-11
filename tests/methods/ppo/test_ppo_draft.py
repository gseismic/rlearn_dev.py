import gymnasium as gym
import time
from rlearn_dev.methods.ppo.draft import PPOAgent
from rlearn_dev.utils.eval_agent import eval_agent_performance
from rlearn_dev.utils.seed import seed_all

# make reproducible
g_seed = None
seed_all(g_seed) # do NOT forget PPOAgent(.., seed=g_seed)

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def test_ppo_naive_env2():
    # env setup
    num_envs = 5
    # env_id = 'CliffWalking-v0'
    env_id = 'CartPole-v1'
    capture_video = False
    run_name = "main"

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # loss: mainly vloss
    # 不能区分gamma和gae_lambda
    # tips: increate update_epochs
    # 当update_epochs很大的时候，clip_vloss可能必要
    # 如果gae_lambda*gamma很小，长步长在将降低到计算机精度以下
    gamma = 0.99
    gae_lambda = 0.99
    config = {
        'learning_rate': 0.00025,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'ent_coef': 0.01*0.5,
        'clip_vloss': False,
        'clip_coef': 0.2,
        'clip_coef_v': 20.0, # clip
        'update_epochs': 200, # 200
        'num_minibatches': 10, # minibatch_size: batch_size/num_minibatches
    }
    max_epochs = 5
    # 小步迭代: num_envs * steps_per_epoch
    # too small steps_per_epoch will make value not stable
    steps_per_epoch = 500

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
    test_ppo_naive_env2()