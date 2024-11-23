import gymnasium as gym
from rlearn_dev.methods.sac.naive import SACAgent as Agent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

env_id = 'Ant-v5'
# env = gym.make(env_id, render_mode="rgb_array", xml_file='./ant_v5.xml')
env = gym.make(env_id, render_mode="rgb_array", width=500, height=500)
env = RecordVideo(env, video_folder="videos/ant_random", name_prefix="eval",
                  episode_trigger=lambda x: True)


agent = Agent.load(path='./final_models/ant_sac2.pth', env=env)

num_eval_episodes = 30
# env = RecordVideo(env, video_folder="videos/ant_sac_eval", name_prefix="eval",
#                   episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = agent.predict(obs, deterministic=True)  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')
