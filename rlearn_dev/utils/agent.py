
def generate_episode(env, agent, deterministic, max_episode_steps=None):
    """生成一个episode"""
    state, _ = env.reset()
    episode_states, episode_actions, episode_rewards = [], [], []
    user_truncated = False
    ep_step = 0
    while not (terminated or truncated):
        action, _ = agent.predict(state, deterministic=deterministic)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_states.append(copy.deepcopy(state))
        episode_actions.append(copy.deepcopy(action))
        episode_rewards.append(reward)
        state = next_state
        ep_step += 1
        if max_episode_steps and ep_step >= max_episode_steps:
            user_truncated = True
            break

        
    rv = {
        "observations": episode_states,
        "actions": episode_actions,
        "rewards": episode_rewards
    }
    return rv

def generate_n_episodes(env, agent, num_episodes, de)