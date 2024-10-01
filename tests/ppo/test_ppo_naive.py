import gymnasium as gym
import numpy as np
import torch
from rlearn_dev.methods.ppo.naive.ppo_agent import PPOAgent

def test_ppo_cartpole():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    
    num_episodes = 1000
    max_steps = 500
    update_interval = 20
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        
        for step in range(max_steps):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        if (episode + 1) % update_interval == 0:
            agent.update(states, actions, log_probs, rewards, dones)
            states, actions, rewards, log_probs, dones = [], [], [], [], []
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
        
        if episode_reward >= 495:
            print(f"Solved in {episode + 1} episodes!")
            break
    
    env.close()

if __name__ == "__main__":
    test_ppo_cartpole()
