import torch
import gym
import numpy as np
from ppo_continuous_fixed_rk4_env import Agent

def test_agent(env_name, model_path, num_episodes=10):
    env = gym.make(env_name)
    agent = Agent(env.observation_space.shape[0], env.action_space)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent(state)
            action = action.cpu().numpy()[0]
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    test_agent(env_name="CartPole-v1", model_path="path_to_saved_model")