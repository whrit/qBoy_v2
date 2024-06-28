import numpy as np
import matplotlib.pyplot as plt
import logging
from src.visualize import plot_learning_curves

def q_learning_learning_loop(env, agent, learning_rate: float, discount_factor: float, episodes: int,
                             min_epsilon_allowed: float, initial_epsilon_value: float,
                             batch_size: int = 32, decay_method="exponential", replay_frequency: int = 4) -> tuple:
    epsilon = initial_epsilon_value
    epsilon_decay_factor = np.power(min_epsilon_allowed / epsilon, 1 / episodes)

    reward_across_episodes = []
    epsilons_across_episodes = []

    for episode in range(episodes):
        observations, _ = env.reset()
        terminated = np.array([False] * env.num_envs)
        truncated = np.array([False] * env.num_envs)
        episode_rewards = np.zeros(env.num_envs)
        
        step_count = 0
        while not np.all(terminated) and not np.all(truncated):
            actions = [agent.step({"observation": observation}) for observation in observations]
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            episode_rewards += rewards
            
            for obs, action, reward, next_obs, term, trun in zip(observations, actions, rewards, next_observations, terminated, truncated):
                agent.update_qvalue({"observation": obs}, action, reward, {"observation": next_obs}, term or trun)
                
            observations = next_observations

            # Perform experience replay
            if step_count % replay_frequency == 0:
                agent.replay()

            step_count += 1

        if decay_method == "exponential":
            epsilon = max(min_epsilon_allowed, epsilon * epsilon_decay_factor)
        else:
            epsilon = max(min_epsilon_allowed, epsilon - (initial_epsilon_value - min_epsilon_allowed) / episodes)

        reward_across_episodes.append(np.mean(episode_rewards))
        epsilons_across_episodes.append(epsilon)

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            logging.info(f"Episode {episode + 1}/{episodes} - Mean Reward: {np.mean(episode_rewards)} - Epsilon: {epsilon}")

    plot_learning_curves(reward_across_episodes, epsilons_across_episodes)

    return agent, reward_across_episodes, epsilons_across_episodes