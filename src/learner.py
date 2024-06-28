import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from src.agent import QLearningAgent
import random

def q_learning_learning_loop(env, learning_rate: float, discount_factor: float, episodes: int,
                             min_epsilon_allowed: float, initial_epsilon_value: float,
                             buffer_size: int = 10000, batch_size: int = 32) -> tuple:
    """Learning loop to train Agent to reach GOAL state in the environment using Q-Learning Algorithm.

    Args:
        env (gymnasium.Env): Object of Grid Environment.
        learning_rate (float): Learning rate used in Q-Learning algorithm.
        discount_factor (float): Discount factor to quantify the importance of future reward.
        episodes (int): Number of episodes to train.
        min_epsilon_allowed (float): Minimum epsilon value allowed during training.
        initial_epsilon_value (float): Initial epsilon value at the start of training.
        buffer_size (int): Size of the experience replay buffer.
        batch_size (int): Size of the batch for experience replay.

    Returns:
        tuple[QLearningGreedyAgent, list, list]: Returns a tuple containing agent,
                                                 cumulative rewards across episodes,
                                                 epsilon used across episodes respectively.
    """

    agent = QLearningAgent(env, learning_rate=learning_rate, discount_factor=discount_factor)
    print("Initial Q-Table: {0}".format(agent.q_table))

    epsilon = initial_epsilon_value
    epsilon_decay_factor = np.power(min_epsilon_allowed / epsilon, 1 / episodes)

    reward_across_episodes = []
    epsilons_across_episodes = []

    replay_buffer = deque(maxlen=buffer_size)

    for episode in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False

        current_state = obs
        reward_per_episode = 0
        epsilons_across_episodes.append(epsilon)

        while not terminated and not truncated:
            current_action = agent.step(current_state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(current_action)

            reward_per_episode += reward
            replay_buffer.append((current_state, current_action, reward, next_obs, terminated))

            if len(replay_buffer) >= batch_size:
                batch = np.array(random.sample(replay_buffer, batch_size), dtype=object)
                for b_state, b_action, b_reward, b_next_state, b_done in batch:
                    if b_done:
                        target = b_reward
                    else:
                        target = b_reward + discount_factor * np.max(agent.q_table[b_next_state])

                    agent.q_table[b_state, b_action] += learning_rate * (target - agent.q_table[b_state, b_action])

            current_state = next_obs

        epsilon = max(min_epsilon_allowed, epsilon * epsilon_decay_factor)
        reward_across_episodes.append(reward_per_episode)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} - Reward: {reward_per_episode} - Epsilon: {epsilon}")

    print("Trained Q-Table: {0}".format(agent.q_table))

    # Plotting rewards and epsilon decay
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(reward_across_episodes, label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Rewards over Episodes')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilons_across_episodes, label='Epsilon')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return agent, reward_across_episodes, epsilons_across_episodes
