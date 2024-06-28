import numpy as np
import gymnasium as gym
import gym_trading_env
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import yfinance as yf
from collections import deque
import pandas as pd

from src.agent import DoubleQLearningAgent
from src.environment import make_env, StockTradingEnvironment
from src.utils import save_pickle, load_pickle, run_learned_policy, run_multiple_episodes, plot_training_progress
from src.data import fetch_stock_data, prepare_data, debug_dataframe
from src.ensemble import AgentEnsemble
from src.visualize import visualize_signals, plot_learning_curves
from src.metrics import calculate_metrics

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting the training script")

    # GPU/CPU configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = '/GPU:0'
        logger.info("Using GPU for training")
    else:
        device = '/CPU:0'
        logger.info("Using CPU for training")

    # Get Data
    logger.info("Fetching stock data for AAPL")
    stock_data = fetch_stock_data('AAPL', '2019-06-20', '2024-06-21', 'AAPL_data.csv')
    logger.info("Debugging raw stock data")
    debug_dataframe(stock_data, "Raw Stock Data", save_csv=True, output_file="raw_stock_data.csv")

    # Data Preprocessing
    logger.info("Preprocessing stock data")
    preprocessed_data, scaler, optimal_features, stock_data = prepare_data('AAPL', '2019-06-20', '2024-06-21', n_select=15, debug=True)
    logger.info(f"Selected {len(optimal_features)} optimal features")

    # Environment Setup
    logger.info("Setting up the training environment")
    num_envs = 3
    file_path = 'preprocessed_data.csv'
    envs = gym.vector.SyncVectorEnv([lambda: make_env(file_path, number_of_days_to_consider=20, n_select=15) for _ in range(num_envs)])

    logger.info(f"Observation space: {envs.single_observation_space}")
    logger.info(f"Action space: {envs.single_action_space}")

    # Agent Initialization
    logger.info("Initializing agent ensemble")
    num_agents = 5
    learning_rate = 0.001
    discount_factor = 0.95
    ensemble = AgentEnsemble(num_agents, 
                             observation_space=envs.single_observation_space, 
                             action_space=envs.single_action_space,
                             learning_rate=learning_rate, 
                             discount_factor=discount_factor,
                             epsilon=1.0, 
                             epsilon_min=0.01, 
                             epsilon_decay=0.995,
                             buffer_size=10000, 
                             batch_size=32)

    # Training
    logger.info("Starting training process")
    num_episodes = 10000
    replay_frequency = 32
    reward_across_episodes = []
    epsilons_across_episodes = []

    # Early stopping parameters
    patience = 100
    min_delta = 0.001
    window_size = 20

    # Initialize variables for early stopping
    best_reward = float('-inf')
    wait = 0
    reward_window = deque(maxlen=window_size)

    for episode in range(num_episodes):
        observations, _ = envs.reset()
        terminated = np.array([False] * num_envs)
        truncated = np.array([False] * num_envs)
        episode_rewards = np.zeros(num_envs)
        
        step_count = 0
        while not np.all(terminated) and not np.all(truncated):
            actions = [ensemble.act({"observation": obs}) for obs in observations]
            next_observations, rewards, terminated, truncated, _ = envs.step(actions)
            episode_rewards += rewards
            
            # Vectorized update
            states, actions, rewards, next_states, dones, td_errors = ensemble.update(
                tf.convert_to_tensor(observations, dtype=tf.float32),
                tf.convert_to_tensor(actions, dtype=tf.int32),
                tf.convert_to_tensor(rewards, dtype=tf.float32),
                tf.convert_to_tensor(next_observations, dtype=tf.float32),
                tf.convert_to_tensor(terminated | truncated, dtype=tf.bool)
            )
            
            # Update memories outside of the TensorFlow graph
            ensemble.update_memories(states, actions, rewards, next_states, dones, td_errors)
            
            observations = next_observations

            # Perform experience replay less frequently
            if step_count % replay_frequency == 0:
                ensemble.replay()

            step_count += 1

        mean_reward = np.mean(episode_rewards)
        reward_across_episodes.append(mean_reward)
        epsilons_across_episodes.append(ensemble.agents[0].epsilon)

        ensemble.decay_epsilon()

        # Early stopping check
        reward_window.append(mean_reward)
        if len(reward_window) == window_size:
            current_reward = np.mean(reward_window)
            if current_reward > best_reward + min_delta:
                best_reward = current_reward
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping triggered at episode {episode + 1}")
                    break

        if (episode + 1) % 100 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes} - Mean Reward: {mean_reward:.4f} - Epsilon: {ensemble.agents[0].epsilon:.4f}")

    logger.info(f"Training completed after {episode + 1} episodes")

    # Evaluation
    logger.info("Starting evaluation on multiple episodes")
    total_reward_learned_policy = run_multiple_episodes(envs, ensemble, num_episodes=30)
    logger.info(f"Average reward over 30 episodes: {total_reward_learned_policy:.4f}")

    # Visualization
    logger.info("Generating learning curves")
    plot_learning_curves(reward_across_episodes, epsilons_across_episodes)
    plot_training_progress(reward_across_episodes, epsilons_across_episodes, total_reward_learned_policy)

    # Single Environment Evaluation
    logger.info("Evaluating on single environment")
    stock_trading_environment = StockTradingEnvironment('./preprocessed_data.csv', number_of_days_to_consider=30)
    stock_trading_environment.train = False
    total_reward = run_learned_policy(stock_trading_environment, ensemble, verbose=True)
    logger.info(f"Total reward on test environment: {total_reward:.4f}")

    logger.info("Generating predictions for visualization")
    obs, _ = stock_trading_environment.reset()
    terminated, truncated = False, False
    predictions = []
    while not terminated and not truncated:
        action = ensemble.act({"observation": obs})
        obs, reward, terminated, truncated, info = stock_trading_environment.step(action)
        predictions.append(action)

    logger.info("Visualizing trading signals")
    visualize_signals(stock_data.iloc[-len(predictions):], np.array(predictions))

    # Performance Metrics
    logger.info("Calculating performance metrics")
    y_true = stock_data['signal'].iloc[-len(predictions):].map({'buy': 0, 'sell': 1, 'none': 2}).values
    accuracy, precision, recall, f1 = calculate_metrics(y_true, predictions)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Save Ensemble
    logger.info("Saving the trained ensemble")
    save_pickle(ensemble, 'aapl_ensemble_agent.pkl')
    logger.info("Training script completed successfully")

if __name__ == "__main__":
    main()