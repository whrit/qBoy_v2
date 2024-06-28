import pickle
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_pickle(q_table, object_name):
    with open(object_name, "wb") as f:
        pickle.dump(q_table, f)
    logging.info('Pickle file saved as %s', object_name)

def load_pickle(object_name):
    with open(object_name, "rb") as f:
        deserialized_dict = pickle.load(f)
    logging.info('Pickle file %s loaded', object_name)
    return deserialized_dict

def run_learned_policy(env, agent):
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    logging.info("Initial state: %s", obs)
    
    total_reward = 0
    steps = 0
    
    while not terminated and not truncated:
        action = agent.act({"observation": obs})  # Assuming agent has an 'act' method
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        logging.info("Action: %s, Reward: %f, Terminated: %s, Truncated: %s", 
                     action, reward, terminated, truncated)
        time.sleep(0.1)  # Reduced sleep time
        logging.info('**************')
        steps += 1

    logging.info("Total Reward Collected Over the Episode: %f in Steps: %d", total_reward, steps)
    return total_reward

def run_learned_policy_suppressed_printing(env, agent):
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    total_reward = 0
    steps = 0
    
    while not terminated and not truncated:
        action = agent.act({"observation": obs})  # Assuming agent has an 'act' method
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
    return total_reward

def plot_grid(env, agent, reward_across_episodes: list, epsilons_across_episodes: list) -> None:
    """Plot main and extra plots in a 2x2 grid."""
    
    env.train = False
    total_reward_learned_policy = [run_learned_policy_suppressed_printing(env, agent) for _ in range(30)]
    
    plt.figure(figsize=(15, 10))

    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(reward_across_episodes, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward Value')
    plt.title('Rewards Per Episode (Training)')
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(total_reward_learned_policy, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward Value')
    plt.title('Rewards Per Episode (Learned Policy Evaluation)')
    plt.grid()

    # Extra plots
    plt.subplot(2, 2, 3)
    plt.plot(reward_across_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward Per Episode (Training)')
    plt.title('Cumulative Reward vs Episode')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(epsilons_across_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Values')
    plt.title('Epsilon Decay')
    plt.grid()

    plt.tight_layout()
    plt.show()