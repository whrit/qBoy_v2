import pickle
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_pickle(obj, object_name):
    with open(object_name, "wb") as f:
        pickle.dump(obj, f)
    logging.info('Pickle file saved as %s', object_name)

def load_pickle(object_name):
    with open(object_name, "rb") as f:
        deserialized_obj = pickle.load(f)
    logging.info('Pickle file %s loaded', object_name)
    return deserialized_obj

def run_learned_policy(env, agent, verbose=False):
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    if verbose:
        logging.info("Initial state: %s", obs)
    
    total_reward = 0
    steps = 0
    
    while not terminated and not truncated:
        action = agent.act({"observation": obs})
        
        if verbose:
            logging.info("Agent opts to take the following action: %s", action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if verbose:
            logging.info("New Observation: %s; Immediate Reward: %f, Termination Status: %s, Info: %s", 
                         obs, reward, terminated, info)
            time.sleep(1)
            logging.info('**************')
        
        steps += 1

    if verbose:
        logging.info("Total Reward Collected Over the Episode: %f in Steps: %d", total_reward, steps)

    return total_reward

def run_multiple_episodes(env, agent, num_episodes=30):
    total_rewards = []
    for _ in range(num_episodes):
        total_reward = run_learned_policy(env, agent)
        total_rewards.append(total_reward)
    return total_rewards

def plot_training_progress(reward_across_episodes, epsilons_across_episodes, total_reward_learned_policy):
    plt.figure(figsize=(15, 10))

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