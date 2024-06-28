import matplotlib.pyplot as plt
import numpy as np

def visualize_signals(data, predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(data['datetime'], data['close'], label='Close Price', linewidth=1)
    
    buy_signals = data[predictions == 0]
    sell_signals = data[predictions == 1]
    
    plt.scatter(buy_signals['datetime'], buy_signals['close'], label='Buy Signal', marker='^', color='green', alpha=1, s=100)
    plt.scatter(sell_signals['datetime'], sell_signals['close'], label='Sell Signal', marker='v', color='red', alpha=1, s=100)
    
    plt.title('Stock Price with Predicted Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(reward_across_episodes, epsilons_across_episodes):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(reward_across_episodes, label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
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