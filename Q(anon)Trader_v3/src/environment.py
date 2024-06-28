import gymnasium as gym
import gym_trading_env
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.data import preprocess_data, debug_dataframe

np.random.seed(69)

class StockTradingEnvironment:
    """This class wraps the gym-trading-env environment."""

    def __init__(self, df: pd.DataFrame, **kwargs: Any):
        self.df = df.copy()  # Ensure we're working with a copy
        self.train = kwargs.get('train', True)
        self.number_of_days_to_consider = kwargs.get('number_of_days_to_consider', 20)
        self.positions = kwargs.get('positions', [-1, 0, 1])
        self.windows = kwargs.get('windows', None)
        self.trading_fees = kwargs.get('trading_fees', 0)
        self.borrow_interest_rate = kwargs.get('borrow_interest_rate', 0)
        self.portfolio_initial_value = kwargs.get('portfolio_initial_value', 1000)
        self.initial_position = kwargs.get('initial_position', 'random')
        self.max_episode_duration = kwargs.get('max_episode_duration', 'max')
        self.verbose = kwargs.get('verbose', 1)
        self.n_select = kwargs.get('n_select', 15)

        # Preprocess data using the function from data.py
        X_scaled, self.scaler, self.optimal_features = preprocess_data(self.df, self.n_select)
        self.df_processed = pd.DataFrame(X_scaled, columns=self.optimal_features, index=self.df.index)

        # Add back the basic columns if they're not in optimal_features
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in self.optimal_features:
                self.df_processed[col] = self.df[col]

        print("Columns after preprocessing:", self.df_processed.columns)

        # Initialize the gym-trading-env environment
        self.env = gym.make(
            'TradingEnv',
            df=self.df_processed,
            positions=self.positions,
            windows=self.windows,
            trading_fees=self.trading_fees,
            borrow_interest_rate=self.borrow_interest_rate,
            portfolio_initial_value=self.portfolio_initial_value,
            initial_position=self.initial_position,
            max_episode_duration=self.max_episode_duration,
            verbose=self.verbose
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Add custom metrics
        self.env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
        self.env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))

        # Initialize tracking variables
        self.previous_portfolio_value = self.portfolio_initial_value
        self.max_portfolio_value = self.portfolio_initial_value
        self.holding_duration = 0

    def custom_reward_function(self, history: Dict[str, Any]) -> float:
        current_portfolio_value = history["portfolio_valuation"][-1]
        
        # Sharpe ratio component
        returns = np.diff(history["portfolio_valuation"]) / history["portfolio_valuation"][:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9)  # Add small epsilon to avoid division by zero
        
        # Drawdown component
        drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value
        
        # Position holding duration component
        if history["position"][-1] != 0:
            self.holding_duration += 1
        else:
            self.holding_duration = 0
        holding_penalty = -0.001 * self.holding_duration  # Penalize long holdings
        
        # Trading activity component
        trading_penalty = -0.001 * (history["position"][-1] != history["position"][-2])
        
        # Combine components
        reward = 10 * sharpe_ratio - 5 * drawdown + holding_penalty + trading_penalty
        
        # Update tracking variables
        self.previous_portfolio_value = current_portfolio_value
        self.max_portfolio_value = max(self.max_portfolio_value, current_portfolio_value)
        
        return reward

    def reset(self):
        observation, info = self.env.reset()
        self.previous_portfolio_value = self.portfolio_initial_value
        self.max_portfolio_value = self.portfolio_initial_value
        self.holding_duration = 0
        return observation, info

    def step(self, action: int):
        observation, _, terminated, truncated, info = self.env.step(action)

        # Fetch the history from the environment
        history = self.env.unwrapped.history

        # Calculate custom reward
        reward = self.custom_reward_function(history)

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = 'human'):
        self.env.render()

    def close(self):
        self.env.close()

def make_env(file_path: str, **kwargs: Any) -> gym.Env:
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print(f"DataFrame head after reading preprocessed data:\n{df.head()}")
    env = StockTradingEnvironment(df, **kwargs).env
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    return env