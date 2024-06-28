import gymnasium as gym
import gym_trading_env
import pandas as pd
import numpy as np
from src.features import select_optimal_indicators, apply_optimal_indicators

np.random.seed(69)

class StockTradingEnvironment:
    """This class wraps the gym-trading-env environment."""

    def __init__(self, df, **kwargs):
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

        # Adding optimal technical indicators
        self.df = self._add_optimal_indicators(self.df)
        print("Columns after adding optimal technical indicators:", self.df.columns)

        # Initialize the gym-trading-env environment
        self.env = gym.make(
            'TradingEnv',
            df=self.df,
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

        # Add custom metrics
        self.env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
        self.env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))

    def _add_optimal_indicators(self, df):
        try:
            print("Shape of DataFrame before adding indicators:", df.shape)
            optimal_features = select_optimal_indicators(df, n_select=self.n_select)
            df_with_indicators = apply_optimal_indicators(df, optimal_features)
            print("Shape of DataFrame after adding indicators:", df_with_indicators.shape)
            print(f"Selected {len(optimal_features)} optimal features: {optimal_features}")
            
            # Fill NaN values
            df_filled = df_with_indicators.fillna(0)
            
            # Check if any NaN values remain
            if df_filled.isna().any().any():
                print("Warning: NaN values still present after filling.")
                print(df_filled.isna().sum())
            
            return df_filled
        except Exception as e:
            print(f"Error adding optimal technical indicators: {e}")
            # Return the original DataFrame if an error occurs
            return df

    def custom_reward_function(self, history):
        # Logarithmic change in portfolio valuation
        log_return = np.log(history["portfolio_valuation"][-1] / history["portfolio_valuation"][-2])
        
        # Drawdown calculation
        max_val = np.max(history["portfolio_valuation"][:])
        drawdown = (max_val - history["portfolio_valuation"][-1]) / max_val

        # Add a component to reward consistent growth
        consistency_bonus = 0.1 if log_return > 0 else 0
        
        # Penalize excessive trading
        trading_penalty = 0.001 * (history["position"][-1] != history["position"][-2])

        # Reward is adjusted by subtracting the drawdown penalty and adding consistency bonus
        reward = log_return - drawdown + consistency_bonus - trading_penalty
        return reward

    def reset(self):
        observation, info = self.env.reset()
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)

        # Fetch the history from the environment
        history = self.env.unwrapped.history

        # Calculate custom reward
        reward = self.custom_reward_function(history)

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()

def make_env(file_path, **kwargs):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    print(f"DataFrame head before adding optimal TA features:\n{df.head()}")
    env = StockTradingEnvironment(df, **kwargs).env
    print(f"DataFrame head after adding optimal TA features:\n{env.unwrapped.df.head()}")
    return env