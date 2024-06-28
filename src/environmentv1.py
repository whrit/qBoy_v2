import math
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from ta import add_all_ta_features

np.random.seed(69)

class StockTradingEnvironment(gymnasium.Env):
    """This class implements the Stock Trading environment."""

    def __init__(self, file_path, train=True, number_of_days_to_consider=30):
        self.file_path = file_path
        self.stock_data = pd.read_csv(self.file_path)
        self.train = train

        # Adding technical indicators
        self.stock_data = self._add_technical_indicators(self.stock_data)

        # Splitting the data into train and test datasets.
        self.training_stock_data = self.stock_data.iloc[:int(0.8 * len(self.stock_data))]
        self.testing_stock_data = self.stock_data.iloc[int(0.8 * len(self.stock_data)):].reset_index(drop=True)

        self.number_of_days_to_consider = number_of_days_to_consider
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.number_of_days_to_consider, len(self.training_stock_data.columns)-1), dtype=np.float32)  # Excluding 'Date'
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold

        self.reset()

    def _add_technical_indicators(self, df):
        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
        df.fillna(0, inplace=True)
        return df

    def reset(self):
        self.investment_capital = 100000
        self.number_of_shares = 0
        self.book_value = 0
        self.total_account_value = self.investment_capital
        self.total_account_value_list = []
        self.timestep = 0

        if self.train:
            self.data = self.training_stock_data
        else:
            self.data = self.testing_stock_data

        self.max_timesteps = len(self.data) - self.number_of_days_to_consider
        return self._get_observation(), {}

    def _get_observation(self):
        end = self.timestep + self.number_of_days_to_consider
        obs = self.data.iloc[self.timestep:end].copy()
        obs = obs.drop(columns=['date'])  # Assuming Date is one of the columns
        obs_normalized = obs / obs.max()
        return obs_normalized.values

    def step(self, action):
        penalty = 0
        current_price = self.data['open'].iloc[self.timestep + self.number_of_days_to_consider]

        if action == 0:  # Buy
            if self.number_of_shares > 0:
                penalty = -10

            shares_to_buy = math.floor(self.investment_capital / current_price)
            cost = shares_to_buy * current_price
            if shares_to_buy > 0 and cost <= self.investment_capital:
                self.number_of_shares += shares_to_buy
                self.book_value += shares_to_buy * current_price
                self.investment_capital -= cost
                reward = penalty
            else:
                reward = -10

        elif action == 1:  # Sell
            if self.number_of_shares > 0:
                revenue = self.number_of_shares * current_price
                reward = (revenue - self.book_value) / self.book_value * 100 if self.book_value > 0 else -10
                self.investment_capital += revenue
                self.number_of_shares = 0
                self.book_value = 0
            else:
                reward = -10

        elif action == 2:  # Hold
            if self.book_value > 0:
                reward = ((self.number_of_shares * current_price) - self.book_value) / self.book_value * 100
            else:
                reward = -1

        self.stock_value = self.number_of_shares * current_price
        self.total_account_value = self.investment_capital + self.stock_value
        self.total_account_value_list.append(self.total_account_value)

        self.timestep += 1
        terminated = self.timestep >= self.max_timesteps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        plt.figure(figsize=(15, 10))
        plt.plot(self.total_account_value_list, color='lightseagreen', linewidth=7)
        plt.xlabel('Days', fontsize=32)
        plt.ylabel('Total Account Value', fontsize=32)
        plt.title('Total Account Value over Time', fontsize=38)
        plt.grid()
        plt.show()

    def _calculate_price_trend(self):
        """Calculate the trend of the stock price over the number of days to consider."""
        price_increase_list = []
        for i in range(self.number_of_days_to_consider):
            if self.data['close'].iloc[self.timestep + 1 + i] - self.data['close'].iloc[self.timestep + i] > 0:
                price_increase_list.append(1)
            else:
                price_increase_list.append(0)
        price_increase = np.sum(price_increase_list) / self.number_of_days_to_consider >= 0.5
        return price_increase

    def _get_discrete_observation(self, price_increase, stock_held):
        """Convert the price trend and stock holding status to a discrete observation."""
        observation = [price_increase, stock_held]
        if np.array_equal(observation, [True, False]):
            return 0
        if np.array_equal(observation, [True, True]):
            return 1
        if np.array_equal(observation, [False, False]):
            return 2
        if np.array_equal(observation, [False, True]):
            return 3