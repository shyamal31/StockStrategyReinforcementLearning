# import subprocess
# subprocess.check_call(["pip", "install", "gym"])

import gym
import numpy as np
from gym import spaces
from utils import get_initial_mid_spread
from benchmark import Benchmark

class MultiDayTradingExecutionEnv(gym.Env):
    def __init__(self, data, initial_mid_spread, target_shares):
        super().__init__()
        self.data = data
        self.daily_data = self.split_data_by_day()
        self.initial_mid_spread = initial_mid_spread
        self.target_shares = target_shares
        self.remaining_shares = target_shares
        self.current_step = 0
        self.benchmark = Benchmark(self.data)
        self.action_space = spaces.Discrete(101)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

    def split_data_by_day(self):
        days = []
        self.data['date'] = self.data['timestamp'].dt.date
        for date, group in self.data.groupby('date'):
            days.append(group.reset_index(drop=True))
        return days

    def reset(self):
        if not hasattr(self, 'episode_day_index'):
            self.episode_day_index = 0
        if self.episode_day_index >= len(self.daily_data):
            self.episode_day_index = 0
        self.current_day_data = self.daily_data[self.episode_day_index]
        self.current_step = 0
        self.remaining_shares = self.target_shares
        self.trade_schedule = []
        self.episode_day_index += 1
        self.initial_mid_spread = get_initial_mid_spread(self.current_day_data)
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.current_day_data):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        row = self.current_day_data.iloc[self.current_step]
        state = [
            self.remaining_shares, len(self.current_day_data) - self.current_step,
            row['bid_price_1'], row['bid_size_1'], row['bid_price_2'], row['bid_size_2'],
            row['bid_price_3'], row['bid_size_3'], row['bid_price_4'], row['bid_size_4'],
            row['bid_price_5'], row['bid_size_5'], row['ask_price_1'], row['ask_size_1'],
            row['ask_price_2'], row['ask_size_2'], row['ask_price_3'], row['ask_size_3'],
            row['ask_price_4'], row['ask_size_4'], row['ask_price_5'], row['ask_size_5'],
            row['open'], row['high'], row['low'], row['close'], row['volume'],
            row['log_return'], row['volatility']
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.current_step >= len(self.current_day_data):
            done = True
            reward = 0
            return self._get_state(), reward, done, {}

        row = self.current_day_data.iloc[self.current_step]
        execution_price = row['bid_price_1']
        shares_to_sell = min(action, self.remaining_shares)
        slippage, market_impact = self.benchmark.compute_components(
            alpha=4.439584e-06, shares=shares_to_sell, idx=self.current_step
        )
        transaction_cost = slippage + market_impact
        reward = -transaction_cost

        self.remaining_shares -= shares_to_sell
        self.current_step += 1
        done = (self.remaining_shares <= 0) or (self.current_step >= len(self.current_day_data))
        
        trade = {
            "date": row['timestamp'].date(),
            "timestamp": row['timestamp'],
            "share_size": shares_to_sell,
            "execution_price": execution_price,
            "remaining_shares": self.remaining_shares,
            "transaction_cost": transaction_cost,
            "action": action
        }
        self.trade_schedule.append(trade)

        return self._get_state(), reward, done, {}
