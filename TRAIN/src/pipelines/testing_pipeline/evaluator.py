import torch
import pandas as pd
from src.model_creation.environment import MultiDayTradingExecutionEnv
from src.model_creation.model import DQNModel
from src.utils.utils import get_initial_mid_spread
from src.utils.benchmark import Benchmark

class Evaluator:
    def __init__(self, test_data, config, model_path="artifacts/dqn_model.pth"):
        self.test_data = test_data
        self.config = config
        self.env = MultiDayTradingExecutionEnv(
            test_data,
            initial_mid_spread=get_initial_mid_spread(test_data),
            target_shares=config["total_inventory"]
        )
        self.model = DQNModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def evaluate_model(self):
        total_cost = 0
        all_trade_schedules = []
        with torch.no_grad():
            for episode in range(len(self.env.daily_data)):
                state = self.env.reset()
                state = torch.FloatTensor(state).unsqueeze(0)
                done = False
                episode_cost = 0
                while not done:
                    action = torch.argmax(self.model(state)).item()
                    next_state, reward, done, _ = self.env.step(action)
                    state = torch.FloatTensor(next_state).unsqueeze(0)
                    episode_cost += -reward
                total_cost += episode_cost
                all_trade_schedules.append(pd.DataFrame(self.env.trade_schedule))
        all_trade_schedules_df = pd.concat(all_trade_schedules, ignore_index=True)
        all_trade_schedules_df.to_csv("artifacts/all_trade_schedules.csv", index=False)

        twap_cost, vwap_cost = self.evaluate_benchmark(self.test_data, target_shares=1000)
        result = {'TWAP Benchmark Cost': [twap_cost], 'VWAP Benchmark Cost': [vwap_cost], "RL_strategy_cost": [total_cost]}
        result_df = pd.DataFrame(result)
        result_df.to_csv('artifacts/custom_dqn_result.csv')

    
    def evaluate_benchmark(self, data, target_shares):
        benchmark = Benchmark(data)
        twap_trades = benchmark.get_twap_trades(data, target_shares)
        vwap_trades = benchmark.get_vwap_trades(data, target_shares)
        
        # Simulate TWAP and VWAP costs
        twap_slippage, twap_impact = benchmark.simulate_strategy(twap_trades, data, preferred_timeframe=390)
        twap_total_cost = sum(twap_slippage) + sum(twap_impact)
        vwap_slippage, vwap_impact = benchmark.simulate_strategy(vwap_trades, data, preferred_timeframe=390)
        vwap_total_cost = sum(vwap_slippage) + sum(vwap_impact)

        return twap_total_cost, vwap_total_cost

    def inference(self, one_day_data):
        """
        Inference function for a single day's trading data.
        
        Parameters:
        one_day_data (pd.DataFrame): Data for a single trading day.

        Returns:
        dict: A dictionary containing timestamps and number of shares to sell at each step.
        """
        # Initialize environment for a single day
        env = MultiDayTradingExecutionEnv(
            one_day_data,
            initial_mid_spread=get_initial_mid_spread(one_day_data),
            target_shares=self.config["total_inventory"]
        )

        trade_schedule = []
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            done = False
            while not done:
                # Predict action using trained model
                action = torch.argmax(self.model(state)).item()
                
                # Perform the action in the environment
                next_state, _, done, _ = env.step(action)
                
                # Record the trade schedule in the desired format
                trade_info = {
                    "timestamp": env.current_day_data.iloc[env.current_step - 1]['timestamp'],
                    "shares_to_sell": action
                }
                trade_schedule.append(trade_info)
                
                # Update state
                state = torch.FloatTensor(next_state).unsqueeze(0)

        return trade_schedule
