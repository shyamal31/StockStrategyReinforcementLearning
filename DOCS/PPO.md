---

# Trade Execution using Proximal Policy Optimization (PPO)

A reinforcement learning-based approach using **Proximal Policy Optimization (PPO)** for optimizing multi-day trade execution. The goal is to dynamically execute a target number of shares while minimizing transaction costs across a trading day, by learning an optimal trading policy.

## 1. Algorithm Architecture

### PPO Model
PPO is a popular reinforcement learning algorithm that uses a policy-gradient approach to learn optimal policies. This implementation utilizes the **Stable-Baselines3 PPO** library to train the agent, leveraging PPO’s capacity to balance exploration and exploitation while preventing large policy updates, which can destabilize learning. The PPO model is configured as follows:
- **Policy Network**: Uses an MLP (Multi-Layer Perceptron) policy, where the policy and value functions share neural network layers.
- **Entropy Coefficient** (`ent_coef=0.01`): Adds exploration by penalizing deterministic policies, encouraging diverse action choices.
- **Learning Rate** (`learning_rate=1e-3`): Controls the model’s update step size for faster convergence.
- **Gamma** (`gamma=0.95`): Discount factor, slightly lowered to prioritize immediate rewards and minimize short-term transaction costs.
- **Clip Range** (`clip_range=0.3`): Limits the extent of each policy update to prevent large, destabilizing changes, ensuring a stable learning process.

### Multi-Day Trading Execution Environment
The trading environment is built using OpenAI’s `gym` library and simulates multiple trading days. It includes features to support PPO’s continuous action space requirements:
- **State Representation**: The state vector contains relevant trading information, including:
  - Inventory status (remaining shares),
  - Market data (bid-ask prices and volumes at multiple levels),
  - Time left in the trading day,
  - Historical volatility, log returns, and OHLCV data.
- **Action Space**: Uses a continuous action space where actions range from `0` (no trade) to `target_shares`. The action represents the number of shares to sell at each step.
- **Reward Function**: Designed to penalize high transaction costs by applying slippage and market impact costs as negative rewards. A scaled reward structure is used to incentivize smaller, strategic trades that minimize overall costs.

### Execution Flow
1. **Data Splitting**: The historical data is divided into individual trading days, creating episodes for training and evaluation.
2. **Reset**: For each episode, the environment resets with the market data for a specific day and the target shares.
3. **Action Execution**: At each time step, the agent selects an action that dictates the fraction of remaining shares to execute. The model converts this continuous action into an integer number of shares to sell.
4. **Reward Calculation**: Transaction costs (slippage and market impact) are calculated based on the action taken and serve as the negative reward.
5. **Logging and Analysis**: The trade schedule is recorded for each day, with the cumulative reward calculated as the total transaction cost for the day.

## 2. Training and Hyperparameter Fine-Tuning

Key hyperparameters and their roles are as follows:
- **Entropy Coefficient** (`ent_coef=0.01`): Adjusts the level of exploration by encouraging action variability, helpful in the early stages of training.
- **Learning Rate** (`learning_rate=1e-3`): Increased to accelerate learning, balancing exploration with stable convergence.
- **Gamma** (`gamma=0.95`): Slightly reduced to prioritize immediate transaction costs, making the agent more responsive to immediate trade impacts.
- **Clip Range** (`clip_range=0.3`): A larger clip range allows more flexibility in policy updates, promoting adaptation to diverse market conditions.

## Execution Steps

### Training the PPO Model
To train the model, the `train_ppo` function initializes the environment with market data and trains the PPO agent over a specified number of timesteps.

```python
# Training the PPO model
train_ppo(data, target_shares=1000, total_timesteps=10000)
```

### Evaluating the PPO Model
After training, the `evaluate_ppo` function allows the agent to execute trades based on the learned policy and evaluates its performance. It outputs the trade schedule and total transaction costs for each trading day, saving the results to a CSV file.

```python
# Evaluation of the PPO model
evaluate_ppo(data, model_path="ppo_trading_model", target_shares=1000)
```

## Execution Workflow
1. **Load Data**: Load and preprocess historical market data for trade execution.
2. **Training**: Train the PPO agent to minimize transaction costs by strategically executing trades over multiple days.
3. **Evaluation**: Evaluate the agent’s performance on new trading data, logging the trade schedule and transaction costs for each day.
4. **Results**: The combined trade schedules and costs are saved as `ppo_trade_schedules.csv` for further analysis.

---
