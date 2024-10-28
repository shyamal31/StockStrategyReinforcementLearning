---

# Custom DQN for Multi-Day Trading Execution

A custom DQN (Deep Q-Network) approach for optimizing trade execution across multiple days. It adapts a reinforcement learning environment to minimize transaction costs by dynamically selecting the number of shares to execute at each step within a day. The agent learns an optimal trading strategy that balances the trade-off between execution costs and remaining shares in a multi-day trading environment.

## 1. Algorithm Architecture

### DQN Model
The DQN model is a three-layer fully connected neural network with the following specifications:
- **Input Layer**: Accepts a feature vector representing the state, which includes market data, bid-ask spreads, and inventory status.
- **Hidden Layers**: Two hidden layers with 128 units each, using ReLU activation functions for non-linearity.
- **Output Layer**: Outputs Q-values for each possible action, representing the expected cost or reward for executing trades of various sizes.

### DQN Agent
The agent is responsible for decision-making, memory management, and learning. It has several key components:
- **Memory Replay Buffer**: A deque buffer stores the agent’s past experiences (`state, action, reward, next_state, done`), enabling efficient learning by sampling diverse experiences.
- **Epsilon-Greedy Policy**: Controls exploration versus exploitation, gradually shifting from random exploration to relying on learned Q-values as epsilon decays.
- **Target Network**: A clone of the main model that periodically updates to stabilize Q-value estimates, allowing for more consistent learning.

### Multi-Day Trading Execution Environment
The custom environment is built using OpenAI’s `gym` library and simulates a multi-day trading scenario where:
- **State**: The state vector includes inventory status, time left in the trading day, and relevant market data (e.g., bid-ask prices, historical prices, and volatility).
- **Action**: The action space includes 101 discrete actions, representing different quantities of shares to execute at each step, including a "do nothing" action.
- **Reward**: The reward structure is designed to minimize transaction costs. If the agent executes a trade, it incurs a cost based on slippage and market impact, which serves as a negative reward to incentivize efficient trading. When the agent takes no action, it receives a minor penalty to discourage inactivity.

### Execution Flow
1. **Data Splitting by Day**: Historical data is grouped by day, creating episodes for multi-day training.
2. **Reset**: Each episode (trading day) initializes the state with the day’s market data and target shares to execute.
3. **Action Execution and Step Calculation**: At each step, the agent chooses an action that determines the number of shares to sell. The execution price, transaction cost, and remaining shares are calculated.
4. **Reward Calculation**: Transaction costs, which consist of slippage and market impact, are used as negative rewards. A small penalty is applied for idle actions.
5. **Replay and Optimization**: The agent samples experiences from memory, updates Q-values using Bellman’s equation, and optimizes the network using MSE loss.

## 2. Fine-Tuning Hyperparameters

Key hyperparameters and their roles are as follows:

- **Learning Rate** (`lr=1e-3`): Controls the update step size, balancing quick learning and stable convergence.
- **Discount Factor** (`gamma=0.99`): Determines the agent’s foresight by weighting future rewards, encouraging the agent to prioritize long-term cost minimization.
- **Epsilon Decay** (`epsilon_decay=0.999`, `epsilon_min=0.01`): Gradually shifts from exploration to exploitation, allowing the agent to start with exploratory actions and rely on learned Q-values as training progresses.
- **Replay Buffer Capacity** (`maxlen=10000`): Sets the size of the memory buffer, ensuring enough experiences for learning while limiting memory usage.
- **Batch Size** (`batch_size=64`): Specifies the number of experiences sampled during each learning step, balancing learning stability and computational efficiency.

## Execution Steps

1. **Environment Setup**: The environment (`MultiDayTradingExecutionEnv`) processes historical trading data and enables simulation of daily trading actions.
2. **Training Loop**: The agent trains over multiple episodes (days), learning to execute trades optimally across different market conditions.
3. **Evaluation and Analysis**: The model's performance is analyzed based on transaction costs incurred, enabling an evaluation of the agent's strategy over time.

