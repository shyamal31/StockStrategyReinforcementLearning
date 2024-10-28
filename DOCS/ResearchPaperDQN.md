---

# Optimized Trade Execution with DQN

This project implements a DQN-based approach for optimized trade execution, inspired by the research paper "Reinforcement Learning for Optimized Trade Execution." The goal is to minimize transaction costs while executing a target number of shares across a trading day using reinforcement learning. We adapt DQN (Deep Q-Network) to handle the challenges of real-world market data, utilizing a replay buffer and target network for stability.

## 1. Algorithm Architecture

### Deep Q-Network (DQN) Model
The architecture consists of:
   - **DQN Neural Network**: A 3-layer fully connected neural network with ReLU activations. It takes in the state (`[time, inventory]`) and outputs Q-values for each action, representing the expected cost of executing a trade at a given step.
   - **Replay Buffer**: A memory buffer stores experiences (`state, action, reward, next_state, done`) to avoid correlation in training data. Each training step samples a batch of experiences for efficient learning.
   - **Target Network**: A duplicate of the DQN network, updated at intervals to stabilize Q-value estimation. It mitigates instability by decoupling target updates from frequent changes in the policy network.

### Execution Logic
1. **State Space**: The state consists of `[time, remaining inventory]`, tracking both the number of time steps (or minutes) left in the day and the remaining shares to be executed.
2. **Action Space**: Actions are defined by the time (minute) and number of shares to sell. Each action corresponds to selling a portion of the remaining inventory at the market price at a given time step.
3. **Reward**: Negative transaction costs are used as rewards, with costs calculated based on the number of shares executed and the market price (ask price at that minute).
4. **Optimization**: The agent learns a policy to minimize cumulative transaction costs over the day by strategically selecting actions through DQN optimization.

## 2. Fine-Tuning Hyperparameters

Hyperparameters are set as follows to balance exploration, stability, and learning efficiency:

- **Learning Rate** (`learning_rate = 0.001`): Controls the step size of updates to the policy network. A moderately low value ensures stable learning.
- **Discount Factor** (`gamma = 0.99`): Determines the weight given to future rewards, encouraging the agent to consider long-term transaction costs.
- **Epsilon and Decay** (`epsilon = 1.0`, `epsilon_decay = 0.995`, `epsilon_min = 0.01`): Epsilon-greedy strategy for exploration-exploitation trade-off. Initially, exploration is high, and it gradually decays to encourage exploitation as learning progresses.
- **Batch Size** (`batch_size = 64`): Number of experiences sampled from the replay buffer per training step, ensuring sufficient diversity for stable learning.
- **Target Update Frequency** (`target_update = 10`): Number of episodes between updates to the target network. Frequent updates reduce instability in Q-value estimates.
- **Replay Buffer Capacity** (`memory_capacity = 10000`): Determines the size of the replay buffer, storing the agent’s experiences for training.

## Execution Steps

1. **Data Loading and Grouping**: Loads market data, groups it by day, and defines an action space based on the number of available minutes and target inventory per day.
2. **Training DQN Agent**: For each day, the agent goes through multiple episodes to learn the optimal policy that minimizes transaction costs.
3. **Trade Schedule Extraction**: After training, an optimal trade schedule is generated for each day based on the learned Q-values.
4. **Transaction Cost Calculation**: Transaction costs are computed for each day based on the agent’s trading decisions. Results are visualized by plotting the total transaction cost over time.

---
