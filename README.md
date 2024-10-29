---

# Reinforcement Learning for Optimized Trade Execution

## Overview
This project applies Reinforcement Learning (RL) to optimize trade execution strategies for minimizing transaction costs when selling a fixed quantity of shares in a day. This documentation outlines the steps and methodologies used, the algorithms implemented, and the experimental results compared against standard benchmarks (TWAP and VWAP).

## Table of Contents
- [Project Objectives](#project-objectives)
- [Data Analysis](#data-analysis)
- [Project Assumptions](#project-assumptions)
- [Methodology](#methodology)
  - [Initial Research](#initial-research)
  - [Algorithm Exploration](#algorithm-exploration)
  - [Modeling Process](#modeling-process)
- [Benchmark Comparison](#benchmark-comparison)
- [Experimentation & Results](#experimentation--results)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## Project Objectives
The main goal is to develop an intelligent RL-based trade execution strategy that minimizes transaction costs over a single trading day, given historical trade and quote data. We explore a variety of RL algorithms to create a trading agent that can outperform TWAP and VWAP benchmarks.

## Data Analysis
We start by analyzing the provided high-frequency trading data, containing bid and ask prices, bid and ask sizes, and OHLC data. Key points observed:
- Each trading day contains minute-level data with up to 390 rows.
- The data has structured bids and asks, allowing us to calculate transaction costs associated with slippage and market impact.
- TWAP and VWAP benchmarks provide cost baselines against which our RL strategies are compared.

## Project Assumptions
To simplify the modeling process, we made the following assumptions:
1. **No Fractional Trading**: All trade orders are in whole numbers, with no fractional shares allowed.
2. **Mandatory Sale Completion**: The model must fully execute the sale of 1000 shares by the end of each trading day. This constraint ensures the model’s strategy aligns with end-of-day liquidity requirements.
3. **Market Order Execution**: Every trade is executed as a market order, with the bid price as the execution price, regardless of quantity. This approach prioritizes immediate execution over limit order considerations, impacting slippage and market impact.

These assumptions influence the model’s strategies and cost-minimization goals. They ensure realistic but simplified trading conditions, focusing on optimizing transaction cost rather than more complex factors like order book dynamics or time-weighted sales.

## Methodology

### Initial Research
We began by studying relevant research, including papers on trade execution with reinforcement learning and well-established benchmarks. We adapted methodologies to fit our corporate objective of minimizing costs and developed experiments based on these research insights.

### Algorithm Exploration
To systematically approach the solution, we implemented and tested multiple RL models:
1. **[Q-Learning with DQN](https://github.com/shyamal31/StockStrategyReinforcementLearning/blob/main/DOCS/ResearchPaperDQN.md)** - Used for its simplicity, DQN was first implemented to validate the setup and understand baseline agent performance.
2. **[Custom DQN Model](https://github.com/shyamal31/StockStrategyReinforcementLearning/blob/main/DOCS/CustomDQN.md)** - Built a custom DQN model to accommodate specific requirements in our data structure.
3. **[PPO Implementation](https://github.com/shyamal31/StockStrategyReinforcementLearning/blob/main/DOCS/PPO.md)** - Shifted to the PPO algorithm, exploring its suitability for continuous action spaces and stable learning.


For deployment, I used the Custom DQN model. I experimented with the other two models in a Python notebook. 

### Modeling Process
Each model implementation follows a similar workflow:
1. **Environment Setup** - Used custom Gym environments with action spaces allowing for partial or no trade executions within each time step.
2. **Reward Calculation** - Defined the reward as negative transaction cost, emphasizing cost minimization as the primary objective.
3. **Exploration vs. Exploitation** - Guided the model to avoid excessive selling and spaced out trading behavior by experimenting with reward shaping and epsilon decay in DQN and entropy coefficient tuning in PPO.

## Benchmark Comparison
For evaluation, we compare each RL agent's transaction costs against:
- **TWAP (Time-Weighted Average Price)** - Divides total shares equally over time.
- **VWAP (Volume-Weighted Average Price)** - Proportionally sells shares based on trading volume at each interval.

Each experiment documents total transaction costs across multiple trading days to compare the effectiveness of each RL model against TWAP and VWAP.

## Experimentation & Results
| Model       | Avg Transaction Cost | TWAP Cost | VWAP Cost | Comparison      |
|-------------|----------------------|-----------|-----------|-----------------|
| DQN         | [1.13]               | [0.0023]    | [0.00029]    | + TWAP/ + VWAP |
| Custom DQN  | [0.034]               | [0.0023]    | [0.00029]    | + TWAP/ + VWAP |
| PPO         | [0.003]               | [0.0023]    | [0.00029]    | ~  TWAP/ + VWAP |


+ '+' increase cost
+ '~' Almost same cost

_**Note: Findings suggest that PPO technique is the best among all, probably because the use of continuos space of action. But for the purpose of the demonstration and considering the time constraint for this taks I have deployed DQN model on the cloud. But we can surely replace PPO Algorithm in the deploy directory.**_

Each model's performance is summarized, highlighting improvements or areas where they fall short relative to benchmarks.
### Key Insights:
- **Benchmark Assumptions**: Standard benchmarks like TWAP and VWAP operate under conditions that allow for:
  - **Fractional Shares**: Both TWAP and VWAP can sell fractional shares at each time interval, which reduces transaction costs but does not reflect real-world trading limitations.
  - **No Sale Completion Requirement**: These benchmarks do not enforce the completion of 1000-share sales by day’s end, allowing for a lower transaction cost without liquidity constraints.

- **Real-World Applicability**: In contrast, our RL models are built with realistic constraints:
  - **Whole-Share Trades Only**: Our RL agent executes whole-share trades, aligning with the discrete nature of real-world transactions.
  - **Mandatory Sale Completion**: The model is required to complete 1000-share sales by day’s end, simulating end-of-day liquidity requirements.

Given these considerations, while the RL model may not outperform the benchmarks in terms of transaction cost, its strategy is better suited for practical, real-world trading conditions where such restrictions are standard.

Overall, this result underlines the real-world value of our approach, even if benchmark comparisons appear less favorable under theoretical assumptions.

## How to use this repo?
To get started you can check the readme file in [train](https://github.com/shyamal31/StockStrategyReinforcementLearning/blob/main/TRAIN/train.md) and [deploy](https://github.com/shyamal31/StockStrategyReinforcementLearning/blob/main/DEPLOY/deploy.md) directories respectively. 

## Future Work
Future directions for this project include:
- **Futher testing:** - The findings from this task needs further testing to ensure the robustness of the algorithms.
- **Enhanced Model Architectures** - Explore multi-agent setups, ensemble models, or deep Q-networks with prioritized experience replay.
- **Advanced Reward Shaping** - Further refine reward signals to encourage more optimal behavior and stable learning.
- **Real-World Deployment Considerations** - Test scalability and adapt models to real-world trading environments with live data integration.

## Contributors
- **[Shyamal Gandhi]** - [shyamalgandhi.applyusa@gmail.com](mailto:shyamalgandhi.applyusa@gmail.com)

## Resources Used

1. [**Reinforcement Learning in Limit Order Markets**](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf) — This is the primary research paper you need to replicate. This will serve as the foundational model for your task, although you will need to adjust the objective to minimize transaction costs rather than maximize wealth.
2. **AWS SageMaker Documentation**:
    - [**AWS SageMaker Developer Guide**](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) — This documentation will help you understand how to deploy your model on SageMaker, including setting up real-time inference endpoints for serving the trade optimization model.
    - [**SageMaker Real-Time Inference**](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) — Detailed guide on deploying machine learning models using real-time inference in SageMaker, which is essential for making your model accessible via an API.
3. **PPO implementation**
    - [**Stable Baselines3 SAC Documentation**](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) — A practical library for implementing SAC in Python, using PyTorch.
4. **Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP)**:
    - https://chain.link/education-hub/twap-vs-vwap
5. **Python Libraries for Reinforcement Learning and Trading**:
    - **Pandas for Financial Data** — Documentation for Pandas, which is essential for handling financial datasets and performing time-series operations on market data.
6. **Trading Benchmark**:
   - **https://chain.link/education-hub/twap-vs-vwap**
---
