import torch
from src.model_creation.environment import MultiDayTradingExecutionEnv
from src.model_creation.model import Agent
from src.utils.utils import get_initial_mid_spread

class TradingExecutionTrainer:
    def __init__(self, train_data, config):
        self.train_data = train_data
        self.config = config
        self.env = MultiDayTradingExecutionEnv(
            train_data,
            initial_mid_spread=get_initial_mid_spread(train_data),
            target_shares=config["total_inventory"]
        )
        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n
        self.agent = Agent(input_size, output_size)

    def train_model(self, epochs=1):
        for e in range(epochs):
            for episode in range(len(self.env.daily_data)):
                state = self.env.reset()
                state = torch.FloatTensor(state).unsqueeze(0)
                done = False
                while not done:
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    self.agent.replay(batch_size=32)

        torch.save(self.agent.model.state_dict(), "/Users/shyamalgandhi/Desktop/Shyamal/blockhouse final submission/Train/artifacts/dqn_model.pth") #change file name to the destination where your destination
