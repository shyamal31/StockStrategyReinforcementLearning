import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import random
from collections import deque

class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, input_size, output_size):
        self.model = DQNModel(input_size, output_size)
        self.target_model = DQNModel(input_size, output_size)
        self.update_target_model()
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.model.fc3.out_features)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.cat([s[0] for s in minibatch])
        targets = self.model(states).detach()

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                with torch.no_grad():
                    next_q_values = self.target_model(next_state)
                target += self.gamma * torch.max(next_q_values[0]).item()
            targets[i][action] = target

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
