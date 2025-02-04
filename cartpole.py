import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Initialize environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 64
gamma = 0.95          # Discount rate
epsilon = 1.0         # Exploration rate
epsilon_decay = 0.995 # Decay rate for exploration
epsilon_min = 0.01    # Minimum exploration rate
learning_rate = 0.001
n_episodes = 1000

# Define the neural network model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            target = torch.tensor(reward).float().to(device)  # Ensure target is Float
            if not done:
                target += gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[action] = target
            loss = nn.functional.mse_loss(self.model(state)[action], target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
# Main training loop
agent = DQNAgent(state_size, action_size)
scores = []
for e in range(n_episodes):
    state, _ = env.reset()
    score = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            agent.update_target_model()
            print(f"Episode {e+1}/{n_episodes}, Score: {score}, Epsilon: {epsilon:.2}")
            break
    scores.append(score)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    agent.replay(batch_size)

# Plot the results
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Training Progress of DQN on CartPole")
plt.show()

# Display trained agent
state, _ = env.reset()
for _ in range(500):
    env.render()
    action = agent.act(state)
    state, _, done, _, _ = env.step(action)
    if done:
        break
env.close()
