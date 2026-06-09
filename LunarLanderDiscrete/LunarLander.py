# Research here: https://gymnasium.farama.org/environments/box2d/lunar_lander/

import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from torch.distributions import Categorical

class LunarLanderBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        # Returns probabilities for the 4 actions
        return torch.softmax(self.layer3(x), dim=-1)

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

LunarLanderBrain = LunarLanderBrain()
llbrain_optimizer = optim.Adam(LunarLanderBrain.parameters(), lr=0.001)

for gameId in range(7000):
    state, info = env.reset()

    saved_log_probs = []
    rewards = []

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probabilities = LunarLanderBrain(state_tensor)
        
        weighted_coin = Categorical(probabilities)
        action = weighted_coin.sample()
        
        saved_log_probs.append(weighted_coin.log_prob(action))

        state, reward, terminated, truncated, info = env.step(action.item())
        # action is an tensor made by Categorical, so we use .item() to get the actual number

        rewards.append(reward)
        # here there is a need since reward is changing every time so no pf steps can do the trick


        if terminated or truncated:
            break

    losses = []
    # stores = -1 * log(probability of taking that action) * how good was the action (reward)

    R = 0
    gamma = 0.99
    total_reward = sum(rewards)
    # how much future rewards matter at this step

    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
        
    # Reward Normalization (Baseline)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    losses = []
    # stores = -1 * log(probability of taking that action) * how good was the action (reward)
    for log_prob, R in zip(saved_log_probs, returns):
        losses.append(-log_prob * R)
    
    total_loss = torch.stack(losses).sum()

    llbrain_optimizer.zero_grad()
    total_loss.backward()
    llbrain_optimizer.step()

    print(f"Game: {gameId}, Score: {total_reward}")

# Save the brain so you don't have to retrain it!
torch.save(LunarLanderBrain.state_dict(), "lunar_lander_brain.pth")
print("Training Complete. Model Saved!")

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
for demoId in range(100):
    
    state, info = env.reset()

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probabilities = LunarLanderBrain(state_tensor)
        # During testing, we don't want it to randomly guess anymore.
        # We want it to pick the absolute highest probability action!
        action = torch.argmax(probabilities).item()
        
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break