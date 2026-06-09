import torch
import torch.nn as nn

# Module 1: The Brain

class CartBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4,128)
        self.layer2 = nn.Linear(128,2)

    def forward(self,input):
        output = self.layer1(input)
        output = torch.relu(output)
        output = self.layer2(output)
        return torch.softmax(output,dim=1)

# Module 2: The Agent's Setup & Memory

import torch.optim as optim
import gymnasium as gym

CartBrain = CartBrain()
cart_optimizer = optim.Adam(CartBrain.parameters(), lr=0.001)

env = gym.make("CartPole-v1")

# Module 3: Playing the Game (The Forward Pass)

from torch.distributions import Categorical

for gameId in range(1000):
    
    state, info = env.reset()

    saved_log_probs = []
    rewards = []

    for stepId in range(500):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probabilities = CartBrain(state_tensor)
        
        weighted_coin = Categorical(probabilities)
        action = weighted_coin.sample()
        
        saved_log_probs.append(weighted_coin.log_prob(action))

        state, reward, terminated, truncated, info = env.step(action.item())
        # action is an tensor made by Categorical, so we use .item() to get the actual number

        rewards.append(reward)
        # here there is no need since reward is +1 every time so no pf steps can do the trick


        if terminated or truncated:
            break
    
    losses = []
    # stores = -1 * log(probability of taking that action) * how good was the action (reward)

    R = 0
    gamma = 0.99
    total_reward = 0
    # how much future rewards matter at this step

    for r in rewards[::-1]:
        R = r + gamma * R
        total_reward += r
        losses.append(-1 * saved_log_probs.pop() * R)
    
    total_loss = torch.stack(losses).sum()
    
    cart_optimizer.zero_grad()
    total_loss.backward()
    cart_optimizer.step()
    
    print(f"Game: {gameId}, Score: {total_reward}")
    

env = gym.make("CartPole-v1", render_mode="human")
for demoId in range(100):
    
    state, info = env.reset()

    for stepId in range(500):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probabilities = CartBrain(state_tensor)
        
        weighted_coin = Categorical(probabilities)
        action = weighted_coin.sample()
        
        state, reward, terminated, truncated, info = env.step(action.item())
        
        if terminated or truncated:
            break