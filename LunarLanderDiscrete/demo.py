import torch
import torch.nn as nn
import gymnasium as gym

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

def main():
    # Initialize environment with render_mode="human"
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
                   
    # Initialize and load the brain
    brain = LunarLanderBrain()
    brain.load_state_dict(torch.load("lunar_lander_brain.pth", weights_only=True))
    brain.eval() # Set to evaluation mode
    
    print("Brain loaded! Starting demo for 10 iterations...")
    
    # Run for 10 iterations
    for demoId in range(10):
        state, info = env.reset()
        total_reward = 0
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            with torch.no_grad():
                probabilities = brain(state_tensor)
                
            # Pick the absolute highest probability action
            action = torch.argmax(probabilities).item()
            
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Iteration {demoId + 1} finished with total reward: {total_reward:.2f}")
                break
                
    env.close()
    print("Demo completed!")

if __name__ == "__main__":
    main()
