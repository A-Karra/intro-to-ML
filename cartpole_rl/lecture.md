# Reinforcement Learning: Solving CartPole

We are going to learn by analogy! In every module below, I will explain a concept and show you exactly how to write the code for a **Self-Driving Car** AI playing a game called `CarRacing-v0`. 

Your job is to read the Self-Driving Car code, understand how it works, and then write the equivalent code for your **CartPole** environment in `train.py`.

---

## Module 1: The Brain (Policy Network)

Before an agent can learn, it needs a neural network brain. In Policy Gradients, the brain looks at the current game state and outputs the percentage chance of taking each possible action.

### Example: Self-Driving Car
Imagine an AI driving a car. 
- **Input:** 5 sensors (Distance to left wall, distance to right wall, speed, distance to car ahead, current steering angle). 
- **Hidden Layer:** We use 64 neurons to do the math. *(How do we pick 64? It's a guessing game! We usually pick a power of 2 like 32, 64, or 128. If the number is too small, the brain isn't smart enough. If it's too big, it runs slowly and over-memorizes. 64 is a safe middle ground for a simple problem).*
- **Output:** 3 actions (Turn Left, Turn Right, Hit Gas).
- **Softmax:** We use `torch.softmax(x, dim=1)` to convert the raw numbers into clean percentages. Softmax has a special trick: it uses exponents to "exaggerate the winner". 
  - *Example:* If the raw numbers are `5`, `18`, and `17`, normal percentages would give them `12.5%`, `45%`, and `42.5%`. 
  - But Softmax turns them into `0.0001%`, `73%`, and `27%`. 
  - It heavily rewards the biggest number (`18`), keeps the close runner-up alive (`17`), and crushes bad options to dust (`5`). This allows the AI to be confident while still occasionally exploring good alternatives!

Here is the exact code for the Car's brain:

```python
import torch
import torch.nn as nn

class CarBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 inputs, 64 hidden neurons
        self.layer1 = nn.Linear(5, 64)
        # 64 hidden neurons, 3 outputs
        self.layer2 = nn.Linear(64, 3)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.softmax(x, dim=1)
```

### Assignment 1
Open `train.py`. Write a class called `PolicyNet` that does the same thing, but for **CartPole**. 
*(Hint: CartPole has 4 inputs and 2 possible actions. You can use 128 hidden neurons instead of 64).*

---

## Module 2: The Agent's Setup & Memory

### What is an Optimizer?
An optimizer is the engine that actually updates the weights of your neural network. When we calculate the "Loss" (how wrong the network was), the optimizer looks at that Loss and calculates exactly which direction to turn every single knob in the brain to do better next time. 

### Why do we need it?
Without an optimizer, a neural network is just a static math formula. It can play the game, but it can never learn. The optimizer is the actual "learning" mechanism.

### Different Types of Optimizers
There are several ways to turn the knobs:
1. **SGD (Stochastic Gradient Descent):** The oldest and simplest. It looks at the Loss and takes a rigid, fixed-size step down the hill. It works, but it can be very slow and easily get stuck in dead ends.
2. **RMSprop:** A smarter version of SGD. If a knob is constantly being turned in the same direction, RMSprop speeds it up. If a knob is bouncing back and forth, it slows it down.
3. **Adam (Adaptive Moment Estimation):** The modern gold standard. It combines the best parts of SGD and RMSprop. It tracks "momentum" (like a snowball rolling down a hill, gaining speed) and adapts the learning rate for every single individual knob perfectly. 

### Why Adam?
We choose **Adam** because it is incredibly fast, stable, and almost always works perfectly out-of-the-box for Reinforcement Learning. It automatically figures out the best way to turn the knobs without us having to do any complex tuning.

### The Agent's Memory
Additionally, because we only learn *after* the game is completely over, our agent needs a memory! During the game, it needs to save exactly which actions it chose (specifically, the "Log Probability" of the action, which is required for the math formula) and exactly what rewards it received at every step.

### Example: Self-Driving Car
```python
import torch.optim as optim
import gymnasium as gym

# 1. Create the brain
car_model = CarBrain()

# 2. Create the optimizer (learning rate = 0.005)
car_optimizer = optim.Adam(car_model.parameters(), lr=0.005)

# 3. Create the environment
car_env = gym.make('CarRacing-v0')
```

### Assignment 2
Below your class in `train.py`, do the equivalent for CartPole. 
Create your `policy` network, create an Adam optimizer for it with a learning rate of `0.01`, and make the `CartPole-v1` environment.

---

## Module 3: Playing the Game (The Forward Pass)

Before we write the code to play the game, let's break down the new PyTorch and Gymnasium concepts you will see:

- **`state`, `info`, and `.reset()`:** To start a new game, we call `state, info = car_env.reset()`. The environment returns the very first `state` array (the starting numbers of the sensors) and an `info` dictionary (which contains extra debugging details that we usually just ignore).
- **Tensors:** A Tensor is just PyTorch's fancy version of a list/array of numbers. PyTorch neural networks refuse to look at normal Python lists; they only understand Tensors. So, we convert the `state` array into a tensor using `torch.tensor(state)`. The `.unsqueeze(0)` just adds an extra bracket around it because PyTorch always expects data to arrive in batches (like a list of lists).
- **`Categorical` Class (A Dry Run):** If our brain outputs percentages like `[80% Left, 20% Right]`, we shouldn't just force it to go Left every time. It needs to explore! Here is exactly what happens mathematically:
  1. **The Math Coin:** `Categorical(probs)` creates a mathematical "weighted coin" using our `[0.80, 0.20]` percentages.
  2. **Sampling:** `m.sample()` flips the coin. It randomly spits out `0` (Left) 80% of the time, and `1` (Right) 20% of the time. We pass this chosen action to the environment using `.item()` to convert it from a PyTorch Tensor back to a normal Python integer.
  3. **The Computational Graph (`log_prob`):** Since you built your own custom autograd engine (`PocketNet`), let's look at what is actually happening. 
  In your autograd engine, every mathematical operation creates a `Value` object that stores its `_prev` parents so you can backpropagate later. The neural network's output `probs` is a Tensor with `requires_grad=True` (it is a `Value` object whose parents are the network weights).
  If you use `math.random` to pick `action = 0`, that action is just a raw integer. It has no parents. If you try to call `.backward()` on a raw integer, your autograd engine will crash because the chain rule is broken.
  To do the backward pass, we need to calculate `Loss = -log(probability) * Reward`. 
  `Categorical.log_prob(action)` is just a PyTorch shortcut that does this:
  `chosen_prob = probs[action]` (Extracts the `0.80` Tensor, keeping its `_prev` pointers intact).
  `log_prob_value = torch.log(chosen_prob)` (Applies the math `log()` function, creating a new Tensor whose `_prev` points to `chosen_prob`).
  Because `log_prob_value` was built using actual math operations on the network's output, the entire computational graph remains perfectly intact! When you call `loss.backward()`, PyTorch's autograd engine starts at `log_prob_value`, applies the derivative of `log()`, moves backwards to `chosen_prob`, backwards through the Softmax, and finally back to the Linear layer weights, accumulating `.grad` exactly like your custom engine does!

### Example: Self-Driving Car (Step-by-Step)
```python
from torch.distributions import Categorical

# Play 100 games
for game in range(100):
    
    # Start a brand new game
    state, info = car_env.reset()
    
    # Empty our memories for this specific game
    saved_log_probs = []
    rewards = []
    
    # Play the game for a maximum of 200 steps
    for step in range(200):
        
        # 1. Convert the game state into a PyTorch Tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # 2. Ask the brain for the action percentages
        probs = car_model(state_tensor)
        
        # 3. Create a weighted coin and flip it
        m = Categorical(probs)
        action = m.sample()
        
        # 4. Save the log probability of that action into memory
        saved_log_probs.append(m.log_prob(action))
        
        # 5. Tell the game to execute the action, and get the new results
        state, reward, terminated, truncated, info = car_env.step(action.item())
        
        # 6. Save the reward into memory
        rewards.append(reward)
        
        # If we crashed (terminated) or ran out of time (truncated), stop playing!
        if terminated or truncated:
            break
```

### Assignment 3
Write the exact same training loops for CartPole in `train.py`. Use **500** total games (episodes), and **500** maximum steps per game.

---

## Module 4: The Credit Assignment Problem & Returns

We can't just give every action the total score of the game. An action only takes credit for the rewards that happened **after** it. We also use a Discount Factor (`gamma = 0.99`) to prioritize immediate survival over distant survival.

### Example: Self-Driving Car
After the inner step loop breaks, the car has a list of `rewards`. We calculate the discounted returns by looping backwards through the rewards.

```python
    # THIS GOES AT THE BOTTOM OF THE OUTER LOOP
    # (after the game finishes)
    
    returns = []
    R = 0
    # Loop backwards through the rewards
    for r in rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
```

### Assignment 4
At the bottom of your outer episode loop, write the exact same logic to convert your CartPole `rewards` list into a `returns` list.

---

## Module 5: The Magic Learning Trick (Backward Pass)

This is where the AI learns. We use the fake Loss formula:
**Loss = -1 * log_prob * Return**

PyTorch turns the weights to make the Loss as negative as possible. If the Return was a huge positive number, PyTorch automatically tweaks the weights to make that specific action more likely next time!

### Example: Self-Driving Car
```python
    # THIS GOES IMMEDIATELY AFTER CALCULATING RETURNS
    policy_loss = []
    
    # Pair up every log_prob with its matching Return
    for log_prob, R in zip(saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
        
    # Sum them all together
    # (torch.stack takes our Python list of separate tensors and glues them into one big PyTorch tensor so we can call .sum() on it)
    loss = torch.stack(policy_loss).sum()
    
    # Perform Gradient Descent to update the weights!
    car_optimizer.zero_grad()
    loss.backward()
    car_optimizer.step()
    
    # Print progress
    print(f"Game: {game}, Score: {sum(rewards)}")
```

### Assignment 5
Do the equivalent for your CartPole agent inside your outer loop. 
Once finished, run your code! You should see the score slowly rise until it hits 500 consistently!

---

## FAQ

### What is `m.log_prob(action)` doing?

To understand what `m.log_prob(action)` (or `weighted_coin.log_prob(action)`) is doing, let's break it down into two parts: **what it calculates** and **why we need it for Reinforcement Learning**.

#### 1. What is it calculating?
When your neural network outputs probabilities (like 80% chance to go Left, 20% chance to go Right), you use `Categorical` to create a weighted coin based on those probabilities. 

When you flip that coin and pick an action, `log_prob(action)` simply looks up the probability of the action you just took, and takes the **natural logarithm** of that number.

*   If your network gave Left an 80% probability (0.8), and your agent picked Left, `log_prob(action)` calculates `log(0.8)`.
*   If your agent picked Right, it calculates `log(0.2)`.

*(Note: In mathematics, "log" usually implies base-10, but in programming and machine learning, it always means the natural logarithm, base-e).*

#### 2. Why do we need it? (The "Policy Gradient" Trick)
In the REINFORCE algorithm, you are trying to teach the neural network to increase the probability of actions that lead to high rewards and decrease the probability of actions that lead to low or negative rewards. 

You might wonder, *"Why use the logarithm? Why not just use the raw 0.8 probability?"*

Here is why `log_prob` is the secret sauce of policy gradients:

*   **It prevents the computer from rounding to zero (Underflow):** If you play a game for 500 steps, the probability of that exact sequence of 500 actions is P1 * P2 * P3 ... all the way to P500. Multiplying 500 decimals (like 0.8 * 0.7 * 0.9...) results in a number so tiny that a computer simply rounds it to exactly `0.0`. When you use logarithms, you can **add** the numbers instead of multiplying them (`log(0.8) + log(0.7) + log(0.9)...`), which keeps the numbers safely within the computer's limits.
*   **It makes the Calculus easier:** To update the neural network's weights, PyTorch needs to calculate gradients (derivatives). The derivative of a logarithm has a very neat mathematical property that makes calculating the gradient of the entire policy network extremely simple and efficient.

#### How you will use it next
Later in your code, you will take this `log_prob` and multiply it by the total reward the agent got. 
*   `log_prob(action) * BIG_REWARD` = PyTorch knows to strongly increase the weights that caused that action.
*   `log_prob(action) * SMALL_REWARD` = PyTorch only slightly increases them.