# PocketNet

A tiny, scalar-valued autograd engine and neural network library built completely from scratch in Python. 

This project was inspired by and built after watching [Andrej Karpathy's Micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0). I implemented the core automatic differentiation logic and neural network abstractions myself, extending the math operations and adding a clean API for building multi-layer perceptrons (MLPs).

## Project Structure

- `engine.py`: The core autograd engine. Contains the `Value` class which wraps standard floats, tracks their history through a computational graph, and computes gradients using backpropagation (`.backward()`).
- `NeuralNet.py`: The neural network API. Contains `Neuron`, `Layer`, and `NeuralNet` classes that utilize the `Value` engine to build fully connected layers with `tanh` activations.
- `example.py`: A complete training loop demonstration. It builds a 3-layer MLP, defines a Mean Squared Error (MSE) loss, and trains the network on a small dataset using Stochastic Gradient Descent (SGD).

## Features
- Fully functional Backpropagation (`.backward()`) on a dynamically built computational graph.
- Support for arithmetic operations: `+`, `-`, `*`, `/`, `**` (power), and `tanh` activation.
- Method to reset gradients (`.zero_grad()`) before each optimization step.
- An easy-to-use API for creating neural networks of arbitrary shapes.

## Usage Example

You can train a neural network just by running `example.py`. Here is a quick look at how the API works:

```python
from engine import Value
from NeuralNet import NeuralNet

# 1. Initialize a neural net: 3 inputs, two hidden layers of 4, and 3 outputs
model = NeuralNet(num_in=3, hidden_layers=[4, 4], num_out=3)

# 2. Forward pass with some input
inputs = [1, 5, 8]
predictions = model(inputs)

# 3. Compute Loss (MSE)
targets = [0, 1, 0]
losses = [(p - Value(t))**2 for p, t in zip(predictions, targets)]
loss = sum(losses, Value(0))

# 4. Backward pass (compute gradients)
loss.zero_grad()
loss.backward()

# 5. Gradient Descent step
for p in model.parameters():
    p.data -= 0.05 * p.grad
```

## Running the Example

Simply run the example script to watch the loss decrease over 1000 iterations:

```bash
python example.py
```

## Learning Notes

I have included my personal study notes in the `notes/` directory (along with the `micrograd-lecture.ipynb` notebook). These contain the math derivations, calculus, and step-by-step logic I used to understand backpropagation and neural networks from the ground up. Feel free to use them if you're learning along!
