import numpy as np
from engine import Value

class Neuron:
    def __init__(self, num_in):
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(num_in)]
        self.bias = Value(np.random.uniform(-1, 1))

    def __call__(self, input_arr):
        output = sum((wi*xi for wi, xi in zip(self.weights, input_arr)), self.bias).tanh()
        return output

    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:

    def __init__(self, num_in, num_neurons):
        self.neurons = [Neuron(num_in) for _ in range(num_neurons)]

    def __call__(self, input_arr):
        output = [n(input_arr) for n in self.neurons]
        return output

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class NeuralNet:

    def __init__(self, num_in, hidden_layers=[], num_out=1):
        sizes = [num_in] + hidden_layers + [num_out]
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    
    def __call__(self, input_arr):
        output = input_arr
        for l in self.layers:
            output = l(output)
        return output

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]