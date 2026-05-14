import numpy as np

class Value:
    def __init__(self, data, _children = (), _op = None, label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        # suppose c = a + b
        # when c._backward is called, a and b's gradients are set according to the op and c's gradient
        # this is a function that defines the backward behaviour for differentiation at the local level basically
        # this has to be different for different functions
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"[ Label: {self.label}, Data: {self.data}, Grad: {self.grad} ]"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def back_of_out():
            self.grad += out.grad * 1.0 # because if out = a + b, d(L)/da = dL/da = dL/d(out) * d(out)/da = dL/d(out) * 1
            other.grad += out.grad * 1.0 # because if out = a + b, d(L)/da = dL/da = dL/d(out) * d(out)/da = dL/d(out) * 1
        
        out._backward = back_of_out
        return out 
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        
        def back_of_out():
            self.grad += out.grad * 1.0 # because if out = a - b, d(L)/da = dL/da = dL/d(out) * d(out)/da = dL/d(out) * 1
            other.grad += out.grad * -1.0 # because if out = a - b, d(L)/db = dL/db = dL/d(out) * d(out)/db = dL/d(out) * -1
        
        out._backward = back_of_out
        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        return self * other**(-1)

    def __rtruediv__(self, other):
        return Value(other) / self

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def back_of_out():
            self.grad += out.grad * other.data  # accumulate: d(out)/d(self) = other.data, so dL/d(self) += dL/d(out) * other.data
            other.grad += out.grad * self.data  # accumulate: d(out)/d(other) = self.data, so dL/d(other) += dL/d(out) * self.data
        out._backward = back_of_out
        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        # this only supports integer powers
        assert isinstance(other, int)
        out = Value(self.data**other, (self,), f'**{other}')
        
        def back_of_out():
            self.grad += other * self.data**(other-1) * out.grad # d(out)/d(self) = other * self**(other-1)
        out._backward = back_of_out

        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), 'tanh')
        
        def back_of_out():
            self.grad += (1 - out.data**2) * out.grad  # accumulate: derivative of tanh is 1 - tanh^2, so dL/d(self) += dL/d(out) * (1 - tanh(self)^2)
        out._backward = back_of_out
        return out 
    
    def backward(self):
        topo = []
        
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        # toposort of tree where all edges point upward is basically like postorder of tree
        for node in reversed(topo):
            node._backward()


    def zero_grad(self):
        # reset gradients for the entire graph to avoid pollution from previous backward passes
        visited = set()
        def reset(node):    
            if node not in visited:
                visited.add(node)
                node.grad = 0.0
                for child in node._prev:
                    reset(child)
        reset(self)