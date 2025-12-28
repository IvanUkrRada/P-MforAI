"""
Optimization algorithms for neural network training.

Provides 2 gradient descent optimizers:
- SGD: Stochastic Gradient Descent
- SGDMomentum: SGD with momentum for accelerated convergence
"""
import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms.
    
    All optimizers must implement the update() method to modify
    network parameters based on computed gradients.
    """
    def update(self, params: dict, grads: dict):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Updates parameters using: θ = θ - η *dL
    where η is the learning rate(lr) and ∇L is the gradient.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params: dict, grads: dict):
        """
        Perform SGD parameter update.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for k in params:
            if k.startswith("W"):
                i = k[1:]
                params[k] -= self.lr * grads[f"dW{i}"]
            elif k.startswith("b"):
                i = k[1:]
                params[k] -= self.lr * grads[f"db{i}"]



class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        """
        Initialize SGD with Momentum optimizer.
        
        Args:
            lr: Learning rate (default: 0.01)
            beta: Momentum coefficient, typically 0.9 (default: 0.9)
        """
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params: dict, grads: dict):
        """
        Perform momentum-based parameter update.
        Accumulates exponentially weighted average of past gradients
        to accelerate convergence and dampen oscillations.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for k in params:
            if k.startswith("W"):
                i = k[1:]
                g = grads[f"dW{i}"]
            elif k.startswith("b"):
                i = k[1:]
                g = grads[f"db{i}"]
            else:
                continue

            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])

            # momentum update
            self.v[k] = self.beta * self.v[k] + g
            params[k] -= self.lr * self.v[k]