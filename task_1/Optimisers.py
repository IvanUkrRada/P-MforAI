"""
Code for optimiser that I can import in neural network.
"""
import numpy as np

class Optimizer:
    def update(self, params: dict, grads: dict):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params: dict, grads: dict):
        for k in params:
            if k.startswith("W"):
                i = k[1:]
                params[k] -= self.lr * grads[f"dW{i}"]
            elif k.startswith("b"):
                i = k[1:]
                params[k] -= self.lr * grads[f"db{i}"]



