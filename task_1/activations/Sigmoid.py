import numpy as np

class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        positive_x = np.maximum(0, x)   # To avoid runtime error with overflow.
        output = 1 / (1 + np.exp(-positive_x))
        self.cache = output
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        output = self.cache
        return dout * output * (1 - output)
    