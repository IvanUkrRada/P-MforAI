import numpy as np

class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        self.cache = output
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        output = self.cache
        return dout * output * (1 - output)
    