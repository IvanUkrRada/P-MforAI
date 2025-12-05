# Softmax.py
import numpy as np

class Softmax:
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # numeric stability: subtract row-wise max
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.cache = probs
        return probs

    def backwards(self, dout: np.ndarray) -> np.ndarray:
        s = self.cache  # softmax output
        # Jacobian–vector product: dx = s * (dout - (dout * s) summed over classes)
        dot = np.sum(dout * s, axis=1, keepdims=True)
        dx = s * (dout - dot)
        return dx
