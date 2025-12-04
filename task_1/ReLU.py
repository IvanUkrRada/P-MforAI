import numpy as np

class ReLU:
    def __init__ (self):
        self.cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backwards (self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        return dout * (x > 0)