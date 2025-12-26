import numpy as np

class ReLU:
    """
    Rectified Linear Unit activation function.
    
    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0, else 0
    """
    def __init__ (self):
        self.cache = None
        

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input array
            
        Returns:
            Activated output, same shape as input
        """
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradient.
        
        Args:
            dout: Upstream gradient
            
        Returns:
            Gradient with respect to input
        """
        x = self.cache
        return dout * (x > 0)
