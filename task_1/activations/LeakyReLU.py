import numpy as np

class LeakyReLU:
    """
    Leaky ReLU activation function.
    
    Forward: f(x) = x if x > 0, else alpha * x
    Backward: f'(x) = 1 if x > 0, else alpha
    """
    def __init__ (self, alpha=0.01):
        self.cache = None
        self.alpha = alpha
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input array
            
        Returns:
            Activated output, same shape as input
        """
        self.cache = x
        return np.maximum(x, x * self.alpha)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradient.
        
        Args:
            dout: Upstream gradient
            
        Returns:
            Gradient with respect to input
        """
        x = self.cache
        dx = np.where(x > 0, 1, self.alpha)
        return dout * dx
