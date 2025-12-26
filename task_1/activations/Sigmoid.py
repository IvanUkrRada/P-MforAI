import numpy as np

class Sigmoid:
    """
    Sigmoid activation function. (S == sigma below)
    
    Forward: S(x) = 1 / (1 + e^(-x))
    Backward: S'(x) = S(x) * (1 - S(x))
    """
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with numerical stability.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid output, same shape as input
        """
        output = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        self.cache = output
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradient.
        
        Args:
            dout: Upstream gradient
            
        Returns:
            Gradient with respect to input
        """
        output = self.cache
        return dout * output * (1 - output)
    