import numpy as np


class Tanh:
    """
    Hyperbolic tangent activation function.
    
    Forward: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    Backward: d/dx tanh(x) = 1 - tanh²(x)
    """
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using numpy's built in stable tanh.
            
        Args:
            x: Input array of any shape
            
        Returns:
            tanh(x), same shape as input
        """
        tanh_x = np.tanh(x)
        self.cache = tanh_x
        return tanh_x
        

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradient.
        
        Args:
            dout: Upstream gradient
            
        Returns:
            Gradient with respect to input
        """
        tanh_x = self.cache
        return dout * (1 - (tanh_x ** 2))
