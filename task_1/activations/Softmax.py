# Softmax.py
import numpy as np

class Softmax:
    """
    Softmax activation function.
    
    Forward: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    Converts logits to probability distribution over classes.
    """
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input logits, shape (batch_size, num_classes)
            
        Returns:
            Probability distribution, shape (batch_size, num_classes)
            Each row sums to 1.0
        """
        # numeric stability: subtract row-wise max
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.cache = probs
        return probs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        When used with cross-entropy, the gradient is already computed
        in backward_pass as (y_pred - y_true) in NeuralNetwork class.
        """
        return dout
    