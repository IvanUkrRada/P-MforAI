# Softmax with Cross Entropy

import numpy as np


class SoftmaxCrossEntropy:
    """
    Combined Softmax activation and Cross-Entropy loss layer.
    
    Softmax: P(y_i) = exp(x_i) / Σ exp(x_j)
    Cross-Entropy: L = -Σ y_true * log(y_pred)
    
    Combined gradient simplifies to: y_pred - y_true
    """
    def __init__(self):
        self.probs = None 
        self.cache_labels = None 

    def forward(self, logits: np.ndarray, cache_labels: np.ndarray) -> np.ndarray:
        """
        Forward pass computing softmax probabilities and cross-entropy loss.

        Args:
            logits: Raw network outputs, shape (batch_size, num_classes): For understanding it's just x but 
                When a single sample  x that has gone through trenches with couple of hidden layers and 
                when it passes through linear layer and reaches SoftMax it's called "logits".
            cache_labels: one-hot encoded, shape (batch_size, num_classes)
                Remember it has to be one-hot encoded
        
        Returns:
            Int scalar loss value
        """
        # Softmax calculation
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.cache_labels = cache_labels

        # Cross-entropy loss calculation
        batch_size = logits.shape[0]
        log_likelihood = -np.log(np.sum(self.probs * cache_labels, axis=1))
        loss = np.sum(log_likelihood) / batch_size
        return loss


    def backward(self):
        """
        Backward pass not implemented here.
        
        The gradient (probs - labels) is computed directly in the 
        NeuralNetwork class backward_pass method for efficiency.
        """
        return
