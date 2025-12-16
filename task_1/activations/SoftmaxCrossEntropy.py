# Softmax with Cross Entropy

import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None 
        self.cache_labels = None 

    def forward(self, logits: np.ndarray, cache_labels: np.ndarray) -> np.ndarray:
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


    # NOTE: Don't need to use this function in NN. It's used in the backwards code. Could change later on.
    def backward(self) -> np.ndarray:
        """
        Calculates dX (gradient wrt logits/input to this layer).
        The result is dA of the previous layer (or dZ of the current layer in your naming convention)
        """
        batch_size = self.cache_labels.shape[0]
        # The simplified gradient: (Predicted Probs - True Labels) / Batch Size
        dX = (self.probs - self.cache_labels) / batch_size
        return dX