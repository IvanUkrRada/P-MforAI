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


    # NOTE: Don't need to use this function in NN. It's used in the backwards code inside NN class. 
    def backward(self):
        """
        When used with cross-entropy, the gradient is already computed
        in backward_pass as (y_pred - y_true) in NeuralNetwork class.
        """
        return