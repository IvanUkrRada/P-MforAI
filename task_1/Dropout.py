"""
Inverted Dropout regularization layer.

Dropout prevents overfitting by randomly dropping units during training.
Inverted dropout scales activations during training so test-time behavior
remains unchanged without requiring scaling.
"""
import numpy as np

class Dropout:
    """
    Inverted Dropout layer for regularization.
    
    Forward (training): output = input * mask / (1 - p)
    Forward (testing):  output = input
    Backward:           gradient = upstream_grad * mask
    """
    def __init__(self, p=0.5):
        if p < 0 or p >= 1:
            raise ValueError("Dropout keep probability p must be in [0,1)")
        self.p = p          # keep probability
        self.mask = None   # mask for backprop

    def forward(self, x, training=True):
        """
        Forward pass with inverted dropout.
        
        Args:
            x: Input array
            training: If True, apply dropout; if False, pass through
            
        Returns:
            Output array, same shape as input
        """
        if not training or self.p == 0:
            return x        # no dropout at test time

        # inverted dropout mask
        keep_prob = 1.0 - self.p
        self.mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        return x * self.mask

    def backward(self, dout):
        """
        Backward pass - apply mask to gradients.
        
        Args:
            dout: Upstream gradient
            
        Returns:
            Gradient with respect to input
        """
        return dout * self.mask
