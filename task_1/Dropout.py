"""
Docstring for task_1.activations.Dropout
Task 1d: Implement dropout

d. Present dropout in report. implement inverted dropout. forward and backward pass should be implemented. Present in report

Note: Test performance is critical, it is preferable to leaving the forward pass unchanged at test time. 
therefore, in most implementations inverted dropout is employed to overcome undesirable property of the original input.

"""
import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        if p <= 0 or p > 1:
            raise ValueError("Dropout keep probability p must be in (0,1)")
        self.p = p          # keep probability
        self.mask = None   # mask for backprop

    def forward(self, x, training=True):
        if not training:
            return x        # no dropout at test time

        # inverted dropout mask
        self.mask = (np.random.rand(*x.shape) < self.p) / self.p
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
