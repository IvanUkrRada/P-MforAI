'''
Docstring for task_1.NeuralNetwork
What's expected?

e. Implement a fully parametrizable neural network class
You should implement a fully-connected NN class where with number of hidden
layers, units, activation functions can be changed. In addition, you can add dropout or
regularizer (L1 or L2). Report the parameters used (update rule, learning rate, decay,
epochs, batch size) and include the plots in your report.
'''
import numpy as np

class NeuralNetwork:
    """
        Fully parametrizable Neural Network.
        
        Parameters:
        - input_size: int (e.g., 784 for MNIST, 3072 for CIFAR-10 flattened)
        - layers: list of ints → [64, 32] means two hidden layers with 64 and 32 units
        - output_size: int (number of classes)
        - activation: list → 'relu' or 'sigmoid' for all hidden layers (try with others if time for report),
                        or list like ['relu', 'sigmoid', 'softmax'] to specify per layer 
                        i.e. for 2nd last, last hidden layer and output layer
        - dropout_rates: None or list of floats (0 to 1), same length as hidden_layers
        - regularisation: float, either L1 or L2
        - seed: for reproducibility. It makes it so that random are not so random.
    """
    def __init__(self, input_size: int, layers: list[int], activations: list[str], dropout_rates=None,
                regularisation=None, reg_lambda=0.01, seed=None):
        self.input_size = input_size        # It's 3072 for cifar 10 image when flattened.
        self.layers = layers            
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.regularisation = regularisation # It's either L1 or L2.
        self.reg_lambda = reg_lambda
        self.hidden_layer_len = len(self.layers) - 1
        
        if seed is not None:
            np.random.seed(seed)
        
        self.initialise_weights() 
        
    def initialise_weights(self):
        pass
        
    def forward(self, X, training=True):
        pass
        
    def backward(self, X, y):
        pass
        
    def update_weights(self, optimizer):
        pass
        
    def train(self, X, y, X_val=None, y_val=None, 
              epochs=20, batch_size=64, lr=0.01, decay=0.0, optimizer=None):
        pass
        
    def predict(self, X):
        pass

