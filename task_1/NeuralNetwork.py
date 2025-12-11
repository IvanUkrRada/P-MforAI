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
import activations as act

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
    def __init__(self, input_size: int, layers: list[int], output_size: int, 
                 activations: list[str], dropout_rates=None, regularisation=None, 
                 reg_lambda=0.01, seed=0):
        self.input_size = input_size        # It's 3072 for cifar 10 image when flattened.
        self.output_size = output_size
        self.layers = layers            
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.regularisation = regularisation # It's either L1 or L2.
        self.reg_lambda = reg_lambda

        np.random.seed(seed)
        
        self.W, self.b = {}, {}
        self.initialise_weights() 

    '''
    Reference: Material used to learn about different kinds of initializer techniques.
    https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/'''
    def initialise_weights(self):
        # Collection of dimensions of all layers.
        dims = [self.input_size] + self.layers + [self.output_size]

        # He-Normal initialisation is better for ReLU
        for i in range(1, len(dims)):
            current_activation = self.activations[i-1]
            
            prev_layer = dims[i-1] 
            curr_layer = dims[i]

            if current_activation == "relu":
                # Applying He-Normal Initialiser (specially for ReLU). 
                scale = np.sqrt(2.0 / prev_layer)
                self.W[i] = np.random.randn(curr_layer, prev_layer) * scale
                self.b[i] = np.zeros((1, curr_layer))

            elif current_activation == "sigmoid" or current_activation == "tanh":
                # Applying Xavier/Glorot Normal Initialiser (specifically for sigmoid/tanh)
                scale = np.sqrt(1.0 / prev_layer) 
                self.W[i] = np.random.randn(curr_layer, prev_layer) * scale
                self.b[i] = np.zeros((1, curr_layer))

            elif current_activation == "softmax":
                # Applying Xavier initialization ()
                scale = np.sqrt(1.0 / prev_layer)
                self.W[i] = np.random.randn(curr_layer, prev_layer) * scale
                self.b[i] = np.zeros((1, curr_layer))

            else:
                # Fallback for unsupported/unknown activation functions
                print(f"Warning: Unknown activation '{current_activation}' at layer {i}. Defaulting to standard random initialisation.")
                self.W[i] = np.random.randn(curr_layer, prev_layer) * 0.01
                self.b[i] = np.zeros((1, curr_layer))


    # For testing purposes:
    def get_W(self):
        return self.W
    def get_b(self):
        return self.b

    def get_activation(self, name):
        if name == "relu":
            return act.ReLU()
        elif name == "sigmoid":
            return act.Sigmoid()
        elif name == "softmax":
            return act.Softmax()
        else:
            raise ValueError(f"Unknown activation: {name}")
        
        
    def forward_pass_no_dropout(self, X, training=True):
        pass
        
    def backward(self, X, y):
        pass
        
    def update_weights(self, optimizer):
        if optimizer is None:
            pass
        pass
        
    def train(self, X, y, X_val=None, y_val=None, 
              epochs=20, batch_size=64, lr=0.01, decay=0.0, optimizer=None):
        pass
        
    def predict(self, X):
        pass


model = NeuralNetwork(5, [2, 1], 5, ["relu", "sigmoid", "softmax"])
print(f"Weight: {model.get_W()}")
print(f"bias : {model.get_b()}")

'''
Can have a NN class that extends the superclass which applies dropout in hidden layers. (i.e. different model[])
'''

# {1: array([[0.], [0.]]), 
#  2: array([[0.]]), 
#  3: array([[0.],[0.],[0.],[0.],[0.]])
# }

            # self.W[i] = np.random.randn(dims[i], dims[i-1]) * np.sqrt(2.0/dims[i-1])


# Weight: {1: array([[-0.58176609,  0.36755163, -0.43474151,  0.65772275, -0.40864235], [-0.16379515,  1.83227004, -0.21435161,  0.67958921,  0.71505445]]), 
#        2: array([[1.84532779, 0.05482559]]), 
#        3: array([[-0.18495464],[-2.07575926],[ 1.45916958],[-1.3764414 ],[ 1.24237644]])
#        }
