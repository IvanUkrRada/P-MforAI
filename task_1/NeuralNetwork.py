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
from activations.ReLU import ReLU
from activations.Sigmoid import Sigmoid
from activations.Softmax import Softmax
from Dropout import Dropout

class NeuralNetwork:
    """
        Fully parametrizable Neural Network.
        
        Parameters:
        - input_size: int (e.g., 784 for MNIST, 3072 for CIFAR-10 flattened)
        - hidden_layers: list of ints → [64, 32] means two hidden layers with 64 and 32 units
        - output_size: int (number of classes)
        - activation: list → 'relu' or 'sigmoid' for all hidden layers (try with others if time for report),
                        or list like ['relu', 'sigmoid', 'softmax'] to specify per layer 
                        i.e. for 2nd last, last hidden layer and output layer
        - dropout_rates: None or list of floats (0 to 1), same length as hidden_layers
        - regularisation: float, either L1 or L2
        - seed: for reproducibility. It makes it so that random are not so random.
    """
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int, 
                 activations: list[str], dropout_rates=None, regularisation=None, 
                 reg_lambda=0.01, seed=0):
        self.input_size = input_size        # It's 3072 for cifar 10 image when flattened.
        self.output_size = output_size
        self.hidden_layers = hidden_layers            
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.regularisation = regularisation # It's either L1 or L2.
        self.reg_lambda = reg_lambda
        self.len_hidden_layers = len(self.hidden_layers)

        np.random.seed(seed)

        # Input Validation: Ensuring the number of activations provided matches the number of layers
        expected_activations_count = self.len_hidden_layers + 1 # All hidden layers + the final output layer
        if expected_activations_count != len(self.activations):
            raise ValueError(
                f"Mismatch in number of activations provided. Expected {expected_activations_count} "
                f"activations (1 for each of {self.len_hidden_layers} hidden layers + 1 output layer), "
                f"but got {len(self.activations)}."
            )
        
        self.W, self.b = {}, {}
        self.initialise_weights() 

        self.dropout_layers = {}
        if self.dropout_rates is not None:
            if len(self.dropout_rates) != self.len_hidden_layers:
                raise ValueError("dropout rates must match number of hidden layers")

            for i, p in enumerate(self.dropout_rates, start=1):
                self.dropout_layers[i] = Dropout(p)

    '''
    Reference: Material used to learn about different kinds of initializer techniques.
    https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/
    '''   
   
    def initialise_weights(self):
        # Collection of dimensions of all layers.
        dims = [self.input_size] + self.hidden_layers + [self.output_size]

        # He-Normal initialisation is better for ReLU
        for i in range(1, len(dims)):
            current_activation = self.activations[i-1]
            
            prev_layer = dims[i-1] 
            curr_layer = dims[i]

            if current_activation == "relu":
                # Applying He-Normal Initialiser (specially for ReLU). 
                scale = np.sqrt(2.0 / prev_layer)
                self.W[i] = np.random.randn(curr_layer, prev_layer) * scale
                self.b[i] = np.zeros((curr_layer, 1))

            elif current_activation == "sigmoid" or current_activation == "tanh":
                # Applying Xavier/Glorot Normal Initialiser (specifically for sigmoid/tanh)
                scale = np.sqrt(1.0 / prev_layer) 
                self.W[i] = np.random.randn(curr_layer, prev_layer) * scale
                self.b[i] = np.zeros((curr_layer, 1))

            elif current_activation == "softmax":
                # Applying Xavier initialization ()
                scale = np.sqrt(1.0 / prev_layer)
                self.W[i] = np.random.randn(curr_layer, prev_layer) * scale
                self.b[i] = np.zeros((curr_layer, 1))

            else:
                # Fallback for unsupported/unknown activation functions
                print(f"Warning: Unknown activation '{current_activation}' at layer {i}. Defaulting to standard random initialisation.")
                self.W[i] = np.random.randn(curr_layer, prev_layer) * 0.01


    def get_activation(self, name):
        if name == "relu":
            return ReLU()
        elif name == "sigmoid":
            return Sigmoid()
        elif name == "softmax":
            return Softmax()
        else:
            raise ValueError(f"Unknown activation: {name}")

    
    def forward_pass(self, X, training=True):  # Added dropout
        A = X.T
        self.cache = {"A0": A}  # A0 represents activation of input layer.
        self.activation_objects = {}

        for i in range(1, self.len_hidden_layers + 1):
            Z = self.W[i] @ A + self.b[i]
            self.cache[f"Z{i}"] = Z

            act = self.get_activation(self.activations[i-1])
            A = act.forward(Z)
            self.activation_objects[i] = act

            if training and self.dropout_rates is not None and i in self.dropout_layers:
                A = self.dropout_layers[i].forward(A, training = True)

            self.cache[f"A{i}"] = A

        return A
        

    def backward_pass(self, X, y_true):
        m = X.shape[0]
        y_true = y_true.T

        y_pred = self.cache[f"A{self.len_hidden_layers}"]

        # derivative of Loss wrt A for Softmax + Cross-entropy layer.
        print(f"Y_prediction: {y_pred}")
        dA = y_pred - y_true

        # Collection cache for all gradients calculated.
        self.grads = {}

        for i in reversed(range(1, self.len_hidden_layers + 1)):

            # Activation backward
            act = self.activation_objects[i]
            dZ = act.backward(dA)

            # Linear backward
            A_prev = self.cache[f"A{i-1}"]

            self.grads[f"dW{i}"] = (dZ @ A_prev.T) / m
            self.grads[f"db{i}"] = np.sum(dZ, axis=1, keepdims=True) / m

            # gradient to propagate backward
            dA = self.W[i].T @ dZ

            if self.dropout_rates is not None and i in self.dropout_layers:
                dA = self.dropout_layers[i].backward(dA)

        
    def update_weights(self):
        for i in range(1, self.len_hidden_layers+1):
            self.W[i] -= self.lr * self.grads[f"dW{i}"]
            self.b[i] -= self.lr * self.grads[f"db{i}"]
        
    def predict(self, X):
        output = self.forward_pass(X, training=False)
        return np.argmax(output, axis=0)
    
    # For testing purposes:
    def get_W(self):
        return self.W
    def get_b(self):
        return self.b
    
    def train(self, X, y, X_val, y_val, 
                epochs=10, batch_size=64, lr=0.01, decay=0.0, optimizer=None):
        n = X.shape[0] 

        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                self.forward_pass(X_batch, training=True)
                self.backward_pass(X_batch, y_batch)

            # Evaluating...
            prediction = self.predict(X_val)
            acc = np.mean(prediction == y_val)
            print(f"Epoch {epoch} / {epochs}, Accuracy: {acc:.4f * 100}")

        print("Training Completed!!!")




model1 = NeuralNetwork(5, [2, 1], 5, ["relu", "sigmoid", "softmax"])
# print(f"Weight: {model1.get_W()[1] . T}")
# print(f"bias : {model1.get_b()}")

# model1.train(model1, )


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

#Testing Dropout function

import numpy as np

#Test input
X = np.random.randn(10, 5) 

# Model with dropout at 50%
model = NeuralNetwork(
    input_size=5,
    hidden_layers=[4],  #one hidden layer
    output_size=3,
    activations=["relu", "softmax"],
    dropout_rates=[0.5]  #dropout on hidden layer
)

print(" === Forward pass with training=True (dp on) ===")
out1 = model.forward_pass(X, training=True)
out2 = model.forward_pass(X, training=True)

print(out1)
print("\nSecond pass:")
print(out2)

print("\n=== Forward pass with training=False (dp Off) ===")
out_eval = model.forward_pass(X, training=False)
print(out_eval)

