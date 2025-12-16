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
        - input_size: int (3072 for CIFAR-10 flattened)
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
                self.W[i] = np.random.randn(prev_layer, curr_layer) * scale
                self.b[i] = np.zeros((1, curr_layer))

            elif current_activation == "sigmoid" or current_activation == "tanh":
                # Applying Xavier/Glorot Normal Initialiser (specifically for sigmoid/tanh)
                scale = np.sqrt(1.0 / prev_layer) 
                self.W[i] = np.random.randn(prev_layer, curr_layer) * scale
                self.b[i] = np.zeros((1, curr_layer))

            elif current_activation == "softmax":
                # Applying Xavier initialization ()
                scale = np.sqrt(1.0 / prev_layer)
                self.W[i] = np.random.randn(prev_layer, curr_layer) * scale
                self.b[i] = np.zeros((1, curr_layer))

            else:
                # Fallback for unsupported/unknown activation functions
                print(f"Warning: Unknown activation '{current_activation}' at layer {i}. Defaulting to standard random initialisation.")
                self.W[i] = np.random.randn(prev_layer, curr_layer) * 0.01
                self.b[i] = np.zeros((1, curr_layer))


    def get_activation(self, name):
        if name == "relu":
            return ReLU()
        elif name == "sigmoid":
            return Sigmoid()
        elif name == "softmax":
            return Softmax()
        else:
            raise ValueError(f"Unknown activation: {name}")

    
    def forward_pass(self, X, training=True):
        A = X   # A shape is now (32, 3072) i.e. (batch_size, features). Previously X.T caused a lot of shapes issues.
        self.cache = {"A0": A}  # A0 represents activation of input layer.
        self.activation_objects = {}

        # When testing softmax was ignored hence forward not working properly hence fixed problem with +2.
        for i in range(1, self.len_hidden_layers + 2):   
            Z = A @ self.W[i] + self.b[i]

            act = self.get_activation(self.activations[i-1])
            A = act.forward(Z)
            self.activation_objects[i] = act

            if training and self.dropout_rates is not None and i in self.dropout_layers:
                A = self.dropout_layers[i].forward(A, training = True)

            self.cache[f"A{i}"] = A

        return A
        

    def backward_pass(self, X, y_true):
        m = X.shape[0]

        y_pred = self.cache[f"A{self.len_hidden_layers + 1}"]

        # derivative of Loss wrt A for Softmax + Cross-entropy layer.
        dA = y_pred - y_true

        # Collection cache for all gradients calculated.
        self.grads = {}

        for i in reversed(range(1, self.len_hidden_layers + 2)):

            # Activation backward
            if i == self.len_hidden_layers + 1:
                # Softmax + Cross-Entropy
                dZ = dA
            else:
                # This block only runs hidden layers.
                act = self.activation_objects[i]
                dZ = act.backward(dA)


            # Applying dropout mast to dZ.
            if self.dropout_rates is not None and i <= self.len_hidden_layers:
                # The dropout mask (32, 4) must be applied to dZ (32, 4), not the later dA
                # Adding mask to dA later changed the shape of the dA gradient.
                dZ = self.dropout_layers[i].backward(dZ)

            # Linear backward
            A_prev = self.cache[f"A{i-1}"]

            # Recalculating dW and db for Batch-first convention
            self.grads[f"dW{i}"] = (A_prev.T @ dZ) / m 
            # db calculation: Sum across the batch (axis=0). db shape: (1, curr_features)
            self.grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m 


            # gradient to propagate backward
            dA = dZ @ self.W[i].T


    # lr is learning rate for the NN.
    def update_weights(self, lr):
        for i in range(1, self.len_hidden_layers + 2):
            self.W[i] -= lr * self.grads[f"dW{i}"]
            self.b[i] -= lr * self.grads[f"db{i}"]
        
    def predict(self, X):
        output = self.forward_pass(X, training=False)
        predictions = np.argmax(output, axis=1)
        return predictions
    
    # For testing purposes:
    def get_W(self):
        return self.W
    def get_b(self):
        return self.b
    
    def train(self, X, y, X_val, y_val, 
                epochs=1, batch_size=64, lr=0.01, decay=0.0, optimizer=None):
        n = X.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                self.forward_pass(X_batch, training=True)
                self.backward_pass(X_batch, y_batch)
                self.update_weights(lr)
            
            # Evaluating...
            prediction = self.predict(X_val)
            y_val_labels = np.argmax(y_val, axis=1)
            acc = np.mean(prediction == y_val_labels)
            
            print(f"Epoch {epoch+1} / {epochs}, Accuracy: {(acc * 100):.4f}%")

        print("Training Completed!!!")





'''
Can have a NN class that extends the superclass which applies no dropout in hidden layers. (i.e. different model[])
'''
