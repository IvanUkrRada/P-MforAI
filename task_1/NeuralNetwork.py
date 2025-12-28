"""
Fully parametrizable feedforward neural network for classification.

Supports configurable architecture, multiple activation functions,
dropout regularization, and L1/L2 weight regularization.
"""
import numpy as np
from activations.ReLU import ReLU
from activations.Sigmoid import Sigmoid
from activations.Softmax import Softmax, SoftmaxCrossEntropy
from activations.Tanh import Tanh
from activations.LeakyReLU import LeakyReLU
from Dropout import Dropout

class NeuralNetwork:
    """
        Fully parametrizable Neural Network for classification.
        Structure for understanding (for model with 3 hidden_layers): 
            Input data (3072) → (Input layer / First Hidden) (512, ReLU) → Hidden (256, ReLU) → 
            Hidden (128, ReLU) → Output (10, Softmax)
        
        Args:
            input_size: Input feature dimension (e.g., 3072 for CIFAR-10)
            hidden_layers: List of hidden layer sizes, e.g., [512, 256, 128]
            output_size: Number of output classes
            activations: List of activation names for hidden layers, e.g., ['relu', 'relu', 'relu']
            dropout_rates: List of dropout rates per hidden layer, or None
            regularisation: 'L1', 'L2', or None
            reg_lambda: Regularization strength (default: 0.01)
            seed: Random seed for reproducibility (default: 0)
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
        expected_activations_count = self.len_hidden_layers
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
        """
        Initialize weights and biases based on activation function.
        
        Uses He initialization for ReLU/LeakyReLU and Xavier/Glorot
        initialization for Sigmoid/Tanh/output layer.
        """
        # Collection of dimensions of all layers.
        dims = [self.input_size] + self.hidden_layers + [self.output_size]

        for i in range(1, len(dims)):            
            prev_layer = dims[i-1] 
            curr_layer = dims[i]

            if i <= self.len_hidden_layers:
                current_activation = self.activations[i-1]
            else:
                current_activation = "output"  # Output / Softmax layer
            

            if current_activation == "relu":
                # Applying He-Normal Initialiser (specially better for ReLU according to reference). 
                scale = np.sqrt(1.0 / prev_layer)
            
            elif current_activation in ["leaky_relu", "leakyrelu", "lrelu"]:
                alpha = 0.01
                scale = np.sqrt(1.0 / (prev_layer * (1 + alpha**2)))

            elif current_activation == "sigmoid" or current_activation == "tanh":
                # Applying Xavier/Glorot Normal Initialiser (specifically for sigmoid/tanh)
                scale = np.sqrt(2.0 / (prev_layer + curr_layer))

            elif current_activation == "output":
                # Applying Xavier initialization.
                scale = np.sqrt(2.0 / (prev_layer + curr_layer))


            else:
                # Fallback for unsupported/unknown activation functions
                print(f"Warning: Unknown activation '{current_activation}' at layer {i}. Defaulting to standard random initialisation.")
                scale = 0.01

            self.W[i] = np.random.randn(prev_layer, curr_layer) * scale
            self.b[i] = np.zeros((1, curr_layer))



    def get_activation(self, name):
        """
        Get activation function instance by name.
        
        Args:
            name: Activation name ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            
        Returns:
            Activation function instance
        """
        if name == "relu":
            return ReLU()
        elif name == "sigmoid":
            return Sigmoid()
        elif name == "softmax":
            return Softmax()
        elif name == "tanh":
            return Tanh()
        elif name in ["leaky_relu", "leakyrelu", "lrelu"]: # Aliases so there is less confusion.
            return LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    
    def forward_pass(self, X, training=True):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            training: If True, apply dropout; if False, inference mode
            
        Returns:
            Logits (if training) or probabilities (if not training)
        """
        A = X   # A shape is now (32, 3072) i.e. (batch_size, features). Previously X.T caused a lot of shapes issues.
        self.cache = {"A0": A}  # A0 represents activation of input layer.
        self.activation_objects = {}

        # Process hidden layers (not output layer yet)
        for i in range(1, self.len_hidden_layers + 1):   
            Z = A @ self.W[i] + self.b[i]

            act = self.get_activation(self.activations[i-1])
            A = act.forward(Z)
            self.activation_objects[i] = act

            if training and self.dropout_rates is not None and i in self.dropout_layers:
                A = self.dropout_layers[i].forward(A, training = True)

            self.cache[f"A{i}"] = A

        # Processing output/softmax layer.
        i = self.len_hidden_layers + 1
        Z = A @ self.W[i] + self.b[i]
        self.cache[f"Z{i}"] = Z 
        
        # For prediction, we need probabilities
        if not training:
            softmax = Softmax()
            A = softmax.forward(Z)
            self.cache[f"A{i}"] = A
        
        return Z if training else A
        

    def backward_pass(self, X, y_true):
        """
        Backward propagation to compute gradients.
        
        Computes gradients for all weights and biases including
        regularization terms if applicable.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            y_true: One-hot encoded labels, shape (batch_size, output_size)
        """
        m = X.shape[0]

        y_pred = self.cache[f"A{self.len_hidden_layers + 1}"]

        # derivative of Loss wrt A for Softmax + Cross-entropy layer.
        dA = (y_pred - y_true)

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

            # Adding regularisation on weights.
            if self.regularisation == "L2":
                # L2 regularization: add λ*W to gradient
                self.grads[f"dW{i}"] += (self.reg_lambda / m) * self.W[i]
            elif self.regularisation == "L1":
                # L1 regularization: add λ*sign(W) to gradient
                self.grads[f"dW{i}"] += (self.reg_lambda / m) * np.sign(self.W[i])
            else:
                pass

            # db calculation: Sum across the batch (axis=0). db shape: (1, curr_features)
            self.grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m 


            # gradient to propagate backward
            dA = dZ @ self.W[i].T


    def update_weights(self, lr):
        """
        Updates weights. Called during training for every epoch.
        
        Args:
            lr: Learning rate of the model.
        """
        for i in range(1, self.len_hidden_layers + 2):
            self.W[i] -= lr * self.grads[f"dW{i}"]
            self.b[i] -= lr * self.grads[f"db{i}"]
        

    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data, shape (batch_size, input_size)
            
        Returns:
            Predicted class labels, shape (batch_size,)
        """
        output = self.forward_pass(X, training=False)
        predictions = np.argmax(output, axis=1)
        return predictions
    

    def compute_loss(self, y_true):
        """
        Compute cross-entropy loss using SoftmaxCrossEntropy.
        This is just for monitoring - doesn't affect training.
        """
        """
        Compute total loss (cross-entropy + regularization).
        
        Args:
            y_true: One-hot encoded labels
            
        Returns:
            total_loss = entropy loss + regularisation loss
        """
        # Get the logits (pre-softmax activations) from output layer
        logits = self.cache[f"Z{self.len_hidden_layers + 1}"]

        # Calculate loss
        loss_fn = SoftmaxCrossEntropy()
        cross_entropy_loss = loss_fn.forward(logits, y_true)

        # Storing probs for backward pass
        self.cache[f"A{self.len_hidden_layers + 1}"] = loss_fn.probs
        
        # Applying regularization
        reg_loss = 0.0
        if self.regularisation is not None:
            for i in range(1, self.len_hidden_layers + 2):
                if self.regularisation == "L2":
                    # L2: sum of squared weights
                    reg_loss += np.sum(self.W[i] ** 2)
                elif self.regularisation == "L1":
                    # L1: sum of absolute weights
                    reg_loss += np.sum(np.abs(self.W[i]))
            
            # Scaling reg_loss. Without it total_loss was shown in thousands.
            reg_loss *= self.reg_lambda / (2 if self.regularisation == "L2" else 1)

        total_loss = cross_entropy_loss + reg_loss

        return total_loss
    

    def get_params(self):
        """
        Function to get params for optimizer.
        """
        params = {}
        for i in range(1, self.len_hidden_layers + 2):
            params[f"W{i}"] = self.W[i]
            params[f"b{i}"] = self.b[i]
        return params


    def train(self, X, y, X_val, y_val,
              epochs=50, batch_size=64, lr=0.01, decay=0.0, optimizer=None):
        n = X.shape[0]
        """
        Train the neural network.
        
        Args:
            X: Training data, shape (n_samples, input_size)
            y: Training labels (one-hot), shape (n_samples, output_size)
            X_val: Validation data
            y_val: Validation labels (one-hot)
            epochs: Number of training epochs (default: 1)
            batch_size: Mini-batch size (default: 64)
            lr: Learning rate (default: 0.01)
            decay: Learning rate decay factor (default: 0.0)
            optimizer: Optimizer instance (SGD, SGDMomentum), or None for vanilla SGD
        Returns:
            Dictionary containing training history:
                - 'train_loss': List of training losses per epoch
                - 'train_acc': List of training accuracies per epoch
                - 'val_loss': List of validation losses per epoch
                - 'val_acc': List of validation accuracies per epoch
        """

        # Initialize history tracking
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # 1. Fetch parameters references once if using an optimizer
        if optimizer:
            params = self.get_params()

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            epoch_loss = 0
            num_batches = 0

            # Adding decaying learning rate if decay inputted as parameter (Step decay used).
            curr_lr = lr - (1 + decay * epoch)

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward_pass(X_batch, training=True)

                batch_loss = self.compute_loss(y_batch)
                epoch_loss += batch_loss
                num_batches += 1

                self.backward_pass(X_batch, y_batch)

                # 2. Optimization Step
                if optimizer:
                    # The optimizer updates the arrays inside 'params' in-place.
                    # Since 'params' points to self.W and self.b, the model updates automatically.
                    optimizer.update(params, self.grads)
                else:
                    # Fallback to basic SGD if no optimizer provided
                    self.update_weights(curr_lr)

            # Training accuracy
            train_prediction = self.predict(X)
            y_train_labels = np.argmax(y, axis=1)
            train_acc = np.mean(train_prediction == y_train_labels)
            
            # Validation accuracy
            val_prediction = self.predict(X_val)
            y_val_labels = np.argmax(y_val, axis=1)
            val_acc = np.mean(val_prediction == y_val_labels)

            # Compute validation loss
            self.forward_pass(X_val, training=False)
            val_loss = self.compute_loss(y_val)

            avg_train_loss = epoch_loss / num_batches

            # Store metrics
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1} / {epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {(train_acc * 100):.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {(val_acc * 100):.2f}%")

        print("Training Completed!!!")
        return history