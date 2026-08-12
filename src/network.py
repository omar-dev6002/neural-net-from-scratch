import numpy as np
from layers import Dense
from activations import relu, sigmoid, softmax, leaky_relu
from activations import sigmoid_derivative, relu_derivative, leaky_relu_derivative


# Mapping each activation function to its matching derivative function
DERIVATIVES = {
     sigmoid : sigmoid_derivative,
     relu : relu_derivative,
     leaky_relu : leaky_relu_derivative
}



class NeuralNetwork:
    '''
    Chains Dense layers together with activations, configurable per layer.
    '''
    def __init__(self, layer_sizes, activations):
         """
            layer_sizes: list like [3, 4, 1] -> 3 inputs, 4 hidden neurons, 1 output
            activations: list of activation functions, one per layer (excluding input)
            e.g. [relu, sigmoid] for a hidden layer + output layer
        """
         assert len(layer_sizes) - 1 == len(activations), \
         "Need one activation per layer (not counting the input size)"

         self.layers = []

         for i in range(len(layer_sizes) - 1):
              self.layers.append(Dense(n_inputs = layer_sizes[i], n_neurons = layer_sizes[i+1]))

         self.activations = activations
         


    def forward(self,x):
        """
            x: input data, shape (n_features,)
            returns: output of the network after forward pass
        """

        self.z_values = []   # pre - activation value(needed for derivatives)
        self.a_values = [x]  # Store activations for backward pass, activated outputs, needed for weight gradients 
        for layer , activation in zip(self.layers, self.activations):
             z = layer.forward(x)
             x = activation(z)
             self.z_values.append(z)
             self.a_values.append(x) # Store activated output for backward pass
        return x
    
    def backward(self, y_true, learning_rate, l2_lambda = 0.0):
            """
               y_true: true labels, shape (n_outputs,)
               learning_rate: learning rate for weight updates
            """

            y_pred = self.a_values[-1]   # Get the last activation (output of the network)
            output_activation = self.activations[-1]
            output_derivative = DERIVATIVES[output_activation]
            
            # start the chain: error at the output layer, using z (not a!)
            delta = (y_pred - y_true) * output_derivative(self.z_values[-1])

            # walk backward through layers
            for i in reversed(range(len(self.layers))):
                 delta_prev = self.layers[i].backward(delta, learning_rate, l2_lambda = l2_lambda)
                 if i > 0:
                      hidden_activation = self.activations[i - 1]
                      hidden_derivative = DERIVATIVES[hidden_activation]
                      delta = delta_prev * hidden_derivative(self.z_values[i - 1])

                      



