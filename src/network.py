import numpy as np
from layers import Dense
from activations import relu, sigmoid, softmax

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
        for layer , activation in zip(self.layers, self.activations):
             z = layer.forward(x)
             x = activation(z)
        return x
    
    