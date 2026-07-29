import numpy as np

class Dense:
    '''
    A fully connected (dense) layer: every i/p connects to every neuron 
    '''
    def __init__(self, n_inputs, n_neurons):

        # Small random weights to start 
        self.W = 0.01 * np.random.randn(n_neurons, n_inputs)
        self.b = np.zeros(n_neurons)

    def forward(self,x):

        self.x = x
        self.z = np.dot(self.W,x) + self.b
        return self.z

