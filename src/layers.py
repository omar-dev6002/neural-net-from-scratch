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

    def backward(self, delta, learning_rate):
        '''
        delta: the error term (dL/dz) for THIS layer, already computed
        Returns: dL/dx, the error to pass to the PREVIOUS layer
        '''
        dW = np.outer(delta, self.x)      # dL/dW = delta · x^T
        db = delta                        # dL/db = delta
        dx = np.dot(self.W.T, delta)       # dL/dx = W^T · delta, passed backward

        # Update weights and biases of this layer
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dx
    
