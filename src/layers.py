import numpy as np

class Dense:
    '''
    A fully connected (dense) layer: every i/p connects to every neuron 
    '''
    def __init__(self, n_inputs, n_neurons):

        # Small random weights to start 
        self.W = 0.01 * np.random.randn(n_neurons, n_inputs)
        self.b = np.zeros(n_neurons)

        # day 5
        # momentum "velocity" terms — same shape as weights/biases, start at zero
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b) 
        

    def forward(self,x):

        self.x = x
        self.z = np.dot(self.W,x) + self.b
        return self.z

    

    def backward(self, delta, learning_rate, beta = 0.9, l2_lambda = 0.0):           # l2_lambda=0.0 default means regularization is "off" unless you explicitly turn it on
        '''
        delta: the error term (dL/dz) for THIS layer, already computed
        l2_lambda: L2 regularization strength (0.0 = no regularization)
        Returns: dL/dx, the error to pass to the PREVIOUS layer
        '''
        
        dW = np.outer(delta, self.x)      # dL/dW = delta · x^T
        db = delta                        # dL/db = delta
        dx = np.dot(self.W.T, delta)       # dL/dx = W^T · delta, passed backward


        # L2 regularization: add penalty gradient (2 * lambda * W) to discourage large weights

        dW += 2 * l2_lambda * self.W


        # Day 5: Momentum update
        # momentum update: blend new gradient with past velocity

        self.vW = beta * self.vW + (1 - beta) * dW
        self.vb = beta * self.vb + (1 - beta) * db

        

        # Update weights and biases of this layer
        self.W -= learning_rate * self.vW
        self.b -= learning_rate * self.vb

        return dx
    