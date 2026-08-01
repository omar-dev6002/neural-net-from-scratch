import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return  np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def sigmoid_derivative(a):
    return a * (1 - a)        # Takes the ALREADY-ACTIVATED value a, not z.
    
