import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return  np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def sigmoid_derivative(z):
    a  = sigmoid(z)
    return a * (1 - a)         

def leaky_relu(z, alpha = 0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha = 0.01 ):
    return np.where(z > 0, 1, alpha)

