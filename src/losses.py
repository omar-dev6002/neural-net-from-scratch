import numpy as np 

def mse(y_pred, y_true):
    '''
    Mean Squared Error loss function
    '''
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size
