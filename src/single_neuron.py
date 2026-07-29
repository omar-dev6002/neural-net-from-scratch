import numpy as np
from activations import sigmoid 


# One neuron , 3 i/p
np.random.seed(42)                   
x =  np.array([1.0, 0.5, -1.5])      # i/p feature
w = np.random.randn(3)               # weights, one per input feature 
b = np.random.randn()                # bias


# Forward pass through a single neuron
z = np.dot(w,x) + b
a =  sigmoid(z)


print(f"z (pre activation): {z:.4f}")
print(f"a (activated o/p): {a:.4f}")

