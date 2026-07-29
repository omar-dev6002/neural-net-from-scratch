import numpy as np
from layers import Dense 
from activations import sigmoid, relu

np.random.seed(42)

x = np.array([1.0, 0.5, -1.5])

layer_1 = Dense(n_inputs = 3, n_neurons = 4)  # hidden layer: 3 in , 4 neurons

z_1 = layer_1.forward(x)
a_1 = relu(z_1)

print(f"z1 (pre-activation,4 neurons): {z_1}")
print(f"a1 (after ReLU): {a_1}")


             