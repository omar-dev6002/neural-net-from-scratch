import numpy as np
from network import NeuralNetwork
from activations import relu, sigmoid


np.random.seed(42)

x = np.array([1.0, 0.5, -1.5])

net = NeuralNetwork(layer_sizes = [3,4,1], activations = [relu, sigmoid] )

output = net.forward(x)

print(f"Network output: {output}")

