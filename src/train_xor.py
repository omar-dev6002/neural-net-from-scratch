import numpy as np
from network import NeuralNetwork
from activations import sigmoid


np.random.seed(7)  # For reproducibility

X  = np.array([[0,0], [0,1], [1,0], [1,1]])

Y =np.array([0,1,1,0])

net =  NeuralNetwork(layer_sizes = [2, 4, 1], activations = [sigmoid, sigmoid])

epochs  = 10000
learning_rate = 1.0

for epoch in range(epochs):
    total_loss = 0

    indices = np.random.permutation(len(X))   # For shuffling the data 

    for idx in indices:
        x , y = X[idx], Y[idx]
        y_pred = net.forward(x)                         # Forward pass
        total_loss += (y_pred[0] - y) ** 2              # Accumlate loss
        net.backward(np.array([y]), learning_rate)     # backward pass

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / 4:.4f}")

print("\nFinal predictions: ")
for x, y in zip(X, Y):
    pred = net.forward(x)
    print(f"Input: {x}, True: {y}, Predictied: [{pred[0]:.4f}]")

