import numpy as np
from network import NeuralNetwork
from activations import sigmoid



np.random.seed(8)


X = np.array([[0,0], [0,1], [1,0], [1,1]])

Y = np.array([0,1,1,0])

net = NeuralNetwork(layer_sizes= [ 2, 4, 1], activations = [sigmoid, sigmoid])

epochs = 10000

learning_rate = 1.0

batch_size = 2

for epoch in range(epochs):
    total_loss = 0

    indices = np.random.permutation(len(X))    # For shuffling the data

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # process in mini batches 

    for start in range (0, len(X), batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end].reshape(-1, 1)

        y_pred = net.forward(X_batch)                    # Batched forward pass
        total_loss += np.sum((y_pred - Y_batch) ** 2)    # Accumulate loss
        net.backward(Y_batch, learning_rate)             # One weight update per batch

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / 4:.4f}")

print("\nfinal predictions: ")

for x, y in zip(X, Y):
    pred = net.forward(x)
    print(f"Input: {x}, True: {y}, Predictied: [{pred[0]:.4f}]")
    



