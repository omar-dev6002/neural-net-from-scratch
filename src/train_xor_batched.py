import numpy as np
from network import NeuralNetwork
from activations import sigmoid



np.random.seed(8)


X = np.array([[0,0], [0,1], [1,0], [1,1]])

Y = np.array([0,1,1,0])

net = NeuralNetwork(layer_sizes= [ 2, 4, 1], activations = [sigmoid, sigmoid])

epochs = 3000

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
        Y_batch = Y_shuffled[start:end]


        for x, y in zip(X_batch, Y_batch):
            y_pred = net.forward(x)                      # Forward pass
            total_loss += (y_pred[0] - y) ** 2           # Accumulate loss
            net.backward(np.array([y]), learning_rate)   # backward pass (now uses momentum)

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / 4:.4f}")

print("\nfinal predictions: ")

for x, y in zip(X, Y):
    pred = net.forward(x)
    print(f"Input: {x}, True: {y}, Predictied: [{pred[0]:.4f}]")
    



