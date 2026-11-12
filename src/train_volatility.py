import numpy as np
import pandas as pd
from network import NeuralNetwork
from activations import sigmoid, leaky_relu
from losses import mse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- Load the processed dataset ---
data = pd.read_csv("../data/nifty50_model_data.csv", index_col = 0)

features = ['Return_lag1', 'Return_lag2', 'Volatility_lag1', 'Volatility_ma10_lag1']

target = 'Volatility' 



X = data[features].values  # shape(n_samples, 4)
y = data[target].values    # shape(n_samples, )

# --- Chronological train/val/test split (70/10/20, NO shuffling) ---

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- Normalize using TRAIN stats only (never leak test stats into training) ---

X_mean, X_std =  X_train.mean(axis=0), X_train.std(axis=0)

X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

y_min, y_max = y_train.min(), y_train.max()
y_train_norm = (y_train - y_min) / (y_max - y_min)
y_val_norm = (y_val - y_min) / (y_max - y_min)
y_test_norm = (y_test - y_min) / (y_max - y_min)


print(f"X_train_norm mean/std:{X_train_norm.mean():.4f}, {X_train_norm.std():.4f} ")
print(f"y_train_norm mean/std:{y_train_norm.mean():.4f}, {y_train_norm.std():.4f} ")



# --- Build and train the network ---
np.random.seed(42)


net = NeuralNetwork(layer_sizes = [4, 8, 1], activations =[leaky_relu, sigmoid])
# 4 inputs -> 8 hidden neurons (leaky_ReLU) -> 1 output (sigmoid, since i will rescale later)


epochs = 200    # the number of times the entire training dataset is passed through the network
learning_rate = 0.01
l2_lambda = 0.0      # L2 regularization strength — small penalty on large weights

n_samples = len(X_train_norm)


# --- Train with early stopping on VALIDATION loss (keep best weights) ---

best_val_loss = float('inf')
best_weights = None
patience = 30
epochs_without_improvement = 0

for epoch in range(epochs):
    total_loss = 0
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train_norm[indices]
    y_shuffled = y_train_norm[indices]


    for i in range(n_samples):
        x = X_shuffled[i]
        y_true = np.array([y_shuffled[i]])
        y_pred = net.forward(x)
        total_loss += mse(y_pred, y_true)
        net.backward(y_true, learning_rate, l2_lambda= l2_lambda)

    avg_loss = total_loss / n_samples

    # compute validation loss for early stopping
    val_loss = 0.0
    for i in range(len(X_val_norm)):
        y_pred = net.forward(X_val_norm[i])
        val_loss += mse(y_pred, np.array([y_val_norm[i]]))
    val_loss /= len(X_val_norm)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = [(layer.W.copy(), layer.b.copy()) for layer in net.layers]
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch % 20 == 0:
        print(f"Epoch : {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f} (best: {best_val_loss:.6f})")

    if epochs_without_improvement >= patience:
        print(f"\n early stopping at epoch {epoch} - no improvement for {patience} epochs")
        break


#       --- Restore best weights before evaluating ---
for layer, (W,b) in zip(net.layers, best_weights):
    layer.W = W
    layer.b = b


#               --- Evaluate on test set (once, at the end) ---
test_loss = 0.0
predictions = []

for i in range(len(X_test_norm)):
    x = X_test_norm[i]
    y_true = float(y_test_norm[i])
    y_pred = float(net.forward(x)[0])

    predictions.append(y_pred)
    test_loss += (y_pred - y_true) ** 2

test_loss /= len(X_test_norm)
print(f"\nFinal Test Loss (normalized scale): {test_loss:.6f}")

# --- Baseline comparison: naive persistence predictor ---
# predict today's volatility = yesterday's volatility, in the SAME MinMax scale as y
naive_prediction = (X_test[:, 2] - y_min) / (y_max - y_min)
naive_loss = np.mean((naive_prediction - y_test_norm)**2)
print(f"Naive Baseline Loss (persistence, same scale): {naive_loss:.6f}")


print(f"\n---- Comparison ----")
print(f"From-scratch NN Test Loss :   {test_loss:.6f}")
print(f"Naive Baseline Test Loss :    {naive_loss:.6f}")
if naive_loss < test_loss:
    print(f"Persistence baseline beats the network.")
else:
    print(f"Network beats the persistence baseline.")


        
# --- Visualize predictions vs actual volatility ---
# Convert predictions back to actual predictions list (already have `predictions` from earlier)


plt.figure(figsize=(12,5))
plt.plot(y_test_norm, label = 'Actual Volatility (normalized)', color = 'black', linewidth = 1.5)
plt.plot(predictions, label = 'Predictied Volatility (My NN)', color = 'orange', linewidth = 1.5, alpha = 0.8)
plt.title('Nifty 50 realized Volatility: Predictied vs actual (Test set)')
plt.xlabel('Days (Test period)')
plt.ylabel('Volatility (normalized 0 - 1)')
plt.legend()
plt.tight_layout()
plt.savefig('../notebooks/volatility_predictions.png', dpi = 150)
plt.show()
print("\nSaved plot to notebooks/volatility_predictions.png")
