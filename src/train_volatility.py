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



def train_sgd(X_tr, y_tr, X_va, y_va, epochs=200, lr=0.01, l2_lambda=0.0, patience=30, log_every=None):
    np.random.seed(42)
    net = NeuralNetwork(layer_sizes=[4, 8, 1], activations=[leaky_relu, sigmoid])
    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0
    n_samples = len(X_tr)

    for epoch in range(epochs):
        total_loss = 0.0
        indices = np.random.permutation(n_samples)
        for i in indices:
            y_true = np.array([y_tr[i]])
            y_pred = net.forward(X_tr[i])
            total_loss += mse(y_pred, y_true)
            net.backward(y_true, lr, l2_lambda=l2_lambda)

        avg_loss = total_loss / n_samples
        val_loss = np.mean([
            mse(net.forward(X_va[i]), np.array([y_va[i]]))
            for i in range(len(X_va))
        ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [(layer.W.copy(), layer.b.copy()) for layer in net.layers]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if log_every and epoch % log_every == 0:
            print(f"Epoch : {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f} (best: {best_val_loss:.6f})")

        if epochs_without_improvement >= patience:
            if log_every:
                print(f"\n early stopping at epoch {epoch} - no improvement for {patience} epochs")
            break

    for layer, (W, b) in zip(net.layers, best_weights):
        layer.W = W
        layer.b = b
    return net


# --- Train direct model with early stopping on VALIDATION loss ---
epochs = 200
learning_rate = 0.01
l2_lambda = 0.0
patience = 30

net = train_sgd(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm,
    epochs=epochs, lr=learning_rate, l2_lambda=l2_lambda, patience=patience, log_every=20
)


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


# --- Follow-up 1: predict change in volatility (y_t - Volatility_lag1) ---
dy_train = y_train - X_train[:, 2]
dy_val = y_val - X_val[:, 2]
dy_min, dy_max = dy_train.min(), dy_train.max()
dy_train_norm = (dy_train - dy_min) / (dy_max - dy_min)
dy_val_norm = (dy_val - dy_min) / (dy_max - dy_min)

net_res = train_sgd(
    X_train_norm, dy_train_norm, X_val_norm, dy_val_norm,
    epochs=epochs, lr=learning_rate, l2_lambda=l2_lambda, patience=patience
)
dy_preds_norm = np.array([float(net_res.forward(X_test_norm[i])[0]) for i in range(len(X_test_norm))])
dy_preds = dy_preds_norm * (dy_max - dy_min) + dy_min
res_preds_norm = ((X_test[:, 2] + dy_preds) - y_min) / (y_max - y_min)
res_test_loss = np.mean((res_preds_norm - y_test_norm) ** 2)
print(f"Residual NN (y_t - y_{{t-1}}) Test Loss : {res_test_loss:.6f}")


# --- Follow-up 2: predict log-volatility ---
ly_train = np.log(y_train)
ly_val = np.log(y_val)
ly_min, ly_max = ly_train.min(), ly_train.max()
ly_train_norm = (ly_train - ly_min) / (ly_max - ly_min)
ly_val_norm = (ly_val - ly_min) / (ly_max - ly_min)

net_log = train_sgd(
    X_train_norm, ly_train_norm, X_val_norm, ly_val_norm,
    epochs=epochs, lr=learning_rate, l2_lambda=l2_lambda, patience=patience
)
ly_preds_norm = np.array([float(net_log.forward(X_test_norm[i])[0]) for i in range(len(X_test_norm))])
ly_preds = ly_preds_norm * (ly_max - ly_min) + ly_min
log_preds_norm = (np.exp(ly_preds) - y_min) / (y_max - y_min)
log_test_loss = np.mean((log_preds_norm - y_test_norm) ** 2)
print(f"Log-volatility NN Test Loss :           {log_test_loss:.6f}")


        
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
