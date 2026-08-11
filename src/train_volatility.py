import numpy as np
import pandas as pd
from network import NeuralNetwork
from activations import relu, sigmoid 
from losses import mse, mse_derivative 
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


# --- Load the processed dataset ---
data = pd.read_csv("../data/nifty50_model_data.csv", index_col = 0)
data = data.dropna()   # remove leftover header artifact rows from the MultiIndex CSV export

features = ['Return_lag1', 'Return_lag2', 'Volatility_lag1']

target = 'Volatility' 



X = data[features].values  # shape(n_samples, 3)
y = data[target].values    # shape(n_samples, )

# --- Chronological train/test split (80/20, NO shuffling) ---

split_idx = int(len(X) * 0.8)   # split index for 80% train and 20% test

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

# --- Normalize using TRAIN stats only (never leak test stats into training) ---

X_mean, X_std =  X_train.mean(axis=0), X_train.std(axis=0)
y_mean, y_std =  y_train.mean(axis=0), y_train.std(axis=0)

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

y_min, y_max = y_train.min(), y_train.max()
y_train_norm = (y_train - y_min) / (y_max - y_min)
y_test_norm = (y_test - y_min) / (y_max - y_min)


print(f"X_train_norm mean/std:{X_train_norm.mean():.4f}, {X_train_norm.std():.4f} ")
print(f"y_train_norm mean/std:{y_train_norm.mean():.4f}, {y_train_norm.std():.4f} ")


print(data.head())
print(data.dtypes)
print(f"\nNaN count per column:\n{data.isna().sum()}")



# --- Build and train the network ---
np.random.seed(42)


net = NeuralNetwork(layer_sizes = [3, 8, 1], activations =[relu, sigmoid])
# 3 inputs -> 8 hidden neurons (ReLU) -> 1 output (sigmoid, since i will rescale later)


epochs = 200    # the number of times the entire training dataset is passed through the network
learning_rate = 0.01
batch_size = 32
l2_lambda = 0.00001      # L2 regularization strength — small penalty on large weights

n_samples = len(X_train_norm)


# --- Train with early stopping (keep best weights) ---

best_loss = float('inf')
best_weights = None
patience = 30
epochs_witout_improvement = 0

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

    if avg_loss < best_loss :
        best_loss = avg_loss
        best_weights = [(layer.W.copy(), layer.b.copy()) for layer in net.layers]
        epochs_witout_improvement = 0
    else:
        epochs_witout_improvement += 1

    if epoch % 20 == 0:
        print(f"Epoch : {epoch}, Train Loss: {avg_loss:.6f} (best: {best_loss:.6f})")

    if epochs_witout_improvement >= patience:
        print(f"\n early stopping at epoch {epoch} - no improvement for {patience} epochs")
        break


#       --- Restore best weights before evaluating ---
for layer, (W,b) in zip(net.layers, best_weights):
    layer.W = W
    layer.b = b


#               --- Evaluate on test set ---
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

# --- Baseline comparison: naive "tomorrow = today" predictor ---
naive_prediction = X_test_norm[:, 2]                   # Volatility_lag1 IS our naive guess for today's volatility
naive_loss = np.mean((naive_prediction - y_test_norm)**2)
print(f"Naive Baseline Loss (predict using yesterday's volatility): {naive_loss:.6f}")



# --- Sanity check against sklearn's MLPRegressor ---


sklearn_model = MLPRegressor(
    hidden_layer_sizes = (8,),
    activation = 'relu',
    max_iter = 1000,
    random_state = 42  
)

sklearn_model.fit(X_train_norm, y_train_norm)
sklearn_preds = sklearn_model.predict(X_test_norm)
sklearn_test_loss = np.mean((sklearn_preds - y_test_norm) ** 2)



print(f"\n---- Comparision ----")
print(f"Your from - scratch NN Test Loss :   {test_loss:.6f}")
print(f"sklearn MLPRegressor Test Loss :     {sklearn_test_loss:.6f}")
print(f"Naive Baseline Test Loss :           {naive_loss:.6f}")


        
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
