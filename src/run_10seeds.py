import sys
import os
import time
import numpy as np
import pandas as pd
from network import NeuralNetwork
from activations import relu, sigmoid, leaky_relu

# Resolve dataset path relative to this file
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty50_model_data.csv")
data = pd.read_csv(data_path, index_col=0)

f4 = ['Return_lag1', 'Return_lag2', 'Volatility_lag1', 'Volatility_ma10_lag1']
f3 = ['Return_lag1', 'Return_lag2', 'Volatility_lag1']
y = data['Volatility'].values

# Chronological 70/10/20 split
n = len(y)
te, ve = int(n * 0.7), int(n * 0.8)
y_tr, y_va, y_te = y[:te], y[te:ve], y[ve:]
ymin, ymax = y_tr.min(), y_tr.max()
y_tr_n = (y_tr - ymin) / (ymax - ymin)
y_va_n = (y_va - ymin) / (ymax - ymin)
y_te_n = (y_te - ymin) / (ymax - ymin)


def prep_X(cols):
    X = data[cols].values
    X_tr, X_va, X_te = X[:te], X[te:ve], X[ve:]
    Xm, Xs = X_tr.mean(0), X_tr.std(0)
    return X_tr, X_va, X_te, (X_tr - Xm) / Xs, (X_va - Xm) / Xs, (X_te - Xm) / Xs


def run_sgd(X_tr, y_tr, X_va, y_va, act, seed):
    np.random.seed(seed)
    net = NeuralNetwork([X_tr.shape[1], 8, 1], [act, sigmoid])
    y_tr_list = [np.array([v]) for v in y_tr]
    y_va_2d = y_va.reshape(-1, 1)
    best_val, best_w, no_imp = 1e9, None, 0
    for ep in range(200):
        for i in np.random.permutation(len(X_tr)):
            net.forward(X_tr[i])
            net.backward(y_tr_list[i], 0.01)
        vl = np.mean((net.forward(X_va) - y_va_2d) ** 2)
        if vl < best_val:
            best_val, best_w, no_imp = vl, [(l.W.copy(), l.b.copy()) for l in net.layers], 0
        else:
            no_imp += 1
        if no_imp >= 30:
            break
    for l, (W, b) in zip(net.layers, best_w):
        l.W, l.b = W, b
    return net


def evaluate_mode(mode):
    t0 = time.time()
    mses = []
    if mode in ("4_lrelu", "4_relu", "3_lrelu", "3_relu"):
        cols = f4 if mode.startswith("4_") else f3
        act = leaky_relu if mode.endswith("_lrelu") else relu
        _, _, _, Xtr_n, Xva_n, Xte_n = prep_X(cols)
        for seed in range(10):
            net = run_sgd(Xtr_n, y_tr_n, Xva_n, y_va_n, act, seed)
            mses.append(np.mean((net.forward(Xte_n).flatten() - y_te_n) ** 2))
    elif mode == "residual":
        X_tr, X_va, X_te, Xtr_n, Xva_n, Xte_n = prep_X(f4)
        dy_tr, dy_va = y_tr - X_tr[:, 2], y_va - X_va[:, 2]
        dy_min, dy_max = dy_tr.min(), dy_tr.max()
        dy_tr_n, dy_va_n = (dy_tr - dy_min) / (dy_max - dy_min), (dy_va - dy_min) / (dy_max - dy_min)
        for seed in range(10):
            net = run_sgd(Xtr_n, dy_tr_n, Xva_n, dy_va_n, leaky_relu, seed)
            dy_p = net.forward(Xte_n).flatten() * (dy_max - dy_min) + dy_min
            y_p_n = ((X_te[:, 2] + dy_p) - ymin) / (ymax - ymin)
            mses.append(np.mean((y_p_n - y_te_n) ** 2))
    elif mode == "logvol":
        _, _, _, Xtr_n, Xva_n, Xte_n = prep_X(f4)
        ly_tr, ly_va = np.log(y_tr), np.log(y_va)
        ly_min, ly_max = ly_tr.min(), ly_tr.max()
        ly_tr_n, ly_va_n = (ly_tr - ly_min) / (ly_max - ly_min), (ly_va - ly_min) / (ly_max - ly_min)
        for seed in range(10):
            net = run_sgd(Xtr_n, ly_tr_n, Xva_n, ly_va_n, leaky_relu, seed)
            ly_p = net.forward(Xte_n).flatten() * (ly_max - ly_min) + ly_min
            y_p_n = (np.exp(ly_p) - ymin) / (ymax - ymin)
            mses.append(np.mean((y_p_n - y_te_n) ** 2))
    print(f"{mode:12s}: mean={np.mean(mses):.6f}, std={np.std(mses):.6f} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["4_lrelu", "4_relu", "3_lrelu", "3_relu", "residual", "logvol"]
    _, _, X4_te, _, _, _ = prep_X(f4)
    naive_n = (X4_te[:, 2] - ymin) / (ymax - ymin)
    print(f"persistence : mean={np.mean((naive_n - y_te_n)**2):.6f}, std=0.000000", flush=True)
    for m in modes:
        evaluate_mode(m)

