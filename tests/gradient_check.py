"""
Finite-difference gradient check for Dense layer and NeuralNetwork.
Compares analytical gradients from backward() against central differences.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from layers import Dense
from network import NeuralNetwork
from activations import sigmoid, relu, leaky_relu
from losses import mse


def numerical_gradient_network(net, x, y_true, param_layer_idx, param_name, param_i, param_j=None, eps=1e-7):
    """Central-difference gradient for one scalar parameter in the network."""
    layer = net.layers[param_layer_idx]
    param = getattr(layer, param_name)

    def loss_fn(y_pred):
        if x.ndim == 1:
            return 0.5 * np.sum((y_pred - y_true) ** 2)
        else:
            return 0.5 * np.sum((y_pred - y_true) ** 2) / x.shape[0]

    if param_j is not None:
        old_val = param[param_i, param_j]
        param[param_i, param_j] = old_val + eps
        loss_plus = loss_fn(net.forward(x.copy()))

        param[param_i, param_j] = old_val - eps
        loss_minus = loss_fn(net.forward(x.copy()))

        param[param_i, param_j] = old_val
    else:
        old_val = param[param_i]
        param[param_i] = old_val + eps
        loss_plus = loss_fn(net.forward(x.copy()))

        param[param_i] = old_val - eps
        loss_minus = loss_fn(net.forward(x.copy()))

        param[param_i] = old_val

    return (loss_plus - loss_minus) / (2 * eps)


def check_gradients(activations_list, label, batch_size=None, n_checks=5, eps=1e-7):
    """
    Build a small network, do one forward+backward, then compare analytical
    gradients against central differences for a random subset of parameters.
    If batch_size is None, tests 1D per-sample SGD path; otherwise tests 2D mini-batch path.
    Returns True if all checks pass.
    """
    np.random.seed(42)
    net = NeuralNetwork(layer_sizes=[3, 4, 1], activations=activations_list)
    if batch_size is None:
        x = np.random.randn(3)
        y_true = np.array([np.random.rand()])
    else:
        x = np.random.randn(batch_size, 3)
        y_true = np.random.rand(batch_size, 1)

    # forward + backward to get analytical gradients
    # backward uses (y_pred - y_true) * act'(z), which is the gradient of 0.5*(y-ytrue)^2
    # so our numerical check uses 0.5 * sum((y_pred - y_true)^2) as the loss
    net.forward(x.copy())

    # save weights before backward (backward mutates them)
    saved = []
    for layer in net.layers:
        saved.append((layer.W.copy(), layer.b.copy(), layer.vW.copy(), layer.vb.copy()))

    # we need to extract the analytical gradients before the weight update
    # so we do a manual forward + compute delta + get dW/db without updating
    # Alternatively, we compute numerical gradients and compare against
    # the weight change divided by learning rate, accounting for momentum.
    # Simpler: temporarily set beta=0 and lr=1, then dW = W_before - W_after

    # restore weights
    for layer, (W, b, vW, vb) in zip(net.layers, saved):
        layer.W = W.copy()
        layer.b = b.copy()
        layer.vW = np.zeros_like(W)
        layer.vb = np.zeros_like(b)

    net.forward(x.copy())

    # with beta=0, vW = dW, and W_new = W - lr * dW
    # so dW = (W_old - W_new) / lr
    lr = 1.0
    w_before = [(l.W.copy(), l.b.copy()) for l in net.layers]
    net.backward(y_true, learning_rate=lr, l2_lambda=0.0)
    w_after = [(l.W.copy(), l.b.copy()) for l in net.layers]

    # but backward uses beta=0.9 by default, so we need to account for that
    # vW = beta*0 + (1-beta)*dW = (1-beta)*dW
    # W_new = W - lr * vW = W - lr*(1-beta)*dW
    # dW = (W_old - W_new) / (lr * (1-beta))
    beta = 0.9
    analytical_grads = []
    for (wb, bb), (wa, ba) in zip(w_before, w_after):
        dW = (wb - wa) / (lr * (1 - beta))
        db = (bb - ba) / (lr * (1 - beta))
        analytical_grads.append((dW, db))

    # restore weights for numerical gradient computation
    for layer, (W, b, vW, vb) in zip(net.layers, saved):
        layer.W = W.copy()
        layer.b = b.copy()
        layer.vW = np.zeros_like(W)
        layer.vb = np.zeros_like(b)

    max_diff = 0.0
    errors = []

    for layer_idx in range(len(net.layers)):
        dW_anal, db_anal = analytical_grads[layer_idx]

        # check a few weight entries
        rows, cols = dW_anal.shape
        for _ in range(min(n_checks, rows * cols)):
            i = np.random.randint(rows)
            j = np.random.randint(cols)
            num_grad = numerical_gradient_network(net, x.copy(), y_true, layer_idx, 'W', i, j, eps)
            anal_grad = dW_anal[i, j]
            diff = abs(num_grad - anal_grad)
            denom = max(abs(num_grad) + abs(anal_grad), 1e-15)
            rel_diff = diff / denom
            max_diff = max(max_diff, diff)
            if diff > 1e-5 and rel_diff > 1e-4:
                errors.append(f"  Layer {layer_idx} W[{i},{j}]: analytical={anal_grad:.10f}, numerical={num_grad:.10f}, diff={diff:.2e}")

        # check bias entries
        for i in range(len(db_anal)):
            num_grad = numerical_gradient_network(net, x.copy(), y_true, layer_idx, 'b', i, eps=eps)
            anal_grad = db_anal[i]
            diff = abs(num_grad - anal_grad)
            denom = max(abs(num_grad) + abs(anal_grad), 1e-15)
            rel_diff = diff / denom
            max_diff = max(max_diff, diff)
            if diff > 1e-5 and rel_diff > 1e-4:
                errors.append(f"  Layer {layer_idx} b[{i}]: analytical={anal_grad:.10f}, numerical={num_grad:.10f}, diff={diff:.2e}")

    status = "PASS" if len(errors) == 0 else "FAIL"
    print(f"[{status}] {label}: max abs diff = {max_diff:.2e}")
    for e in errors:
        print(e)
    return len(errors) == 0


def main():
    all_pass = True

    print("--- Gradient checks (per-sample, SGD mode) ---")
    all_pass &= check_gradients([sigmoid, sigmoid], "sigmoid + sigmoid (SGD)")
    all_pass &= check_gradients([relu, sigmoid], "relu + sigmoid (SGD)")
    all_pass &= check_gradients([leaky_relu, sigmoid], "leaky_relu + sigmoid (SGD)")

    print("\n--- Gradient checks (mini-batch mode, batch_size=4) ---")
    all_pass &= check_gradients([sigmoid, sigmoid], "sigmoid + sigmoid (batch=4)", batch_size=4)
    all_pass &= check_gradients([relu, sigmoid], "relu + sigmoid (batch=4)", batch_size=4)
    all_pass &= check_gradients([leaky_relu, sigmoid], "leaky_relu + sigmoid (batch=4)", batch_size=4)

    print()
    if all_pass:
        print("All gradient checks passed.")
    else:
        print("SOME GRADIENT CHECKS FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
