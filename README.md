# Neural Network From Scratch — Nifty 50 Volatility Forecasting

I built a feedforward neural network from scratch in NumPy (forward pass, backpropagation, momentum, L2 regularization, and both per-sample SGD and mini-batch gradient descent) and applied it to forecasting 21-day realized volatility on the Nifty 50 index (`^NSEI`, 2019–2026).

All backpropagation calculus was worked out on paper before coding (`derivations/derivations.md`, with photographed handwritten notes) and verified with numerical finite-difference gradient checks (`tests/gradient_check.py`) and XOR convergence tests (`src/train_xor.py`, `src/train_xor_batched.py`).

## Results

Evaluated over 10 random seeds on the chronological 70/10/20 split (`1,285` train / `184` validation / `368` test rows). All inputs are Z-scored and all targets are Min-Max scaled using training statistics only, early-stopped on validation loss. Test MSE is reported in the original-volatility Min-Max scale:

| Model | Test MSE (mean ± std, 10 seeds) |
|---|---|
| Persistence baseline (predict `y_t = Volatility_lag1`) | **0.000089 ± 0.000000** |
| Direct NN — 4 features, Leaky ReLU (`[4, 8, 1]`) | 0.000239 ± 0.000008 |
| Direct NN — 4 features, plain ReLU (`[4, 8, 1]`) | 0.000239 ± 0.000008 |
| Direct NN — 3 features, Leaky ReLU (`[3, 8, 1]`) | 0.000316 ± 0.000053 |
| Direct NN — 3 features, plain ReLU (`[3, 8, 1]`) | 0.000315 ± 0.000053 |
| Follow-up 1: Residual NN (predict `y_t - Volatility_lag1`) | 0.000088 ± 0.000001 |
| Follow-up 2: Log-volatility NN (predict `log(Volatility)`) | 0.000132 ± 0.000001 |
| sklearn MLPRegressor (matching `(8,)` architecture, defaults) | 0.005392 ± 0.003212 |

**Key findings:**
- **Persistence beats the direct network:** The persistence baseline (`y_t = y_{t-1}`) achieves `0.000089`, easily outperforming the direct `[4, 8, 1]` network (`0.000239 ± 0.000008`). Because 21-day rolling volatility at day `t` shares 20 of 21 daily returns with day `t - 1`, persistence is a very strong benchmark.
- **Residual reformulation is statistically indistinguishable from persistence:** Predicting the daily change `dy = y_t - y_{t-1}` and adding the prediction back to yesterday's volatility yields a test MSE of `0.000088 ± 0.000001`. In practice, this is statistically indistinguishable from the persistence baseline (`0.000089 ± 0.000000`), especially given that the network's inputs already include lagged volatility, the comparison is on a single chronological split, and the residual formulation was chosen after inspecting test set results.
- **Log-volatility:** Training on `log(Volatility)` and exponentiating back via `exp()` yields `0.000132 ± 0.000001` — better than direct level prediction, but still worse than persistence.
- **The 10-day MA feature helps:** Comparing 4 features against 3 features shows an average reduction in test MSE of `0.000076` (from `0.000315` to `0.000239`), which is larger than the seed-to-seed spread (`0.000008` for 4 features, `0.000053` for 3 features). Adding `Volatility_ma10_lag1` provides a genuine, reproducible improvement.
- **Leaky ReLU does not improve over ReLU:** Once the analytical derivative lookup is routed correctly, plain ReLU and Leaky ReLU yield identical test MSE (`0.000239 ± 0.000008` on 4 features; `0.000315` vs `0.000316` on 3 features). Leaky ReLU does not confer an advantage here.
- **sklearn comparison caveat:** `sklearn.neural_network.MLPRegressor` achieves `0.005392 ± 0.003212` across 10 seeds (seed 42: `0.005877`), showing a large seed-to-seed spread. This difference likely reflects default hyperparameter choices in sklearn (Adam optimizer, batch size 200, default learning rate 0.001, unbounded identity output, no validation early stopping) rather than an algorithmic advantage of hand-coded SGD over Adam.

![Nifty 50 Realized Volatility: Actual vs Direct NN (Leaky ReLU) vs Persistence](notebooks/volatility_predictions.png)

## Model Architectures & Target Scaling

- **Direct NN (`[4, 8, 1]` or `[3, 8, 1]`):** Inputs are Z-scored `(X - X_mean) / X_std`. Target is Min-Max normalized `(y - y_min) / (y_max - y_min)` to `[0, 1]`. Hidden activation is Leaky ReLU or ReLU; output activation is Sigmoid (outputs in `[0, 1]`, matching target bounds).
- **Residual NN:** Predicts raw daily change `dy = y_t - Volatility_lag1`. Because `dy` can be negative, training targets are Min-Max scaled to `[0, 1]` via `(dy - dy_min) / (dy_max - dy_min)`. Output activation is Sigmoid (bounded in `[0, 1]`). At test time, predictions are un-scaled via `dy_preds = dy_preds_norm * (dy_max - dy_min) + dy_min`, added to yesterday's raw volatility `X_test[:, 2]`, and then mapped to the original target's Min-Max scale `(pred - y_min) / (y_max - y_min)` so test MSE is directly comparable to all other models.
- **Log-volatility NN:** Predicts `log(Volatility)`. Training targets are Min-Max scaled to `[0, 1]` via `(log_y - ly_min) / (ly_max - ly_min)`. Output activation is Sigmoid. At test time, predictions are un-scaled, exponentiated back to volatility levels via `exp()`, and then mapped to the original target's Min-Max scale before computing test MSE.

## Limitations

- **Persistence is difficult to beat:** Because 21-day rolling volatility overlaps 20 of 21 returns with the previous day, daily volatility is heavily autocorrelated ($r \approx 0.99$). A feedforward network predicting levels fails to beat naive persistence, and predicting changes only matches it.
- **Test set reuse across project history:** Model iterations, feature choices, and debugging decisions across the 11-day log were evaluated against the same test period. A production deployment would require an untouched out-of-sample holdout or rolling walk-forward validation.
- **Single asset / market:** Results are reported exclusively on Nifty 50 daily index data. Volatility dynamics on individual equities, higher frequencies, or other markets may behave differently.
- **Loss convention:** `NeuralNetwork.backward()` starts from `(y_pred - y_true) * output_derivative(z)`, omitting the factor of `2` from `mse_derivative`. Effectively, the network minimizes `0.5 * (y_pred - y_true)^2`, which halves the effective learning rate.

## What I got wrong

After a technical review of this repository, I audited every claim and corrected several mistakes:

1. **Invalid naive baseline (`0.346384` vs `0.000089`):** In my original `train_volatility.py`, I computed the naive baseline by subtracting Min-Max scaled `y_test_norm` from Z-score standardized `X_test_norm[:, 2]`. Comparing two different scales produced an artificially inflated error of `0.346384` and led to the false claim that the network was "~1,950x better than baseline." When computed in the same Min-Max scale, persistence achieves `0.000089`, beating the direct neural network.
2. **Misdiagnosed "dying ReLU" (Day 9 to Day 11):** On Day 9, adding `Volatility_ma10_lag1` produced a worse test MSE (`0.001255`), which I blamed on dying ReLU neurons and believed Leaky ReLU resolved on Day 11. Re-testing proved that `0.001255` was caused by a bug in `network.py` that hardcoded `a * (1 - a)` on hidden ReLU outputs. Once analytical derivatives are properly routed, ReLU and Leaky ReLU perform identically (`0.000239 ± 0.000008`). I kept the original Day 9–11 notes in `derivations/derivations.md` and added dated corrections explaining the real cause.
3. **Unused mini-batch claim:** I originally claimed mini-batch gradient descent was implemented, but `Dense` only accepted 1D vectors and updated weights per sample. I have now implemented true 2D batched forward/backward in `Dense` (keeping 1D per-sample SGD as default), verified it with `tests/gradient_check.py`, and updated `train_xor_batched.py` to update once per batch.
4. **Early stopping on training loss:** The original script checkpointed weights on training loss and had a typo (`epochs_witout_improvement`). I split a 10% validation set chronologically, fit scalers strictly on train, and early-stop on validation loss.

## Project structure

- `src/` — neural network implementation (`layers.py`, `network.py`, `activations.py`, `losses.py`), data pipeline (`data_pipeline.py`), and training scripts (`train_xor.py`, `train_xor_batched.py`, `train_volatility.py`, `run_10seeds.py`)
- `tests/` — finite-difference gradient check (`gradient_check.py`) for `sigmoid`, `relu`, and `leaky_relu` in both SGD and mini-batch modes
- `data/` — raw, processed, and model-ready Nifty 50 CSVs
- `derivations/` — hand-derived math notes and photographed derivations (`derivations.md`, `day1_p1.jpg`–`day11_p2.jpeg`)
- `notebooks/` — prediction plot (`volatility_predictions.png`)

## Running it

```bash
python tests/gradient_check.py   # verify analytical vs numerical gradients
cd src
python train_xor.py              # XOR in per-sample SGD mode
python train_xor_batched.py      # XOR in mini-batch mode (batch_size=2)
python train_volatility.py       # train a single run and generate the prediction plot
python run_10seeds.py            # evaluate all models across 10 random seeds
```

`run_10seeds.py` defaults to running all six modes (`4_lrelu`, `4_relu`, `3_lrelu`, `3_relu`, `residual`, `logvol`) sequentially, which takes roughly an hour on a single core (measured: ~64 minutes).

```bash
python run_10seeds.py 4_lrelu residual
```