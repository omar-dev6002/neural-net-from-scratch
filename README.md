# Neural Network From Scratch — Nifty 50 Volatility Forecasting

A neural network built entirely from scratch in NumPy — forward pass,
backpropagation, mini-batch gradient descent, and momentum all
hand-derived and implemented without any ML framework — applied to
forecasting realized volatility on the Nifty 50 index.

## Why this project

Most "NN from scratch" projects stop at MNIST. This one goes further in
two ways: the math is fully hand-derived (see `derivations/derivations.md`
for photographed, worked-out calculus across 11 days of development), and
the applied case study uses real Indian market data (Nifty 50) rather
than a toy dataset — a genuine regression forecasting problem, not
classification.

## What's implemented

- Single neuron → configurable `Dense` layer → chainable `NeuralNetwork` class
- Manually derived and coded backpropagation (verified via XOR)
- Mini-batch gradient descent with momentum
- Multiple activation functions (Sigmoid, ReLU, Leaky ReLU) with a
  generalized derivative-lookup system, not hardcoded to one activation
- Manual L2 regularization
- Applied to Nifty 50 realized volatility forecasting:
  - Real data pulled via `yfinance`
  - Lagged features engineered with strict no-lookahead-bias discipline
  - Chronological train/test split (no shuffling across time)
  - Min-max target normalization (matched to sigmoid's output range)
  - Early stopping with best-weight restoration

## Results

| Model | Test Loss (normalized MSE) |
|---|---|
| Naive baseline (predict yesterday's volatility) | 0.346384 |
| 3 features, ReLU | 0.000777 |
| 4 features, ReLU | 0.001255 (worse — dying ReLU, see below) |
| **4 features, Leaky ReLU (final model)** | **0.000177** |
| sklearn MLPRegressor (default hyperparameters) | 0.005700 |

The final model is **~1950x better than the naive baseline**. Note: the
sklearn comparison is not fully controlled (different optimizer, learning
rate schedule, and early stopping setup) — see `derivations/derivations.md`
(Day 8) for a full breakdown. It should not be read as "hand-rolled SGD
beats Adam."

![Predictions vs actual](notebooks/volatility_predictions.png)

### The debugging story behind the final result

This result came from a genuine multi-day debugging arc, not a single
attempt:

1. **Day 7-8:** built a 3-feature model, validated it beats naive and
   sklearn baselines (with caveats).
2. **Day 9:** hypothesized that adding a longer-window volatility average
   would help the model distinguish calm periods from post-spike
   settling. Tested it — performance got *worse*, and predictions became
   a flat line during calm periods. Diagnosed this as dying ReLU: hidden
   neurons permanently outputting zero for that input region.
3. **Day 10:** tried L2 regularization to address instability seen in
   training curves. Found it either had no effect (too weak) or
   collapsed the model to near-constant predictions (too strong) —
   another honestly-reported negative result.
4. **Day 11:** revisited the Day 9 hypothesis, suspecting the *feature*
   wasn't the problem — the *activation function* was. Switched the
   hidden layer to Leaky ReLU (allows small negative gradients, so
   neurons can't get permanently stuck) and re-tested the 4-feature
   model. Result: the original hypothesis was correct all along — it had
   been masked by a dead-neuron bug, not disproven.

Full derivations and reasoning for each day are in
`derivations/derivations.md`.

## Project structure

src/ - all runnable code (layers, network, training scripts)
data/ - raw and processed Nifty 50 datasets
derivations/ - hand-derived math and debugging notes, photographed, day by day
notebooks/ - plots and exploratory analysis

## Running it

```bash
cd src
pip install -r requirements.txt
python data_pipeline.py       # pulls and processes Nifty 50 data
python train_volatility.py    # trains and evaluates the final model
```