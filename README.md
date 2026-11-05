# Deep Learning — LAB 1: Neural Networks for Regression

ESIEA × SKEMA · MSc AIBT

Lab exploring **regression with neural networks** (`MLPRegressor` from scikit-learn), compared against linear regression, with a focus on the training process and model selection.

## Contents

| File | Description |
|------|-------------|
| `LAB1_neural_networks_regression.ipynb` | The completed lab notebook (code + answers). |
| `LAB1.pdf` | The lab subject / instructions. |
| `LAB1_datasets/` | `LAB1_dataset1.txt` (167 samples) and `LAB1_dataset2.txt` (177 samples), each with 1 feature and 1 target. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
jupyter notebook LAB1_neural_networks_regression.ipynb
```

## Summary of the work

**Part 1 — Dataset.** Loaded both datasets and plotted them. Dataset 1 is essentially linear (a simple linear regression reaches R² ≈ 0.95), so an MLP is **not** appropriate for it — a 2-parameter linear model already captures it. Dataset 2 is clearly non-linear, which makes it the right candidate for a neural network.

**Part 2 — Visualization.** Built a `plot_predictions` helper (true values vs. predictions). Confirmed the linear fit on dataset 1 (y ≈ −1.16x + 2.00, R² ≈ 0.95). Trained an MLP (one hidden layer, 30 ReLU neurons) on dataset 2: R² ≈ 0.84 vs. 0.12 for a linear model, producing the expected piecewise-linear curve. Counted the network's parameters: **91 total** (60 weights + 31 biases).

**Part 3 — Training process.** Analysed the loss curve (model slightly undertrained at the stopping point). Compared SGD with several momentum values:
- momentum 0.3 → very slow, underfitted;
- momentum 0.999 → fast but unstable, oscillates and overshoots;
- **momentum 0.99 → best compromise** (fast, smooth convergence).

**Part 4 — Model selection.** Used a train/test split and studied how the test score depends on the hidden-layer size. Averaging R² over 20 random seeds for 2–30 neurons showed that performance plateaus (mean R² ≈ 0.82, low variance) from about **15 neurons**, which is the chosen architecture: enough capacity without being sensitive to initialisation.
