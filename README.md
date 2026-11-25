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

---

# LAB 3: Image Classification (CIFAR-10)

Folder: [`LAB3/`](LAB3/)

| File | Description |
|------|-------------|
| `LAB3_SO_HOEGY_ANDRIAMANAMIRIJA.ipynb` | Executed notebook. The answers to Q3, Q7, Q9 and Q12 are marked with ⭐. |
| `SO_HOEGY_ANDRIAMANAMIRIJA.keras` | Trained CNN (Q6). |
| `train_cnn.py` | Script with no input that trains the same CNN on the training data only and saves it (Q8): `python train_cnn.py` |

**Part 1 — Custom CNN.** CIFAR-10 is loaded, the labels are one-hot encoded and the pixels are scaled to [0, 1]. The model is VGG-style: three blocks of 2 × Conv3×3 + BatchNorm + ReLU, with 32, 64 and 128 filters, each block followed by MaxPool and Dropout (0.2, 0.3, 0.4). These feed a Dense-128 head with dropout 0.5 and a 10-way softmax, for 552k parameters. Training uses data augmentation (random flips and shifts, sources cited in the notebook) and Adam with ReduceLROnPlateau. All choices were made on a 5k validation split taken from the training set. After 30 epochs: **87.0 % validation accuracy and 86.4 % test accuracy**. The weakest classes are cat (68 %) and dog (76 %), which the model mostly confuses with each other. Q9 explains why a new test set would most likely score lower: selection bias from tuning, sampling noise and distribution shift.

**Part 2 — Pretrained models.** MobileNetV2 and ResNet50, both trained on ImageNet, take 224 × 224 × 3 inputs. CIFAR test images are upscaled to that size and passed through each model's own `preprocess_input`. The pipeline runs, but the predictions are mostly wrong: only 3 to 4 of the 12 images get a sensible label from each model. Three things explain this: blurry upscaled 32 × 32 images, ImageNet's 1,000 fine-grained classes that do not match the 10 CIFAR classes, and a different data distribution. Transfer learning (fine-tuning on CIFAR-10) would be the proper fix.
