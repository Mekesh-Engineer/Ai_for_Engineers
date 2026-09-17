# Vanilla RNN Sequential Data Modeling Project — Experiment 5

## 📋 Overview
A complete, modular Vanilla Recurrent Neural Network (RNN) sequential time-series forecasting pipeline applied to **Household Energy Consumption Forecasting** (2,160 hourly observations across 90 days featuring 24-hour diurnal patterns, morning/evening peak demand, 168-hour weekly load shifts, and temperature-dependent HVAC dynamics). The project constructs a 2-layer Stacked Vanilla RNN architecture (`StackedRNNRegressor`) in PyTorch (`torch.nn.RNN`), utilizes sliding lookback windowing ($w=24$), performs MinMaxScaler normalization, trains with early stopping and gradient clipping, evaluates test regression metrics (MSE, RMSE, MAE, MAPE, $R^2$), calculates residual autocorrelation (ACF), executes 24-hour recursive multi-step forecasting, and produces publication-quality visualizations.

---

## 🗂️ Project Structure
```
rnn_sequential_modeling_project/
├── config/
│   └── hyperparameters.json       # Central configuration file
├── data/
│   ├── raw/
│   │   └── household_energy_consumption.csv # 2,160 hourly energy consumption records
│   └── processed/                 # Processed tensors & arrays
├── models/
│   └── rnn_model.pt               # Saved trained PyTorch Vanilla RNN weights
├── notebooks/
│   └── rnn_training.ipynb         # Interactive Jupyter notebook
├── results/
│   ├── training_history.png       # Loss curves (Train vs Val MSE)
│   ├── predictions_vs_actual.png  # Test set actual vs predicted energy trajectory
│   ├── error_analysis.png         # Residual histogram, scatter & cumulative error
│   ├── autocorrelation_plot.png   # ACF plot of prediction residuals
│   ├── forecast_future.png        # 24-hour recursive future forecast trajectory
│   ├── test_case_verification.png # Single-sequence verification diagnostic plot
│   ├── predictions.csv            # Per-timestep actual vs predicted energy values
│   └── metrics_report.txt         # Comprehensive text evaluation report
├── src/
│   ├── data_loader.py             # DataLoaderModule & energy dataset generator
│   ├── time_series_preprocessing.py # MinMaxScaler & clean interpolation
│   ├── sequence_generator.py      # Sliding window generator (w=24) & split
│   ├── model_builder.py           # StackedRNNRegressor (PyTorch Vanilla RNN)
│   ├── training.py                # Training loop with BPTT & early stopping
│   ├── evaluation.py              # Regression metrics, ACF & future forecast
│   └── visualization.py           # Publication-quality time-series diagnostic plots
├── main.py                        # End-to-end pipeline runner
├── verify_test_case.py            # Test sequence inference & verification tool
└── README.md
```

---

## ⚙️ Installation

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

---

## 🚀 Quick Start

### 1. Run the full end-to-end pipeline
```bash
cd rnn_sequential_modeling_project
py main.py
```

### 2. Override epochs or batch size
```bash
py main.py --epochs 30 --batch_size 64
```

### 3. Verify a specific test sequence window
```bash
py verify_test_case.py --sample_index 0
py verify_test_case.py --sample_index 10
```

### 4. Run the Jupyter notebook
```bash
cd notebooks
jupyter notebook rnn_training.ipynb
```

---

## 🔬 Methodology & Workflow

| Step | Module | Description |
|------|--------|-------------|
| 1. Configuration | `config/` | Define lookback window (24 hrs), batch size (32), learning rate, and paths |
| 2. Data Acquisition | `data_loader.py` | Load 2,160 hourly energy consumption records |
| 3. Preprocessing | `time_series_preprocessing.py` | MinMaxScaler normalization to $[0.0, 1.0]$ and missing step interpolation |
| 4. Sequence Windowing | `sequence_generator.py` | Create 3D sliding window tensors `(N, 24, 1)` & 80/10/10 temporal split |
| 5. RNN Architecture | `model_builder.py` | RNN(64, tanh) $\to$ Dropout(0.2) $\to$ RNN(32, tanh) $\to$ Dropout(0.2) $\to$ Dense(1) |
| 6. Training (BPTT) | `training.py` | Adam optimizer, MSE loss, gradient clipping, early stopping monitoring |
| 7. Evaluation | `evaluation.py` | Inverse scaling, MSE, RMSE, MAE, MAPE, $R^2$, residual autocorrelation |
| 8. Multi-Step Forecast | `evaluation.py` | 24-hour recursive future energy forecasting beyond dataset boundary |
| 9. Visualisation | `visualization.py` | 5 diagnostic plots (curves, trajectory, residuals, ACF, forecast) |

---

## 🏗️ Model Architecture (`StackedRNNRegressor`)

```
Input Sequence Tensor (Batch Size × 24 Timesteps × 1 Feature)
    ↓
[Vanilla RNN Layer 1 (input_size=1, hidden_size=64, nonlinearity='tanh', batch_first=True)]   → Output: (Batch, 24, 64)
    ↓
[Dropout (p=0.2)]
    ↓
[Vanilla RNN Layer 2 (input_size=64, hidden_size=32, nonlinearity='tanh', batch_first=True)]  → Output: (Batch, 24, 32)
    ↓
[Dropout (p=0.2)]
    ↓
[Extract Last Timestep Hidden Vector (-1)]                                                     → Output: (Batch, 32)
    ↓
[Linear Dense Output (32 → 1)]                                                                 → Output: Predicted Next-Hour Energy (kWh)
```

---

## 📊 Verified Performance Benchmarks

| Metric | Measured Model Value |
|--------|-----------------------|
| Root Mean Squared Error (RMSE) | **0.4755 kWh** |
| Mean Absolute Error (MAE) | **0.2827 kWh** |
| Mean Absolute Percentage Error (MAPE) | **13.98%** |
| $R^2$ Forecast Determination Score | **0.6499** |
| Mean Residual Error (Bias) | **-0.0662 kWh** |

---

## 🖼️ Visualisations Produced

1. **Training History Plot**: MSE loss curves over epochs for training and validation sets.
2. **Predictions vs. Actual Energy Trajectory**: Test set actual energy vs. Vanilla RNN predicted energy over time.
3. **Residual Error Analysis**: 3-panel figure showing residual histogram, scatter plot vs. actual, and cumulative absolute error.
4. **Residual Autocorrelation (ACF) Plot**: ACF lag plot confirming residual white-noise properties.
5. **24-Hour Future Forecast Trajectory**: Autoregressive multi-step energy forecast extending 24 hours into the future.
6. **Test Case Verification Diagnostic Plot**: Single sequence inference breakdown and error inspection.

---

## ❓ Viva Voce Questions & Answers

1. **What is a Vanilla Recurrent Neural Network (RNN)?**  
   *Answer*: An RNN is a neural network architecture designed for sequential data processing that maintains an internal recurrent hidden state vector $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ acting as a continuous memory of prior temporal context.

2. **What is Backpropagation Through Time (BPTT)?**  
   *Answer*: BPTT unrolls Recurrent Neural Network feedback loops across temporal sequences, computing gradients backward from the final timestep through all preceding recurrent states using the chain rule of differentiation.

3. **Why do standard Vanilla RNNs suffer from Vanishing Gradients over long sequences?**  
   *Answer*: Error gradients involve matrix multiplication chain products of the recurrent weight matrix $W_{hh}^T$. If the spectral radius (largest eigenvalue) of $W_{hh} < 1$, gradients decay exponentially to zero across extended time steps.

4. **What is the Exploding Gradient problem in RNNs and how is it mitigated?**  
   *Answer*: When recurrent weights exceed unity, backpropagated gradients grow exponentially, causing numerical instability (`NaN`). It is mitigated via **Gradient Norm Clipping** (`torch.nn.utils.clip_grad_norm_`).

5. **Why must time-series data be split sequentially rather than using random train-test splitting?**  
   *Answer*: Random splitting introduces temporal data leakage, allowing the model to observe future timestamps when predicting past ones, yielding unrealistically optimistic metrics.

6. **Explain the 3D tensor input shape required by PyTorch RNN layers: `(batch_size, seq_len, input_size)`.**  
   *Answer*: `batch_size` is the mini-batch sample count; `seq_len` is the historical lookback window length ($w=24$); `input_size` is the number of parallel feature dimensions measured at each step ($1$ for univariate time series).

