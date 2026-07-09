# eeg-eye-state-classification
EEG eye-state classification using LSTM, Random Forest, and a custom Transformer architecture with SHAP interpretability and Diagnostic Attention Heatmaps.

## Overview
This project builds an EEG eye-state classification pipeline on the UCI EEG Eye State dataset, comparing Random Forest models across four feature representations against an LSTM and a custom temporal Transformer on raw EEG.

## Results
| Model | Features | ROC-AUC | Std |
|-------|----------|---------|-----|
| Random Forest | FFT Bandpower | 0.52 | 0.065 |
| Random Forest | Alpha/Beta Ratio | 0.46 | 0.058 |
| Random Forest | Combined | 0.59 | 0.039 |
| Random Forest | Statistical | 0.62 | 0.028 |
| LSTM | Raw EEG | 0.68 | 0.082 |
| Transformer | Raw EEG (Windows + Instance Norm) | 0.70 | 0.115 |

## Key Findings
1. Statistical time-domain features outperform spectral features for classical trees.
2. A custom temporal Transformer analyzing raw, uncompressed voltage windows achieves the best overall performance (**0.70 ROC-AUC**), outperforming LSTM (**0.68**) and Random Forest (**0.62**) baselines.
3. Attention heatmaps show that individual Transformer heads specialize natively, as some performed Alpha/Beta frequency banding and others globally broadcasted frontal oculomotor blinking spikes.
4. SHAP analysis revealed that classification is driven primarily by oculomotor artifacts at frontal electrodes rather than occipital alpha rhythm suppression, suggesting consumer-grade EEG hardware cannot reliably capture the alpha-blocking phenomenon with sufficient SNR.

## Transformer and Interpretability Experiments
Initial experiments using standard Blocked Cross-Validation yielded an inflated ROC-AUC of 0.81 due to severe class imbalances in the sequential splits (e.g., a 77:11 class ratio in the final fold).

To enforce strict methodological rigor, the validation strategy was updated to **5-fold Stratified Group K-Fold (SGKF)**. Windows were grouped into contiguous macro-blocks to prevent overlap leakage, while SGKF ensured a representative balance of eyes-open and eyes-closed states across all folds.

Initial Transformer iterations utilized an autoregressive casual mask (like GPT), which produced "attention sinks" on timestep 0 and failed to capture the structure of a whole window. Removing lower-triangular masking converted the architecture into a **Bidirectional Encoder**, allowing attention to correlate temporal features across the entire 128-timestep sequence simultaneously.

Initial global fold z-scoring across the dataset caused severe waveform distortion across time. Instead, **window-level instance normalization** (standardizing every 128-timestep window independently to zero-mean and unit-variance per channel). This eliminated baseline drift, reduced cross-fold variance from 0.137 to 0.115, and stabilized model generalization.

An automated Bayesian hyperparameter search using Optuna (Tree-structured Parzen Estimator with Median Pruning) was conducted to find an architecture that prevents overfitting on the dataset.

The optimized architecture is a 2-layer Transformer with 16-dimensional embeddings and 4 attention heads, trained via AdamW (lr 1.78e-4, weight decay 0.021, dropout 0.358). It utilizes RoPE positional encoding, label smoothing (0.1), and early stopping monitored via an Exponential Moving Average (EMA) of the validation loss.

## SHAP and Diagnostic Attention Heatmaps
By extracting attention weight matrices from the encoder, we visualize how it processes raw voltage without convolutional pooling:
- **Heads 0 and 3** track oscillatory temporal rhythms, showing clear diagonal frequency banding across the entire window
- **Heads 1 and 2** broadcast localized frontal blink artifacts across the entire sequence


![SHAP Summary](shap_summary.png)

## Pipeline
- Data loading (117 seconds, 14 channels, 14980 samples)
- Sliding windows segmentation (window = 128 samples, stride = 32) with per-window mean subtraction to remove DC drift
- Artifact rejection (threshold = 150uV)
- Blocked cross-validation to prevent model from training on data temporally adjacent to test data and inflate performance
- 5-fold division, each fold serving as test set once while remaining folds form training set
- StandardScaler fitted on training data only per fold
- SHAP Analysis applied to best-performing Random Forest

**Transformer:**
- 5-fold Stratified Group K-Fold (SGKF) to prevent data leakage and handle non-stationarity
- Window-level instance normalization (z-scoring along time dimension per window/channel independently)
- Automated Bayesian hyperparameter search via Optuna with hyperband early pruning

## Requirements
- Python 3.8+
- numpy, pandas, torch, scikit-learn, scipy, seaborn, matplotlib, optuna, shap

## Usage
Download EEG Eye State.arff from the UCI repository and place it in the same directory as the script. Then run:
EEGEyeState.py

## Paper
Full research paper included in this repository as EEG_Paper.pdf
