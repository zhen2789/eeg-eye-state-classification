# eeg-eye-state-classification
EEG eye-state classification using LSTM, Random Forest, and transformer architecture with SHAP interpretability analysis

## Overview
This project builds an EEG eye-state classification pipeline on the UCI 
EEG Eye State dataset, comparing Random Forest models across four feature 
representations against an LSTM on raw EEG. Key finding: statistical 
time-domain features outperform spectral features, and the transformer achieves the 
best overall performance (0.81 ROC-AUC), suggesting attention over raw temporal windows
captures discriminative structure that handcrafted features may miss.

## Results
| Model | Features | ROC-AUC | Std |
|-------|----------|---------|-----|
| Random Forest | FFT Bandpower | 0.52 | 0.065 |
| Random Forest | Alpha/Beta Ratio | 0.46 | 0.058 |
| Random Forest | Combined | 0.59 | 0.039 |
| Random Forest | Statistical | 0.62 | 0.028 |
| LSTM | Raw EEG | 0.68 | 0.082 |
| Transformer | Raw EEG (windowed) | 0.81 | 0.068 |

## Key Finding
SHAP analysis revealed that classification is driven primarily by 
oculomotor artifacts at frontal electrodes rather than occipital alpha 
rhythm suppression — suggesting consumer-grade EEG hardware cannot 
reliably capture the alpha-blocking phenomenon with sufficient SNR.

## Transformer Experiment
A 3-layer transformer with 64-dimensional embeddings was trained on 
raw EEG windows of shape (batch, 128, 14) using an AdamW optimizer 
with weight decay 1e-2 and a learning rate of 1e-4. The model also utilized RoPE positional encoding and 
patience-based early stopping. It trained with 5-fold blocked cross-validation, 
achieving a mean ROC-AUC of 0.81. However, this mean was computed using the 
first 4 folds, as fold 5 was excluded from the mean due to severe class 
imbalance in the test split (77 class-0 vs. 11 class-1 windows). Fold 
variance is also high (0.068 std), meaning the model generalizes well on 
some temporal segments and less well on others, reflecting the 
non-stationarity of EEG. The transformer outperformed both the LSTM (0.68) 
and Random Forest (0.62) baselines, suggesting that attention over raw 
temporal windows captures discriminative structure that handcrafted 
features may miss.

## Pipeline
- Data loading (117 seconds, 14 channels, 14980 samples)
- Sliding windows segmentation (window = 128 samples, stride = 32) with per-window mean subtraction to remove DC drift
- Artifact rejection (threshold = 150uV)
- Blocked cross-validation to prevent model from training on data temporally adjacent to test data and inflate performance
- 5-fold division, each fold serving as test set once while remaining folds form training set
- StandardScaler fitted on training data only per fold
- SHAP Analysis applied to best-performing Random Forest
- The transformer experiment uses a separate notebook with its own pipeline including blocked CV on raw windowed EEG with per-fold normalization, patience-based early stopping, and weight decay.

## Requirements
- Python 3.8+
- numpy
- pandas
- torch
- scikit-learn
- scipy
- shap

## Usage
Download EEG Eye State.arff from the UCI repository and place it in the 
same directory as the script. Then run:
EEGEyeState.py

## Paper
Full research paper included in this repository as EEG_Paper.pdf

![SHAP Summary](shap_summary.png)
