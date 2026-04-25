# eeg-eye-state-classification
EEG eye-state classification using LSTM and Random Forest with SHAP interpretability analysis

## Overview
This project builds an EEG eye-state classification pipeline on the UCI 
EEG Eye State dataset, comparing Random Forest models across four feature 
representations against an LSTM on raw EEG. Key finding: statistical 
time-domain features outperform spectral features, and LSTM achieves the 
best overall performance (0.68 ROC-AUC), suggesting oculomotor artifacts 
drive classification more than alpha rhythm suppression.

## Results
| Model | Features | ROC-AUC | Std |
|-------|----------|---------|-----|
| Random Forest | FFT Bandpower | 0.52 | 0.065 |
| Random Forest | Alpha/Beta Ratio | 0.46 | 0.058 |
| Random Forest | Combined | 0.59 | 0.039 |
| Random Forest | Statistical | 0.62 | 0.028 |
| LSTM | Raw EEG | 0.68 | 0.082 |

## Key Finding
SHAP analysis revealed that classification is driven primarily by 
oculomotor artifacts at frontal electrodes rather than occipital alpha 
rhythm suppression — suggesting consumer-grade EEG hardware cannot 
reliably capture the alpha-blocking phenomenon with sufficient SNR.

## Pipeline
- Data loading (117 seconds, 14 channels, 14980 samples)
- Sliding windows segmentation (window = 128 samples, stride = 32) with per-window mean subtraction to remove DC drift
- Artifact rejection (threshold = 150uV)
- Blocked cross-validation to prevent model from training on data temporally adjacent to test data and inflate performance
- 5-fold division, each fold serving as test set once while remaining folds form training set
- StandardScaler fitted on training data only per fold
- SHAP Analysis applied to best-performing Random Forest

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
