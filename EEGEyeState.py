# Generated from: EEGEyeState.ipynb
# Converted at: 2026-04-25T13:03:42.126Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

pip install shap

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis
import shap

data, meta = arff.loadarff("EEG Eye State.arff")
df = pd.DataFrame(data)
df["eyeDetection"] = df["eyeDetection"].astype(int)

X_raw = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

window_size = 128
stride = 32

def create_windows(data, labels, window_size = 128, stride = 32):
    X, y = [], []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start : start + window_size]
        window_mean = np.mean(window, axis = 0)
        window_centered = window - window_mean
        X.append(window_centered)
        label_window = labels[start : start + window_size]
        y.append(np.bincount(label_window.astype(int)).argmax())
    return np.array(X), np.array(y)

X_win, y_win = create_windows(X_raw, y_raw)

def reject_artifacts(X_win, y_win, threshold = 150):
    mask = np.max(np.abs(X_win), axis = (1,2)) < threshold
    return X_win[mask], y_win[mask]

X_win, y_win = reject_artifacts(X_win, y_win)
print("Windows after artifact rejection: ", len(X_win))
print("Remaining class distribution: ", np.bincount(y_win))

def extract_bandpower(raw_windows):
    freq_bins = np.fft.rfft(raw_windows, axis = 1)
    power = np.abs(freq_bins)**2
    
    delta = power[:, 1:4, :]
    theta = power[:, 4:8, :]
    alpha = power[:, 8:13, :]
    beta = power[:, 13:31, :]
    
    output_delta = np.mean(delta, axis = 1)
    output_theta = np.mean(theta, axis = 1)
    output_alpha = np.mean(alpha, axis = 1)
    output_beta = np.mean(beta, axis = 1)

    final_delta = np.log1p(output_delta)
    final_theta = np.log1p(output_theta)
    final_alpha = np.log1p(output_alpha)
    final_beta = np.log1p(output_beta)
    
    result = np.concatenate((final_delta, final_theta, final_alpha, final_beta), axis = 1)
    return result

def extract_statistical_features(X_win):
    mean = np.mean(X_win, axis = 1)
    std = np.std(X_win, axis = 1)
    mn = np.min(X_win, axis = 1)
    mx = np.max(X_win, axis = 1)
    skewness = skew(X_win, axis = 1)
    kurt = kurtosis(X_win, axis = 1)
    return np.concatenate([mean, std, mn, mx, skewness, kurt], axis = 1)

clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
X_spectral = extract_statistical_features(X_win)

K = 5
fold_size = len(X_spectral) // K
auc_scores = []
gap = 4

for i in range(K):
    test_start = i * fold_size
    test_end = test_start + fold_size

    X_test_fold = X_spectral[test_start:test_end]
    y_test_fold = y_win[test_start:test_end]

    left_end = max(0, test_start - gap)
    right_start = min(len(X_spectral), test_end + gap)

    X_train_fold = np.concatenate((X_spectral[:left_end], X_spectral[right_start:]))
    y_train_fold = np.concatenate((y_win[:left_end], y_win[right_start:]))

    print(f"Fold {i+1} | class dist: {np.bincount(y_test_fold)} | ", end="")

    scaler = StandardScaler()
    scaler.fit(X_train_fold)
    X_train_fold = scaler.transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)
    trained = clf.fit(X_train_fold, y_train_fold)
    y_prob = clf.predict_proba(X_test_fold)[:, 1]
    roc_auc = roc_auc_score(y_test_fold, y_prob)
    auc_scores.append(roc_auc)

print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

channels = "AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4".split()
descriptors = ["mean", "std", "min", "max", "skew", "kurt"]
feature_names = [f"{desc}_{ch}" for desc in descriptors for ch in channels]
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X_spectral)
clf_full = RandomForestClassifier(n_estimators = 100, random_state = 42)
clf_full.fit(X_scaled, y_win)
explainer = shap.TreeExplainer(clf_full)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values[:, :, 1], X_scaled, feature_names = feature_names)

def extract_ratio_features(raw_windows):
    bins = np.fft.rfft(raw_windows, axis = 1)
    power = np.abs(bins)**2
    alpha = power[:, 8:13, :]
    beta = power[:, 13:31, :]
    alpha_mean = np.mean(alpha, axis = 1)
    beta_mean = np.mean(beta, axis = 1)
    ratio = alpha_mean / (beta_mean + 1e-10)
    log_ratio = np.log1p(ratio)
    return log_ratio

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis

torch.manual_seed(42)
np.random.seed(42)

data, meta = arff.loadarff("EEG Eye State.arff")
df = pd.DataFrame(data)
df["eyeDetection"] = df["eyeDetection"].astype(int)

X_raw = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

window_size = 128
stride = 32

def create_windows(data, labels, window_size = 128, stride = 32):
    X, y = [], []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start : start + window_size]
        window_mean = np.mean(window, axis = 0)
        window_centered = window - window_mean
        X.append(window_centered)
        label_window = labels[start : start + window_size]
        y.append(np.bincount(label_window.astype(int)).argmax())
    return np.array(X), np.array(y)

X_win, y_win = create_windows(X_raw, y_raw)

def reject_artifacts(X_win, y_win, threshold = 150):
    mask = np.max(np.abs(X_win), axis = (1,2)) < threshold
    return X_win[mask], y_win[mask]

X_win, y_win = reject_artifacts(X_win, y_win)

auc_scores = []
K = 5
fold_size = len(X_win) // K
gap = 4
    
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        
    def __len__(self): return len(self.X)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class EEGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 14
        hidden_size = 32
        num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.dropout = nn.Dropout(p = 0.3)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        z = self.fc(x)
        return z

for i in range(K):
    test_start = i * fold_size
    test_end = test_start + fold_size

    X_test_fold = X_win[test_start:test_end]
    y_test_fold = y_win[test_start:test_end]

    left_end = max(0, test_start - gap)
    right_start = min(len(X_win), test_end + gap)

    X_train_fold = np.concatenate((X_win[:left_end], X_win[right_start:]))
    y_train_fold = np.concatenate((y_win[:left_end], y_win[right_start:]))

    print(f"Fold {i+1} | class dist: {np.bincount(y_test_fold)} | ", end="")

    model = EEGClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(EEGDataset(X_train_fold, y_train_fold), batch_size = 32, shuffle = False)
    test_loader = DataLoader(EEGDataset(X_test_fold, y_test_fold), batch_size = 32, shuffle = False)
    
    for epoch in range(30):
        model.train()
        epoch_loss = 0                        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            probs = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())
    roc_auc = roc_auc_score(all_labels, all_probs)
    auc_scores.append(roc_auc)

print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")