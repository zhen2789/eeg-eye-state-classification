"""
EEG Eye-State Classification
Transformer Architecture on Raw EEG (Windowed)
Dataset: UCI EEG Eye State
Author: Gary Zhen
Date: July 2026
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv('EEG-Eye-State.csv')
df["eyeDetection"] = df["eyeDetection"].astype(int)

batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 25
learning_rate = 2.2682050682897235e-05
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 16
n_head = 4
n_layer = 2
dropout = 0.5923249428368587

n = int(len(df) * 0.8)
train_data = df[:n]
test_data = df[n:]

X_raw = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

def create_windows(data, labels, window_size=128, stride=32, macro_block_size=1000):
    X, y, groups = [], [], []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start : start + window_size]
        window_mean = np.mean(window, axis=0)
        window_centered = window - window_mean
        X.append(window_centered)
        label_window = labels[start : start + window_size]
        y.append(np.bincount(label_window.astype(int)).argmax())
        groups.append(start // macro_block_size)
    return np.array(X), np.array(y), np.array(groups)

def reject_artifacts(X_win, y_win, groups, threshold=150):
    mask = np.max(np.abs(X_win), axis=(1,2)) < threshold
    return X_win[mask], y_win[mask], groups[mask]

def get_batch(split):
    X = X_train if split == 'train' else X_test
    y = y_train if split == 'train' else y_test
    ix = torch.randint(len(X), (batch_size, ))
    xb = X[ix]
    yb = y[ix]
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

def get_data_folds(X, y, groups, K=5):
    sgkf = StratifiedGroupKFold(n_splits=K)
    folds = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train_np, y_train_np = X[train_idx], y[train_idx]
        X_test_np, y_test_np = X[test_idx], y[test_idx]

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.long)

        fold_mean = X_train.mean(dim=0)
        fold_std = X_train.std(dim=0).clamp(min=1e-8)
        X_train = (X_train - fold_mean) / fold_std
        X_test = (X_test - fold_mean) / fold_std

        class_counts = np.bincount(y_train_np)
        weights = len(y_train_np) / (2.0 * class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)

        folds.append((X_train, y_train, X_test, y_test, class_weights))

    return folds

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        aucs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
            losses[k] = loss.item()
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == Y).float().mean()
            accs[k] = acc.item()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = Y.cpu().numpy()
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = 0.5
            aucs[k] = auc
        out[split] = losses.mean(), accs.mean(), aucs.mean()
    model.train()
    return out

class LayerNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.beta = torch.ones(dim)
        self.gamma = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class RoPE(nn.Module):

    def __init__(self, d, max_seq_len):
        super().__init__()
        m = torch.arange(max_seq_len)
        theta = 10000 ** -(torch.arange(0,d,2).float() / d)
        angles = torch.outer(m, theta).repeat_interleave(2, dim=-1)
        self.register_buffer('angles', angles)

    def forward(self, q):
        seq_len = q.shape[1]
        angles = self.angles[:seq_len]
        q_swapped = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        final_output = (q * torch.cos(angles)) + (q_swapped * torch.sin(angles))
        return final_output

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.rope = RoPE(head_size, max_seq_len=block_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.rope(self.key(x))
        q = self.rope(self.query(x))
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Linear(14, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, 2)

    def forward(self, idx, targets=None):
        B,T,C = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        xmean = x.mean(dim=1)
        logits = self.lm_head(xmean)
        return logits

model = Model()
m = model.to(device)

# Stratified Group K-Fold

X_win, y_win, groups = create_windows(X_raw, y_raw)
X_win, y_win, groups = reject_artifacts(X_win, y_win, groups)
folds = get_data_folds(X_win, y_win, groups, K=5)
auc_scores = []

for i, (X_train_fold, y_train_fold, X_test_fold, y_test_fold, class_weights) in enumerate(folds):
    print(f"----Starting Fold {i+1}----")

    X_train, y_train = X_train_fold, y_train_fold
    X_test, y_test = X_test_fold, y_test_fold

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.027514211721453685)
    class_weights = class_weights.to(device)

    smoothed_test_loss = None
    alpha = 0.3
    
    best_test_loss = float('inf')
    best_auc = 0
    best_model_state = None
    counter = 0

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            train_loss, train_acc, train_auc = losses['train']
            test_loss, test_acc, test_auc = losses['test']
            
            current_test_loss = test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
            
            print(f"step {iter}: train loss {train_loss:.4f} | test loss {test_loss:.4f} | test auc {test_auc:.4f}")

            if smoothed_test_loss is None:
                smoothed_test_loss = current_test_loss
            else:
                smoothed_test_loss = alpha * current_test_loss + (1 - alpha) * smoothed_test_loss

            if smoothed_test_loss < best_test_loss:
                best_test_loss = smoothed_test_loss
                best_auc = test_auc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1

            if counter == 5:
                model.load_state_dict(best_model_state)
                break

        xb, yb = get_batch('train')
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, weight=class_weights, label_smoothing=0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    auc_scores.append(float(best_auc))

print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
