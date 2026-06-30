# Generated from: eeg-transformer-fixed.ipynb
# Converted at: 2026-06-30T13:41:12.714Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.metrics import roc_auc_score

df = pd.read_csv('EEG-Eye-State.csv')
df["eyeDetection"] = df["eyeDetection"].astype(int)

batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 50
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 3
dropout = 0.3

n = int(len(df) * 0.8)
train_data = df[:n]
test_data = df[n:]

X_raw = df.iloc[:, :-1].values # (14980, 14)
y_raw = df.iloc[:, -1].values # (14980)

def create_windows(data, labels, window_size=128, stride=32):
    X, y = [], []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start : start + window_size]
        window_mean = np.mean(window, axis=0)
        window_centered = window - window_mean
        X.append(window_centered)
        label_window = labels[start : start + window_size]
        y.append(np.bincount(label_window.astype(int)).argmax())
    return np.array(X), np.array(y)

X_win, y_win = create_windows(X_raw, y_raw)

# X_win (465, 128, 14)
# y_win (465, )

def reject_artifacts(X_win, y_win, threshold=150):
    mask = np.max(np.abs(X_win), axis=(1,2)) < threshold
    return X_win[mask], y_win[mask]

X_win, y_win = reject_artifacts(X_win, y_win)

# X_win (441, 128, 14)
# y_win (441, )

def get_batch(split):
    X = X_train if split == 'train' else X_test
    y = y_train if split == 'train' else y_test
    ix = torch.randint(len(X), (batch_size, ))
    xb = X[ix]
    yb = y[ix]
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

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
            logits, loss = model(X, Y)
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
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

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
        m = torch.arange(max_seq_len) # (max_seq_len, )
        theta = 10000 ** -(torch.arange(0,d,2).float() / d) # (1, d/2)
        angles = torch.outer(m, theta).repeat_interleave(2, dim=-1) # (max_seq_len, d)
        self.register_buffer('angles', angles)

    def forward(self, q):
        seq_len = q.shape[1] # (seq_len), gets sequence length of query tensor q
        angles = self.angles[:seq_len] # gets angles up to a specific sequence length
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
        k = self.rope(self.key(x)) # (B,T,C)
        q = self.rope(self.query(x)) # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # (B,T,T)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,C)
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
        B,T,C = idx.shape # (B,T,14)
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        x = tok_emb
        x = self.blocks(x) # (B,T,n_embd)
        x = self.ln_f(x) # (B,T,n_embd)
        xmean = x.mean(dim=1) # (B,n_embd)
        logits = self.lm_head(xmean) # (B,2)
        if targets is None:
            loss = None

        else:
            targets = targets.view(B)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = Model()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-2)

# blocked CV

K = 5
fold_size = len(X_win) // K
auc_scores = []
gap = 4
X = torch.tensor(X_win, dtype = torch.float32)
y = torch.tensor(y_win, dtype = torch.long)

for i in range(K):
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    test_start = i * fold_size
    test_end = test_start + fold_size

    X_test_fold = X[test_start:test_end]
    y_test_fold = y[test_start:test_end]

    left_end = max(0, test_start - gap)
    right_start = min(len(X), test_end + gap)

    X_train_fold = torch.cat((X[:left_end], X[right_start:]))
    y_train_fold = torch.cat((y[:left_end], y[right_start:]))

    print(f"Fold {i+1} | class dist: {np.bincount(y_test_fold.numpy())} | ", end="")

    fold_mean = X_train_fold.mean(dim=0)
    fold_std = X_train_fold.std(dim=0).clamp(min=1e-8)
    X_train_fold = (X_train_fold - fold_mean) / fold_std
    X_test_fold = (X_test_fold - fold_mean) / fold_std
    
    X_train, y_train = X_train_fold, y_train_fold
    X_test, y_test = X_test_fold, y_test_fold

    best_auc = 0
    best_model_state = None
    counter = 0

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            train_loss, train_acc, train_auc = losses['train']
            test_loss, test_acc, test_auc = losses['test']
            print(f"step {iter}: train loss {train_loss:.4f} | test loss {test_loss:.4f} | test auc {test_auc:.4f}")

            if test_auc > best_auc and not torch.isnan(test_auc):
                best_auc = test_auc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                
            if counter == 5:
                break
            
        xb, yb = get_batch('train')
        xb = xb + torch.randn_like(xb) * 0.05
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    auc_scores.append(float(best_auc))

print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")