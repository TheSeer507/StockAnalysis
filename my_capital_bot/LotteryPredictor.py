import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

data_text = """\
Oct. 04 27 23 36 13 01
Oct. 19 30 12 08 03 17
Oct. 35 28 04 20 01 25
Nov. 23 32 16 11 31 04
Nov. 09 03 18 11 33 27
Nov. 19 03 36 05 28 18
Nov. 07 36 19 27 37 31
Nov. 04 18 14 05 29 28
Nov. 08 33 09 36 23 01
Nov. 28 31 01 11 10 38
Nov. 30 14 38 36 31 07
Dic. 33 18 29 12 07 10
Dic. 35 11 07 19 16 01
Dic. 26 10 17 04 25 08
Dic. 28 07 05 02 33 03
Dic. 02 33 01 17 08 05
Dic. 07 26 10 15 03 25
Dic. 07 06 05 01 21 19
Dic. 21 03 07 27 18 04
Dic. 30 13 14 25 03 02
Ene. 29 17 25 19 27 02
Ene. 22 14 25 06 08 29
Ene. 36 14 03 30 02 12
Ene. 08 16 20 21 06 27
Ene. 07 18 14 04 21 09
Ene. 34 08 07 21 13 10
Ene. 02 32 36 34 10 20
Ene. 08 15 11 12 37 33
Ene. 09 12 33 19 25 36
Feb. 05 04 11 15 07 20
Feb. 16 37 13 29 32 02
Feb. 01 19 03 31 09 28
Feb. 26 22 15 37 28 23
Feb. 35 14 29 27 26 30
Feb. 11 36 02 17 24 06
Feb. 36 09 04 11 02 23
Feb. 30 09 29 22 16 35
Mar. 07 17 09 11 02 26
Mar. 20 15 38 02 06 23
Mar. 35 19 29 34 25 27
Mar. 30 02 09 38 37 31
Mar. 12 33 14 03 04 29
Mar. 14 23 25 26 34 12
Mar. 14 03 30 09 12 13
Mar. 21 07 02 26 19 20
Mar. 16 28 02 03 38 20
Abr. 25 31 24 10 04 12
Abr. 06 18 04 12 37 15
Abr. 04 27 03 26 20 21
Abr. 29 37 32 11 16 07
Abr. 23 35 28 37 11 22
Abr. 02 32 37 23 18 16
Abr. 22 25 20 11 21 19
Abr. 31 38 10 12 36 25
Abr. 24 33 19 36 12 38
May. 18 09 25 06 28 23
May. 05 31 33 10 15 24
May. 17 07 03 08 35 25
May. 02 25 05 35 18 38
May. 18 23 08 02 12 30
May. 10 16 18 11 37 33
May. 28 36 20 16 04 05
May. 30 04 20 16 25 27
Jun. 32 18 27 36 38 04
Jun. 26 25 36 06 17 20
Jun. 03 33 12 29 34 26
Jun. 08 20 34 24 21 30
Jun. 32 04 09 26 13 23
Jun. 11 30 15 29 17 21
Jun. 29 23 01 25 15 26
Jun. 14 27 20 29 04 13
Jun. 34 19 37 04 03 05
Jul. 13 33 31 03 15 26
Jul. 15 08 26 11 23 16
Jul. 30 27 33 32 07 05
Jul. 28 04 09 07 13 02
Jul. 10 27 25 26 38 15
Jul. 25 24 32 11 07 05
Jul. 28 20 16 11 36 10
Jul. 37 20 08 33 15 07
Jul. 04 34 18 32 24 35
Ago. 07 31 38 35 23 33
Ago. 26 25 02 35 30 20
Ago. 05 14 26 27 23 30
Ago. 09 25 24 29 35 32
Ago. 29 27 04 30 14 26
Ago. 23 30 11 07 08 28
Ago. 02 26 20 18 04 21
Ago. 34 18 01 08 35 14
Ago. 11 12 15 05 14 03
Sep. 02 15 28 13 25 31
Sep. 05 20 07 27 24 01
Sep. 37 25 13 06 36 32
Sep. 22 34 14 33 20 08
Sep. 08 06 14 37 10 13
Sep. 09 16 22 28 34 31
Sep. 24 22 04 19 06 28
Sep. 25 32 31 04 03 26
Oct. 38 09 03 33 18 23
Oct. 31 26 29 23 04 20
Oct. 02 29 36 07 30 32
Oct. 13 18 38 07 17 06
Oct. 22 12 09 11 23 35
Oct. 04 17 01 15 02 32
Oct. 22 26 33 17 29 28
Oct. 36 33 28 23 27 17
Oct. 36 13 06 14 35 25
Nov. 15 25 17 19 33 03
Nov. 29 06 19 18 34 27
Nov. 11 34 01 35 04 14
Nov. 14 31 27 03 05 36
Nov. 31 15 23 20 35 09
Nov. 03 09 11 14 30 29
Nov. 20 27 35 28 34 02
Nov. 07 02 20 22 13 01
Nov. 13 26 23 10 11 32
Dic. 27 30 03 28 09 24
Dic. 37 17 18 08 25 10
Dic. 09 06 21 33 01 30
Dic. 16 08 37 11 20 18
Dic. 35 38 09 11 08 15
Dic. 29 34 07 24 08 14
Dic. 37 28 20 24 38 27
Dic. 17 02 26 12 28 11
Ene. 32 12 21 17 27 36
Ene. 33 15 22 03 18 29
Ene. 31 26 25 34 19 28
Ene. 20 05 28 14 35 24
Ene. 21 34 16 35 19 04
Ene. 21 09 11 12 18 16
Ene. 05 17 12 24 33 22
Ene. 13 12 28 03 34 07
Ene. 02 21 33 09 01 29
Feb. 05 21 35 27 13 12
Feb. 36 08 21 24 35 13
Feb. 38 08 20 37 03 29
Feb. 18 37 27 21 06 02
Feb. 32 27 12 18 09 15
Feb. 11 13 35 21 27 24
Feb. 07 21 03 22 32 14
Mar. 33 25 04 06 07 26
Mar. 24 13 08 30 23 31
Mar. 05 28 36 06 01 13
Mar. 15 08 38 04 34 30
Mar. 01 06 05 31 02 24
Mar. 28 38 22 15 24 09
Mar. 08 35 32 24 17 22
Mar. 31 19 13 12 11 09
Mar. 16 36 35 05 24 12
Apr. 29 10 25 14 05 03
Apr. 20 08 35 21 28 12
Apr. 31 14 16 07 21 20
Apr. 34 10 36 32 12 24
Apr. 09 17 01 05 36 16
Apr. 07 08 22 36 20 29
Apr. 07 02 06 30 32 17
Apr. 34 09 04 17 36 05
May. 20 05 12 01 14 22
May. 25 19 09 12 27 13
May. 05 36 01 33 27 20
May. 27 02 09 10 12 17
May. 30 11 27 10 07 12
May. 20 32 35 23 04 38
May. 13 05 03 32 14 37
May. 34 18 27 17 13 22
May. 36 19 18 11 27 04
Jun. 07 10 33 30 05 15
Jun. 33 31 26 07 10 01
Jun. 11 26 32 22 17 16
Jun. 07 08 28 38 33 06
Jun. 24 07 26 02 01 37
Jun. 16 31 24 36 30 29
Jun. 34 32 24 01 20 28
Jun. 30 21 15 38 07 12
Jul. 09 33 31 27 12 02
Jul. 03 10 29 26 37 14
""".strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2) PARSE INTO NUMPY  (one-hot per draw)  — pool = 38 numbers
# ─────────────────────────────────────────────────────────────────────────────
NUMBERS = 38
draws = []
for line in data_text.splitlines():
    parts = line.strip().split()
    numbers = list(map(int, parts[1:]))  # ignore month label
    one_hot = np.zeros(NUMBERS, dtype=np.float32)
    one_hot[[n - 1 for n in numbers]] = 1.0
    draws.append(one_hot)
draws = np.stack(draws)        # shape [N_draws, 38]

# ─────────────────────────────────────────────────────────────────────────────
# 3) BUILD SEQUENCES                    (seq → next_draw)
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 10                   # last 10 draws → predict the 11-th
X_seq, y_seq = [], []
for i in range(len(draws) - SEQ_LEN):
    X_seq.append(draws[i:i+SEQ_LEN])      # shape [SEQ_LEN, 38]
    y_seq.append(draws[i+SEQ_LEN])        # shape [38]
X_seq = np.stack(X_seq)                   # [samples, SEQ_LEN, 38]
y_seq = np.stack(y_seq)                   # [samples, 38]

# ─────────────────────────────────────────────────────────────────────────────
# 4) CHRONOLOGICAL TRAIN / VAL / TEST SPLIT  (70 / 15 / 15 %)
# ─────────────────────────────────────────────────────────────────────────────
n_samples = len(X_seq)
train_end = int(0.70 * n_samples)
val_end   = int(0.85 * n_samples)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val,   y_val   = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test,  y_test  = X_seq[val_end:],         y_seq[val_end:]

# convert to torch
def to_tensor(a): return torch.tensor(a, dtype=torch.float32)
X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)
X_val_t,   y_val_t   = to_tensor(X_val),   to_tensor(y_val)
X_test_t,  y_test_t  = to_tensor(X_test),  to_tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=16, shuffle=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5) NETWORK DEFINITION  (LSTM → Dense 38)
# ─────────────────────────────────────────────────────────────────────────────
class LotteryLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=NUMBERS,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, NUMBERS)

    def forward(self, x):                 # x: [B, SEQ_LEN, 38]
        out, _ = self.lstm(x)             # out: [B, SEQ_LEN, H]
        last = out[:, -1, :]              # take output at final timestep
        return self.fc(last)              # logits: [B, 38]

model     = LotteryLSTM()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────────────────────────────────────────
# 6) TRAINING LOOP WITH EARLY STOP
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS   = 100
PATIENCE = 8
best_val = np.inf
patience = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # validation loss
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()

    print(f"Epoch {epoch:3d}    val_loss = {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        best_weights = model.state_dict()
    else:
        patience += 1
        if patience >= PATIENCE:
            print("→ Early stop")
            break

model.load_state_dict(best_weights)

# ─────────────────────────────────────────────────────────────────────────────
# 7) EVALUATION — top-6 accuracy
# ─────────────────────────────────────────────────────────────────────────────
def top6_accuracy(y_true, logits):
    probs = torch.sigmoid(logits).cpu().numpy()
    correct = 0
    for i in range(len(y_true)):
        top6 = np.argsort(probs[i])[-6:]
        truth = np.where(y_true[i].cpu().numpy() == 1)[0]
        correct += len(set(top6) & set(truth)) / 6.0
    return correct / len(y_true)

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
test_acc = top6_accuracy(y_test_t, test_logits)
print(f"\nTEST top-6 accuracy   = {test_acc:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8)  NEXT-DRAW PREDICTIONS  — five tickets, 6 numbers each
# ─────────────────────────────────────────────────────────────────────────────
last_seq = torch.tensor(draws[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)  # [1,SEQ_LEN,38]
with torch.no_grad():
    logits = model(last_seq)
prob = torch.sigmoid(logits).squeeze().numpy()        # [38]
ranked = prob.argsort()[::-1]                         # high→low

print("\nFive candidate tickets for the next draw:")
for i in range(5):
    ticket = np.sort(ranked[i*6:(i+1)*6] + 1)         # convert 0-based to 1-38
    print(f"  Ticket {i+1}: {ticket.tolist()}")