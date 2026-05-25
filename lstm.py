from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH = 64
EPOCHS = 20
LR = 1e-3

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Zariadenie: {device}")

# 1. Data
dataset = load_dataset("TUKE-KEMT/toxic-sk")
df_tr, df_val = train_test_split(
    dataset["train"].to_pandas(),
    test_size=0.1,
    random_state=SEED,
    stratify=dataset["train"].to_pandas()["label"],
)
df_test = dataset["test"].to_pandas()


# 2. Slovnik
def tokenize(text):
    return str(text).lower().split()


words = [w for t in df_tr["text"] for w in tokenize(t)]
vocab = ["<PAD>", "<UNK>"] + [w for w, _ in Counter(words).most_common(20000)]
w2i = {w: i for i, w in enumerate(vocab)}


def encode(text, max_len=128):
    ids = [w2i.get(w, 1) for w in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


# 3. Dataset
class TD(Dataset):
    def __init__(self, df):
        self.X = torch.tensor([encode(t) for t in df["text"]], dtype=torch.long)
        self.y = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


tr_dl = DataLoader(TD(df_tr), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(TD(df_val), batch_size=BATCH)
te_dl = DataLoader(TD(df_test), batch_size=BATCH)


# 4. Model
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 128, padding_idx=0)
        self.lstm = nn.LSTM(
            128, 256, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True
        )
        self.fc = nn.Linear(512, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        _, (h, _) = self.lstm(self.drop(self.emb(x)))
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


model = LSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 5. Trenovanie
best_f1, best_patience = 0.0, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for X, y in tr_dl:
        optimizer.zero_grad()
        loss = criterion(model(X.to(device)), y.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in val_dl:
            preds += model(X.to(device)).argmax(1).cpu().tolist()
            trues += y.tolist()

    f1 = f1_score(trues, preds, average="binary", pos_label=1)
    acc = accuracy_score(trues, preds)
    print(f"Epocha {epoch:2d} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "model_lstm_best.pt")
        best_patience = 0
    else:
        best_patience += 1
        if best_patience >= 3:
            print("Early stopping")
            break

# 6. Test
model.load_state_dict(torch.load("model_lstm_best.pt"))
model.eval()
preds, trues = [], []
with torch.no_grad():
    for X, y in te_dl:
        preds += model(X.to(device)).argmax(1).cpu().tolist()
        trues += y.tolist()

acc = round(accuracy_score(trues, preds), 4)
precision, recall, f1, _ = precision_recall_fscore_support(
    trues, preds, average="binary", pos_label=1
)
precision = round(precision, 4)
recall = round(recall, 4)
f1 = round(f1, 4)

print(f"\nVysledky LSTM (toxic-sk):")
print(f"  Accuracy  : {acc}")
print(f"  Precision : {precision}")
print(f"  Recall    : {recall}")
print(f"  F1        : {f1}")

with open("model_lstm_vysledky.txt", "w") as f:
    f.write(f"LSTM | toxic-sk\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1: {f1}\n")
