import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Zariadenie: {device}")

COLORS = {"toxic": "#e74c3c", "neutral": "#2ecc71"}

all_texts = [
    "Si úplný idiot a mal by si zmiznúť z internetu.",
    "Toto je absolútna hlúposť, čo hovoríš.",
    "Nechápem ako môžeš byť taký sprostý.",
    "Myslím že táto politika má svoje výhody aj nevýhody.",
    "Súhlasím s tým čo bolo povedané na začiatku.",
    "Zaujímavý pohľad na vec, ďakujem za príspevok.",
]
text_labels = ["Toxický", "Toxický", "Toxický", "Neškodný", "Neškodný", "Neškodný"]


def clean_token(token):
    token = token.replace("Ġ", " ").replace("▁", " ").strip()
    try:
        token = token.encode("latin-1").decode("utf-8")
    except:
        pass
    return token


# KONFUZNE MATICE
def plot_cm(labels, preds, model_name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Neškodný", "Toxický"]
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=12, pad=10)
    ax.set_xlabel("Predikcia")
    ax.set_ylabel("Skutočnosť")
    plt.tight_layout()
    fname = f"figures/cm_{model_name.lower().replace('-', '_').replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


# LIME
def run_lime(predict_fn, model_name):
    print(f"  LIME — {model_name}...")
    safe = model_name.lower().replace("-", "_").replace(" ", "_")
    explainer = LimeTextExplainer(class_names=["Neškodný", "Toxický"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (text, label) in enumerate(zip(all_texts, text_labels)):
        exp = explainer.explain_instance(
            text, predict_fn, num_features=6, num_samples=300
        )
        words = [x[0] for x in exp.as_list()]
        weights = [x[1] for x in exp.as_list()]
        colors = [COLORS["toxic"] if w > 0 else COLORS["neutral"] for w in weights]
        pred = predict_fn([text])[0]
        ax = axes[i]
        ax.barh(words, weights, color=colors, edgecolor="white")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_title(
            f"{label} (Toxic={pred[1]:.2f})\n{text[:45]}...", fontsize=8, pad=5
        )
        ax.set_xlabel("LIME váha", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle(f"LIME vysvetlenia — {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f"figures/lime_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


# SHAP
def run_shap(predict_fn, model_name):
    print(f"  SHAP — {model_name}...")
    safe = model_name.lower().replace("-", "_").replace(" ", "_")
    try:
        masker = shap.maskers.Text(r"\W+")
        shap_explainer = shap.Explainer(predict_fn, masker)
        shap_values = shap_explainer(all_texts)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, (text, label) in enumerate(zip(all_texts, text_labels)):
            tokens = shap_values[i].data
            vals = shap_values[i].values[:, 1]
            filtered = [(t, v) for t, v in zip(tokens, vals) if t.strip()]
            if filtered:
                words, weights = zip(*filtered)
                colors = [
                    COLORS["toxic"] if w > 0 else COLORS["neutral"] for w in weights
                ]
                ax = axes[i]
                ax.barh(words, weights, color=colors, edgecolor="white")
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_title(f"{label}\n{text[:45]}...", fontsize=8, pad=5)
                ax.set_xlabel("SHAP hodnota", fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
        plt.suptitle(f"SHAP vysvetlenia — {model_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        fname = f"figures/shap_{safe}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {fname}")
    except Exception as e:
        print(f"  SHAP chyba: {e}")


# INTEGRATED GRADIENTS
def run_ig(model, tokenizer, model_name):
    print(f"  IG — {model_name}...")
    safe = model_name.lower().replace("-", "_").replace(" ", "_")

    def get_ig(text, target_class=1):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=True
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        embeddings = model.roberta.embeddings.word_embeddings(input_ids)

        def forward_func(emb):
            outputs = model.roberta(inputs_embeds=emb, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier.out_proj(
                model.classifier.dropout(
                    torch.tanh(model.classifier.dense(model.classifier.dropout(pooled)))
                )
            )
            return logits

        ig = IntegratedGradients(forward_func)
        baseline = torch.zeros_like(embeddings)
        attributions, _ = ig.attribute(
            embeddings, baseline, target=target_class, return_convergence_delta=True
        )
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / (torch.norm(attributions) + 1e-8)
        tokens = [clean_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0])]
        return tokens, attributions.cpu().detach().numpy()

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, (text, label) in enumerate(zip(all_texts, text_labels)):
            target = 1 if label == "Toxický" else 0
            tokens, attrs = get_ig(text, target_class=target)
            filtered = [
                (t, a)
                for t, a in zip(tokens, attrs)
                if t not in ["<s>", "</s>", "<pad>", ""] and t.strip()
            ]
            if filtered:
                words, weights = zip(*filtered)
                colors = [
                    COLORS["toxic"] if w > 0 else COLORS["neutral"] for w in weights
                ]
                ax = axes[i]
                ax.barh(words, weights, color=colors, edgecolor="white")
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_title(f"{label}\n{text[:45]}...", fontsize=8, pad=5)
                ax.set_xlabel("IG atribúcia", fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
        plt.suptitle(f"Integrated Gradients — {model_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        fname = f"figures/ig_{safe}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {fname}")
    except Exception as e:
        print(f"  IG chyba: {e}")


# PFI TRANSFORMER
def run_pfi_transformer(model, tokenizer, hf_test, model_name):
    print(f"  PFI — {model_name}...")
    torch.manual_seed(42)
    np.random.seed(42)
    safe = model_name.lower().replace("-", "_").replace(" ", "_")
    model.eval()
    pad_id = tokenizer.pad_token_id
    all_input_ids = pad_sequence(
        [torch.tensor(x) for x in hf_test["input_ids"]],
        batch_first=True,
        padding_value=pad_id,
    ).to(device)
    all_attention = pad_sequence(
        [torch.tensor(x) for x in hf_test["attention_mask"]],
        batch_first=True,
        padding_value=0,
    ).to(device)
    all_labels = list(hf_test["labels"])
    baseline_preds = []
    with torch.no_grad():
        for i in range(0, len(all_input_ids), 32):
            logits = model(
                input_ids=all_input_ids[i : i + 32],
                attention_mask=all_attention[i : i + 32],
            ).logits
            baseline_preds.extend(logits.argmax(1).cpu().numpy())
    baseline_f1 = f1_score(all_labels, baseline_preds, average="binary")
    print(f"    Baseline F1: {baseline_f1:.4f}")
    importances = []
    for pos in range(15):
        ids_perm = all_input_ids.clone()
        idx = torch.randperm(len(ids_perm))
        ids_perm[:, pos] = ids_perm[idx, pos]
        perm_preds = []
        with torch.no_grad():
            for i in range(0, len(ids_perm), 32):
                logits = model(
                    input_ids=ids_perm[i : i + 32],
                    attention_mask=all_attention[i : i + 32],
                ).logits
                perm_preds.extend(logits.argmax(1).cpu().numpy())
        perm_f1 = f1_score(all_labels, perm_preds, average="binary")
        importances.append(baseline_f1 - perm_f1)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["toxic"] if v > 0 else COLORS["neutral"] for v in importances]
    ax.bar(range(len(importances)), importances, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pozícia tokenu v sekvencii")
    ax.set_ylabel("Pokles F1 skóre")
    ax.set_title(f"Permutation Feature Importance — {model_name}", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fname = f"figures/pfi_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


# PFI LSTM
def run_pfi_lstm(lstm_model, vocab, df_test):
    print(f"  PFI — LSTM...")
    torch.manual_seed(42)
    np.random.seed(42)
    w2i = {w: i for i, w in enumerate(vocab)}

    def encode(text, max_len=128):
        ids = [w2i.get(w, 1) for w in str(text).lower().split()][:max_len]
        return ids + [0] * (max_len - len(ids))

    texts = df_test["text"].tolist()
    labels = df_test["label"].tolist()
    lstm_model.eval()
    X = torch.tensor([encode(t) for t in texts], dtype=torch.long).to(device)
    with torch.no_grad():
        preds = lstm_model(X).argmax(1).cpu().numpy()
    baseline_f1 = f1_score(labels, preds, average="binary")
    print(f"    Baseline F1: {baseline_f1:.4f}")
    importances = []
    for pos in range(15):
        X_perm = torch.tensor([encode(t) for t in texts], dtype=torch.long)
        idx = torch.randperm(len(X_perm))
        X_perm[:, pos] = X_perm[idx, pos]
        with torch.no_grad():
            preds_perm = lstm_model(X_perm.to(device)).argmax(1).cpu().numpy()
        perm_f1 = f1_score(labels, preds_perm, average="binary")
        importances.append(baseline_f1 - perm_f1)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["toxic"] if v > 0 else COLORS["neutral"] for v in importances]
    ax.bar(range(len(importances)), importances, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pozícia tokenu v sekvencii")
    ax.set_ylabel("Pokles F1 skóre")
    ax.set_title("Permutation Feature Importance — LSTM", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fname = "figures/pfi_lstm.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


# POROVNANIE METRIK
def plot_porovnanie():
    def read_results(path):
        d = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        key = key.strip().lower()
                        if key in {"accuracy", "precision", "recall", "f1"}:
                            try:
                                d[key] = float(val.strip())
                            except ValueError:
                                pass
        except FileNotFoundError:
            print(f"    ⚠ Súbor {path} nenájdený")
            return None
        # Skontroluj, či máme všetky 4 metriky
        missing = {"accuracy", "precision", "recall", "f1"} - d.keys()
        if missing:
            print(f"    ⚠ V {path} chýba: {missing}")
            return None
        return d

    result_paths = {
        "LSTM": "model_lstm_vysledky.txt",
        "SlovakBERT": "model_slovakbert_simple/vysledky.txt",
        "XLM-RoBERTa": "model_xlmr/vysledky.txt",
    }

    results = {}
    for model_name, path in result_paths.items():
        r = read_results(path)
        if r is not None:
            results[model_name] = r

    if not results:
        print("Žiadne výsledky neboli načítané, graf nebude vygenerovaný.")
        return

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Správnosť", "Presnosť", "Návratnosť", "F1 skóre"]
    model_colors = {
        "LSTM": "#3498db",
        "SlovakBERT": "#e74c3c",
        "XLM-RoBERTa": "#2ecc71",
    }
    # Pouzi iba modely, ktore mame nacitane
    available_models = {m: c for m, c in model_colors.items() if m in results}

    x = np.arange(len(metrics))
    width = 0.8 / len(available_models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(available_models.items()):
        vals = [results[model][m] for m in metrics]
        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=model,
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # Dynamicky nastav y-os podla minimalnej hodnoty
    all_vals = [v for model in results.values() for v in model.values()]
    y_min = max(0, min(all_vals) - 0.05)
    ax.set_ylim(y_min, 1.02)

    ax.set_xticks(x + width * (len(available_models) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Hodnota metriky")
    ax.set_title(
        "Porovnanie výkonnosti modelov na testovacej množine", fontsize=13, pad=15
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/porovnanie_metrik.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ figures/porovnanie_metrik.png")


# LSTM MODEL ARCHITEKTURA
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(
            128, 256, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True
        )
        self.fc = nn.Linear(512, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        _, (h, _) = self.lstm(self.drop(self.emb(x)))
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


# HLAVNY BLOK
if __name__ == "__main__":
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("Generujem porovnanie metrik...")
    plot_porovnanie()

    # Nacitaj dataset
    print("\nNacitavam dataset...")
    dataset = load_dataset("TUKE-KEMT/toxic-sk")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    df_tr, df_val = train_test_split(
        df_train, test_size=0.1, random_state=42, stratify=df_train["label"]
    )

    def make_ds(df, tok):
        ds = HFDataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        ds = ds.map(
            lambda b: tok(b["text"], max_length=128, truncation=True, padding=False),
            batched=True,
            remove_columns=["text"],
        )
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch")
        return ds

    # ---- SlovakBERT ----
    print("\n" + "=" * 60)
    print("Nacitavam SlovakBERT...")
    sb_tokenizer = AutoTokenizer.from_pretrained("model_slovakbert_simple/best")
    sb_model = AutoModelForSequenceClassification.from_pretrained(
        "model_slovakbert_simple/best"
    )
    sb_model.eval().to(device)

    def predict_sb(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = sb_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            logits = sb_model(**inputs).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    print("SlovakBERT XAI...")
    run_lime(predict_sb, "SlovakBERT")
    run_shap(predict_sb, "SlovakBERT")
    run_ig(sb_model, sb_tokenizer, "SlovakBERT")

    hf_test_sb = make_ds(df_test, sb_tokenizer)
    run_pfi_transformer(sb_model, sb_tokenizer, hf_test_sb, "SlovakBERT")

    # Confusion matrix SlovakBERT
    pad_id = sb_tokenizer.pad_token_id
    all_ids = pad_sequence(
        [torch.tensor(x) for x in hf_test_sb["input_ids"]],
        batch_first=True,
        padding_value=pad_id,
    ).to(device)
    all_att = pad_sequence(
        [torch.tensor(x) for x in hf_test_sb["attention_mask"]],
        batch_first=True,
        padding_value=0,
    ).to(device)
    preds_sb = []
    with torch.no_grad():
        for i in range(0, len(all_ids), 32):
            logits = sb_model(
                input_ids=all_ids[i : i + 32], attention_mask=all_att[i : i + 32]
            ).logits
            preds_sb.extend(logits.argmax(1).cpu().numpy())
    plot_cm(list(hf_test_sb["labels"]), preds_sb, "SlovakBERT")

    # ---- XLM-RoBERTa ----
    print("\n" + "=" * 60)
    print("Nacitavam XLM-RoBERTa...")
    xlmr_tokenizer = AutoTokenizer.from_pretrained("model_xlmr/best")
    xlmr_model = AutoModelForSequenceClassification.from_pretrained("model_xlmr/best")
    xlmr_model.eval().to(device)

    def predict_xlmr(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = xlmr_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            logits = xlmr_model(**inputs).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    print("XLM-RoBERTa XAI...")
    run_lime(predict_xlmr, "XLM-RoBERTa")
    run_shap(predict_xlmr, "XLM-RoBERTa")
    run_ig(xlmr_model, xlmr_tokenizer, "XLM-RoBERTa")

    hf_test_xlmr = make_ds(df_test, xlmr_tokenizer)
    run_pfi_transformer(xlmr_model, xlmr_tokenizer, hf_test_xlmr, "XLM-RoBERTa")

    # Confusion matrix XLM-R
    pad_id = xlmr_tokenizer.pad_token_id
    all_ids = pad_sequence(
        [torch.tensor(x) for x in hf_test_xlmr["input_ids"]],
        batch_first=True,
        padding_value=pad_id,
    ).to(device)
    all_att = pad_sequence(
        [torch.tensor(x) for x in hf_test_xlmr["attention_mask"]],
        batch_first=True,
        padding_value=0,
    ).to(device)
    preds_xlmr = []
    with torch.no_grad():
        for i in range(0, len(all_ids), 32):
            logits = xlmr_model(
                input_ids=all_ids[i : i + 32], attention_mask=all_att[i : i + 32]
            ).logits
            preds_xlmr.extend(logits.argmax(1).cpu().numpy())
    plot_cm(list(hf_test_xlmr["labels"]), preds_xlmr, "XLM-RoBERTa")

    # ---- LSTM ----
    print("\n" + "=" * 60)
    print("Nacitavam LSTM...")
    words = [w for t in df_tr["text"] for w in str(t).lower().split()]
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in Counter(words).most_common(20000)]
    w2i_lstm = {w: i for i, w in enumerate(vocab)}

    lstm_model = LSTMClassifier(len(vocab)).to(device)
    lstm_model.load_state_dict(torch.load("model_lstm_best.pt", map_location=device))
    lstm_model.eval()

    def encode_lstm(text, max_len=128):
        ids = [w2i_lstm.get(w, 1) for w in str(text).lower().split()][:max_len]
        return ids + [0] * (max_len - len(ids))

    def predict_lstm(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        X = torch.tensor([encode_lstm(t) for t in texts], dtype=torch.long).to(device)
        with torch.no_grad():
            return torch.softmax(lstm_model(X), dim=-1).cpu().numpy()

    print("LSTM XAI...")
    run_lime(predict_lstm, "LSTM")
    run_shap(predict_lstm, "LSTM")
    run_pfi_lstm(lstm_model, vocab, df_test)

    # Confusion matrix LSTM
    X_test = torch.tensor(
        [encode_lstm(t) for t in df_test["text"]], dtype=torch.long
    ).to(device)
    with torch.no_grad():
        preds_lstm = lstm_model(X_test).argmax(1).cpu().numpy()
    plot_cm(df_test["label"].tolist(), preds_lstm, "LSTM")

    print("\n" + "=" * 60)
    print("Vsetky vizualizacie hotove!")
    print("Subory su v priecinku figures/")
