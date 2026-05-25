import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "gerulata/slovakbert"
SAVE_DIR = "model_slovakbert"
SEED = 42

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Zariadenie: {device}")

# 1. Data
dataset = load_dataset("TUKE-KEMT/toxic-sk")
df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()
df_tr, df_val = train_test_split(
    df_train, test_size=0.1, random_state=SEED, stratify=df_train["label"]
)

# 2. Tokenizacia
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize(batch):
    return tokenizer(batch["text"], max_length=128, truncation=True, padding=False)


def make_ds(df):
    ds = HFDataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds


hf_tr = make_ds(df_tr)
hf_val = make_ds(df_val)
hf_test = make_ds(df_test)

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
)
model.to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# 4. Trenovanie
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    weight_decay=0.1,
    warmup_steps=200,
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    seed=SEED,
    report_to="none",
    fp16=(device == "cuda"),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_tr,
    eval_dataset=hf_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# 5. Vysledky
trainer.save_model(f"{SAVE_DIR}/best")
tokenizer.save_pretrained(f"{SAVE_DIR}/best")

results = trainer.evaluate(eval_dataset=hf_test)
print(f"\nVysledky SlovakBERT (toxic-sk):")
print(f"  Accuracy : {results['eval_accuracy']}")
print(f"  F1       : {results['eval_f1']}")

with open(f"{SAVE_DIR}/vysledky.txt", "w") as f:
    f.write(f"SlovakBERT | toxic-sk\n")
    f.write(f"Accuracy: {results['eval_accuracy']}\n")
    f.write(f"Precision: {results['eval_precision']}\n")
    f.write(f"Recall: {results['eval_recall']}\n")
    f.write(f"F1: {results['eval_f1']}\n")
