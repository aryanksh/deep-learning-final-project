import time
import os
import sys
import html

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# --------------------
# Device / environment info 
# --------------------
print(torch.cuda.is_available())
if torch.cuda.is_available():
    # if all GPUs are visible and 5 is free
    try:
        torch.cuda.set_device(5)
    except Exception as e:
        print(f"Warning: could not set CUDA device 5: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("Python executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


# Base project directory (../ from this script), and results directory for plots/logs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results_bert-lr0.00005")
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------
# Loading & Preprocessing Data
# --------------------
ds_train = load_dataset("cardiffnlp/tweet_eval", "emotion", split="train")
ds_test = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")
ds_val = load_dataset("cardiffnlp/tweet_eval", "emotion", split="validation")

print("Example train row:", ds_train[0])



# Preprocessing

def lower_text(example):
    example["text"] = str(example["text"]).lower()
    return example


ds_train = ds_train.map(lower_text)
ds_test = ds_test.map(lower_text)
ds_val = ds_val.map(lower_text)


def apply_preprocess(example):
    text = example["text"]

    new_text = []

    # change all tags to users to "@user" and all links to "http"
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)

    cleaned_text = " ".join(new_text)
    cleaned_text = html.unescape(cleaned_text)

    example["text"] = cleaned_text

    return example


ds_train = ds_train.map(apply_preprocess)
ds_val = ds_val.map(apply_preprocess)
ds_test = ds_test.map(apply_preprocess)


# --------------------
# Tokenization (BERT)
# --------------------
BERT_MODEL = "google-bert/bert-base-uncased"

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=False)


def bert_tokenization(example):
    return bert_tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )


ds_train_tokenized_bert = ds_train.map(bert_tokenization, batched=True)
ds_test_tokenized_bert = ds_test.map(bert_tokenization, batched=True)
ds_val_tokenized_bert = ds_val.map(bert_tokenization, batched=True)

# Change format of BERT tokenized datasets into tensors, so that we can use PyTorch
# The `input_ids`, `token_type_ids`, and `attention_mask` columns will be the actual inputs to the model
ds_train_tokenized_bert.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)
ds_test_tokenized_bert.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)
ds_val_tokenized_bert.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)

print("Tokenized train format:", ds_train_tokenized_bert.format)


# 1. Define a filter to find None values
def find_none(example):
    return example["text"] is None


# 2. Apply it to the training set
bad_rows = ds_train.filter(find_none)

print(f"Total rows in train: {len(ds_train)}")
print(f"Rows with None:      {len(bad_rows)}")


# --------------------
# Training BERT Model
# --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


print("\n" + "=" * 30)
print(" TRAINING MODEL: BERT")
print("=" * 30)

model_bert = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL, num_labels=4
)

args_bert = TrainingArguments(
    output_dir="/data/jiang/anandvh/deep-learning-final-project/results_bert",
    num_train_epochs=3,  # 3 loops is standard
    per_device_train_batch_size=16,  # Reduce to 8 if you get CUDA OOM error
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_first_step=True,
    save_total_limit=1,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    weight_decay=0.01,
    report_to="none",  # Disable wandb logging to keep output clean
)

trainer_bert = Trainer(
    model=model_bert,
    args=args_bert,
    train_dataset=ds_train_tokenized_bert,
    eval_dataset=ds_val_tokenized_bert,
    compute_metrics=compute_metrics,
)

trainer_bert.train()


# Evaluation (validation + test)
eval_results = trainer_bert.evaluate()
print("Validation results:", eval_results)

test_results = trainer_bert.evaluate(ds_test_tokenized_bert)
print("Test results:", test_results)

logs = pd.DataFrame(trainer_bert.state.log_history)


# Training curves 
# deduplicated by epoch to avoid vertical lines

train_logs = logs[logs.get("loss").notnull()][["epoch", "loss"]]
eval_logs = logs[logs.get("eval_loss").notnull()][
    ["epoch", "eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
]

# Deduplicate by epoch to avoid multiple entries at the same x-position
train_by_epoch = train_logs.groupby("epoch", as_index=False).last()
eval_by_epoch = eval_logs.groupby("epoch", as_index=False).last()

# Save numeric results (test_results dict + epoch-wise metrics) to results.txt
results_file = os.path.join(RESULTS_DIR, "results.txt")
merged = pd.merge(train_by_epoch, eval_by_epoch, on="epoch", how="inner")
with open(results_file, "w") as f:
    f.write(str(test_results) + "\n\n")
    f.write(
        "Epoch\tTraining Loss\tValidation Loss\tAccuracy\tF1\tPrecision\tRecall\n"
    )
    for _, row in merged.iterrows():
        epoch = int(row["epoch"])
        f.write(
            f"{epoch}\t"
            f"{row['loss']:.6f}\t"
            f"{row['eval_loss']:.6f}\t"
            f"{row['eval_accuracy']:.6f}\t"
            f"{row['eval_f1']:.6f}\t"
            f"{row['eval_precision']:.6f}\t"
            f"{row['eval_recall']:.6f}\n"
        )

# Training & validation loss plot
plt.figure(figsize=(6, 4))
plt.plot(train_by_epoch["epoch"], train_by_epoch["loss"], label="train_loss")
plt.plot(eval_by_epoch["epoch"], eval_by_epoch["eval_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BERT: Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_validation_loss.png"), dpi=150)
plt.show()

# Validation accuracy and F1 plot
plt.figure(figsize=(6, 4))
plt.plot(eval_by_epoch["epoch"], eval_by_epoch["eval_f1"], label="val_macro_f1")
plt.plot(eval_by_epoch["epoch"], eval_by_epoch["eval_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("BERT: Validation Accuracy & Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "validation_accuracy_f1.png"), dpi=150)
plt.show()


# --------------------
# Final test evaluation, confusion matrix, per-class F1
# --------------------
test_pred = trainer_bert.predict(ds_test_tokenized_bert)
y_true = test_pred.label_ids
y_pred = np.argmax(test_pred.predictions, axis=-1)

# get label names directly from the dataset
label_names = ds_train.features["label"].names

print("Test classification report:")
print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("BERT Confusion Matrix (Normalized)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# Per-class F1 bar plot
report = classification_report(
    y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
)

per_class_f1 = [report[label]["f1-score"] for label in label_names]

plt.figure(figsize=(6, 4))
plt.bar(label_names, per_class_f1)
plt.ylim(0, 1.0)
plt.ylabel("F1-score")
plt.title("BERT: Per-class F1 on TweetEval Emotion (test)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_class_f1.png"), dpi=150)
plt.show()


