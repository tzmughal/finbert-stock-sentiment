"""Train a classifier head on FROZEN FinBERT embeddings.

Honest scope note: this is NOT full end-to-end fine-tuning of FinBERT.
The BERT encoder weights are frozen; only a logistic-regression head on
the 768-d [CLS] embedding is trained (train split) and evaluated on the
same held-out test split as the zero-shot model, so the two results are
directly comparable. Full fine-tuning (updating all ~110M parameters)
was not run: this machine has 1.5 shared vCPUs and ~1 GB RAM available,
and Adam states for 110M parameters do not fit that budget. See README.

Writes results/metrics_finetune_head.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = ROOT / "models" / "finbert"
MODEL_PATH = str(LOCAL_MODEL) if (LOCAL_MODEL / "model.safetensors").exists() or (LOCAL_MODEL / "pytorch_model.bin").exists() else MODEL_NAME
LABELS = ["positive", "neutral", "negative"]


@torch.no_grad()
def embed(texts, tokenizer, model, batch_size=32, max_length=128):
    model.eval()
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        out = model(**enc).last_hidden_state[:, 0, :]  # [CLS]
        vecs.append(out.numpy())
    return np.vstack(vecs)


def main() -> None:
    torch.set_num_threads(2)
    train = pd.read_csv(ROOT / "data" / "processed" / "phrasebank_train.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "phrasebank_test.csv")
    print(f"train={len(train)} test={len(test)}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    print(f"model loaded in {time.time() - t0:.1f}s")
    t1 = time.time()
    cache_tr = ROOT / "data" / "processed" / "embeddings_train.npy"
    cache_te = ROOT / "data" / "processed" / "embeddings_test.npy"
    if cache_tr.exists() and cache_te.exists():
        X_train = np.load(cache_tr)
        X_test = np.load(cache_te)
        print(f"loaded cached embeddings from {cache_tr.parent}")
    else:
        X_train = embed(train["text"].astype(str).tolist(), tokenizer, model)
        X_test = embed(test["text"].astype(str).tolist(), tokenizer, model)
        np.save(cache_tr, X_train)
        np.save(cache_te, X_test)
    print(f"embeddings ready; dim={X_train.shape[1]} (embed time {time.time() - t1:.1f}s)")

    clf = LogisticRegression(max_iter=1000)  # multinomial by default for lbfgs
    clf.fit(X_train, train["label"])
    pred = clf.predict(X_test)
    acc = accuracy_score(test["label"], pred)
    report = classification_report(test["label"], pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    majority_acc = (test["label"] == test["label"].value_counts().idxmax()).mean()

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "model": "LogisticRegression head on frozen ProsusAI/finbert [CLS] embeddings "
                 "(encoder frozen; NOT full fine-tuning)",
        "accuracy": round(float(acc), 4),
        "majority_baseline_accuracy": round(float(majority_acc), 4),
        "per_class": {lab: {k: round(float(v), 4) for k, v in report[lab].items()
                            if k in ("precision", "recall", "f1-score", "support")}
                      for lab in LABELS},
    }
    out = ROOT / "results" / "metrics_finetune_head.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
