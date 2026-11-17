"""Score text with pretrained FinBERT (ProsusAI/finbert).

This is the pre-trained FinBERT checkpoint: BERT-base fine-tuned by ProsusAI
on the Financial PhraseBank (Araci 2019, arXiv:1908.10063). It is used here
AS-IS for inference — no weights are updated in this script. A separate
script (finetune_head.py) trains a classifier head on frozen FinBERT
embeddings; full end-to-end fine-tuning was not run (see README).

Usage:
    python src/score_sentiment.py --input data/processed/phrasebank_test.csv \
        --text-col text --output data/processed/phrasebank_test_scored.csv
    python src/score_sentiment.py --input data/processed/news_sample.csv \
        --text-col headline --output data/processed/news_scored.csv

Output adds: predicted_label, p_negative, p_neutral, p_positive,
score (= p_positive - p_negative, in [-1, 1]).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
ROOT = Path(__file__).resolve().parents[1]
# Local weights live in models/finbert/ (downloaded once with resumable curl;
# the HF Hub downloader kept stalling on this connection). Prefer the local
# copy when present so runs never re-download.
LOCAL_MODEL = ROOT / "models" / "finbert"
MODEL_PATH = str(LOCAL_MODEL) if (LOCAL_MODEL / "model.safetensors").exists() or (LOCAL_MODEL / "pytorch_model.bin").exists() else MODEL_NAME


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    torch.set_num_threads(2)
    df = pd.read_csv(args.input)
    texts = df[args.text_col].astype(str).tolist()
    print(f"scoring {len(texts)} texts with {MODEL_PATH} (CPU, batch={args.batch_size})")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"model loaded in {time.time() - t0:.1f}s; labels={model.config.id2label}")

    labels, probs = [], []
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i : i + args.batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)
            idx = p.argmax(dim=-1).tolist()
            labels.extend([model.config.id2label[j] for j in idx])
            probs.extend(p.tolist())
            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  {min(i + args.batch_size, len(texts))}/{len(texts)}")
    elapsed = time.time() - t1
    print(f"inference: {len(texts)} texts in {elapsed:.1f}s "
          f"({len(texts) / elapsed:.1f} texts/sec)")

    import numpy as np
    # Column order follows the model's own id2label, not an assumption.
    prob_arr = pd.DataFrame({
        f"p_{model.config.id2label[j]}": [row[j] for row in probs]
        for j in sorted(model.config.id2label)
    })
    df["predicted_label"] = labels
    df = pd.concat([df, prob_arr], axis=1)
    df["score"] = df["p_positive"] - df["p_negative"]
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print(df["predicted_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
