"""Prepare the Financial PhraseBank labelled dataset.

Source: Malo et al. (2014), "Good Debt or Bad Debt: Detecting Semantic
Orientations in Economic Texts", Journal of the Association for Information
Science and Technology 65(4).
Downloaded as FinancialPhraseBank-v1.0.zip from the Hugging Face dataset
mirror takala/financial_phrasebank (data/FinancialPhraseBank-v1.0.zip).

File used: Sentences_50Agree.txt — 4,845 sentences where at least 50% of
annotators agreed on the label (positive / neutral / negative).
Each line is: <sentence>@<label>

Output: data/processed/phrasebank_train.csv and phrasebank_test.csv,
a stratified 80/20 split (random_state=42).
"""
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "FinancialPhraseBank-v1.0" / "Sentences_50Agree.txt"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = []
    for line in SRC.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        text, label = text.strip(), label.strip()
        if text and label in {"positive", "neutral", "negative"}:
            rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    print(f"parsed {len(df)} unique labelled sentences")
    print(df["label"].value_counts().to_string())

    train, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    train.to_csv(OUT / "phrasebank_train.csv", index=False)
    test.to_csv(OUT / "phrasebank_test.csv", index=False)
    print(f"train={len(train)} test={len(test)}")
    print("test label counts:")
    print(test["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
