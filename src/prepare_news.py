"""Prepare the per-ticker news headline sample.

Source: David-Chew-HL/Tech-Stocks-News on Hugging Face — files
{AAPL,MSFT,NVDA,TSLA}_Alpaca_NLP.csv (Alpaca news headlines, columns:
headline, date). The four raw CSVs are ~535 MB in total and are NOT
committed; download them with:
    python src/download_news.py
(or fetch the four files from the dataset page into data/raw/).

Steps: stream each file, keep headlines dated 2023-01-01..2024-01-30,
drop empty headlines, de-duplicate by (ticker, normalised headline)
— wire headlines are heavily syndicated — then take a seeded random
sample of at most --per-ticker headlines per ticker so FinBERT scoring
is feasible on CPU. Sampling is documented in the run log and in
data/processed/news_sample_counts.json; it is NOT hidden.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ticker", type=int, default=1500)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-01-30")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts = {}
    frames = []
    for ticker in TICKERS:
        path = RAW / f"{ticker}_Alpaca_NLP.csv"
        raw_n = in_range = dedup_n = 0
        seen = set()
        rows = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_n += 1
                headline = (row.get("headline") or "").strip()
                date = (row.get("date") or "")[:10]
                if not headline or not (args.start <= date <= args.end):
                    continue
                in_range += 1
                key = " ".join(headline.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"ticker": ticker, "date": date, "headline": headline})
        dedup_n = len(rows)
        df = pd.DataFrame(rows)
        if len(df) > args.per_ticker:
            df = df.sample(n=args.per_ticker, random_state=args.seed)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        frames.append(df)
        counts[ticker] = {
            "raw_rows": raw_n,
            "in_date_range_nonempty": in_range,
            "after_dedup": dedup_n,
            "sampled": len(df),
        }
        print(f"{ticker}: raw={raw_n} in_range={in_range} dedup={dedup_n} sampled={len(df)}")

    sample = pd.concat(frames, ignore_index=True)
    sample.to_csv(OUT / "news_sample.csv", index=False)
    (OUT / "news_sample_counts.json").write_text(json.dumps(counts, indent=2))
    print(f"total sampled headlines: {len(sample)}")
    print(sample.groupby("ticker").size().to_string())
    print(f"date range: {sample['date'].min()} to {sample['date'].max()}")


if __name__ == "__main__":
    main()
