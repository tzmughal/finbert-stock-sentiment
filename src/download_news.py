"""Download the four Alpaca news CSVs from Hugging Face.

Dataset: David-Chew-HL/Tech-Stocks-News, files {TICKER}_Alpaca_NLP.csv.
Total ~535 MB. Files are git-ignored; this script just (re)fetches them
into data/raw/ so the pipeline is reproducible from scratch.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]

for t in TICKERS:
    p = hf_hub_download(repo_id="David-Chew-HL/Tech-Stocks-News",
                        filename=f"{t}_Alpaca_NLP.csv",
                        repo_type="dataset",
                        local_dir=ROOT / "data" / "raw",
                        local_dir_use_symlinks=False)
    print(t, "->", p)
