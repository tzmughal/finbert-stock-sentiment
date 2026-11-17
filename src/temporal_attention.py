"""Temporal attention model over daily sentiment (genuinely implemented).

For each ticker, one row per calendar day: mean FinBERT score, positive
share, negative share, headline count (from the scored news sample) plus
the day's stock return (Yahoo Finance prices via yfinance; Stooq CSV
fallback if yfinance fails). A 5-day window of these daily features is
fed to a small PyTorch model: linear projection -> tanh -> learned
attention pooling over the 5 days -> sigmoid, predicting whether the
NEXT day's close is up (1) or down (0).

Split is by time, not at random: windows whose target day is on or before
2023-09-30 train the model; later windows are the test set. Test accuracy
is reported against a majority-class baseline. This is a small
demonstration model on a sampled headline set — it is not a trading
signal, and the README says so.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
SEQ_LEN = 5
SPLIT_DATE = "2023-09-30"


def load_prices() -> pd.DataFrame:
    """Daily closes from Yahoo Finance's chart API (no key).

    Note: the yfinance Python client could not reach Yahoo from this VM
    (its cookie/crumb handshake timed out for every ticker), so prices are
    fetched with plain HTTPS calls to query1.finance.yahoo.com instead.
    Timestamps are exchange-local trading days.
    """
    import json
    import urllib.request

    cache = ROOT / "data" / "processed" / "prices.csv"
    if cache.exists():
        px = pd.read_csv(cache, parse_dates=["date"])
        if len(px) > 100:
            px["date"] = pd.to_datetime(px["date"]).dt.date.astype(str)
            return px
        cache.unlink()  # stale/empty cache from a failed fetch; refetch
    frames = []
    for t in TICKERS:
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{t}?period1=1672531200&period2=1706745600&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        res = data["chart"]["result"][0]
        ts = res["timestamp"]
        closes = res["indicators"]["quote"][0]["close"]
        d = pd.DataFrame({
            "date": pd.to_datetime(ts, unit="s").date.astype(str),
            "ticker": t,
            "close": closes,
        }).dropna()
        frames.append(d)
        print(f"{t}: {len(d)} trading days of closes")
    px = pd.concat(frames, ignore_index=True)
    if len(px) == 0:
        raise RuntimeError("no prices fetched from Yahoo chart API")
    px.to_csv(cache, index=False)
    return px


class TemporalAttention(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, seq, features)
        h = torch.tanh(self.proj(x))                       # (b, s, h)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # (b, s)
        ctx = (h * w.unsqueeze(-1)).sum(dim=1)              # (b, h)
        return torch.sigmoid(ctx.sum(dim=1, keepdim=True)), w


def build_sequences(daily: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    X, y, meta = [], [], []
    feats = ["mean_score", "pos_share", "neg_share", "log_n", "return"]
    for ticker, g in daily.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        arr = g[feats].to_numpy(dtype=np.float32)
        dates = g["date"].tolist()
        nxt = g["next_return"].to_numpy()
        for i in range(len(g) - SEQ_LEN):
            X.append(arr[i:i + SEQ_LEN])
            y.append(1 if nxt[i + SEQ_LEN] > 0 else 0)
            meta.append({"ticker": ticker, "target_date": dates[i + SEQ_LEN]})
    return np.stack(X), np.array(y), meta


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    news = pd.read_csv(ROOT / "data" / "processed" / "news_scored.csv",
                       parse_dates=["date"])
    news["date"] = pd.to_datetime(news["date"]).dt.date.astype(str)
    daily = (news.groupby(["ticker", "date"])
             .agg(mean_score=("score", "mean"),
                  pos_share=("predicted_label", lambda s: (s == "positive").mean()),
                  neg_share=("predicted_label", lambda s: (s == "negative").mean()),
                  n=("score", "size"))
             .reset_index())
    daily["log_n"] = np.log1p(daily["n"])
    daily.to_csv(ROOT / "data" / "processed" / "news_daily.csv", index=False)
    print(f"daily rows: {len(daily)} ({daily.groupby('ticker').size().to_dict()})")

    px = load_prices()
    px = px.sort_values(["ticker", "date"])
    px["return"] = px.groupby("ticker")["close"].pct_change()
    px["next_return"] = px.groupby("ticker")["return"].shift(-1)
    merged = daily.merge(px[["ticker", "date", "return", "next_return"]],
                         on=["ticker", "date"], how="left")
    merged = merged.dropna(subset=["return", "next_return"])
    print(f"daily rows with prices: {len(merged)}")

    X, y, meta = build_sequences(merged)
    if len(X) == 0:
        raise RuntimeError("no 5-day windows could be built (no overlapping "
                           "sentiment/price days)")
    meta_df = pd.DataFrame(meta)
    is_train = meta_df["target_date"] <= SPLIT_DATE
    Xtr, ytr = torch.tensor(X[is_train]), torch.tensor(y[is_train], dtype=torch.float32)
    Xte, yte = torch.tensor(X[~is_train]), torch.tensor(y[~is_train], dtype=torch.float32)
    print(f"windows: train={len(Xtr)} test={len(Xte)} "
          f"(test up-share={yte.mean():.3f})")

    # Standardise on train only.
    mu, sd = Xtr.mean(dim=(0, 1)), Xtr.std(dim=(0, 1)) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = TemporalAttention(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    for epoch in range(60):
        model.train()
        opt.zero_grad()
        pred, _ = model(Xtr)
        loss = loss_fn(pred.squeeze(-1), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_te, w_te = model(Xte)
        p_tr, _ = model(Xtr)
    acc_te = ((p_te.squeeze(-1) > 0.5).float() == yte).float().mean().item()
    acc_tr = ((p_tr.squeeze(-1) > 0.5).float() == ytr).float().mean().item()
    majority = max(yte.mean().item(), 1 - yte.mean().item())
    attn = w_te.mean(dim=0).tolist()  # mean attention per position (oldest..newest)

    metrics = {
        "model": "PyTorch temporal attention: Linear(5 feats->32) + learned "
                 "softmax attention over 5 days -> next-day up/down",
        "seq_len_days": SEQ_LEN,
        "features": ["mean_score", "pos_share", "neg_share", "log_n", "return"],
        "train_windows": int(len(Xtr)),
        "test_windows": int(len(Xte)),
        "split": f"train target_date <= {SPLIT_DATE}, test after",
        "train_accuracy": round(acc_tr, 4),
        "test_accuracy": round(acc_te, 4),
        "majority_baseline_test_accuracy": round(majority, 4),
        "mean_attention_weight_oldest_to_newest_day": [round(float(a), 3) for a in attn],
    }
    out = ROOT / "results" / "metrics_temporal_attention.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
