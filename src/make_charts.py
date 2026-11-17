"""Charts for the README — built ONLY from this run's real outputs.

Reads data/processed/news_scored.csv, news_daily.csv,
results/metrics_phrasebank.json, results/metrics_finetune_head.json and
results/metrics_temporal_attention.json and writes PNGs to screenshots/.
Run after the pipeline:  python src/make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "font.size": 10})
ROOT = Path(__file__).resolve().parents[1]
SHOT = ROOT / "screenshots"
SHOT.mkdir(parents=True, exist_ok=True)
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
COLORS = {"positive": "#2e7d32", "neutral": "#90a4ae", "negative": "#c62828"}


def chart_sentiment_by_ticker(news: pd.DataFrame) -> None:
    counts = (news.groupby(["ticker", "predicted_label"]).size()
              .unstack(fill_value=0).reindex(TICKERS))
    for lab in ("positive", "neutral", "negative"):
        if lab not in counts:
            counts[lab] = 0
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(TICKERS))
    for lab in ("positive", "neutral", "negative"):
        ax.bar(TICKERS, pct[lab], bottom=bottom, label=lab, color=COLORS[lab])
        for i, v in enumerate(pct[lab]):
            if v >= 5:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center",
                        va="center", color="white", fontsize=9)
        bottom += pct[lab].values
    ax.set_title("FinBERT sentiment distribution by ticker\n"
                 f"(scored headlines, n={len(news)})")
    ax.set_ylabel("% of headlines")
    ax.legend(title="FinBERT label", loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3)
    fig.tight_layout()
    fig.savefig(SHOT / "sentiment_distribution_by_ticker.png")
    plt.close(fig)


def chart_sentiment_over_time(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in TICKERS:
        g = daily[daily["ticker"] == t].copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.sort_values("date")
        roll = g.set_index("date")["mean_score"].rolling("14D").mean()
        ax.plot(roll.index, roll.values, label=t, linewidth=1.5)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Mean FinBERT sentiment score over time (14-day rolling mean)\n"
                 "score = P(positive) - P(negative)")
    ax.set_ylabel("mean sentiment score")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(SHOT / "sentiment_over_time.png")
    plt.close(fig)


def chart_confusion(metrics: dict) -> None:
    cm = np.array(metrics["confusion_matrix_rows_true_cols_pred"])
    labels = metrics["confusion_matrix_labels"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — zero-shot FinBERT on PhraseBank test\n"
                 f"(n={metrics['n_test']}, accuracy={metrics['accuracy']:.3f})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(SHOT / "confusion_matrix_phrasebank.png")
    plt.close(fig)


def chart_model_comparison(pb: dict, head: dict, ta: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = ["Zero-shot\nFinBERT", "FinBERT embeddings\n+ logistic head"]
    accs = [pb["accuracy"], head["accuracy"]]
    base = [pb["majority_baseline_accuracy"], head["majority_baseline_accuracy"]]
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w / 2, accs, width=w, color="#1565c0", label="model")
    axes[0].bar(x + w / 2, base, width=w, color="#b0bec5", label="majority baseline")
    axes[0].set_xticks(x, names, fontsize=9)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("PhraseBank test accuracy")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for i, v in enumerate(accs):
        axes[0].text(i - w / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    names2 = ["Temporal attention\n(next-day direction)", "Majority\nbaseline"]
    vals = [ta["test_accuracy"], ta["majority_baseline_test_accuracy"]]
    axes[1].bar([0, 1], vals, color=["#6a1b9a", "#b0bec5"])
    axes[1].set_xticks([0, 1], names2, fontsize=9)
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Temporal attention, test windows n={ta['test_windows']}")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(SHOT / "model_comparison.png")
    plt.close(fig)


def main() -> None:
    import json
    news = pd.read_csv(ROOT / "data" / "processed" / "news_scored.csv")
    daily = pd.read_csv(ROOT / "data" / "processed" / "news_daily.csv")
    pb = json.loads((ROOT / "results" / "metrics_phrasebank.json").read_text())
    head = json.loads((ROOT / "results" / "metrics_finetune_head.json").read_text())
    ta = json.loads((ROOT / "results" / "metrics_temporal_attention.json").read_text())
    chart_sentiment_by_ticker(news)
    chart_sentiment_over_time(daily)
    chart_confusion(pb)
    chart_model_comparison(pb, head, ta)
    for p in sorted(SHOT.glob("*.png")):
        print(p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
