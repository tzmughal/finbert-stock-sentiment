"""Run the full pipeline end to end. Each step is idempotent.

    python run_pipeline.py [--skip-scoring]

Steps: prepare PhraseBank -> prepare news sample -> score PhraseBank test
-> evaluate -> train classifier head -> score news sample -> temporal
attention model -> charts.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "bin" / "python"


def run(*args: str) -> None:
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([str(PY), *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scoring", action="store_true",
                    help="skip the two FinBERT scoring steps (reuse existing outputs)")
    args = ap.parse_args()

    run("src/prepare_phrasebank.py")
    run("src/prepare_news.py")
    if not args.skip_scoring:
        run("src/score_sentiment.py", "--input", "data/processed/phrasebank_test.csv",
            "--text-col", "text", "--output", "data/processed/phrasebank_test_scored.csv")
    run("src/evaluate.py")
    run("src/finetune_head.py")
    if not args.skip_scoring:
        run("src/score_sentiment.py", "--input", "data/processed/news_sample.csv",
            "--text-col", "headline", "--output", "data/processed/news_scored.csv",
            "--max-length", "64")
    run("src/temporal_attention.py")
    run("src/make_charts.py")
    print("\nPipeline complete. See README.md for the verified results.")


if __name__ == "__main__":
    main()
