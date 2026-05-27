"""
Download SimpleQA and prepare it for evaluation.

SimpleQA is hosted by OpenAI as a single CSV. Each row has a topic tag,
question, and reference answer. We:

1. Download the CSV.
2. Filter to the domains in config.DOMAINS_TO_KEEP (drops "Other").
3. Assign a stable item_id and save as parquet for fast loading.
4. Print a summary of items per domain so you can sanity-check.

Run this once locally before launching any Modal jobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

# Allow imports from the project root when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DOMAINS_TO_KEEP, SIMPLEQA_CSV_URL


def download_simpleqa(csv_path: Path) -> None:
    """Fetch the SimpleQA CSV from OpenAI's blob storage."""
    print(f"Downloading SimpleQA from {SIMPLEQA_CSV_URL} ...")
    response = requests.get(SIMPLEQA_CSV_URL, timeout=60)
    response.raise_for_status()
    csv_path.write_bytes(response.content)
    print(f"  saved to {csv_path}  ({len(response.content):,} bytes)")


def prepare() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = DATA_DIR / "simpleqa_raw.csv"
    if not raw_csv.exists():
        download_simpleqa(raw_csv)

    df = pd.read_csv(raw_csv)
    print(f"\nRaw SimpleQA: {len(df):,} rows, columns = {list(df.columns)}")

    # The official CSV has a 'metadata' column that contains a stringified dict
    # with a 'topic' key. Parse it out.
    if "topic" not in df.columns and "metadata" in df.columns:
        df["metadata"] = df["metadata"].apply(eval)  # CSV stores dict-as-string
        df["topic"] = df["metadata"].apply(lambda d: d.get("topic", "Other"))
        df["answer_type"] = df["metadata"].apply(lambda d: d.get("answer_type", ""))

    # Stable item id assigned BEFORE filtering, so we can audit drops later.
    df = df.reset_index(drop=True)
    df["item_id"] = df.index.map(lambda i: f"sqa_{i:05d}")

    print("\nTopic distribution (full dataset):")
    print(df["topic"].value_counts().to_string())

    # Filter to kept domains.
    keep_mask = df["topic"].isin(DOMAINS_TO_KEEP)
    print(f"\nDropping {(~keep_mask).sum():,} items in unwanted domains")
    df = df[keep_mask].copy()

    print(f"\nFinal item count: {len(df):,}")
    print("Per-domain item counts (analysis set):")
    print(df["topic"].value_counts().to_string())

    out_path = DATA_DIR / "simpleqa_prepared.parquet"
    df[["item_id", "topic", "problem", "answer"]].to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")
    return df


if __name__ == "__main__":
    prepare()
