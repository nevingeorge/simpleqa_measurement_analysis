"""
Top-level orchestrator: runs the full local-side analysis pipeline.

Assumes:
  * data/simpleqa_prepared.parquet exists  (run data/prepare_data.py)
  * results/scores_long.parquet exists      (run `./scripts/sync_results.sh`
                                             after all eval jobs have finished)

Then:
  1. Fits the G-study and bootstraps variance components.
  2. Computes per-domain G-coefficients and the D-study.
  3. Computes Construction 2 re-weighted scores and rank changes.
  4. Builds the three headline figures.

Usage:
    python run_all.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.figures import make_all_figures
from analysis.gtheory import run_gstudy
from analysis.reweight import report as reweight_report
from config import FIGURES_DIR, RESULTS_DIR


def main():
    long_path = RESULTS_DIR / "scores_long.parquet"
    if not long_path.exists():
        raise SystemExit(
            f"{long_path} not found.\n"
            "Run `./scripts/sync_results.sh` after all eval jobs have finished."
        )
    long_df = pd.read_parquet(long_path)

    print("=" * 70)
    print("STEP 1: G-study (variance components, per-domain G, D-study)")
    print("=" * 70)
    run_gstudy(long_df)

    print("\n" + "=" * 70)
    print("STEP 2: Construction 2 re-weighting and rank changes")
    print("=" * 70)
    reweight_report(long_df)

    print("\n" + "=" * 70)
    print("STEP 3: Figures")
    print("=" * 70)
    make_all_figures(long_df)

    print("\nDone.")
    print(f"  Tables in: {RESULTS_DIR}")
    print(f"  Figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
