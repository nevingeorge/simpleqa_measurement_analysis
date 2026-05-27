"""
Construction 2 domain re-weighting and rank-change analysis.

Each item gets weight 1/(D * n_d) so every domain contributes equally (1/D)
to the final score regardless of how many items it contains. This is
mathematically equivalent to taking the unweighted mean of per-domain
accuracies, which is what `rebalanced_scores()` computes.

We compare original SimpleQA scores (natural item-count weights) against
the rebalanced scores and report:
  - Per-model original vs rebalanced accuracy
  - Rank changes (positive = moved up under rebalancing)
  - Spearman rank correlation between the two orderings
  - Highlighted results for the flagship trio (see config.FLAGSHIP_KEYS)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

from config import FLAGSHIP_KEYS, RESULTS_DIR


def per_domain_accuracy(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model, per-domain accuracy.

    Parameters
    ----------
    long_df : DataFrame with columns model_key, item_id, topic, score.

    Returns
    -------
    DataFrame with models on rows and domains on columns; cell values are
    mean accuracy (0–1) for that model–domain pair.
    """
    return long_df.pivot_table(
        index="model_key", columns="topic", values="score", aggfunc="mean"
    )


def original_scores(long_df: pd.DataFrame) -> pd.Series:
    """
    Compute the natural SimpleQA score for each model.

    Scores are the mean over all items, which implicitly weights each domain
    by its natural item count (e.g. Science & Technology with 858 items
    contributes ~22% of the score, Video Games with 135 items contributes ~3%).

    Returns a Series indexed by model_key.
    """
    return long_df.groupby("model_key")["score"].mean()


def rebalanced_scores(long_df: pd.DataFrame) -> pd.Series:
    """
    Compute Construction 2 re-weighted scores.

    Each domain contributes exactly 1/D to the final score, regardless of its
    item count. Computed as the unweighted mean of per-domain accuracies.

    Returns a Series indexed by model_key.
    """
    return per_domain_accuracy(long_df).mean(axis=1)


def rank_change_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table comparing original and rebalanced model rankings.

    Columns
    -------
    score_original    : natural SimpleQA score (item-count weighted)
    score_rebalanced  : Construction 2 score (domain-equal weighted)
    delta             : rebalanced - original (positive = score increased)
    rank_original     : rank under natural weighting (1 = best)
    rank_rebalanced   : rank under rebalanced weighting (1 = best)
    rank_delta        : rank_original - rank_rebalanced (positive = moved up)
    is_flagship       : True for models in config.FLAGSHIP_KEYS

    Returns rows sorted by rank_original.
    """
    orig = original_scores(long_df)
    reb = rebalanced_scores(long_df)
    df = pd.DataFrame({
        "score_original": orig,
        "score_rebalanced": reb,
        "delta": reb - orig,
    })
    df["rank_original"] = df["score_original"].rank(ascending=False, method="min").astype(int)
    df["rank_rebalanced"] = df["score_rebalanced"].rank(ascending=False, method="min").astype(int)
    df["rank_delta"] = df["rank_original"] - df["rank_rebalanced"]  # positive = moved up
    df["is_flagship"] = df.index.isin(FLAGSHIP_KEYS)
    return df.sort_values("rank_original")


def report(long_df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """
    Run the full reweighting analysis and save outputs.

    Prints per-model×domain accuracy, the rank-change table, Spearman rank
    correlation, and flagship-trio highlights. Writes two CSVs to out_dir.

    Parameters
    ----------
    long_df : score table (model_key, item_id, topic, score columns).
    out_dir : directory for output CSVs; defaults to config.RESULTS_DIR.

    Returns
    -------
    dict with keys: per_domain_accuracy, rank_change_table,
                    spearman_rho, spearman_p.
    """
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pda = per_domain_accuracy(long_df)
    pda.to_csv(out_dir / "per_model_domain_accuracy.csv")
    print("Per-model x domain accuracy:")
    print(pda.round(3).to_string())

    rct = rank_change_table(long_df)
    rct.to_csv(out_dir / "rank_change_table.csv")
    print("\nRank changes under Construction 2 re-weighting:")
    print(rct.round(3).to_string())

    rho, pval = spearmanr(rct["rank_original"], rct["rank_rebalanced"])
    print(f"\nSpearman rank correlation (original vs rebalanced): "
          f"rho = {rho:.3f}, p = {pval:.3g}")

    print("\nFlagship trio rank changes:")
    print(rct[rct["is_flagship"]].round(3).to_string())

    return {
        "per_domain_accuracy": pda,
        "rank_change_table": rct,
        "spearman_rho": rho,
        "spearman_p": pval,
    }


if __name__ == "__main__":
    long_df = pd.read_parquet(RESULTS_DIR / "scores_long.parquet")
    report(long_df)
