"""
Generalizability theory analysis.

Design:
    score ~ 1 + (1 | model) + (1 | domain) + (1 | model:domain) + (1 | item:domain)

Object of measurement: model
Facet: items nested in domains, crossed with model.

We fit the LMM with statsmodels MixedLM (REML where supported, ML fallback for
the cross-classified pieces). For each domain d we compute the relative
G-coefficient:

    G_d = sigma2_m / (sigma2_m + sigma2_residual / n_i_d)

The D-study inverts that to solve for the minimum n_i_d at G = 0.80 and 0.70.

Cluster bootstrap (resample items within domain) gives CIs on the variance
components and on G_d.

Note on the assumption stretch: G-theory was developed for continuous outcomes.
We treat 0/1 scores as continuous (an ANOVA-style approximation). For domain
ranking and D-study extrapolation this is well-behaved enough; we document it
as a limitation rather than pretending it's exact.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import (
    BOOTSTRAP_N,
    G_THRESHOLD_GROUP,
    G_THRESHOLD_INDIVIDUAL,
    RANDOM_SEED,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Variance component estimation
# ---------------------------------------------------------------------------
@dataclass
class VarianceComponents:
    sigma2_model: float        # between-model variance (the SIGNAL)
    sigma2_domain: float       # between-domain variance
    sigma2_model_domain: float # model x domain interaction (key finding driver)
    sigma2_item_in_domain: float
    sigma2_residual: float     # confounded with model x item residual
    n_obs: int

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_variance_components(long_df: pd.DataFrame) -> VarianceComponents:
    """
    Estimate variance components via a sequence of nested mixed models.

    long_df columns: model_key, item_id, topic, score

    statsmodels MixedLM doesn't directly handle 4 crossed/nested random
    effects, so we use a method-of-moments style decomposition built from
    one-way ANOVA on each grouping factor. This is the standard approach
    in classic G-theory references (Brennan, 2001, Ch. 3) when full
    REML for the full model isn't feasible.

    Steps:
      1. Total variance.
      2. sigma2_model     from one-way model on `model_key`
      3. sigma2_domain    from one-way model on `topic`
      4. sigma2_model_domain from interaction means
      5. sigma2_item_in_domain via item means within domain
      6. sigma2_residual = total - sum of above (clipped at 0)
    """
    df = long_df.copy()
    y = df["score"].astype(float)
    n_obs = len(df)

    grand = y.mean()
    total_var = y.var(ddof=1)

    # --- sigma2_model ---
    # Classic one-way ANOVA estimator: (MS_between - MS_within) / n_per_group.
    # MS_within here is the pooled within-model variance, which serves as an
    # approximation of the residual (model × item interaction confounded with error).
    mod_means = df.groupby("model_key")["score"].agg(["mean", "count"])
    n_models = len(mod_means)
    n_per_model = mod_means["count"].mean()  # balanced design, arithmetic mean is fine
    ss_between_models = ((mod_means["mean"] - grand) ** 2 * mod_means["count"]).sum()
    ms_between = ss_between_models / max(n_models - 1, 1)
    ms_within_model = (
        ((df["score"] - df.groupby("model_key")["score"].transform("mean")) ** 2).sum()
        / max(n_obs - n_models, 1)
    )
    sigma2_model = max((ms_between - ms_within_model) / n_per_model, 0.0)

    # --- sigma2_domain ---
    # Same one-way ANOVA estimator applied to the domain grouping.
    dom_means = df.groupby("topic")["score"].agg(["mean", "count"])
    n_dom = len(dom_means)
    n_per_dom = dom_means["count"].mean()
    ss_between_dom = ((dom_means["mean"] - grand) ** 2 * dom_means["count"]).sum()
    ms_between_dom = ss_between_dom / max(n_dom - 1, 1)
    ms_within_dom = (
        ((df["score"] - df.groupby("topic")["score"].transform("mean")) ** 2).sum()
        / max(n_obs - n_dom, 1)
    )
    sigma2_domain = max((ms_between_dom - ms_within_dom) / n_per_dom, 0.0)

    # --- sigma2_model_domain interaction ---
    # The interaction is the deviation of each model×domain cell mean from the
    # purely additive prediction (grand mean + model effect + domain effect).
    # A large sigma2_model_domain means rankings differ across domains —
    # the key finding that aggregate scores can mask.
    cell_means = df.groupby(["model_key", "topic"])["score"].agg(["mean", "count"])
    additive_pred = (
        df.assign(
            m_eff=df.groupby("model_key")["score"].transform("mean") - grand,
            d_eff=df.groupby("topic")["score"].transform("mean") - grand,
        )
        .assign(pred=lambda x: grand + x["m_eff"] + x["d_eff"])
    )
    interaction_dev = (
        df.groupby(["model_key", "topic"])["score"].mean()
        - additive_pred.groupby(["model_key", "topic"])["pred"].first()
    )
    n_cells = len(interaction_dev)
    avg_cell_n = cell_means["count"].mean()
    # Degrees of freedom for interaction: (n_models-1)(n_domains-1)
    ms_interaction = (interaction_dev ** 2).sum() / max(n_cells - n_models - n_dom + 1, 1)
    # Remove the residual's contribution to the interaction MS before estimating σ²_md.
    sigma2_model_domain = max(ms_interaction - ms_within_model / avg_cell_n, 0.0)

    # --- sigma2_item_in_domain ---
    # Items are nested in domains: variance is computed from item means within
    # each domain, then averaged across domains. The model variance is subtracted
    # because item means are averaged over models, so they include a model-effect
    # contribution of sigma2_model / n_models.
    item_means = df.groupby(["topic", "item_id"])["score"].mean()
    sigma2_item_in_domain = max(
        item_means.groupby(level="topic").var(ddof=1).mean() - ms_within_model / n_models,
        0.0,
    )

    # --- sigma2_residual ---
    # Residual = total variance not explained by the four named components.
    # Clipped at a small epsilon to avoid zero denominators in G-coefficient.
    accounted = (
        sigma2_model + sigma2_domain + sigma2_model_domain + sigma2_item_in_domain
    )
    sigma2_residual = max(total_var - accounted, 1e-6)

    return VarianceComponents(
        sigma2_model=sigma2_model,
        sigma2_domain=sigma2_domain,
        sigma2_model_domain=sigma2_model_domain,
        sigma2_item_in_domain=sigma2_item_in_domain,
        sigma2_residual=sigma2_residual,
        n_obs=n_obs,
    )


# ---------------------------------------------------------------------------
# G-coefficients and D-study
# ---------------------------------------------------------------------------
def g_coefficient(vc: VarianceComponents, n_items: int) -> float:
    """
    Relative G-coefficient (Brennan 2001, eq. 3.27 adapted):

        G = sigma2_m / (sigma2_m + sigma2_residual / n_items)

    We use the residual term (which absorbs model x item interaction) as the
    averageable noise. Domain variance and model x domain interaction are
    excluded because they're constants for a within-domain subscore.
    """
    if vc.sigma2_model <= 0:
        return 0.0
    noise = vc.sigma2_residual / max(n_items, 1)
    return vc.sigma2_model / (vc.sigma2_model + noise)


def required_n_for_g(vc: VarianceComponents, target_g: float) -> int:
    """
    Solve for n_items such that G(n_items) >= target_g.

        target = sigma2_m / (sigma2_m + sigma2_residual / n)
    =>  n      = sigma2_residual / (sigma2_m * (1 - target) / target)
    """
    if vc.sigma2_model <= 0 or target_g >= 1.0:
        return math.inf
    n = vc.sigma2_residual / (vc.sigma2_model * (1.0 - target_g) / target_g)
    return int(math.ceil(n))


def per_domain_table(long_df: pd.DataFrame, vc: VarianceComponents) -> pd.DataFrame:
    """
    Build the per-domain table: current n, G at current n, required n
    at the two thresholds, and the resulting flag.
    """
    rows = []
    for domain, group in long_df.groupby("topic"):
        n_items = group["item_id"].nunique()
        g_now = g_coefficient(vc, n_items)
        n_req_individual = required_n_for_g(vc, G_THRESHOLD_INDIVIDUAL)
        n_req_group = required_n_for_g(vc, G_THRESHOLD_GROUP)
        rows.append({
            "domain": domain,
            "n_current": n_items,
            "G_current": g_now,
            f"n_required_at_{G_THRESHOLD_GROUP:.2f}": n_req_group,
            f"n_required_at_{G_THRESHOLD_INDIVIDUAL:.2f}": n_req_individual,
            "adequate_for_group": n_items >= n_req_group,
            "adequate_for_individual": n_items >= n_req_individual,
        })
    return pd.DataFrame(rows).sort_values("n_current", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_vc(
    long_df: pd.DataFrame, n_boot: int = BOOTSTRAP_N, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Cluster bootstrap for variance component confidence intervals.

    Items are the cluster unit. Within each domain, item IDs are resampled
    with replacement, and all model rows for a sampled item move together.
    This preserves the item-level correlation structure (the same item being
    hard or easy for all models) while treating items as the exchangeable unit
    — consistent with G-theory's assumption that items are random draws from a
    domain universe.

    Resampling within domain (not across domains) keeps domain membership
    constant, so sigma2_domain and sigma2_model_domain estimates reflect
    genuine domain-level heterogeneity rather than resampling artefacts.

    Parameters
    ----------
    long_df : score table with columns model_key, item_id, topic, score.
    n_boot  : number of bootstrap replicates.
    seed    : NumPy random seed for reproducibility (config.RANDOM_SEED).

    Returns
    -------
    DataFrame of shape (n_boot, 6): one row per replicate, columns are the
    five variance components plus boot_iter.
    """
    rng = np.random.default_rng(seed)
    domains = long_df["topic"].unique()
    # Pre-cache item ID arrays per domain to avoid repeated boolean indexing.
    items_by_domain = {d: long_df.loc[long_df["topic"] == d, "item_id"].unique()
                       for d in domains}

    # Index by item_id for O(1) row lookup during resampling.
    indexed = long_df.set_index("item_id", drop=False)

    records = []
    for b in range(n_boot):
        sampled_ids = []
        for d in domains:
            items = items_by_domain[d]
            sampled_ids.extend(rng.choice(items, size=len(items), replace=True))
        # Build the resampled long frame
        resampled = indexed.loc[sampled_ids].reset_index(drop=True)
        # Re-id duplicates so item_id remains unique-ish within the replicate;
        # for variance components computed via group means, dups are fine
        # because they reinforce that item's weight (which is the point of the
        # bootstrap).
        vc = estimate_variance_components(resampled)
        records.append({**vc.as_dict(), "boot_iter": b})
        if (b + 1) % 100 == 0:
            print(f"  bootstrap iteration {b+1}/{n_boot}")
    return pd.DataFrame(records)


def bootstrap_summary(boot_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in boot_df.columns if c.startswith("sigma2_")]
    summary = boot_df[cols].agg(["mean", "std",
                                  lambda x: x.quantile(0.025),
                                  lambda x: x.quantile(0.975)]).T
    summary.columns = ["mean", "sd", "ci_lo_2.5", "ci_hi_97.5"]
    return summary


# ---------------------------------------------------------------------------
# Entry point: full G-study from a long-format scores file
# ---------------------------------------------------------------------------
def run_gstudy(long_df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"G-study on {len(long_df):,} observations "
          f"({long_df['model_key'].nunique()} models x "
          f"{long_df['item_id'].nunique()} items x "
          f"{long_df['topic'].nunique()} domains)")

    vc = estimate_variance_components(long_df)
    print("\nPoint estimates of variance components:")
    for k, v in vc.as_dict().items():
        print(f"  {k:30s} = {v:.6f}")

    per_dom = per_domain_table(long_df, vc)
    print("\nPer-domain table:")
    print(per_dom.to_string(index=False))
    per_dom.to_csv(out_dir / "per_domain_gstudy.csv", index=False)

    print(f"\nRunning cluster bootstrap ({BOOTSTRAP_N} iterations)...")
    boot_df = cluster_bootstrap_vc(long_df)
    boot_df.to_parquet(out_dir / "bootstrap_vc.parquet", index=False)
    summary = bootstrap_summary(boot_df)
    summary.to_csv(out_dir / "variance_components_ci.csv")
    print("\nVariance components with 95% bootstrap CIs:")
    print(summary.to_string())

    return {
        "vc": vc,
        "per_domain": per_dom,
        "bootstrap": boot_df,
        "bootstrap_summary": summary,
    }


if __name__ == "__main__":
    long_path = RESULTS_DIR / "scores_long.parquet"
    if not long_path.exists():
        raise SystemExit(
            "scores_long.parquet not found. Run `./scripts/sync_results.sh` first."
        )
    long_df = pd.read_parquet(long_path)
    run_gstudy(long_df)
