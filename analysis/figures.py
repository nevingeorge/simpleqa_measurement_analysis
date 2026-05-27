"""
Three headline figures for the paper.

Fig 1 — fig1_d_study_item_counts.png
    Horizontal bar chart: per domain, current item count vs. the number of
    items required to reach G ≥ 0.70 (group decisions) and G ≥ 0.80
    (individual decisions). Under-provisioned domains are annotated.

Fig 2 — fig2_model_domain_heatmap.png
    Model × domain accuracy heatmap. Columns are sorted hardest-first;
    rows are sorted best-model-first. Domain columns where G ≥ 0.70 are
    marked with an asterisk to indicate reliable sub-scores.

Fig 3 — fig3_rank_change.png
    Slope plot (spaghetti chart) comparing original SimpleQA ranks with
    Construction 2 rebalanced ranks. The flagship trio is highlighted in
    colour; all other models are grey.

All figures are written to config.FIGURES_DIR by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.gtheory import (
    estimate_variance_components,
    g_coefficient,
    per_domain_table,
)
from analysis.reweight import per_domain_accuracy, rank_change_table
from config import (
    FIGURES_DIR,
    FLAGSHIP_KEYS,
    G_THRESHOLD_GROUP,
    G_THRESHOLD_INDIVIDUAL,
    RESULTS_DIR,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_d_study(long_df: pd.DataFrame, out_path: Path) -> None:
    """
    Fig 1: D-study bar chart — current vs required item counts per domain.

    For each domain, three horizontal bars show:
      - Current item count (blue)
      - Items needed for G ≥ 0.70 group-level decisions (orange)
      - Items needed for G ≥ 0.80 individual decisions (red)

    Domains whose current count falls below the G ≥ 0.70 threshold are
    annotated "under-provisioned". Required counts that exceed the x-axis
    cap (10% above the largest current count) are plotted at the cap.

    Parameters
    ----------
    long_df  : score table (model_key, item_id, topic, score).
    out_path : destination PNG path.
    """
    vc = estimate_variance_components(long_df)
    pd_table = per_domain_table(long_df, vc).sort_values("n_current", ascending=True)

    domains = pd_table["domain"].tolist()
    n_current = pd_table["n_current"].to_numpy()
    n_req_group = pd_table[f"n_required_at_{G_THRESHOLD_GROUP:.2f}"].to_numpy()
    n_req_ind = pd_table[f"n_required_at_{G_THRESHOLD_INDIVIDUAL:.2f}"].to_numpy()

    # Cap infinite required-n at 10% above the largest current count for plotting.
    cap = max(n_current.max(), 2000)
    n_req_group_plot = np.minimum(n_req_group, cap * 1.1)
    n_req_ind_plot = np.minimum(n_req_ind, cap * 1.1)

    y = np.arange(len(domains))
    h = 0.27  # bar height; three bars per domain spaced by h
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(domains) + 2))
    ax.barh(y + h, n_current,        height=h, label="current item count",  color="#3b6ea8")
    ax.barh(y,      n_req_group_plot, height=h, label=f"required at G≥{G_THRESHOLD_GROUP}", color="#e0a050")
    ax.barh(y - h, n_req_ind_plot,   height=h, label=f"required at G≥{G_THRESHOLD_INDIVIDUAL}", color="#c44e4e")

    for i, row in pd_table.reset_index(drop=True).iterrows():
        if not row["adequate_for_group"]:
            ax.text(row["n_current"] + 5, i + h,
                    "  under-provisioned", va="center", fontsize=8, color="#c44e4e")

    ax.set_yticks(y)
    ax.set_yticklabels(domains)
    ax.set_xlabel("items per domain")
    ax.set_title("D-study: current vs required SimpleQA item counts per domain")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_model_domain_heatmap(long_df: pd.DataFrame, out_path: Path) -> None:
    """
    Fig 2: Model × domain accuracy heatmap with reliability annotations.

    Layout:
      - Rows: models, sorted by overall accuracy (best at top).
      - Columns: domains, sorted by mean accuracy across models (hardest left).
      - Cell values: per-model per-domain accuracy, also annotated as text.
      - Column headers include the domain's G-coefficient; columns with
        G ≥ G_THRESHOLD_GROUP (0.70) are marked with an asterisk (*).

    The colour scale runs from 0 to max(0.5, observed max) so that very low
    accuracy models still show contrast between domains.

    Parameters
    ----------
    long_df  : score table (model_key, item_id, topic, score).
    out_path : destination PNG path.
    """
    pda = per_domain_accuracy(long_df)

    # Sort rows (models) by descending overall accuracy.
    overall = long_df.groupby("model_key")["score"].mean().sort_values(ascending=False)
    pda = pda.loc[overall.index]

    # Sort columns (domains) by ascending mean accuracy (hardest on the left).
    pda = pda[pda.mean(axis=0).sort_values().index]

    # Compute per-domain G-coefficient using current item counts.
    vc = estimate_variance_components(long_df)
    n_per_domain = long_df.groupby("topic")["item_id"].nunique().to_dict()
    g_per_domain = {d: g_coefficient(vc, n_per_domain[d]) for d in pda.columns}

    fig, ax = plt.subplots(figsize=(1.0 * len(pda.columns) + 2,
                                     0.45 * len(pda.index) + 2))
    im = ax.imshow(pda.values, aspect="auto", cmap="viridis", vmin=0,
                   vmax=max(0.5, pda.values.max()))
    ax.set_xticks(range(len(pda.columns)))
    ax.set_xticklabels(
        [
            f"{d}\nG={g_per_domain[d]:.2f}"
            + (" *" if g_per_domain[d] >= G_THRESHOLD_GROUP else "")
            for d in pda.columns
        ],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(pda.index)))
    ax.set_yticklabels(pda.index)

    # Annotate each cell with its numeric accuracy value.
    for i, model in enumerate(pda.index):
        for j, dom in enumerate(pda.columns):
            v = pda.iloc[i, j]
            # Use white text on dark cells, black text on light cells.
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if v < 0.3 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("accuracy")
    ax.set_title("Model × domain accuracy   (* marks columns with G ≥ "
                 f"{G_THRESHOLD_GROUP:.2f})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_rank_change(long_df: pd.DataFrame, out_path: Path) -> None:
    """
    Fig 3: Slope plot comparing original vs rebalanced model rankings.

    Each model is a line connecting its original rank (left) to its
    Construction 2 rebalanced rank (right). The y-axis is inverted so
    rank 1 (best) sits at the top. The flagship trio (config.FLAGSHIP_KEYS)
    is drawn in red with a heavier line; all other models are grey.

    Parameters
    ----------
    long_df  : score table (model_key, item_id, topic, score).
    out_path : destination PNG path.
    """
    rct = rank_change_table(long_df)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rct) + 2))

    for model_key, row in rct.iterrows():
        x = [0, 1]
        y = [row["rank_original"], row["rank_rebalanced"]]
        color = "#c44e4e" if row["is_flagship"] else "#888888"
        lw = 2.2 if row["is_flagship"] else 1.0
        ax.plot(x, y, marker="o", color=color, linewidth=lw,
                label=model_key if row["is_flagship"] else None)
        ax.annotate(model_key, (1.02, row["rank_rebalanced"]),
                    fontsize=8, va="center",
                    color=color, weight="bold" if row["is_flagship"] else "normal")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original SimpleQA", "Rebalanced (Construction 2)"])
    ax.invert_yaxis()  # rank 1 at the top
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Model rank changes under domain-equal re-weighting")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_all_figures(long_df: pd.DataFrame, out_dir: Path | None = None) -> None:
    """
    Generate all three headline figures and save them to out_dir.

    Parameters
    ----------
    long_df : score table (model_key, item_id, topic, score).
    out_dir : output directory; defaults to config.FIGURES_DIR.
    """
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building figures ...")
    fig_d_study(long_df, out_dir / "fig1_d_study_item_counts.png")
    fig_model_domain_heatmap(long_df, out_dir / "fig2_model_domain_heatmap.png")
    fig_rank_change(long_df, out_dir / "fig3_rank_change.png")


if __name__ == "__main__":
    long_df = pd.read_parquet(RESULTS_DIR / "scores_long.parquet")
    make_all_figures(long_df)
