"""
Local entrypoints for running eval jobs and syncing results.

Use the wrapper scripts (recommended):

    ./scripts/run_eval.sh --model-key gemma-3-12b --pilot
    ./scripts/sync_results.sh

Or invoke Modal in module mode directly:

    MODAL_BUILD_VALIDATION=ignore modal run -m modal_eval.orchestrator::run_eval --model-key gemma-3-12b --pilot
    MODAL_BUILD_VALIDATION=ignore modal run -m modal_eval.orchestrator::sync_results
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import DATA_DIR, MODELS, MODELS_BY_KEY, RESULTS_DIR
from modal_eval.app import app
from modal_eval.grader import delete_grades, fetch_all_grades, grade_model, grade_model_vllm
from modal_eval.inference import extract_final_answer, fetch_responses, run_model, save_responses


@app.local_entrypoint()
def run_eval(
    model_key: str,
    pilot: bool = False,
    skip_grading: bool = False,
    skip_inference: bool = False,
    force_regrade: bool = False,
    grader_model: str = "gpt-4o-mini",
    n_grader_shards: int = 5,
):
    """Run one model end-to-end (inference + grading)."""
    if model_key not in MODELS_BY_KEY:
        raise SystemExit(
            f"Unknown model key '{model_key}'. Valid keys: "
            f"{list(MODELS_BY_KEY.keys())}"
        )

    items_path = DATA_DIR / "simpleqa_prepared.parquet"
    if not items_path.exists():
        raise SystemExit(
            f"{items_path} not found. Run `python data/prepare_data.py` first."
        )
    items_df = pd.read_parquet(items_path)
    print(f"Loaded {len(items_df):,} items from {items_path}")

    if pilot:
        # Stratified pilot: a few items per topic so we exercise prompt formatting
        # and answer extraction across all domains. Use .head() not .apply() so
        # pandas 3.x keeps the group column (topic).
        items_df = (
            items_df.groupby("topic", group_keys=False)
            .head(10)
            .reset_index(drop=True)
        )
        print(f"PILOT mode: using {len(items_df)} items")

    if skip_inference:
        print("Skipping inference (--skip-inference); loading saved responses")
        responses_df = pd.DataFrame(fetch_responses.remote(model_key))
    else:
        responses_df = run_model(model_key, items_df)
        save_responses.remote(model_key, responses_df.to_dict("records"))

    if skip_grading:
        print("Skipping grading (--skip-grading)")
        return

    if force_regrade:
        print("Deleting existing grades (--force-regrade)")
        delete_grades.remote(model_key)

    # Re-extract from raw (fixes gpt-oss and other reasoning-model parsing updates).
    responses_df["extracted"] = responses_df["raw_response"].apply(
        lambda raw: extract_final_answer(raw, model_key)
    )
    if model_key.startswith("gpt-oss"):
        ext_lens = responses_df["extracted"].str.len()
        print(
            f"Re-extracted gpt-oss answers: median={int(ext_lens.median())} chars, "
            f"max={int(ext_lens.max())} chars"
        )

    grade_df = items_df[["item_id", "topic", "problem", "answer"]].merge(
        responses_df[["item_id", "extracted"]],
        on="item_id",
        how="inner",
    )
    if "/" in grader_model:
        summary = grade_model_vllm.remote(
            model_key, grade_df.to_dict("records"),
            grader_hf_name=grader_model,
            n_shards=n_grader_shards,
        )
    else:
        summary = grade_model.remote(
            model_key, grade_df.to_dict("records"), grader_model=grader_model
        )
    print("\n=== Grading summary ===")
    print(summary)


@app.local_entrypoint()
def sync_results():
    """Pull grades from the Modal volume and write local score files."""
    print("Fetching grades from Modal volume ...")
    rows = fetch_all_grades.remote()
    if not rows:
        raise SystemExit("No grades found on the volume yet.")

    df = pd.DataFrame(rows)
    print(
        f"Got {len(df):,} graded responses across "
        f"{df['model_key'].nunique()} models"
    )

    # NOT_ATTEMPTED counts as 0 (incorrect) in the primary analysis.
    df["score"] = (df["verdict"] == "CORRECT").astype(int)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_path = RESULTS_DIR / "scores_long.parquet"
    df.to_parquet(long_path, index=False)
    print(f"Wrote long-format scores: {long_path}")

    wide = df.pivot_table(
        index="model_key", columns="item_id", values="score", aggfunc="first"
    )
    wide_path = RESULTS_DIR / "score_matrix.parquet"
    wide.to_parquet(wide_path)
    print(
        f"Wrote wide score matrix ({wide.shape[0]} models x "
        f"{wide.shape[1]} items): {wide_path}"
    )

    summary = (
        df.groupby("model_key")["score"]
        .agg(n="count", accuracy="mean")
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary_path = RESULTS_DIR / "per_model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Per-model summary:\n{summary.to_string(index=False)}")

    missing = set(m.key for m in MODELS) - set(df["model_key"].unique())
    if missing:
        print(f"\nWARNING: missing grades for models: {sorted(missing)}")
