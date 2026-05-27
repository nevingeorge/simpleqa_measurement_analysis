# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research pipeline that evaluates 10 open-source LLMs on the SimpleQA benchmark and applies **generalizability theory (G-theory)** to assess whether SimpleQA's domain sampling is valid. The core research question: does SimpleQA have enough items per domain to reliably rank models?

Inference runs on Modal (cloud GPU), grading runs via GPT-4o-mini, and statistical analysis runs locally.

## Setup

```bash
pip install -r requirements.txt
pip install modal && modal setup

# One-time Modal secrets:
modal secret create huggingface-token HF_TOKEN=hf_...
modal secret create openai-api-key OPENAI_API_KEY=sk-...

# Prepare dataset (run once):
python data/prepare_data.py
```

## Key commands

```bash
# Pilot a single model (~90 items, validates the full pipeline):
./scripts/run_eval.sh --model-key gemma-3-12b --pilot

# Full eval for one model:
./scripts/run_eval.sh --model-key <key>

# Sync grades from Modal volume to local disk:
./scripts/sync_results.sh

# Run the full analysis (G-study, D-study, reweighting, figures):
python run_all.py

# Run individual analysis modules directly:
python analysis/gtheory.py
python analysis/reweight.py
python analysis/figures.py
```

Valid model keys: `gemma-3-12b`, `ministral-8b`, `qwen-3.5-9b`, `gpt-oss-20b`, `granite-4.0-h-small`, `gemma-3-27b`, `qwen-3.5-27b`, `gpt-oss-120b`, `llama-3.1-8b`, `ministral-3-14b`

## Pipeline architecture

The pipeline has three distinct stages:

**Stage 1: Modal eval** (`modal_eval/`)
- `app.py` — defines the shared Modal app, two container images (`vllm_image` for inference, `grader_image` for grading), and two persistent volumes (`hf-cache` for model weights, `simpleqa-results` for outputs). All other Modal modules import from here.
- `inference.py` — one `@app.function` is built per distinct GPU spec at import time (via `_build_inference_function()`). `run_model()` dispatches to the right one based on the model's `gpu` and `n_gpu` fields. All prompts are batched into a single `llm.generate()` call for vLLM's continuous batching.
- `grader.py` — calls GPT-4o-mini on Modal (CPU only) using the official simple-evals grading prompt. Retries on rate limit / timeout errors with exponential backoff.
- `orchestrator.py` — local entrypoints that load the prepared dataset, invoke inference remotely, save responses to the Modal volume, invoke grading, and sync grades locally.

**Stage 2: G-theory analysis** (`analysis/gtheory.py`)
- Fits a `score ~ model + domain + model:domain + item:domain` design using a method-of-moments variance decomposition (not full REML, which statsmodels can't handle for 4 crossed/nested effects).
- Computes per-domain G-coefficients and a D-study (how many items would be needed per domain to hit G=0.70 or G=0.80).
- Cluster-bootstraps by resampling items within domain to produce 95% CIs on variance components.

**Stage 3: Reweighting and figures** (`analysis/reweight.py`, `analysis/figures.py`)
- Construction 2 reweighting gives each domain equal weight (1/D), equivalent to taking unweighted mean of per-domain accuracies. Computes rank changes vs. natural SimpleQA weights.
- `figures.py` produces three output figures: D-study bar chart, model×domain heatmap, rank-change slope plot.

**Central config** (`config.py`)
- All model specs (`ModelConfig` dataclass), GPU assignments, paths, analysis constants, and the list of kept domains live here. Add or modify models here only.
- `DOMAINS_TO_KEEP` excludes "Other" because its heterogeneity violates G-theory's within-facet exchangeability assumption.
- `FLAGSHIP_KEYS` (`llama-3.1-8b`, `gpt-oss-120b`, `ministral-3-14b`) are the three models highlighted in rank-change figures.

## Data flow

```
data/simpleqa_prepared.parquet
        ↓ (./scripts/run_eval.sh)
Modal volume: responses/{model_key}.jsonl
        ↓ (grader)
Modal volume: grades/{model_key}.jsonl
        ↓ (./scripts/sync_results.sh)
results/scores_long.parquet  ←─── input to all analysis modules
        ↓ (python run_all.py)
results/*.csv  +  figures/*.png
```

## Important implementation notes

- Modal requires GPU spec to be set at decoration time, so `inference.py` pre-builds one `@app.function` per GPU configuration in `INFERENCE_FUNCTIONS`. If you add a model with a new `(gpu, n_gpu)` combination, add a corresponding entry there.
- Reasoning models (`is_reasoning=True`) have their `<think>...</think>` traces stripped by `extract_final_answer()`. The `gpt-oss` models use `<|channel|>final|...|>` markers instead of think tags.
- `NOT_ATTEMPTED` verdicts are mapped to score=0 in the primary analysis (see `orchestrator.py::sync_results`). The project plan describes a sensitivity analysis treating them as missing.
- The G-theory design treats binary (0/1) scores as continuous — documented as a limitation, not an error.
- `RANDOM_SEED = 20260523` is fixed for bootstrap reproducibility.
