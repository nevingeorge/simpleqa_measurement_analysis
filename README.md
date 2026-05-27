# Domain Validity of SimpleQA via Generalizability Theory

End-to-end pipeline for the paper *"How Domain-Dependent Is SimpleQA? A Generalizability-Theory Analysis."*

We evaluate nine open-source LLMs on the SimpleQA benchmark, decompose score variance using generalizability theory (G-theory), and ask: does SimpleQA's domain sampling give reliable model rankings, and would re-weighting domains to equal contribution change those rankings?

---

## Quick start — skip Modal, use pre-computed results

All inference and grading results are already committed to this repository in `results/scores_long.parquet`. To reproduce every figure and table in the paper without re-running expensive GPU inference:

```bash
git clone <repo-url>
cd simpleqa-validity-analysis

# Create and activate the virtual environment
python3 -m venv sqa_eval
source sqa_eval/bin/activate          # Windows: sqa_eval\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full analysis (~4 minutes; bootstrap takes most of that)
python run_all.py
```

Outputs land in `results/` (CSV tables) and `figures/` (PNG plots).

---

## Repository structure

```
simpleqa-validity-analysis/
├── config.py                  # Central config: model specs, paths, analysis constants
├── run_all.py                 # Top-level driver: G-study → reweight → figures
├── requirements.txt           # Pinned local dependencies
│
├── data/
│   ├── prepare_data.py        # Download SimpleQA, filter domains, write parquet
│   └── simpleqa_prepared.parquet  # 3,851-item analysis dataset (committed)
│
├── analysis/
│   ├── gtheory.py             # Variance component estimation, G/D-study, bootstrap
│   ├── reweight.py            # Construction 2 domain re-weighting, rank changes
│   └── figures.py             # Three headline figures (Fig 1–3)
│
├── modal_eval/
│   ├── app.py                 # Shared Modal app, container images, volumes, secrets
│   ├── inference.py           # vLLM serving (one @app.function per GPU spec)
│   ├── grader.py              # GPT-4o-mini grader (OpenAI API + vLLM fallback)
│   └── orchestrator.py        # Local entrypoints: run_eval, sync_results
│
├── scripts/
│   ├── run_eval.sh            # Wrapper: `modal run` with MODAL_BUILD_VALIDATION=ignore
│   └── sync_results.sh        # Pull grades from Modal volume → local parquet
│
└── results/                   # Pre-computed outputs (committed)
    ├── scores_long.parquet        # 34,659-row (9 models × 3,851 items) score table
    ├── score_matrix.parquet       # Wide format: 9 models × 3,851 items
    ├── per_model_summary.csv      # Per-model accuracy summary
    ├── per_domain_gstudy.csv      # G/D-study table (produced by run_all.py)
    ├── variance_components_ci.csv # Bootstrap CIs on variance components
    ├── per_model_domain_accuracy.csv
    └── rank_change_table.csv
```

---

## Which script produces which result

| Paper output | Script / command |
|---|---|
| Table 1 — variance components + 95% CIs | `python run_all.py` → `results/variance_components_ci.csv` |
| Table 2 — per-domain G and item counts | `python run_all.py` → `results/per_domain_gstudy.csv` |
| Table 3 — rank changes under rebalancing | `python run_all.py` → `results/rank_change_table.csv` |
| Fig 1 — D-study item counts | `python run_all.py` → `figures/fig1_d_study_item_counts.png` |
| Fig 2 — model × domain heatmap | `python run_all.py` → `figures/fig2_model_domain_heatmap.png` |
| Fig 3 — rank-change slope plot | `python run_all.py` → `figures/fig3_rank_change.png` |
| Per-model accuracy (supplement) | `python run_all.py` → `results/per_model_domain_accuracy.csv` |

Individual analysis modules can also be run standalone:

```bash
python analysis/gtheory.py      # G-study only
python analysis/reweight.py     # Reweighting only
python analysis/figures.py      # Figures only (requires results/scores_long.parquet)
```

---

## Expected runtime and compute

| Stage | Where | Time |
|---|---|---|
| `python data/prepare_data.py` | Local | < 30 s |
| Modal inference, per model | Modal (GPU) | 10–40 min depending on GPU |
| `./scripts/sync_results.sh` | Local + Modal | ~1 min |
| `python run_all.py` (bootstrap N=1000) | Local CPU | ~3–4 min |

The full analysis (everything after `sync_results.sh`) runs on a laptop with no GPU.
Bootstrap parallelism is single-threaded by design for reproducibility; runtime scales linearly with `BOOTSTRAP_N` in `config.py`.

---

## Full reproduction from scratch (requires Modal account)

### 1. Environment

```bash
python3 -m venv sqa_eval
source sqa_eval/bin/activate
pip install -r requirements.txt
```

### 2. Modal setup

```bash
# Install the Modal CLI and authenticate
pip install modal
modal setup

# Create secrets (one-time)
modal secret create huggingface-token HF_TOKEN=hf_yourtokenhere
modal secret create openai-api-key OPENAI_API_KEY=sk-yourkeyhere
```

### 3. HuggingFace gated model access

Accept the license on the HuggingFace website for each gated model before running:
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Ministral-8B-Instruct-2410`
- `mistralai/Ministral-3-14B-Instruct-2512`
- `google/gemma-3-27b-it`
- `google/gemma-3-12b-it`

### 4. Prepare the dataset

```bash
python data/prepare_data.py
```

Downloads SimpleQA from OpenAI's public blob storage, filters to the nine kept
domains (drops "Other"), assigns stable item IDs, and writes
`data/simpleqa_prepared.parquet` (3,851 items).

### 5. Run inference (one model at a time)

Pilot a single model first (~90 items, validates the full pipeline):

```bash
./scripts/run_eval.sh --model-key gemma-3-12b --pilot
```

Then run all nine models in full:

```bash
for key in gemma-3-12b ministral-8b qwen-3.5-9b gpt-oss-20b \
           granite-4.0-h-small granite-4.1-8b \
           gemma-3-27b llama-3.1-8b ministral-3-14b; do
  ./scripts/run_eval.sh --model-key "$key"
done
```

Modal automatically picks the right GPU for each model (see `config.py`).
Models can be run concurrently from separate terminal windows.

### 6. Grade and sync

Grading runs automatically after inference for each model. Once all models
are done, pull the results locally:

```bash
./scripts/sync_results.sh
```

### 7. Analysis

```bash
python run_all.py
```

---

## Dataset

SimpleQA is released publicly by OpenAI:
- Paper: [SimpleQA: Measuring Short-Form Factuality in Large Language Models](https://openai.com/research/simpleqa) (Wei et al., 2024)
- Dataset: downloaded automatically by `data/prepare_data.py` from  
  `https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv`

Topic tags are from the Wei et al. post-hoc classification (ChatGPT-generated, included in the CSV metadata field).

---

## Model set

| Model key | HuggingFace ID | GPU | Parameters |
|---|---|---|---|
| `gemma-3-27b` | google/gemma-3-27b-it | H100 × 1 | 27B |
| `granite-4.0-h-small` | ibm-granite/granite-4.0-h-small | A100-80GB × 1 | ~30B (MoE) |
| `qwen-3.5-9b` | Qwen/Qwen3.5-9B | A100-40GB × 1 | 9B |
| `ministral-3-14b` | mistralai/Ministral-3-14B-Instruct-2512 | A100-40GB × 1 | 14B |
| `gemma-3-12b` | google/gemma-3-12b-it | A100-80GB × 1 | 12B |
| `ministral-8b` | mistralai/Ministral-8B-Instruct-2410 | A100-40GB × 1 | 8B |
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | A100-40GB × 1 | 8B |
| `gpt-oss-20b` | openai/gpt-oss-20b | A100-80GB × 1 | 21B (MoE) |
| `granite-4.1-8b` | ibm-granite/granite-4.1-8b | A100-40GB × 1 | 8B |

---

## Reproducibility notes

- **Random seed:** `RANDOM_SEED = 20260523` is set in `config.py` and passed to
  every NumPy `default_rng` call. All stochastic operations (bootstrap) use this seed.
- **Greedy decoding:** `temperature=0` throughout inference; re-running a model
  produces identical responses.
- **Grader:** GPT-4o-mini via the OpenAI API, using the official
  [simple-evals](https://github.com/openai/simple-evals) grading prompt verbatim.
  Model API outputs are non-deterministic at low volume, so small grading
  differences (±1–2 items) are expected if you re-grade from saved raw responses.
- **Pre-computed results:** `results/scores_long.parquet` is committed so the
  local analysis pipeline runs without any cloud credentials.

---

## Attribution

- SimpleQA dataset and grading prompt: [OpenAI simple-evals](https://github.com/openai/simple-evals) (MIT License)
- Generalizability theory framework: Brennan, R. L. (2001). *Generalizability Theory*. Springer.
- Inference serving: [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0)
- Cloud GPU infrastructure: [Modal](https://modal.com)

All analysis code in this repository is original.
