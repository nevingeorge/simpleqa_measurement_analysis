# Domain Validity Analysis of SimpleQA via Generalizability Theory

*A graduate-level research project in AI measurement science*

---

## 1. Research Question and Scope

**Primary research question.** How domain-dependent is SimpleQA, and how would rankings of mid-tier open-source models change if the benchmark were rebalanced so each domain contributed equally to the final score?

**Three sub-questions:**

1. How much do reliability and required item counts vary across SimpleQA's topic domains under a generalizability theory framework?
2. Which domains are over- and under-represented relative to the item counts needed for adequate dependability (G-coefficient ≥ 0.80)?
3. How do per-domain and aggregate scores of three flagship open-source models change when SimpleQA is re-weighted to satisfy domain-equal contribution?

**Scope discipline.** This project makes measurement-quality claims about one benchmark using a restricted-range set of open-source models. It does *not* claim anything about general LLM capability, about other benchmarks, about top proprietary models, or about which model is "actually best." All findings are conditional on the model set, the benchmark, and the statistical assumptions made explicit in the methods.

---

## 2. Why SimpleQA

Four defensible reasons:

- **Single-answer structure.** Each item has one indisputable answer graded correct / incorrect / not attempted. Item scores are effectively binary, which keeps the G-theory variance decomposition tractable.
- **Published topic tags.** Wei et al. (2024) tagged every item with a topic post-hoc using ChatGPT. The nested-in-domain structure is observable.
- **Existing leaderboard.** Many models have published aggregate scores, anchoring my own subset evaluation against known reference points.
- **Documented topical bias.** Google DeepMind's SimpleQA Verified (2025) established that SimpleQA over-indexes on science and technology. This motivates the project but uses *empirical* rebalancing; this project uses *psychometric* rebalancing (G-theory), a complementary methodological angle that no prior work has applied to SimpleQA.

SimpleQA is the right testbed because it's structured enough that any failure of domain validity is interpretable, not noise from a messy benchmark.

---

## 3. Methodological Framework: Generalizability Theory

### The case for G-theory

Standard benchmark practice reports a single aggregate accuracy with at most a binomial confidence interval. This is implicitly a Classical Test Theory measurement: observed score = true score + error, with error treated as a single undifferentiated bucket. It works for one global score on exchangeable items, but breaks down the moment you want to claim things about subscores, ask measurement-design questions, or examine whether models have genuine domain-specific strengths.

G-theory adds five things relevant to this project:

1. **Decomposes error into named, interpretable sources.** Instead of "error," it tells you how much variance comes from items being differentially difficult, from domains being differentially hard, from genuine model × domain interaction, and from residual. Each has a different fix.
2. **Distinguishes signal from noise explicitly.** σ²_m (between-model variance) is what rankings depend on. Everything else is noise that should average out. The G-coefficient is literally the ratio of signal to (signal + averageable noise).
3. **Supports prospective design.** The D-study answers "how many items do I need per domain?" — a question CTT cannot address conditionally.
4. **Handles nested and crossed facets.** Items-nested-in-domains-crossed-with-models is exactly the structure G-theory was built for.
5. **Exposes interaction effects.** A large σ²_md component means rankings depend on which domain you care about — a finding aggregate scores hide.

### Design

The object of measurement is the **model**. The facet is **items nested in domains**. The mixed model is:

```
score ~ 1 + (1 | model) + (1 | domain) + (1 | model:domain) + (1 | item:domain)
```

This yields variance components for: model (σ²_m), domain (σ²_d), model × domain interaction (σ²_md), items nested in domain (σ²_i:d), and residual (σ²_mi:d, confounded with error since one observation per cell).

### G-coefficient (relative)

For per-domain reliability:

$$E\rho^2_d = \frac{\sigma^2_m}{\sigma^2_m + \frac{\sigma^2_{mi:d}}{n_{i,d}}}$$

This is the relative G-coefficient — what matters for ranking models. I'll also report the Phi coefficient (absolute dependability) for completeness, though my use case is rankings.

### D-study

For each domain, solve for n_i,d such that E\rho²_d ≥ 0.80 (individual decisions) and ≥ 0.70 (group-level decisions).

### Honest caveat

G-theory was developed for continuous outcomes in human-subject educational measurement. Applying it to binary item outcomes with LLMs as "subjects" involves assumption-stretching: items aren't sampled from an infinite universe, models aren't a random sample of a population, and binary outcomes mean variance components are noisier than the textbook examples. The analysis is treated as exploratory; CIs come from bootstrapping; point estimates of variance components are not over-interpreted.

---

## 4. Model Set

Ten open-source models, drawn from the top of the Kaggle SimpleQA leaderboard for open-source models, selected to balance three considerations: providing architectural diversity, fitting within the available Modal credit budget, and avoiding the largest frontier-scale open models (e.g., 235B+ MoE and 671B MoE reasoning models) whose multi-GPU serving costs would dominate the project budget. Smaller and mid-sized models were preferred to keep total compute cost low while still spanning a meaningful range of capability levels.

| Model | Type | Parameters (total / active) | Hardware |
|---|---|---|---|
| Llama 4 Scout | MoE, multimodal | 109B / 17B | 1× H100 (int4) |
| gpt-oss-120b | MoE, reasoning | 117B / 5.1B | 1× H100 |
| Mistral Large 2 | dense | 123B | 2× A100 80GB |
| Qwen 3.5 27B | dense, hybrid thinking | 27B | 1× A100 80GB |
| Gemma 3 27B | dense | 27B | 1× A100 80GB |
| gpt-oss-20b | MoE, reasoning | 21B / 3.6B | 1× A10G |
| Granite 4.0 Small | small MoE | ~30B | 1× A100 40GB |
| Qwen 3.5 9B | dense | 9B | 1× A10G |
| Gemma 3 12B | dense | 12B | 1× A10G |
| Ministral 8B | dense | 8B | 1× A10G |

### Flagship trio for narrative comparison

- **Llama 4 Scout** (Meta, MoE multimodal)
- **gpt-oss-120b** (OpenAI, MoE reasoning)
- **Mistral Large 2** (Mistral, large dense)

Three different developers, three different architectural philosophies, similar enough capability levels that rank changes under rebalancing are substantively meaningful rather than dominated by overall capability gaps.

### Acknowledged limitation: restricted range

This model set spans roughly 5–25% accuracy on SimpleQA — a compressed range compared to the full leaderboard (5% to 95%+). σ²_m estimates will be anchored at the top by Llama 4 Scout and gpt-oss-120b; G-coefficients will be lower across the board than they would be with a frontier model included; D-study item-count recommendations will be more demanding. This is documented as a limitation, not papered over. The methodological contribution does not require a frontier model — the goal is to demonstrate G-theory as a measurement-quality tool, and to characterize domain dependence within a coherent tier of open-source models. Within-tier differences are precisely where benchmark consumers struggle most.

### Reasoning model handling

Three models support configurable reasoning effort (gpt-oss-120b, gpt-oss-20b, Qwen 3.5 27B). For consistency, all three are run with **reasoning off / low effort**. SimpleQA is factual recall, where reasoning helps less than on math or code, and keeping reasoning off makes models more comparable. This is documented as a methodological choice; an optional sensitivity analysis runs gpt-oss-120b at high reasoning as a robustness check.

---

## 5. Data and Domain Structure

### Source

SimpleQA full dataset, 4,326 items, from OpenAI's `simple-evals` GitHub repository. Topic tags from the Wei et al. post-hoc classification used as the domain structure.

### Domains

The ~10 topic categories from the Wei et al. taxonomy. The "Other" category is dropped (heterogeneous bucket violates within-facet exchangeability). Final domain set is roughly: Science & Technology, Politics, Art, Geography, History, Sports, Music, TV Shows, Video Games. Final item counts per domain will be confirmed before analysis.

### Scoring

The official `simple-evals` grader (GPT-4o-mini-based classifier) is used for consistency with published leaderboard methodology. Each item is graded correct / incorrect / not_attempted. The main analysis treats "not attempted" as incorrect (0); a sensitivity analysis in the appendix treats it as missing.

---

## 6. Infrastructure: Modal

All 10 models are hosted on Modal using vLLM serving. A single parameterized deployment function handles all models; only the model config and GPU spec change. Critical engineering points:

- **Persistent HuggingFace cache volume** — model weights downloaded once, reused across runs.
- **Batched inference** — all 4,326 prompts in one `llm.generate()` call per model, leveraging vLLM's continuous batching.
- **FP8 or int4 quantization where appropriate** — Llama 4 Scout in int4 fits on 1× H100; gpt-oss-120b runs natively on 1× H100.
- **Capped output tokens** — 256 for non-reasoning models, 2048 for reasoning (with low effort).
- **Pilot on 100 items first** — validate the pipeline for each model before the full 4,326-item run.
- **Save raw responses** — full model outputs stored in a Modal volume for re-analysis and sensitivity checks.

### Cost budget

| Component | Estimated cost |
|---|---|
| 10 model evaluations on Modal | ~$60 in credits |
| Pilot runs and debugging buffer | ~$20 in credits |
| GPT-4o-mini grading (4,326 × 10 = 43,260 calls) | ~$3–5 cash |
| **Total** | **~$80 credits + ~$5 cash** |

This uses ~20% of available Modal credits. Headroom of ~$320 is allocated to optional extensions (see Section 11).

---

## 7. Analysis Pipeline

### Step 1: Data acquisition

- Pull SimpleQA from `simple-evals` repo. Confirm 4,326 items and topic tags present.
- Drop "Other" domain. Confirm final item counts per domain.

### Step 2: Eval runs

- For each model: format SimpleQA prompts, run via Modal vLLM serving, save raw outputs.
- For reasoning models: strip `<think>...</think>` blocks before grading.
- Calibration check: after first 2–3 models complete, compare aggregate score to any published leaderboard score (if available) to verify pipeline correctness. Expected tolerance: within ~2–5 percentage points given grader and sampling variability.

### Step 3: Grading

- Run all model responses through the `simple-evals` GPT-4o-mini grader.
- Construct score matrix: 10 models × 4,326 items, values in {0, 1}, with domain labels on columns.

### Step 4: G-study

- Fit the mixed model using R `lme4::lmer` or Python `pymer4` (R is cleaner for variance components).
- REML estimation. Extract variance components: σ²_m, σ²_d, σ²_md, σ²_i:d, σ²_residual.
- Cluster bootstrap with 1,000 iterations (resample items within domain, refit) to get 95% CIs on each variance component.

### Step 5: Per-domain G-coefficients

- For each domain d, plug in actual n_i,d and compute E\rho²_d.
- Report per-domain G-coefficients with bootstrap CIs.

### Step 6: D-study

- For each domain, solve for required n_i,d at G = 0.70 and G = 0.80.
- Build the headline figure: per domain, three bars — current item count, required at 0.70, required at 0.80. Color-coded over/under-provisioned.

### Step 7: Re-weighted scoring (Construction 2)

For each model:

$$\text{Score}_{\text{rebalanced}} = \frac{1}{D} \sum_{d=1}^{D} \text{Accuracy}_d$$

This is the unweighted mean of per-domain accuracies, equivalent to weighting each item by 1/(D · n_d). Each domain contributes 1/D to the final score regardless of size.

### Step 8: Rank-change analysis

- Compute original and rebalanced scores for all 10 models.
- Focus narrative on the three flagship models: report absolute score changes and rank changes.
- Compute Spearman rank correlation between original and rebalanced rankings across all 10 models.
- Build the second headline figure: 10 × D model-domain accuracy heatmap, with cells annotated by per-domain G-coefficient (high-G cells get a marker indicating "trustworthy").

### Step 9: Effective reliability of rebalanced aggregate

The re-weighted aggregate G-coefficient is a weighted combination of per-domain G-coefficients. It can actually be *lower* than the original aggregate if small domains have poor reliability. Compute explicitly and report. If rebalancing improves rankings but lowers aggregate reliability, that tension is a substantive finding.

---

## 8. Deliverables

- **Written report (~10–15 pages)** structured as: motivation → SimpleQA background → G-theory framework → methods → results (G-study, D-study, model comparison) → limitations → implications.
- **Code repository (GitHub)** with: Modal deployment scripts, data pipeline, G-study fitting code, D-study calculations, figure generation. Reproducible with one command after setting Modal and HF credentials.
- **Three core figures:**
  1. Current item count vs. required item count per domain, at G = 0.70 and G = 0.80.
  2. Model × domain accuracy heatmap with reliability annotations.
  3. Rank/score changes under rebalancing.
- **One summary table:** variance components with bootstrap 95% CIs.
- **Appendix:** sensitivity analyses (not-attempted handling, optional reasoning-effort comparison, optional grader sensitivity).

---

## 9. Practical Implications

Written conservatively, conditional on findings actually supporting them:

- Benchmark developers should report per-domain reliability, not just aggregate accuracy. SimpleQA could publish a "minimum trustworthy subscore" alongside its leaderboard.
- Aggregate leaderboard scores hide domain-specific model strengths. A buyer choosing a model for a domain-heavy use case cannot rely on aggregate SimpleQA rank.
- Domain-balanced re-weighting is a cheap retrofit. Any benchmark with topic tags can be rebalanced post-hoc with no new data collection. The fix is one function call away.
- G-theory is underused in AI evaluation. Most benchmark papers report aggregate accuracy with no variance decomposition. This is a methodological gap that educational measurement already solved decades ago.

---

## 10. Limitations

Explicit and prominent in the report, not buried:

- **One benchmark, ten models, restricted range.** Findings about model × domain interaction strength are specific to SimpleQA and to mid-tier open-source models. No claim is made about other benchmarks, frontier models, or general LLM capability.
- **Topic tags come from an LLM classifier.** The taxonomy has measurement error. A small validation sample is hand-checked; the bulk is not.
- **Restricted-range σ²_m.** All ten models score in roughly 5–25%. This compresses signal variance and makes G-coefficients smaller than they would be with a wider model set.
- **Cost-driven model selection.** The model set was chosen partly to fit within available Modal credits, which ruled out the largest frontier-scale open models (235B+ MoE and 671B MoE reasoning models). A study with unconstrained compute could include those models and would likely produce a wider score range and larger σ²_m.
- **Binary outcomes and small "subject" count.** G-theory variance components are noisier than the textbook continuous-score, large-N case. Bootstrap CIs reflect this; point estimates aren't over-interpreted.
- **"Not attempted" handling is a choice.** Main analysis treats as 0; appendix reports the alternative.
- **Adversarial item selection in SimpleQA.** Items were selected because GPT-4 got them wrong. This compresses difficulty range and may inflate σ²_md artificially.
- **Reasoning models run with low effort.** Results don't speak to what these models can do at full reasoning. Optional sensitivity analysis addresses this for one model.
- **Construct validity is assumed, not tested.** G-theory addresses generalizability and reliability across items and conditions — not whether SimpleQA measures factuality correctly in a deeper Messickian sense. This is a measurement-reliability project, not a construct-validity project.
- **Single grader.** Grader-induced variance is a potential confound. Optional sensitivity analysis with a second grader addresses this if budget allows.
