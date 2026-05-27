"""
Central configuration for the SimpleQA G-theory project.

All model specs, GPU choices, paths, and analysis constants live here so they
can be referenced from both the Modal eval pipeline and the local analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Inside Modal containers
MODAL_VOLUME_PATH = "/cache"
MODAL_RESULTS_PATH = "/results"


# ---------------------------------------------------------------------------
# SimpleQA data
# ---------------------------------------------------------------------------
SIMPLEQA_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)

# Domains we keep. "Other" is dropped (heterogeneous bucket violates the
# within-facet exchangeability assumption of G-theory).
DOMAINS_TO_KEEP = [
    "Science and technology",
    "Politics",
    "Art",
    "Geography",
    "History",
    "Sports",
    "Music",
    "TV shows",
    "Video games",
]


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Specification for one open-source model run on Modal via vLLM."""
    key: str                       # short identifier, used in filenames
    hf_name: str                   # HuggingFace model id
    gpu: str                       # Modal GPU spec, e.g. "H100", "A100-80GB", "A10G"
    n_gpu: int = 1                 # tensor-parallel size
    quantization: str | None = None  # None | "fp8" | "awq" | "gptq" | "bitsandbytes"
    is_reasoning: bool = False     # if True, strip <think>...</think>
    max_model_len: int = 8192      # vLLM context window cap
    max_output_tokens: int = 256   # tight cap; reasoning models override
    gated: bool = False            # requires HF token + license acceptance
    extra_vllm_kwargs: dict = field(default_factory=dict)

    @property
    def output_cap(self) -> int:
        # Reasoning models need room for the trace + the final answer.
        return 2048 if self.is_reasoning else self.max_output_tokens


MODELS: list[ModelConfig] = [
    # --- Top tier of the chosen open-source set ---
    ModelConfig(
        key="llama-3.1-8b",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        gpu="A100-40GB",
        n_gpu=1,
        gated=True,
    ),
    ModelConfig(
        key="ministral-3-14b",
        hf_name="mistralai/Ministral-3-14B-Instruct-2512",
        gpu="A100-40GB",
        n_gpu=1,
        gated=True,
    ),
    # --- Mid tier ---
    ModelConfig(
        key="gemma-3-27b",
        hf_name="google/gemma-3-27b-it",
        gpu="H100",
        n_gpu=1,
        gated=True,
    ),
    ModelConfig(
        key="gpt-oss-20b",
        hf_name="openai/gpt-oss-20b",
        gpu="A100-80GB",
        n_gpu=1,
        is_reasoning=False,
        max_output_tokens=256,
    ),
    ModelConfig(
        key="granite-4.0-h-small",
        hf_name="ibm-granite/granite-4.0-h-small",
        gpu="A100-80GB",
        n_gpu=1,
    ),
    ModelConfig(
        key="granite-4.1-8b",
        hf_name="ibm-granite/granite-4.1-8b",
        gpu="A100-40GB",
        n_gpu=1,
    ),
    # --- Smaller models (anchor the bottom of the difficulty range) ---
    ModelConfig(
        key="qwen-3.5-9b",
        hf_name="Qwen/Qwen3.5-9B",
        gpu="A100-40GB",
        n_gpu=1,
        extra_vllm_kwargs={"language_model_only": True},
    ),
    ModelConfig(
        key="gemma-3-12b",
        hf_name="google/gemma-3-12b-it",
        gpu="A100-80GB",
        n_gpu=1,
        gated=True,
    ),
    ModelConfig(
        key="ministral-8b",
        hf_name="mistralai/Ministral-8B-Instruct-2410",
        gpu="A100-40GB",
        n_gpu=1,
        gated=True,
    ),
]

MODELS_BY_KEY = {m.key: m for m in MODELS}


# ---------------------------------------------------------------------------
# Flagship trio (the headline rank-change story)
# ---------------------------------------------------------------------------
FLAGSHIP_KEYS = ["gemma-3-27b", "ministral-3-14b", "llama-3.1-8b"]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
# We use GPT-4o-mini via the OpenAI API for consistency with the official
# simple-evals grader. Set OPENAI_API_KEY in your environment.
GRADER_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
G_THRESHOLD_INDIVIDUAL = 0.80   # G-coefficient threshold for individual decisions
G_THRESHOLD_GROUP = 0.70        # G-coefficient threshold for group-level decisions
BOOTSTRAP_N = 1000              # cluster-bootstrap iterations for variance CIs
RANDOM_SEED = 20260523          # fixed for reproducibility
