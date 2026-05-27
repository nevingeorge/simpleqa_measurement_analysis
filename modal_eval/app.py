"""
Modal app: shared image, volumes, and secrets used across all inference jobs.

The big win here is using one persistent HuggingFace cache Volume so model
weights are downloaded once and reused on every subsequent run. Without this,
700GB of weights gets re-pulled per cold start and the budget evaporates.
"""
from __future__ import annotations

import modal

APP_NAME = "simpleqa-gtheory"

# A single Modal App for everything in the project. Submodules pull this and
# attach their own @app.function decorators.
app = modal.App(APP_NAME)

# Image: CUDA devel base + vLLM. Modal provides GPU drivers at runtime, but
# vLLM 0.21's FlashInfer sampler JIT-compiles kernels and requires nvcc in the
# image (see Modal's vLLM example: modal.com/docs/examples/vllm_inference).
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "vllm>=0.10.0",         # Qwen3.5 needs recent vLLM (language_model_only, etc.)
        "transformers>=4.51",   # 4.51+ required for Llama-4 architecture
        "huggingface_hub>=0.27",
        "hf_transfer",          # required for HF_XET_HIGH_PERFORMANCE=1
        "openai-harmony>=0.0.8",
        "pandas",
        "pyarrow",
        # torch is intentionally unpinned: vLLM resolves a compatible version
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster HF weight downloads
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
    .add_local_python_source("config")
    .add_local_python_source("modal_eval")
)

# Lighter image for grading (no GPU dependencies).
grader_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.40", "pandas", "pyarrow", "tenacity")
    .add_local_python_source("config")
    .add_local_python_source("modal_eval")
)

# Persistent volume for HuggingFace weights.
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Persistent volume for run outputs (raw responses, grader verdicts, score matrix).
results_volume = modal.Volume.from_name("simpleqa-results", create_if_missing=True)

# HF token (set with: `modal secret create huggingface-token HF_TOKEN=...`)
hf_secret = modal.Secret.from_name("huggingface-token")

# OpenAI API key for the grader
# (set with: `modal secret create openai-api-key OPENAI_API_KEY=...`)
openai_secret = modal.Secret.from_name("openai-api-key")

# Mount points inside the container
HF_CACHE_MOUNT = "/root/.cache/huggingface"
RESULTS_MOUNT = "/results"

VOLUMES = {
    HF_CACHE_MOUNT: hf_cache_volume,
    RESULTS_MOUNT: results_volume,
}
