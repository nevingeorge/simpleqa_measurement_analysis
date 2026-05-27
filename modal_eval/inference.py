"""
vLLM inference on Modal — one global-scope function per GPU spec.

Design notes:

* All prompts are sent in a single llm.generate() call to leverage vLLM's
  continuous batching. This is the difference between a 2-hour run and a
  10-hour run on the same hardware.
* Greedy decoding (temperature=0) so the score matrix is reproducible.
* For reasoning models we cap output tokens at 2048; reasoning traces longer
  than that on SimpleQA factual questions are almost always the model rambling.
* Raw responses are saved to the persistent results volume so we can re-grade,
  audit, or re-analyze without re-running inference.
* Modal requires @app.function decorators to be applied at module (global)
  scope. The shared body lives in _inference_body(); each GPU-specific function
  is a thin wrapper defined at module level.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# Add project root for sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal
import pandas as pd

from config import MODELS_BY_KEY
from modal_eval.app import (
    HF_CACHE_MOUNT,
    RESULTS_MOUNT,
    VOLUMES,
    app,
    hf_secret,
    vllm_image,
)

MINUTES = 60
HOUR = 60 * MINUTES

# SimpleQA's official user prompt template. Keep this identical to the
# simple-evals repo so our scores remain comparable to published numbers.
SIMPLEQA_PROMPT_TEMPLATE = (
    "Answer the following question. Your answer should be a single short "
    "phrase or sentence with no extra commentary.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def format_prompt(question: str) -> str:
    return SIMPLEQA_PROMPT_TEMPLATE.format(question=question.strip())


def extract_final_answer(raw: str, model_key: str | None = None) -> str:
    """Strip reasoning traces; return the short answer string for grading."""
    if not raw:
        return ""

    text = raw
    if model_key and model_key.startswith("gpt-oss"):
        text = _extract_gpt_oss_answer(text)
    elif model_key and model_key.startswith("qwen-3.5"):
        text = _extract_qwen_answer(text)
    elif "assistantfinal" in text or "assistantanalysis" in text:
        text = _extract_gpt_oss_answer(text)
    elif "<|channel|>final|" in text:
        text = text.split("<|channel|>final|", 1)[1]
        text = text.split("|>", 1)[-1] if "|>" in text else text
    elif "<think>" in text:
        text = _extract_qwen_answer(text)

    text = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"</?redacted_thinking>", "", text)
    text = re.sub(r"^\s*Answer:\s*", "", text, flags=re.IGNORECASE)

    if model_key == "llama-3.1-8b" or _looks_repetitive(text):
        text = _collapse_to_short_answer(text)
    return text.strip()


def _looks_repetitive(text: str) -> bool:
    """Detect runaway repetition loops in decoded output."""
    words = text.split()
    if len(words) < 12:
        return False
    return len(set(words)) / len(words) < 0.35


def _collapse_to_short_answer(text: str) -> str:
    """Keep the first concise answer phrase; drop trailing repetition."""
    text = text.strip()
    if not text:
        return text

    # First sentence (up to ~200 chars).
    m = re.match(r"^(.{1,200}?)(?:\.\s|\.$)", text, re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(".") + "."

    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
    if len(first_line) > 200:
        first_line = first_line[:200].rsplit(" ", 1)[0]
    return first_line


def _extract_qwen_answer(raw: str) -> str:
    """Parse Qwen 3.5 thinking output; prefer post-thinking content."""
    text = raw
    im_end = "<|" + "redacted_im_end|>"
    if im_end in text:
        text = text.rsplit(im_end, 1)[-1]
    elif "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif "<think>" in text:
        # Truncated thinking with no closing tag — best-effort from trace.
        text = text.split("<think>", 1)[-1]

    text = re.sub(r"</?redacted_thinking>", "", text)
    text = re.sub(r"</?think>", "", text)

    # Prefer explicit answer markers inside thinking traces.
    for pat in (
        r"(?:\*\*Answer:\*\*|\*\*Final answer:\*\*)\s*([^\n*]{1,120})",
        r"(?:the answer is|answer:)\s*([^\n.]{1,120})",
    ):
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            return hits[-1].strip()

    bolds = re.findall(r"\*\*([^*\n]{1,80})\*\*", text)
    if bolds:
        # Skip meta labels; take the last substantive bold span.
        for candidate in reversed(bolds):
            lower = candidate.lower()
            if lower in {"answer", "final answer", "thinking process"}:
                continue
            if candidate.strip().endswith("?"):
                continue
            return candidate.strip()

    return _collapse_to_short_answer(text)


def _build_prompts(model_key: str, llm, items: list[dict]) -> list[str]:
    """Format prompts with the model chat template when needed."""
    if model_key.startswith("qwen-3.5") or model_key == "llama-3.1-8b":
        tokenizer = llm.get_tokenizer()
        prompts = []
        for item in items:
            user_text = format_prompt(item["problem"])
            messages = [{"role": "user", "content": user_text}]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if model_key.startswith("qwen-3.5"):
                kwargs["enable_thinking"] = False
            prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
        return prompts
    return [format_prompt(item["problem"]) for item in items]


def _extract_gpt_oss_answer(raw: str) -> str:
    """Parse gpt-oss Harmony / vLLM decoded output into a short answer."""
    text = raw

    if "assistantfinal" in text:
        text = text.rsplit("assistantfinal", 1)[-1]
        text = re.split(r"assistantanalysis", text)[0]
    else:
        # No final channel — scan from the end for a short non-meta line.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if len(ln) > 200:
                continue
            lower = ln.lower()
            if lower.startswith(
                ("we need to", "let's search", "the question", "the user asks", "search web")
            ):
                continue
            if re.fullmatch(r"([\w\s\.]+)\1{2,}", ln.replace(" ", "")):
                continue
            return ln

    text = re.sub(r"assistantanalysis+", " ", text)
    text = re.sub(r"\*\*Answer:\*\*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*The answer is[:\s]*", "", text, flags=re.IGNORECASE)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        for ln in lines:
            if len(ln) <= 200 and not ln.lower().startswith(
                ("we need to", "let's search", "the question", "the user asks")
            ):
                return ln
        if len(lines[0]) <= 500:
            return lines[0]

    text = text.strip()
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0]
    return text


def _gpt_oss_prompt_token_ids(question: str, encoding) -> list[int]:
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
    )

    system = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
    developer = DeveloperContent.new().with_instructions(
        "Answer the question with a single short phrase or sentence. "
        "No commentary or reasoning in the final answer."
    )
    conv = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.DEVELOPER, developer),
        Message.from_role_and_content(Role.USER, question.strip()),
    ])
    return encoding.render_conversation_for_completion(conv, Role.ASSISTANT)


def _parse_gpt_oss_completion(encoding, output_tokens, raw_text: str, model_key: str) -> str:
    from openai_harmony import Role

    if output_tokens:
        try:
            messages = encoding.parse_messages_from_completion_tokens(
                output_tokens, Role.ASSISTANT
            )
            finals = []
            for msg in messages:
                channel = getattr(msg, "channel", None) or ""
                if channel not in ("final", "final_answer"):
                    continue
                for part in getattr(msg, "content", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        finals.append(part_text.strip())
            if finals:
                return extract_final_answer(finals[-1], model_key)
        except Exception as exc:
            print(f"[{model_key}] harmony parse fallback: {exc}")
    return extract_final_answer(raw_text, model_key)


def _generate_gpt_oss(llm, items: list[dict], output_cap: int, model_key: str) -> list[dict]:
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    from vllm import SamplingParams

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    prompts = [
        {"prompt_token_ids": _gpt_oss_prompt_token_ids(item["problem"], encoding)}
        for item in items
    ]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_cap,
        stop_token_ids=stop_token_ids,
    )
    print(f"[{model_key}] harmony prompts, reasoning=low, max_tokens={output_cap}")
    outputs = llm.generate(prompts, sampling)

    results = []
    for item, out in zip(items, outputs):
        gen = out.outputs[0]
        raw = gen.text
        extracted = _parse_gpt_oss_completion(
            encoding, gen.token_ids, raw, model_key
        )
        results.append({
            "item_id": item["item_id"],
            "raw_response": raw,
            "extracted": extracted,
        })
    return results


def _inference_body(
    model_key: str,
    items: list[dict],
    hf_name: str,
    n_gpu: int,
    quantization: str | None,
    max_model_len: int,
    output_cap: int,
    extra_vllm_kwargs: dict,
) -> list[dict]:
    """
    Common vLLM inference logic shared by all GPU-specific Modal functions.

    items: list of {"item_id": str, "problem": str}
    Returns: list of {"item_id": str, "raw_response": str, "extracted": str}
    """
    from vllm import LLM, SamplingParams

    print(f"[{model_key}] loading {hf_name} on {n_gpu}x GPU "
          f"(quantization={quantization})")
    t0 = time.time()
    llm = LLM(
        model=hf_name,
        tensor_parallel_size=n_gpu,
        quantization=quantization,
        max_model_len=max_model_len,
        download_dir=HF_CACHE_MOUNT,
        trust_remote_code=True,
        **(extra_vllm_kwargs or {}),
    )
    print(f"[{model_key}] model loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_cap,
    )

    print(f"[{model_key}] generating {len(items):,} responses ...")
    t0 = time.time()
    if model_key.startswith("gpt-oss"):
        results = _generate_gpt_oss(llm, items, output_cap, model_key)
    else:
        prompts = _build_prompts(model_key, llm, items)
        if model_key.startswith("qwen-3.5") or model_key == "llama-3.1-8b":
            print(f"[{model_key}] using chat template"
                  f"{' (thinking off)' if model_key.startswith('qwen-3.5') else ''}")
        outputs = llm.generate(prompts, sampling)
        results = []
        for item, out in zip(items, outputs):
            raw = out.outputs[0].text
            extracted = extract_final_answer(raw, model_key)
            results.append({
                "item_id": item["item_id"],
                "raw_response": raw,
                "extracted": extracted,
            })
    elapsed = time.time() - t0
    print(f"[{model_key}] generation complete in {elapsed/60:.1f} min  "
          f"({len(items)/elapsed:.1f} items/s)")

    return results


# ---------------------------------------------------------------------------
# One @app.function per GPU spec, all at module (global) scope.
# Modal requires this; functions defined inside other functions cannot be
# registered unless serialized=True, which causes name collisions and
# prevents proper image caching.
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image, gpu="A10G:1", volumes=VOLUMES,
    secrets=[hf_secret], timeout=3 * HOUR, scaledown_window=10 * MINUTES,
)
def inference_a10g_1(model_key, items, hf_name, n_gpu, quantization,
                     max_model_len, output_cap, extra_vllm_kwargs):
    return _inference_body(model_key, items, hf_name, n_gpu, quantization,
                           max_model_len, output_cap, extra_vllm_kwargs)


@app.function(
    image=vllm_image, gpu="A100-40GB:1", volumes=VOLUMES,
    secrets=[hf_secret], timeout=3 * HOUR, scaledown_window=10 * MINUTES,
)
def inference_a100_40gb_1(model_key, items, hf_name, n_gpu, quantization,
                          max_model_len, output_cap, extra_vllm_kwargs):
    return _inference_body(model_key, items, hf_name, n_gpu, quantization,
                           max_model_len, output_cap, extra_vllm_kwargs)


@app.function(
    image=vllm_image, gpu="A100-80GB:1", volumes=VOLUMES,
    secrets=[hf_secret], timeout=3 * HOUR, scaledown_window=10 * MINUTES,
)
def inference_a100_80gb_1(model_key, items, hf_name, n_gpu, quantization,
                          max_model_len, output_cap, extra_vllm_kwargs):
    return _inference_body(model_key, items, hf_name, n_gpu, quantization,
                           max_model_len, output_cap, extra_vllm_kwargs)


@app.function(
    image=vllm_image, gpu="A100-80GB:2", volumes=VOLUMES,
    secrets=[hf_secret], timeout=4 * HOUR, scaledown_window=10 * MINUTES,
)
def inference_a100_80gb_2(model_key, items, hf_name, n_gpu, quantization,
                          max_model_len, output_cap, extra_vllm_kwargs):
    return _inference_body(model_key, items, hf_name, n_gpu, quantization,
                           max_model_len, output_cap, extra_vllm_kwargs)


@app.function(
    image=vllm_image, gpu="H100:1", volumes=VOLUMES,
    secrets=[hf_secret], timeout=4 * HOUR, scaledown_window=10 * MINUTES,
)
def inference_h100_1(model_key, items, hf_name, n_gpu, quantization,
                     max_model_len, output_cap, extra_vllm_kwargs):
    return _inference_body(model_key, items, hf_name, n_gpu, quantization,
                           max_model_len, output_cap, extra_vllm_kwargs)


INFERENCE_FUNCTIONS = {
    ("A10G", 1):       inference_a10g_1,
    ("A100-40GB", 1):  inference_a100_40gb_1,
    ("A100-80GB", 1):  inference_a100_80gb_1,
    ("A100-80GB", 2):  inference_a100_80gb_2,
    ("H100", 1):       inference_h100_1,
}


def run_model(model_key: str, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Local-side helper that picks the right Modal function and invokes it.

    items_df must have columns: item_id, topic, problem, answer.
    Returns a DataFrame with the model's responses.
    """
    cfg = MODELS_BY_KEY[model_key]
    key = (cfg.gpu, cfg.n_gpu)
    if key not in INFERENCE_FUNCTIONS:
        raise ValueError(
            f"No inference function configured for {cfg.gpu} x {cfg.n_gpu}. "
            f"Add it to INFERENCE_FUNCTIONS in modal_eval/inference.py."
        )

    items = items_df[["item_id", "problem"]].to_dict("records")

    print(f"\n=== Running {model_key} on {cfg.gpu} x {cfg.n_gpu} ===")
    results = INFERENCE_FUNCTIONS[key].remote(
        model_key=model_key,
        items=items,
        hf_name=cfg.hf_name,
        n_gpu=cfg.n_gpu,
        quantization=cfg.quantization,
        max_model_len=cfg.max_model_len,
        output_cap=cfg.output_cap,
        extra_vllm_kwargs=cfg.extra_vllm_kwargs,
    )
    out_df = pd.DataFrame(results)
    out_df["model_key"] = model_key
    return out_df


# ---------------------------------------------------------------------------
# Modal function: persist responses to the results volume.
# This runs inside Modal so it has direct access to the volume.
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    volumes=VOLUMES,
    timeout=5 * MINUTES,
)
def save_responses(model_key: str, records: list[dict]) -> str:
    """Save the model's responses to the persistent results volume."""
    out_dir = Path(RESULTS_MOUNT) / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.jsonl"
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records):,} responses to {out_path}")
    return str(out_path)


@app.function(
    image=vllm_image,
    volumes=VOLUMES,
    timeout=5 * MINUTES,
)
def fetch_responses(model_key: str) -> list[dict]:
    """Load a model's saved responses from the results volume."""
    out_path = Path(RESULTS_MOUNT) / "responses" / f"{model_key}.jsonl"
    if not out_path.exists():
        raise FileNotFoundError(f"No responses found at {out_path}")
    rows = []
    for line in out_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(f"Loaded {len(rows):,} responses from {out_path}")
    return rows
