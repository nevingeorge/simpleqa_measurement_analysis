"""
Grader: classifies each model response as correct / incorrect / not_attempted.

Two grading backends are supported:
  - OpenAI API (default, gpt-4o-mini): grade_model()
  - vLLM on Modal GPUs (any HuggingFace model): grade_model_vllm()
    Items are sharded across N H100 containers via grade_shard_vllm.map(),
    so throughput scales linearly with n_shards.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Project root imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal

from modal_eval.app import (
    HF_CACHE_MOUNT,
    RESULTS_MOUNT,
    VOLUMES,
    app,
    grader_image,
    hf_secret,
    openai_secret,
    vllm_image,
)

MINUTES = 60
HOUR = 60 * MINUTES
# Stay under org TPM (often 200k); long model outputs (e.g. gpt-oss-120b) use many tokens/call.
GRADE_WORKERS = 15
GRADE_MAX_ATTEMPTS = 15
CHECKPOINT_INTERVAL = 250

# This is the official simple-evals grader prompt, adapted minimally.
# https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
GRADER_TEMPLATE = """\
Your job is to look at a question, a gold target, and a predicted answer, and \
then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

CORRECT: The predicted answer fully contains the gold target without contradicting it.
INCORRECT: The predicted answer contradicts the gold target in any way.
NOT_ATTEMPTED: The predicted answer does not attempt to answer (e.g. "I don't know", \
"I'm not sure", refusals, empty answers, or answers that do not include the gold target \
and do not contradict it).

Respond with exactly one of: CORRECT, INCORRECT, NOT_ATTEMPTED.
Grade:"""


@app.function(image=grader_image, volumes=VOLUMES, timeout=300)
def fetch_all_grades() -> list[dict]:
    """Read every per-model grades JSONL from the results volume."""
    grades_dir = Path(RESULTS_MOUNT) / "grades"
    if not grades_dir.exists():
        return []
    all_rows = []
    for fp in sorted(grades_dir.glob("*.jsonl")):
        for line in fp.read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    return all_rows


def _grades_path(model_key: str) -> Path:
    return Path(RESULTS_MOUNT) / "grades" / f"{model_key}.jsonl"


@app.function(image=grader_image, volumes=VOLUMES, timeout=300)
def delete_grades(model_key: str) -> bool:
    """Remove a model's grades file so grading starts fresh."""
    path = _grades_path(model_key)
    if path.exists():
        path.unlink()
        print(f"[grader/{model_key}] deleted {path}")
        return True
    print(f"[grader/{model_key}] no grades file at {path}")
    return False


def _load_graded(model_key: str) -> dict[str, dict]:
    path = _grades_path(model_key)
    if not path.exists():
        return {}
    graded: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            graded[row["item_id"]] = row
    return graded


def _retry_after_seconds(err: Exception, attempt: int) -> float:
    """Honor OpenAI Retry-After when present; otherwise exponential backoff."""
    resp = getattr(err, "response", None)
    if resp is not None:
        retry_after = resp.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after) + random.uniform(0.1, 0.5)
            except ValueError:
                pass
    return min(120.0, (2**attempt) + random.uniform(0, 1))


def _write_grades(model_key: str, graded: dict[str, dict], item_order: list[str]) -> None:
    path = _grades_path(model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item_id in item_order:
            if item_id in graded:
                f.write(json.dumps(graded[item_id]) + "\n")


@app.function(
    image=grader_image,
    volumes=VOLUMES,
    secrets=[openai_secret],
    timeout=6 * 60 * MINUTES,
)
def grade_model(model_key: str, items_df_records: list[dict], grader_model: str = "gpt-4o-mini") -> dict:
    """
    Grade all responses for one model.

    items_df_records: list of {"item_id", "topic", "problem", "answer",
                               "extracted"} dicts (joined locally before send).
    Returns a summary dict; writes per-item grades to the results volume.
    """
    from openai import OpenAI
    import openai as openai_mod

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def grade_one(question: str, target: str, predicted: str) -> str:
        prompt = GRADER_TEMPLATE.format(
            question=question, target=target, predicted_answer=predicted or "",
        )
        last_err: Exception | None = None
        for attempt in range(GRADE_MAX_ATTEMPTS):
            try:
                resp = client.chat.completions.create(
                    model=grader_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                text = resp.choices[0].message.content.strip().upper()
                # Check INCORRECT before CORRECT: "CORRECT" is a substring of "INCORRECT".
                for label in ("INCORRECT", "NOT_ATTEMPTED", "CORRECT"):
                    if label in text:
                        return label
                return "NOT_ATTEMPTED"
            except openai_mod.RateLimitError as e:
                last_err = e
                time.sleep(_retry_after_seconds(e, attempt))
            except (
                openai_mod.APITimeoutError,
                openai_mod.APIConnectionError,
            ) as e:
                last_err = e
                time.sleep(min(60.0, 2**attempt + random.uniform(0, 1)))
        raise RuntimeError("OpenAI grading failed after retries") from last_err

    item_order = [rec["item_id"] for rec in items_df_records]
    graded = _load_graded(model_key)
    pending = [rec for rec in items_df_records if rec["item_id"] not in graded]
    total = len(items_df_records)

    if graded:
        print(
            f"[grader/{model_key}] resuming: {len(graded):,} graded, "
            f"{len(pending):,} remaining"
        )
    print(
        f"[grader/{model_key}] grading {len(pending):,} responses "
        f"({GRADE_WORKERS} workers)"
    )

    def grade_rec(rec: dict) -> dict:
        verdict = grade_one(rec["problem"], rec["answer"], rec["extracted"])
        return {
            "item_id": rec["item_id"],
            "topic": rec["topic"],
            "model_key": model_key,
            "verdict": verdict,
        }

    already = len(graded)
    t0 = time.time()
    n_done = already
    if pending:
        with ThreadPoolExecutor(max_workers=GRADE_WORKERS) as pool:
            futures = [pool.submit(grade_rec, rec) for rec in pending]
            for fut in as_completed(futures):
                row = fut.result()
                graded[row["item_id"]] = row
                n_done += 1
                if n_done % CHECKPOINT_INTERVAL == 0:
                    _write_grades(model_key, graded, item_order)
                    elapsed = time.time() - t0
                    new_done = n_done - already
                    rate = new_done / elapsed if elapsed else 0
                    remaining = total - n_done
                    eta_min = remaining / rate / 60 if rate else 0
                    print(
                        f"  [{model_key}] graded {n_done:,}/{total:,} "
                        f"({rate:.1f}/s, eta {eta_min:.1f}min)"
                    )

    _write_grades(model_key, graded, item_order)
    out_path = _grades_path(model_key)
    print(f"[grader/{model_key}] saved grades to {out_path}")

    rows = [graded[item_id] for item_id in item_order if item_id in graded]
    n_correct = sum(r["verdict"] == "CORRECT" for r in rows)
    n_incorrect = sum(r["verdict"] == "INCORRECT" for r in rows)
    n_na = sum(r["verdict"] == "NOT_ATTEMPTED" for r in rows)
    summary = {
        "model_key": model_key,
        "n_total": len(rows),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_not_attempted": n_na,
        "accuracy": n_correct / len(rows) if rows else 0.0,
    }
    print(f"[grader/{model_key}] {summary}")
    return summary


# ---------------------------------------------------------------------------
# vLLM grading backend — sharded across multiple H100 containers.
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    gpu="H100:1",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=1 * HOUR,
)
def grade_shard_vllm(shard_items: list[dict], grader_hf_name: str) -> list[dict]:
    """
    Grade one shard of items using a local vLLM model.
    Called in parallel via grade_model_vllm.map(); each call gets its own H100.

    shard_items: list of {item_id, topic, model_key, problem, answer, extracted}
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=grader_hf_name,
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=256,
        download_dir=HF_CACHE_MOUNT,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Disable thinking for Qwen3.x models so the output is just the verdict.
    chat_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "Qwen3" in grader_hf_name:
        chat_kwargs["enable_thinking"] = False

    prompts = []
    for item in shard_items:
        prompt_text = GRADER_TEMPLATE.format(
            question=item["problem"],
            target=item["answer"],
            predicted_answer=item.get("extracted", "") or "",
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], **chat_kwargs
            )
        )

    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10))

    results = []
    for item, out in zip(shard_items, outputs):
        text = out.outputs[0].text.strip().upper()
        for label in ("INCORRECT", "NOT_ATTEMPTED", "CORRECT"):
            if label in text:
                verdict = label
                break
        else:
            verdict = "NOT_ATTEMPTED"
        results.append({
            "item_id": item["item_id"],
            "topic": item["topic"],
            "model_key": item["model_key"],
            "verdict": verdict,
        })
    return results


@app.function(
    image=grader_image,
    volumes=VOLUMES,
    timeout=30 * MINUTES,
)
def grade_model_vllm(
    model_key: str,
    items_df_records: list[dict],
    grader_hf_name: str,
    n_shards: int = 5,
) -> dict:
    """
    Grade all responses for one model using a vLLM model sharded across N H100s.
    Resumes automatically: already-graded items are skipped.
    """
    item_order = [rec["item_id"] for rec in items_df_records]
    graded = _load_graded(model_key)
    pending = [
        {**rec, "model_key": model_key}
        for rec in items_df_records
        if rec["item_id"] not in graded
    ]

    if graded:
        print(
            f"[grader/{model_key}] resuming: {len(graded):,} graded, "
            f"{len(pending):,} remaining"
        )

    if pending:
        n = min(n_shards, len(pending))
        shards = [pending[i::n] for i in range(n)]
        print(
            f"[grader/{model_key}] dispatching {len(pending):,} items across "
            f"{n} shards (~{len(shards[0]):,} items each, grader={grader_hf_name})"
        )
        for shard_results in grade_shard_vllm.map(
            shards, kwargs={"grader_hf_name": grader_hf_name}
        ):
            for row in shard_results:
                graded[row["item_id"]] = row

    _write_grades(model_key, graded, item_order)
    print(f"[grader/{model_key}] saved grades to {_grades_path(model_key)}")

    rows = [graded[item_id] for item_id in item_order if item_id in graded]
    n_correct = sum(r["verdict"] == "CORRECT" for r in rows)
    n_incorrect = sum(r["verdict"] == "INCORRECT" for r in rows)
    n_na = sum(r["verdict"] == "NOT_ATTEMPTED" for r in rows)
    summary = {
        "model_key": model_key,
        "n_total": len(rows),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_not_attempted": n_na,
        "accuracy": n_correct / len(rows) if rows else 0.0,
    }
    print(f"[grader/{model_key}] {summary}")
    return summary
