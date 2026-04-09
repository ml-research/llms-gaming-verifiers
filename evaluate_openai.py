"""
Evaluate an OpenAI model on SLR-Bench via the Chat Completions API.

Outputs are saved in the same format as evaluate_model_vllm.py so that
shortcuts.py can evaluate them without any conversion.

Usage:
    export OPENAI_API_KEY=sk-...
    python evaluate_openai.py --model gpt-4o
    python evaluate_openai.py --model o3 --reasoning-effort high
    python evaluate_openai.py --model gpt-4o --out-path output/eval-openai
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MAX_COMPLETION_TOKENS = 16000


def call_model(client: OpenAI, model: str, prompt: str, reasoning_effort: str | None,
               max_completion_tokens: int) -> tuple[str, int, int]:
    """Call the OpenAI Chat Completions API and return (text, prompt_tokens, completion_tokens)."""
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate an OpenAI model on SLR-Bench.")
    parser.add_argument("--model", required=True, help="OpenAI model ID (e.g. gpt-4o, o3)")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "none"],
                        help="Reasoning effort for o-series and GPT-5 models.")
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS,
                        help=f"Max completion tokens per sample (default: {DEFAULT_MAX_COMPLETION_TOKENS})")
    parser.add_argument("--out-path", default="output/eval-openai",
                        help="Directory to store per-model result folders.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers (default: 8).")
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Evaluate on a subset of N examples (for quick tests).")
    parser.add_argument("--rerun-truncated", action="store_true",
                        help="Re-run only samples that hit the token limit in an existing run.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)

    # Build output directory tag
    tag = args.model.split("/")[-1]
    if args.reasoning_effort:
        tag += f"-effort-{args.reasoning_effort}"
    out_dir = os.path.join(args.out_path, tag)
    outputs_path = os.path.join(out_dir, "model_outputs.json")

    # Handle rerun-truncated / skip logic
    existing_outputs = None
    truncated_ids = None
    if os.path.exists(outputs_path):
        if args.rerun_truncated:
            with open(outputs_path) as f:
                existing_outputs = json.load(f)
            meta_path = os.path.join(out_dir, "meta.json")
            old_max = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    old_max = json.load(f).get("max_completion_tokens")
            if old_max is None:
                old_max = max(r["completion_tokens"] for r in existing_outputs)
            threshold = old_max - 10
            truncated_ids = {r["problem_id"] for r in existing_outputs if r["completion_tokens"] >= threshold}
            print(f"--rerun-truncated: {len(truncated_ids)} truncated samples (old limit={old_max}). Re-running only those.")
        else:
            print(f"Output already exists at {out_dir}. Skipping. Use --rerun-truncated to re-run truncated samples.")
            return

    print(f"Loading SLR-Bench...")
    dataset = load_dataset("AIML-TUDA/SLR-Bench", "v1-All", split="test")
    if args.test_subset:
        dataset = dataset.select(range(min(args.test_subset, len(dataset))))
    if truncated_ids is not None:
        dataset = dataset.filter(lambda x: x["id"] in truncated_ids)
        print(f"Filtered to {len(dataset)} truncated samples.")

    print(f"Evaluating {args.model} on {len(dataset)} problems (workers={args.workers})...")

    def _run(example):
        text, pt, ct = call_model(
            client, args.model, example["prompt"],
            args.reasoning_effort, args.max_completion_tokens,
        )
        return {
            "problem_id": example["id"],
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "model_completion": text,
            "ground_truth": example["ground-truth rule"],
            "reference": {
                "validation_program": example["validation program"],
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            },
        }

    model_outputs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, ex): ex["id"] for ex in dataset}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            try:
                model_outputs.append(fut.result())
            except Exception as e:
                print(f"  Error on problem {futures[fut]}: {e}")

    model_outputs.sort(key=lambda x: x["problem_id"])

    # Merge with existing if rerunning truncated samples
    if existing_outputs is not None and truncated_ids is not None:
        new_by_id = {r["problem_id"]: r for r in model_outputs}
        model_outputs = [new_by_id.get(r["problem_id"], r) for r in existing_outputs]
        print(f"Merged {len(new_by_id)} re-run samples into {len(model_outputs)} total.")

    os.makedirs(out_dir, exist_ok=True)
    with open(outputs_path, "w") as f:
        json.dump(model_outputs, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "model": args.model,
            "tag": tag,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_completion_tokens,
        }, f, indent=2)

    exceeded = sum(1 for r in model_outputs if r["completion_tokens"] >= args.max_completion_tokens - 10)
    print(f"Done. {exceeded}/{len(model_outputs)} outputs hit the token limit.")
    print(f"Saved to {outputs_path}")


if __name__ == "__main__":
    main()
