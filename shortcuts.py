"""
Shortcut detection via Isomorphic Perturbation Testing (IPT).

Loads model outputs from a directory, runs IPT on each prediction,
and prints/saves a performance table and a reward-shortcut table.

Usage:
    python shortcuts.py --output-dir output/eval-openai
    python shortcuts.py --output-dir output/eval-openai --models gpt-4o gpt-5-mini
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from IPT.ipt.verifier import extract_hypothesis_with_meta, verify_ipt
from pricing import get_pricing_v2

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# SLR-Bench: 1000 problems split into four equal complexity tiers.
# Problems are either 0-indexed (0–999) or 1-indexed (1–1000).
_TIERS_0 = [(0, 249, "basic"), (250, 499, "easy"), (500, 749, "medium"), (750, 999, "hard")]
_TIERS_1 = [(1, 250, "basic"), (251, 500, "easy"), (501, 750, "medium"), (751, 1000, "hard")]


def _complexity_from_id(problem_id: int, zero_based: bool) -> str:
    tiers = _TIERS_0 if zero_based else _TIERS_1
    for lo, hi, name in tiers:
        if lo <= problem_id <= hi:
            return name
    return "hard"


class IPTEvaluator:
    """Run IPT on a directory of model outputs and report accuracy + shortcut rates."""

    def __init__(
        self,
        output_dir: str,
        models_filter: Optional[List[str]] = None,
        timeout: int = 5,
        workers: int = 0,
    ):
        self.output_dir = output_dir
        self.models_filter = models_filter
        self.timeout = timeout
        self.workers = workers if workers > 0 else max(1, mp.cpu_count() - 1)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _normalize_outputs(self, raw: Any) -> List[Dict[str, Any]]:
        """Accept both list and dict-keyed JSON formats."""
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            items = []
            for key in sorted(raw, key=lambda k: int(k) if str(k).isdigit() else k):
                item = raw[key]
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                if item.get("problem_id") is None:
                    try:
                        item["problem_id"] = int(key)
                    except (ValueError, TypeError):
                        pass
                items.append(item)
            return items
        return []

    def load_outputs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load model_outputs.json from each subdirectory of output_dir."""
        all_outputs: Dict[str, List[Dict[str, Any]]] = {}
        model_dirs = sorted(
            d for d in glob.glob(os.path.join(self.output_dir, "*")) if os.path.isdir(d)
        )
        for model_dir in model_dirs:
            name = os.path.basename(model_dir)
            if self.models_filter and name not in self.models_filter:
                continue
            path = os.path.join(model_dir, "model_outputs.json")
            if not os.path.exists(path):
                print(f"  ⚠  {name} — no model_outputs.json, skipping")
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception as e:
                print(f"  ⚠  {name} — JSON error: {e}")
                continue
            outputs = self._normalize_outputs(raw)
            if not outputs:
                print(f"  ⚠  {name} — empty, skipping")
                continue
            all_outputs[name] = outputs
        return all_outputs

    # ── IPT Evaluation ─────────────────────────────────────────────────────────

    @staticmethod
    def _run_ipt(args: Tuple) -> Tuple[str, int, Dict[str, Any]]:
        model_name, idx, hypothesis, validation_program, eval_config, timeout = args
        result = verify_ipt(hypothesis, validation_program, eval_config, timeout=timeout)
        return model_name, idx, result

    def evaluate(self, all_outputs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract hypotheses and run IPT on all outputs. Updates dicts in-place."""
        jobs = []
        empty = {
            "extensional_correct": False,
            "isomorphic_correct": False,
            "is_reward_shortcut": False,
            "extensional_partial": 0.0,
            "isomorphic_partial": 0.0,
            "syntax_valid": False,
            "error": None,
        }

        for model_name, outputs in all_outputs.items():
            is_gpt = "gpt" in model_name.lower()
            for idx, output in enumerate(outputs):
                completion = output.get("model_completion", "")
                if not isinstance(completion, str) or not completion.strip():
                    output.update({**empty, "error": "missing completion"})
                    continue

                reference = output.get("reference", {})
                validation_program = reference.get("validation_program", "") if isinstance(reference, dict) else ""
                if not validation_program.strip():
                    output.update({**empty, "error": "missing validation_program"})
                    continue

                eval_config = reference.get("evaluation_config", {}) if isinstance(reference, dict) else {}

                hypothesis, _ = extract_hypothesis_with_meta(
                    completion, enable_line_parsing=not is_gpt
                )
                output["extracted_hypothesis"] = hypothesis
                if not hypothesis:
                    output.update({**empty, "error": "no hypothesis extracted"})
                    continue

                jobs.append((model_name, idx, hypothesis, validation_program, eval_config, self.timeout))

        if not jobs:
            return all_outputs

        if self.workers > 1:
            with mp.Pool(processes=self.workers) as pool:
                it = pool.imap_unordered(self._run_ipt, jobs, chunksize=10)
                if tqdm is not None:
                    it = tqdm(it, total=len(jobs), desc="IPT")
                for model_name, idx, result in it:
                    all_outputs[model_name][idx].update(result)
        else:
            it = jobs if tqdm is None else tqdm(jobs, desc="IPT")
            for args in it:
                model_name, idx, result = self._run_ipt(args)
                all_outputs[model_name][idx].update(result)

        return all_outputs

    # ── Statistics ─────────────────────────────────────────────────────────────

    def build_dataframe(self, all_outputs: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        rows = []
        for model_name, outputs in all_outputs.items():
            ids = [o.get("problem_id") for o in outputs if o.get("problem_id") is not None]
            zero_based = min(ids) == 0 if ids else True

            for output in outputs:
                pid = output.get("problem_id")
                complexity = _complexity_from_id(int(pid), zero_based) if pid is not None else "basic"

                ext = output.get("extensional_correct")
                iso = output.get("isomorphic_correct")
                is_shortcut = bool(output.get("is_reward_shortcut", False))

                rows.append({
                    "model_name": model_name,
                    "problem_id": pid,
                    "complexity": complexity,
                    "extensional_correct": ext,
                    "isomorphic_correct": iso,
                    "is_reward_shortcut": is_shortcut,
                    "syntax_valid": bool(output.get("syntax_valid", False)),
                    "prompt_tokens": output.get("prompt_tokens"),
                    "completion_tokens": output.get("completion_tokens"),
                })
        return pd.DataFrame(rows)

    # ── Tables ─────────────────────────────────────────────────────────────────

    def _token_cost_maps(self, df: pd.DataFrame):
        comp = pd.to_numeric(df.get("completion_tokens"), errors="coerce").fillna(0)
        prompt = pd.to_numeric(df.get("prompt_tokens"), errors="coerce").fillna(0)
        tmp = df[["model_name"]].copy()
        tmp["comp"] = comp
        tmp["prompt"] = prompt

        price_cache = {}
        for m in tmp["model_name"].astype(str).unique():
            p = get_pricing_v2(m)
            price_cache[m] = (float(p.get("input") or 0), float(p.get("output") or 0))

        pairs = tmp["model_name"].astype(str).map(lambda m: price_cache.get(m, (0.0, 0.0)))
        tmp["cost"] = (tmp["prompt"] * pairs.apply(lambda t: t[0]) +
                       tmp["comp"] * pairs.apply(lambda t: t[1])) / 1_000_000

        tokens_by_model = tmp.groupby("model_name")["comp"].sum().to_dict()
        cost_by_model = tmp.groupby("model_name")["cost"].sum().to_dict()
        return tokens_by_model, cost_by_model, float(comp.sum()), float(tmp["cost"].sum())

    def performance_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isomorphic accuracy per complexity tier, plus tokens and cost."""
        tiers = ["basic", "easy", "medium", "hard"]
        tokens_map, cost_map, total_tokens, total_cost = self._token_cost_maps(df)
        syntax_map = df.groupby("model_name")["syntax_valid"].mean().to_dict()

        rows = []
        for model_name in sorted(df["model_name"].unique()):
            mdf = df[df["model_name"] == model_name]
            row = {"model_name": model_name}
            total_iso = 0
            for t in tiers:
                tdf = mdf[mdf["complexity"] == t]
                n = int(tdf["isomorphic_correct"].sum()) if not tdf.empty and tdf["isomorphic_correct"].notna().any() else 0
                row[t] = n
                total_iso += n
            row["total"] = total_iso
            row["syntax"] = syntax_map.get(model_name)
            row["tokens"] = tokens_map.get(model_name)
            row["cost_usd"] = cost_map.get(model_name)
            rows.append(row)

        if rows:
            rows.append({
                "model_name": "SUM",
                **{t: sum(r[t] for r in rows) for t in tiers},
                "total": sum(r["total"] for r in rows),
                "syntax": df["syntax_valid"].mean() if df["syntax_valid"].notna().any() else None,
                "tokens": total_tokens,
                "cost_usd": total_cost,
            })
        return pd.DataFrame(rows)

    def shortcut_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reward shortcut counts (Ns = extensional − isomorphic) per tier."""
        tiers = ["basic", "easy", "medium", "hard"]
        has_ext = df["extensional_correct"].notna().any()
        rows = []

        for model_name in sorted(df["model_name"].unique()):
            mdf = df[df["model_name"] == model_name]
            row = {"model_name": model_name}
            total_ns = 0
            for t in tiers:
                tdf = mdf[mdf["complexity"] == t]
                if tdf.empty or not has_ext:
                    row[t] = None
                    continue
                iso = int(tdf["isomorphic_correct"].sum()) if tdf["isomorphic_correct"].notna().any() else 0
                ext = int(tdf["extensional_correct"].sum()) if tdf["extensional_correct"].notna().any() else 0
                ns = ext - iso
                row[t] = ns
                total_ns += ns
            row["total"] = total_ns
            rows.append(row)

        if rows:
            rows.append({
                "model_name": "SUM",
                **{t: sum(r[t] or 0 for r in rows) for t in tiers},
                "total": sum(r["total"] or 0 for r in rows),
            })
        return pd.DataFrame(rows)

    # ── Printing ───────────────────────────────────────────────────────────────

    def print_results(self, perf: pd.DataFrame, shortcuts: pd.DataFrame) -> None:
        W = 100
        n_models = int((perf["model_name"] != "SUM").sum()) if not perf.empty else 1
        tiers = ["basic", "easy", "medium", "hard"]

        def _fmt_null(x):
            return "-" if x is None or (isinstance(x, float) and pd.isna(x)) else x

        # ── Performance ────────────────────────────────────────────────────────
        if not perf.empty:
            print("\n" + "=" * W)
            print("  TABLE 1  |  PERFORMANCE  (isomorphic accuracy, 250 problems per tier)")
            print("=" * W)

            disp = perf.copy()
            for t in tiers + ["total"]:
                denom = 250 if t != "total" else 1000
                def _pct(val, mn, d=denom, nm=n_models):
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return "-"
                    return int(round(int(val) / (d * (nm if mn == "SUM" else 1)) * 100))
                disp[t] = [_pct(v, mn) for v, mn in zip(disp[t], disp["model_name"])]

            disp["syntax"] = disp["syntax"].apply(
                lambda x: "-" if x is None or (isinstance(x, float) and pd.isna(x))
                else int(round(float(x) * 100))
            )
            disp["tokens"] = disp["tokens"].apply(
                lambda x: "-" if x is None or (isinstance(x, float) and pd.isna(x))
                else f"{float(x) / 1e6:.1f}M"
            )
            disp["cost_usd"] = disp["cost_usd"].apply(
                lambda x: "-" if x is None or (isinstance(x, float) and pd.isna(x))
                else f"${float(x):.2f}"
            )
            disp = disp.rename(columns={
                "basic": "Basic%", "easy": "Easy%", "medium": "Medium%", "hard": "Hard%",
                "total": "Avg%", "syntax": "Syntax%", "tokens": "Tokens", "cost_usd": "Cost",
            })
            print(disp.fillna("-").to_string(index=False))
            print("=" * W)

        # ── Shortcuts ──────────────────────────────────────────────────────────
        if not shortcuts.empty:
            print("\n" + "=" * W)
            print("  TABLE 2  |  REWARD SHORTCUTS  (Ns = extensional − isomorphic, per tier)")
            print("=" * W)
            disp2 = shortcuts.rename(columns={
                "basic": "Basic", "easy": "Easy", "medium": "Medium", "hard": "Hard",
                "total": "Ns_total",
            })
            print(disp2.fillna("-").to_string(index=False))
            print("=" * W)

    # ── Saving ─────────────────────────────────────────────────────────────────

    def save_results(self, df: pd.DataFrame, perf: pd.DataFrame, shortcuts: pd.DataFrame) -> str:
        out = os.path.join(self.output_dir, "ipt_results")
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "detailed_results.csv"), index=False)
        perf.to_csv(os.path.join(out, "performance.csv"), index=False)
        shortcuts.to_csv(os.path.join(out, "shortcuts.csv"), index=False)
        return out

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self) -> None:
        W = 100
        filter_note = f"  (models: {', '.join(self.models_filter)})" if self.models_filter else ""
        print("\n" + "=" * W)
        print("  IPT  ·  Isomorphic Perturbation Testing")
        print(f"  {self.output_dir}{filter_note}")
        print("=" * W)

        all_outputs = self.load_outputs()
        if not all_outputs:
            print("  No model outputs found.")
            return

        n_items = sum(len(v) for v in all_outputs.values())
        print(f"\n  Loaded  {len(all_outputs)} models · {n_items:,} outputs")

        all_outputs = self.evaluate(all_outputs)
        df = self.build_dataframe(all_outputs)

        perf = self.performance_table(df)
        shortcuts = self.shortcut_table(df)

        self.print_results(perf, shortcuts)

        out = self.save_results(df, perf, shortcuts)
        print(f"\n  Saved   {out}/\n")
        print("=" * W + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Isomorphic Perturbation Testing on model outputs."
    )
    parser.add_argument(
        "--output-dir", default="output/eval-openai",
        help="Directory containing per-model output subdirectories (default: output/eval-openai)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict evaluation to these model subdirectory names",
    )
    parser.add_argument(
        "--timeout", type=int, default=5,
        help="Prolog evaluation timeout per sample in seconds (default: 5)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker processes (0 = auto, 1 = single-process)",
    )
    args = parser.parse_args()

    IPTEvaluator(
        output_dir=args.output_dir,
        models_filter=args.models,
        timeout=args.timeout,
        workers=args.workers,
    ).run()


if __name__ == "__main__":
    main()
