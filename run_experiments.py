"""
Batch experiment runner — collects results across human-labelled traces and models.

Outputs three CSVs in results/:
  human_validation.csv   — per (trace, model, mode): human label, judge label, match
  localization_map.csv   — per (trace, model, mode): result_type (steps/global/none/NO_DECOMPOSITION/error)
  partial_trace.csv      — per (trace, mode): first_detected_pct, stable_from_pct, stable

Uses the same judge logic and taxonomy mapping as judge_localise_app.py via experiment_core.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Indices into MAD_human_labelled_dataset.json (0–18). None = all 19.
TRACE_INDICES: list[int] | None = list(range(19))

MODELS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

BACKEND = "genai"

# Retry on 429: wait these many seconds between attempts (3 attempts total)
RETRY_DELAYS = [60, 120]

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

HUMAN_PATH    = ROOT / "data" / "MAST-Data" / "MAD_human_labelled_dataset.json"
DEFS_PATH     = ROOT / "data" / "prompts" / "definitions.txt"
EXAMPLES_PATH = ROOT / "data" / "prompts" / "examples.txt"
RESULTS_DIR   = ROOT / "results"

# Parsed step outputs indexed by (mas_name_key, parser_path)
# GAIA traces are annotated by MagenticOne in MAD.
PARSER_FILES: dict[str, str] = {
    "ChatDev":    "parsers/chatdev_parser/chatdev_output_mad.json",
    "AG2":        "parsers/ag2_parser/ag2_output_mad.json",
    "AppWorld":   "parsers/appworld_parser/appworld_output_mad.json",
    "HyperAgent": "parsers/hyperagent_parser/hyperagent_output_mad.json",
    "MetaGPT":    "parsers/metagpt_parser/metagpt_output_mad.json",
    "GAIA":       "parsers/magenticone_parser/magenticone_output_mad.json",
}

# Modes that are global properties of the trace — localizer skips the LLM call
GLOBAL_MODES = {"1.1", "1.5", "3.1"}

# ── Imports ───────────────────────────────────────────────────────────────────

from LLM_models_interface.llm_interface import (
    build_judge_prompt,
    build_subordinate_localise_prompt,
    parse_14_modes,
    parse_localized_steps,
    LLMJudge,
    JudgeConfig,
    FAILURE_MODES,
)
from experiment_core import (
    build_ground_truth,
    compute_convergence,
    FRACTIONS,
    FRAC_LABELS,
)


# ── Parsed-steps index ────────────────────────────────────────────────────────

def load_parsed_index() -> dict[tuple[str, str], list[dict]]:
    """Load all parser outputs. Returns {(mas_name, str_trace_id): steps}."""
    index: dict[tuple[str, str], list[dict]] = {}
    for mas_name, rel_path in PARSER_FILES.items():
        path = ROOT / rel_path
        if not path.exists():
            print(f"  WARNING: parser output not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for trace in data:
            tid = str(trace["metadata"].get("trace_id", ""))
            index[(mas_name, tid)] = trace["steps"]
    return index


def get_steps(rec: dict, index: dict) -> list[dict] | None:
    """Return parsed steps for a human-labelled record, or None if unavailable.

    AppWorld: human file uses integer trace_ids (0, 5, 11) but the parser uses
    hash-based IDs (e.g. '692c77d_1'). The raw trace text embeds the hash ID
    as '(xxxxxxx_n)' in the task header — we extract it for matching.
    """
    mas = rec["mas_name"]
    tid = str(rec["trace_id"])

    steps = index.get((mas, tid))
    if steps is not None:
        return steps

    if mas == "AppWorld":
        raw = rec.get("trace", "")
        for eid in re.findall(r"\(([a-f0-9]{7}_\d+)\)", raw):
            steps = index.get(("AppWorld", eid))
            if steps is not None:
                return steps

    return None  # triggers NO_DECOMPOSITION


# ── LLM call with retry on 429 ────────────────────────────────────────────────

def dispatch_with_retry(judge: LLMJudge, prompt: str, call_id: str):
    """Call judge._dispatch with exponential backoff on rate-limit errors."""
    delays = [0] + RETRY_DELAYS
    last_exc: Exception | None = None
    for attempt, delay in enumerate(delays):
        if delay:
            print(f"    rate-limit 429 — waiting {delay}s (retry {attempt}/{len(RETRY_DELAYS)})…")
            time.sleep(delay)
        try:
            return judge._dispatch(prompt, call_id)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                last_exc = e
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompts() -> tuple[str, str]:
    defs = DEFS_PATH.read_text() if DEFS_PATH.exists() else ""
    exs  = EXAMPLES_PATH.read_text() if EXAMPLES_PATH.exists() else ""
    return defs, exs


def make_judge(model: str, name: str) -> LLMJudge:
    return LLMJudge(JudgeConfig(
        name=name,
        model=model,
        backend=BACKEND,
        temperature=0.0,
        definitions_path=str(DEFS_PATH),
        examples_path=str(EXAMPLES_PATH),
    ))


def mode_name_from_defs(mode: str, defs: str) -> str:
    m = re.search(rf"{re.escape(mode)}\s+([^\n:]+)", defs)
    return m.group(1).strip() if m else f"Mode {mode}"


# ── Experiment 1: Human Validation ───────────────────────────────────────────

def run_human_validation(
    records: list[dict],
    indices: list[int],
    defs: str,
    exs: str,
) -> list[dict]:
    rows = []
    total = len(indices) * len(MODELS)
    done = 0

    for idx in indices:
        record = records[idx]
        trace_label = f"{record['mas_name']}_tid{record['trace_id']}"
        ground_truth = build_ground_truth(record)
        trace_text = record.get("trace", "")

        for model in MODELS:
            done += 1
            print(f"[{done}/{total}] human_val  trace={trace_label}  model={model}")

            judge = make_judge(model, f"hval_{trace_label}_{model}")
            prompt = build_judge_prompt(trace_text, defs, exs)

            try:
                resp = dispatch_with_retry(judge, prompt, f"hval_{idx}")
                pred = parse_14_modes(resp.raw_text)
            except Exception as e:
                print(f"  ERROR: {e}")
                pred = {m: -1 for m in FAILURE_MODES}

            agree_n = sum(
                1 for m in FAILURE_MODES
                if m in ground_truth and ground_truth[m] == pred.get(m, -1)
            )
            comparable = sum(1 for m in FAILURE_MODES if m in ground_truth)
            agree_pct = round(100 * agree_n / comparable, 1) if comparable else None

            for mode in FAILURE_MODES:
                human_v = ground_truth.get(mode)
                judge_v = pred.get(mode, -1)
                match = None
                if human_v is not None and judge_v != -1:
                    match = int(human_v == judge_v)
                rows.append({
                    "trace_idx": idx,
                    "trace": trace_label,
                    "mas_name": record["mas_name"],
                    "trace_id": record["trace_id"],
                    "round": record.get("round", ""),
                    "model": model,
                    "mode": mode,
                    "human": human_v if human_v is not None else "excluded",
                    "judge": judge_v if judge_v != -1 else "error",
                    "match": match if match is not None else "n/a",
                    "agreement_pct": agree_pct,
                })

    return rows


# ── Experiment 2: Localization Map ────────────────────────────────────────────

def run_localization_map(
    records: list[dict],
    indices: list[int],
    defs: str,
    exs: str,
    parsed_index: dict,
) -> list[dict]:
    rows = []
    total = len(indices) * len(MODELS)
    done = 0

    for idx in indices:
        record = records[idx]
        trace_label = f"{record['mas_name']}_tid{record['trace_id']}"
        trace_text = record.get("trace", "")

        steps = get_steps(record, parsed_index)
        n_steps = len(steps) if steps is not None else 0
        if steps is None:
            print(f"  WARNING: no parsed steps for {trace_label} — marking NO_DECOMPOSITION")

        for model in MODELS:
            done += 1
            print(f"[{done}/{total}] localization  trace={trace_label}  "
                  f"({n_steps} steps)  model={model}")

            judge = make_judge(model, f"loc_{trace_label}_{model}")

            # Full-trace verdict (use raw text — same as human validation)
            prompt_full = build_judge_prompt(trace_text, defs, exs)
            try:
                resp_full = dispatch_with_retry(judge, prompt_full, f"loc_full_{idx}")
                full_pred = parse_14_modes(resp_full.raw_text)
            except Exception as e:
                print(f"  ERROR full judge: {e}")
                full_pred = {m: -1 for m in FAILURE_MODES}

            for mode in FAILURE_MODES:
                baseline = full_pred.get(mode, -1)

                if baseline != 1:
                    rows.append(_loc_row(record, trace_label, model, mode,
                                        baseline, "none", ""))
                    continue

                # No parsed steps → can't localize
                if steps is None:
                    rows.append(_loc_row(record, trace_label, model, mode,
                                        baseline, "NO_DECOMPOSITION", ""))
                    continue

                # Global modes: skip LLM, emit directly
                if mode in GLOBAL_MODES:
                    rows.append(_loc_row(record, trace_label, model, mode,
                                        baseline, "global", "global"))
                    continue

                # Subordinate localizer with real steps
                mname = mode_name_from_defs(mode, defs)
                prompt_sub = build_subordinate_localise_prompt(
                    mode, mname, steps, defs, exs
                )
                try:
                    resp_sub = dispatch_with_retry(
                        judge, prompt_sub, f"loc_sub_{idx}_{mode}"
                    )
                    raw_result = parse_localized_steps(resp_sub.raw_text)
                except Exception as e:
                    print(f"  ERROR sub localizer {mode}: {e}")
                    rows.append(_loc_row(record, trace_label, model, mode,
                                        baseline, "error", ""))
                    continue

                # parse_localized_steps returns str 'global' or list[int]
                if raw_result == "global":
                    result_type = "global"
                    steps_str = "global"
                elif isinstance(raw_result, list) and raw_result:
                    # Validate: keep only indices in [0, n_steps-1]
                    valid = [s for s in raw_result if 0 <= s < n_steps]
                    invalid = [s for s in raw_result if s not in valid]
                    if invalid:
                        print(f"  OOB indices dropped for {mode}: {invalid} "
                              f"(trace has {n_steps} steps, valid 0–{n_steps-1})")
                    if valid:
                        result_type = "steps"
                        steps_str = ",".join(str(s) for s in sorted(valid))
                    else:
                        result_type = "none"
                        steps_str = ""
                else:
                    result_type = "none"
                    steps_str = ""

                rows.append(_loc_row(record, trace_label, model, mode,
                                     baseline, result_type, steps_str))

    return rows


def _loc_row(record, trace_label, model, mode, baseline, result_type, steps_str):
    return {
        "trace_idx": record.get("trace_id"),   # human dataset index
        "trace": trace_label,
        "mas_name": record["mas_name"],
        "trace_id": record["trace_id"],
        "model": model,
        "mode": mode,
        "baseline_verdict": baseline,
        "result_type": result_type,
        "steps": steps_str,
    }


# ── Experiment 3: Partial-Trace Detection ─────────────────────────────────────

def run_partial_trace(
    records: list[dict],
    indices: list[int],
    defs: str,
    exs: str,
) -> list[dict]:
    rows = []
    model = MODELS[0]
    total = len(indices)
    done = 0

    for idx in indices:
        record = records[idx]
        trace_label = f"{record['mas_name']}_tid{record['trace_id']}"
        trace_text = record.get("trace", "")
        words = trace_text.split()
        n_words = len(words)

        done += 1
        print(f"[{done}/{total}] partial_trace  trace={trace_label}  ({n_words} words)")

        verdicts_by_frac: dict[float, dict[str, int]] = {}
        judge = make_judge(model, f"partial_{trace_label}")

        for frac in FRACTIONS:
            n = max(1, round(n_words * frac))
            prefix_text = " ".join(words[:n])
            prompt = build_judge_prompt(prefix_text, defs, exs)
            try:
                resp = dispatch_with_retry(judge, prompt, f"partial_{idx}_frac{frac}")
                verdicts_by_frac[frac] = parse_14_modes(resp.raw_text)
            except Exception as e:
                print(f"  ERROR at {FRAC_LABELS[frac]}: {e}")
                verdicts_by_frac[frac] = {m: -1 for m in FAILURE_MODES}
            print(f"    {FRAC_LABELS[frac]} done")

        for mode in FAILURE_MODES:
            conv = compute_convergence(mode, verdicts_by_frac)
            rows.append({
                "trace_idx": idx,
                "trace": trace_label,
                "mas_name": record["mas_name"],
                "trace_id": record["trace_id"],
                "mode": mode,
                "verdict_25":  verdicts_by_frac.get(0.25, {}).get(mode, -1),
                "verdict_50":  verdicts_by_frac.get(0.50, {}).get(mode, -1),
                "verdict_75":  verdicts_by_frac.get(0.75, {}).get(mode, -1),
                "verdict_100": verdicts_by_frac.get(1.00, {}).get(mode, -1),
                "first_detected_pct": conv["first_detected"] or "absent",
                "stable_from_pct":    conv["stable_from"]    or "absent",
                "stable": (
                    "stable"   if conv["stable"] is True  else
                    "unstable" if conv["stable"] is False else
                    "absent"
                ),
            })

    return rows


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  (no rows, skipping {path.name})")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows → {path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_loc_summary(loc_rows: list[dict]) -> None:
    from collections import defaultdict, Counter
    by_trace_model: dict[tuple, Counter] = defaultdict(Counter)
    for r in loc_rows:
        if r["baseline_verdict"] == 1:
            by_trace_model[(r["trace"], r["model"])][r["result_type"]] += 1

    print("\n" + "=" * 75)
    print(f"  Localization summary (baseline=1 modes only)")
    print(f"  {'Trace':<35} {'Model':<22} steps  global  none  error  NO_DECOMP")
    print("  " + "-" * 73)
    for (trace, model), cnt in sorted(by_trace_model.items()):
        print(f"  {trace:<35} {model:<22} "
              f"{cnt.get('steps',0):<7}"
              f"{cnt.get('global',0):<8}"
              f"{cnt.get('none',0):<6}"
              f"{cnt.get('error',0):<7}"
              f"{cnt.get('NO_DECOMPOSITION',0)}")
    print("=" * 75)


def print_hval_summary(hval_rows: list[dict]) -> None:
    agg: dict[tuple, float] = {}
    for row in hval_rows:
        pct = row.get("agreement_pct")
        if pct is not None:
            agg[(row["trace"], row["model"])] = float(pct)

    print("\n" + "=" * 60)
    print(f"  Human validation agreement")
    print(f"  {'Trace':<35} {'Model':<22} Agreement")
    print("  " + "-" * 58)
    for (trace, model), pct in sorted(agg.items()):
        print(f"  {trace:<35} {model:<22} {pct}%")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["human_validation", "localization", "partial_trace"],
        default=None,
        help="Run only one experiment (default: run all three)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    with open(HUMAN_PATH, encoding="utf-8") as f:
        records = json.load(f)

    indices = TRACE_INDICES if TRACE_INDICES is not None else list(range(len(records)))
    defs, exs = load_prompts()

    print(f"\nRunning on {len(indices)} traces × {len(MODELS)} models")
    print(f"Traces: {indices}  Models: {MODELS}\n")

    run_all = args.only is None

    if run_all or args.only == "human_validation":
        print("── Experiment 1: Human Validation ──────────────────────────")
        hval_rows = run_human_validation(records, indices, defs, exs)
        write_csv(RESULTS_DIR / "human_validation.csv", hval_rows)
        print_hval_summary(hval_rows)

    if run_all or args.only == "localization":
        parsed_index = load_parsed_index()
        print(f"Parsed index loaded: {len(parsed_index)} entries\n")
        print("── Experiment 2: Localization Map ───────────────────────────")
        loc_rows = run_localization_map(records, indices, defs, exs, parsed_index)
        write_csv(RESULTS_DIR / "localization_map.csv", loc_rows)
        print_loc_summary(loc_rows)

    if run_all or args.only == "partial_trace":
        print("── Experiment 3: Partial-Trace Detection ────────────────────")
        partial_rows = run_partial_trace(records, indices, defs, exs)
        write_csv(RESULTS_DIR / "partial_trace.csv", partial_rows)


if __name__ == "__main__":
    main()
