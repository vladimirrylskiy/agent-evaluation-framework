#!/usr/bin/env python3
"""
run_fm_scan.py — Scan multiple MAS frameworks for failure mode distribution.

One LLM call per trace: detects which FMs are present and locates the step
where each occurs, reported as a percentage through the trace.

Usage:
    python run_fm_scan.py --n 1                   # test: 1 trace per framework
    python run_fm_scan.py --n 50                  # overnight: 50 per framework
    python run_fm_scan.py --n 50 --frameworks ChatDev MetaGPT
    python run_fm_scan.py --n 1 --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from LLM_models_interface.llm_interface import (
    LLMJudge,
    JudgeConfig,
    build_judge_locate_prompt,
    parse_judge_locate_response,
    FAILURE_MODES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DEFS_PATH   = ROOT / "data" / "prompts" / "definitions.txt"
RESULTS_DIR = ROOT / "results" / "fm_scan"

PARSER_FILES: dict[str, Path] = {
    "ChatDev":     ROOT / "parsers/chatdev_parser/chatdev_output_mad.json",
    "AG2":         ROOT / "parsers/ag2_parser/ag2_output_mad.json",
    "AppWorld":    ROOT / "parsers/appworld_parser/appworld_output_mad.json",
    "HyperAgent":  ROOT / "parsers/hyperagent_parser/hyperagent_output_mad.json",
    "MetaGPT":     ROOT / "parsers/metagpt_parser/metagpt_output_mad.json",
    "MagenticOne": ROOT / "parsers/magenticone_parser/magenticone_output_mad.json",
    "OpenManus":   ROOT / "parsers/openmanus_parser/openmanus_output_mad.json",
}

RETRY_DELAYS = [60, 120]

# ── Pipeline ──────────────────────────────────────────────────────────────────

def _dispatch(judge: LLMJudge, prompt: str, call_id: str):
    delays = [0] + RETRY_DELAYS
    last_exc = None
    for attempt, delay in enumerate(delays):
        if delay:
            print(f"    rate-limit — waiting {delay}s (retry {attempt})…")
            time.sleep(delay)
        try:
            return judge._dispatch(prompt, call_id)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                last_exc = e
                continue
            raise
    raise last_exc


def run_trace(judge: LLMJudge, trace: dict, definitions: str) -> dict[str, dict]:
    """Single LLM call: detect + locate all FMs in one trace."""
    steps = trace["steps"]
    n_steps = len(steps)
    prompt = build_judge_locate_prompt(steps, definitions)
    r = _dispatch(judge, prompt, "judge_locate")
    return parse_judge_locate_response(r.raw_text, n_steps)


# ── CSV encoding ──────────────────────────────────────────────────────────────

def encode_rows(framework: str, trace_idx: int, n_steps: int,
                parsed: dict[str, dict]) -> list[dict]:
    """One CSV row per FM."""
    rows = []
    for mode in FAILURE_MODES:
        entry = parsed.get(mode, {})
        rows.append({
            "framework": framework,
            "trace_idx": trace_idx,
            "n_steps":   n_steps,
            "fm":        mode,
            "present":   1 if entry.get("present") else 0,
            "step_idx":  entry.get("step_idx", ""),
            "pct":       entry.get("pct", ""),
            "is_global": 1 if entry.get("is_global") else 0,
        })
    return rows


# ── Distribution analysis ─────────────────────────────────────────────────────

def compute_distribution(raw_rows: list[dict], frameworks: list[str]) -> list[dict]:
    """
    Per (framework, fm):
      - n_traces: traces processed
      - n_present: traces where FM was detected
      - prevalence_%
      - mean_pct, median_pct, min_pct, max_pct: position distribution
        (only for non-global detections where pct is available)
    """
    totals:  dict[tuple, int]        = defaultdict(int)
    present: dict[tuple, int]        = defaultdict(int)
    pcts:    dict[tuple, list[float]] = defaultdict(list)

    for r in raw_rows:
        if r["present"] == "error":
            continue
        key = (r["framework"], r["fm"])
        totals[key] += 1
        if str(r["present"]) == "1":
            present[key] += 1
            # Global modes (1.1, 1.5, 3.1) have no meaningful step position
            if r["fm"] not in {"1.1", "1.5", "3.1"} and r["pct"] not in ("", None):
                try:
                    pcts[key].append(float(r["pct"]))
                except (ValueError, TypeError):
                    pass

    rows = []
    for fw in frameworks:
        for mode in FAILURE_MODES:
            key = (fw, mode)
            n_total   = totals.get(key, 0)
            n_present = present.get(key, 0)
            vals      = pcts.get(key, [])
            rows.append({
                "framework":      fw,
                "fm":             mode,
                "n_traces":       n_total,
                "n_present":      n_present,
                "prevalence_pct": round(100 * n_present / n_total, 1) if n_total else "",
                "mean_pct":       round(statistics.mean(vals), 1)                          if vals else "",
                "median_pct":     round(statistics.median(vals), 1)                        if vals else "",
                "std_pct":        round(statistics.stdev(vals), 1) if len(vals) >= 2 else "",
                "min_pct":        round(min(vals), 1)                                      if vals else "",
                "max_pct":        round(max(vals), 1)                                      if vals else "",
            })
    return rows


def print_prevalence_table(dist_rows: list[dict]) -> None:
    """FM × Framework prevalence table."""
    frameworks = list(dict.fromkeys(r["framework"] for r in dist_rows))
    col_w = 16

    header = f"{'FM':<6}" + "".join(f"{fw:>{col_w}}" for fw in frameworks)
    sep    = "-" * len(header)
    print("\n=== FM Prevalence (n_present / n_traces) ===")
    print(sep)
    print(header)
    print(sep)

    by_key = {(r["framework"], r["fm"]): r for r in dist_rows}
    for mode in FAILURE_MODES:
        row_str = f"{mode:<6}"
        for fw in frameworks:
            r     = by_key.get((fw, mode), {})
            n     = r.get("n_present", 0)
            total = r.get("n_traces", 0)
            pct   = r.get("prevalence_pct", "")
            cell  = f"{n}/{total} ({pct}%)" if total else "—"
            row_str += f"{cell:>{col_w}}"
        print(row_str)
    print(sep)


def print_position_table(dist_rows: list[dict]) -> None:
    """FM × Framework position table: median % and std dev for detected FMs."""
    frameworks = list(dict.fromkeys(r["framework"] for r in dist_rows))
    col_w = 16

    header = f"{'FM':<6}" + "".join(f"{fw:>{col_w}}" for fw in frameworks)
    sep    = "-" * len(header)
    print("\n=== FM Position in Trace (median% ± std) — present cases only ===")
    print(sep)
    print(header)
    print(sep)

    by_key = {(r["framework"], r["fm"]): r for r in dist_rows}
    for mode in FAILURE_MODES:
        row_str = f"{mode:<6}"
        for fw in frameworks:
            r      = by_key.get((fw, mode), {})
            median = r.get("median_pct", "")
            std    = r.get("std_pct", "")
            if median == "":
                cell = "—"
            elif std == "":
                cell = f"{median}%"
            else:
                cell = f"{median}% ±{std}"
            row_str += f"{cell:>{col_w}}"
        print(row_str)
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1,
                        help="Traces per framework (default: 1)")
    parser.add_argument("--frameworks", nargs="+", default=list(PARSER_FILES.keys()),
                        choices=list(PARSER_FILES.keys()),
                        help="Frameworks to include (default: all 7)")
    parser.add_argument("--model",       default="gemini-2.5-flash")
    parser.add_argument("--backend",     default="genai")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path     = RESULTS_DIR / f"fm_scan_raw_{stamp}.csv"
    dist_path    = RESULTS_DIR / f"fm_scan_dist_{stamp}.csv"

    definitions = DEFS_PATH.read_text() if DEFS_PATH.exists() else ""

    judge = LLMJudge(JudgeConfig(
        name="fm_scan",
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        definitions_path=str(DEFS_PATH),
        examples_path="",
    ))

    raw_rows: list[dict] = []
    selected = args.frameworks
    total_traces = args.n * len(selected)
    done = 0

    print(f"\nFrameworks : {selected}")
    print(f"Traces     : {args.n} per framework  ({total_traces} total, 1 call each)\n")

    for fw in selected:
        traces = json.loads(PARSER_FILES[fw].read_text())[:args.n]

        for i, trace in enumerate(traces):
            done += 1
            n_steps = len(trace["steps"])
            print(f"[{done}/{total_traces}] {fw} trace {i}  ({n_steps} steps)")
            try:
                parsed = run_trace(judge, trace, definitions)
                raw_rows.extend(encode_rows(fw, i, n_steps, parsed))
            except Exception as e:
                print(f"  ERROR: {e}")
                for mode in FAILURE_MODES:
                    raw_rows.append({
                        "framework": fw, "trace_idx": i, "n_steps": n_steps,
                        "fm": mode, "present": "error",
                        "step_idx": "", "pct": "", "is_global": "",
                    })

    # Write raw CSV
    if raw_rows:
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"\nRaw  → {raw_path}")

    # Compute and write distribution
    dist_rows = compute_distribution(raw_rows, selected)
    with open(dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dist_rows)
    print(f"Dist → {dist_path}\n")

    print_prevalence_table(dist_rows)
    print_position_table(dist_rows)


if __name__ == "__main__":
    main()
