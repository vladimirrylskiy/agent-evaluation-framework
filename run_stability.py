#!/usr/bin/env python3
"""
run_stability.py — Stochastic stability experiment for the 3-framework localizer.

Runs the 4-call pipeline (judge → Forced batch → Semi-Forced batch → Relaxed)
N times on a single trace and reports per-(FM, framework) agreement statistics.

Usage:
    python run_stability.py --trace-idx 0 --n 30
    python run_stability.py --trace-idx 0 --n 30 --model gemini-2.5-pro --temperature 1.0
    python run_stability.py --list-traces
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from LLM_models_interface.llm_interface import (
    LLMJudge,
    JudgeConfig,
    build_judge_prompt,
    parse_14_modes,
    build_forced_batch_prompt,
    build_semi_forced_batch_prompt,
    parse_batch_localise_response,
    build_relaxed_localise_prompt,
    parse_relaxed_steps,
    FAILURE_MODES,
)
from experiment_core import steps_to_text

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFS_PATH   = Path("data/prompts/definitions.txt")
RESULTS_DIR = Path("results/stability")

PARSER_PATHS: dict[str, Path] = {
    "ChatDev":     Path("parsers/chatdev_parser/chatdev_output_mad.json"),
    "AG2":         Path("parsers/ag2_parser/ag2_output_mad.json"),
    "AppWorld":    Path("parsers/appworld_parser/appworld_output_mad.json"),
    "HyperAgent":  Path("parsers/hyperagent_parser/hyperagent_output_mad.json"),
    "MetaGPT":     Path("parsers/metagpt_parser/metagpt_output_mad.json"),
    "MagenticOne": Path("parsers/magenticone_parser/magenticone_output_mad.json"),
    "OpenManus":   Path("parsers/openmanus_parser/openmanus_output_mad.json"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _steps_to_str(steps) -> str:
    """Canonical string for a step result: '3,5', 'global', or ''."""
    if steps == "global":
        return "global"
    if not steps:
        return ""
    return ",".join(str(s) for s in sorted(steps))


def run_once(judge: LLMJudge, trace: dict, definitions: str, debug: bool = False) -> dict:
    """
    Execute one full pipeline iteration on a trace.

    Returns:
        {
          "judge":   {mode: 0|1, ...},
          "forced":  {mode: {'steps': list|'global', 'retracted': bool} | None, ...},
          "semi":    {mode: {'steps': list|'global'|None, 'retracted': bool} | None, ...},
          "relaxed": {step_key: [mode, ...], ...},   # parse_relaxed_steps format
        }
    """
    steps = trace["steps"]
    n_steps = len(steps)
    trace_text = steps_to_text(steps)

    result: dict = {"judge": {}, "forced": {}, "semi": {}, "relaxed": {}}

    # 1. Judge
    p = build_judge_prompt(trace_text, definitions, "")
    r = judge._dispatch(p, "stab_judge")
    result["judge"] = parse_14_modes(r.raw_text)

    present_modes = [m for m in FAILURE_MODES if result["judge"].get(m, 0) == 1]

    if debug:
        print(f"\n  [DEBUG] Judge raw (last 300 chars):\n    ...{r.raw_text[-300:]!r}")
        print(f"  [DEBUG] parse_14_modes → {result['judge']}")
        print(f"  [DEBUG] present_modes  → {present_modes}")

    # 2. Forced batch
    if present_modes:
        p = build_forced_batch_prompt(present_modes, steps, definitions)
        r = judge._dispatch(p, "stab_forced")
        result["forced"] = parse_batch_localise_response(r.raw_text, n_steps)
        if debug:
            print(f"\n  [DEBUG] Forced raw:\n{r.raw_text}")
            print(f"  [DEBUG] parse_batch (forced) → {result['forced']}")
    elif debug:
        print("  [DEBUG] Forced: skipped (no present modes)")

    # 3. Semi-Forced batch
    if present_modes:
        p = build_semi_forced_batch_prompt(present_modes, steps, definitions)
        r = judge._dispatch(p, "stab_semi")
        result["semi"] = parse_batch_localise_response(r.raw_text, n_steps)
        if debug:
            print(f"\n  [DEBUG] Semi raw:\n{r.raw_text}")
            print(f"  [DEBUG] parse_batch (semi)   → {result['semi']}")
    elif debug:
        print("  [DEBUG] Semi: skipped (no present modes)")

    # 4. Relaxed
    p = build_relaxed_localise_prompt(steps, definitions)
    r = judge._dispatch(p, "stab_relaxed")
    result["relaxed"] = parse_relaxed_steps(r.raw_text, n_steps)

    if debug:
        print(f"\n  [DEBUG] Relaxed raw:\n{r.raw_text}")
        print(f"  [DEBUG] parse_relaxed → {result['relaxed']}")

    return result


def encode_row(run_i: int, run: dict) -> list[dict]:
    """Convert one run result into a list of CSV rows (one per FM)."""
    rows = []
    relaxed = run["relaxed"]

    for mode in FAILURE_MODES:
        judge_val = run["judge"].get(mode, 0)

        # Forced: only meaningful when judge said present
        forced_v = run["forced"].get(mode)
        if judge_val == 0:
            forced_str = "~"            # judge said absent — not queried
        elif forced_v is None:
            forced_str = "?"            # queried but parse failed
        else:
            forced_str = _steps_to_str(forced_v["steps"]) or "?"

        # Semi-Forced: only meaningful when judge said present
        semi_v = run["semi"].get(mode)
        if judge_val == 0:
            semi_str = "~"
        elif semi_v is None:
            semi_str = "?"
        elif semi_v["retracted"]:
            semi_str = "RETRACTED"
        else:
            semi_str = _steps_to_str(semi_v["steps"]) or "?"

        # Relaxed: always queried; collect all step keys where this mode appears
        relaxed_hits = sorted(
            [k for k, modes in relaxed.items() if mode in modes],
            key=lambda k: -1 if k == "global" else int(k),
        )
        relaxed_str = ",".join(
            "global" if k == "global" else k for k in relaxed_hits
        ) or ""

        rows.append({
            "run":      run_i,
            "fm":       mode,
            "judge":    judge_val,
            "forced":   forced_str,
            "semi":     semi_str,
            "relaxed":  relaxed_str,
        })
    return rows


# ── Stability analysis ────────────────────────────────────────────────────────

def compute_stability(raw_rows: list[dict], n_runs: int) -> list[dict]:
    """
    For each (FM, framework) compute:
        majority  — most common answer
        agreement — fraction of runs matching the majority (0.0–1.0)
        n_queried — how many runs actually queried this mode (Forced/Semi skip absent)
    """
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for row in raw_rows:
        fm = row["fm"]
        buckets[(fm, "judge")].append(str(row["judge"]))
        buckets[(fm, "forced")].append(row["forced"])
        buckets[(fm, "semi")].append(row["semi"])
        buckets[(fm, "relaxed")].append(row["relaxed"])

    summary = []
    for fm in FAILURE_MODES:
        entry = {"fm": fm}
        for fw in ("judge", "forced", "semi", "relaxed"):
            vals = buckets[(fm, fw)]
            # For forced/semi, exclude "~" runs (judge said absent → not queried)
            if fw in ("forced", "semi"):
                active = [v for v in vals if v != "~"]
            else:
                active = vals
            n_active = len(active)
            if not active:
                majority, agreement = "", 0.0
            else:
                most_common, count = Counter(active).most_common(1)[0]
                majority, agreement = most_common, count / n_active
            entry[f"{fw}_majority"]  = majority
            entry[f"{fw}_agreement"] = round(agreement, 3)
            entry[f"{fw}_n"]         = n_active
        summary.append(entry)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework",   default="ChatDev", choices=list(PARSER_PATHS.keys()),
                   help="MAS framework to load traces from (default: ChatDev)")
    p.add_argument("--trace-idx",   type=int, default=0,
                   help="Index of the trace within the selected framework (default: 0)")
    p.add_argument("--n",           type=int, default=30,
                   help="Number of repetitions (default: 30)")
    p.add_argument("--model",       default="gemini-2.5-flash",
                   help="LLM model name (default: gemini-2.5-flash)")
    p.add_argument("--backend",     default="genai", choices=["genai", "anthropic", "ollama"],
                   help="Backend (default: genai)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (default: 1.0 — use 0.0 for determinism check)")
    p.add_argument("--out-dir",     default=str(RESULTS_DIR),
                   help=f"Output directory (default: {RESULTS_DIR})")
    p.add_argument("--list-traces", action="store_true",
                   help="Print available traces and exit")
    p.add_argument("--debug", action="store_true",
                   help="Print raw LLM responses and parsed results for every run")
    p.add_argument("--print-summary", metavar="CSV",
                   help="Re-print the rich table from an existing summary CSV (no LLM calls)")
    p.add_argument("--fix-semi", metavar="RAW_CSV",
                   help="Re-run only the semi-forced calls using judge verdicts from an existing raw CSV (30 calls)")
    return p.parse_args()


def _fmt_majority(fw: str, majority: str, n_active: int) -> str:
    """Human-readable majority answer for one (framework, mode) cell."""
    if n_active == 0:
        return "N/A"
    if not majority:
        return "not detected"
    if fw == "judge":
        return "present" if majority == "1" else "absent"
    if majority == "?":
        return "parse fail"
    if majority == "RETRACTED":
        return "retracted"
    if majority == "global":
        return "global"
    steps = majority.replace(",", ", ")
    return f"step {steps}" if "," not in majority else f"steps {steps}"


def print_rich_table(summary: list[dict]) -> None:
    """Print agreement % + majority answer for each (FM, framework)."""
    col = 26
    print(f"\n{'FM':<8}  {'Judge':<{col}}  {'Forced':<{col}}  {'Semi':<{col}}  {'Relaxed':<{col}}")
    print("-" * (8 + 4 * (col + 2)))
    for row in summary:
        cells = []
        for fw in ("judge", "forced", "semi", "relaxed"):
            agr = row[f"{fw}_agreement"]
            n   = row[f"{fw}_n"]
            maj = row[f"{fw}_majority"]
            if n == 0:
                cell = "N/A"
            else:
                readable = _fmt_majority(fw, maj, n)
                cell = f"{agr:.0%}  {readable}"
            cells.append(f"{cell:<{col}}")
        print(f"{row['fm']:<8}  " + "  ".join(cells))


def main():
    args = parse_args()

    # ── Re-print mode: read existing CSV, no LLM calls ───────────────────────
    if args.print_summary:
        path = Path(args.print_summary)
        if not path.exists():
            sys.exit(f"ERROR: {path} not found")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for fw in ("judge", "forced", "semi", "relaxed"):
                row[f"{fw}_agreement"] = float(row[f"{fw}_agreement"])
                row[f"{fw}_n"]         = int(row[f"{fw}_n"])
        print_rich_table(rows)
        return

    # ── Fix-semi mode: replay only semi calls from an existing raw CSV ────────
    if args.fix_semi:
        raw_path = Path(args.fix_semi)
        if not raw_path.exists():
            sys.exit(f"ERROR: {raw_path} not found")

        with open(raw_path, newline="") as f:
            all_rows = list(csv.DictReader(f))

        # group rows by run_id
        from collections import defaultdict
        runs: dict[int, dict[str, dict]] = defaultdict(dict)
        for row in all_rows:
            runs[int(row["run"])][row["fm"]] = row

        # infer trace_idx from raw CSV filename (e.g. stability_raw_0_...)
        stem = raw_path.stem  # e.g. "stability_raw_0_20260617_..."
        parts = stem.split("_")
        try:
            trace_idx = int(parts[2])
        except (IndexError, ValueError):
            trace_idx = args.trace_idx

        n_runs = len(runs)
        print(f"\nFix-semi: {n_runs} runs found in {raw_path.name}", flush=True)
        print(f"Trace idx : {trace_idx}  model={args.model}  temp={args.temperature}", flush=True)

        traces_path = PARSER_PATHS.get(args.framework, PARSER_PATHS["ChatDev"])
        if not traces_path.exists():
            sys.exit(f"ERROR: Traces not found at {traces_path}")
        print(f"Loading traces from {traces_path}...", end=" ", flush=True)
        with open(traces_path) as f:
            all_traces = json.load(f)
        trace = all_traces[trace_idx]
        steps = trace["steps"]
        n_steps = len(steps)
        print(f"ok ({n_steps} steps)", flush=True)

        if not DEFS_PATH.exists():
            sys.exit(f"ERROR: Definitions not found at {DEFS_PATH}")
        definitions = DEFS_PATH.read_text()

        print("Initialising LLM judge...", end=" ", flush=True)
        config = JudgeConfig(
            name=f"semi_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model=args.model, backend=args.backend, temperature=args.temperature,
            definitions_path=str(DEFS_PATH), examples_path="",
        )
        judge = LLMJudge(config)
        print("ok\n", flush=True)

        for i, run_id in enumerate(sorted(runs.keys()), 1):
            run_rows = runs[run_id]
            present_modes = [m for m in FAILURE_MODES
                             if run_rows.get(m, {}).get("judge") == "1"]
            print(f"  Run {run_id:2d}/{n_runs} ({len(present_modes)} present modes)...",
                  end=" ", flush=True)
            if present_modes:
                try:
                    p = build_semi_forced_batch_prompt(present_modes, steps, definitions)
                    r = judge._dispatch(p, f"semi_fix_{run_id}")
                    semi_result = parse_batch_localise_response(r.raw_text, n_steps)
                    for mode in FAILURE_MODES:
                        judge_val = run_rows.get(mode, {}).get("judge", "0")
                        if judge_val != "1":
                            new_semi = "~"
                        else:
                            v = semi_result.get(mode)
                            if v is None:
                                new_semi = "?"
                            elif v["retracted"]:
                                new_semi = "RETRACTED"
                            else:
                                new_semi = _steps_to_str(v["steps"]) or "?"
                        run_rows[mode]["semi"] = new_semi
                    retractions = sum(1 for m in present_modes
                                      if run_rows[m]["semi"] == "RETRACTED")
                    print(f"ok  (retractions={retractions})")
                except Exception as e:
                    print(f"ERROR: {e}")
            else:
                print("skipped (judge found nothing)")

        # flatten rows back in original order and write updated raw CSV
        updated_rows = []
        for row in all_rows:
            rid, fm = int(row["run"]), row["fm"]
            updated_rows.append(runs[rid][fm])

        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", "fm", "judge", "forced", "semi", "relaxed"])
            writer.writeheader()
            writer.writerows(updated_rows)

        # recompute summary
        summary = compute_stability(updated_rows, n_runs)
        summary_path = raw_path.parent / raw_path.name.replace("raw", "summary")
        summary_fieldnames = [
            "fm",
            "judge_majority", "judge_agreement", "judge_n",
            "forced_majority", "forced_agreement", "forced_n",
            "semi_majority",   "semi_agreement",   "semi_n",
            "relaxed_majority","relaxed_agreement","relaxed_n",
        ]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary)

        print(f"\nUpdated raw  → {raw_path}")
        print(f"Updated summary → {summary_path}\n")
        print_rich_table(summary)
        return

    # Load traces
    traces_path = PARSER_PATHS[args.framework]
    if not traces_path.exists():
        sys.exit(f"ERROR: Traces not found at {traces_path}")
    with open(traces_path) as f:
        all_traces = json.load(f)

    if args.list_traces:
        for i, t in enumerate(all_traces):
            tid = t.get("metadata", {}).get("trace_id", f"trace_{i}")
            print(f"  [{i:3d}]  {tid}  ({len(t['steps'])} steps)")
        return

    if args.trace_idx >= len(all_traces):
        sys.exit(f"ERROR: --trace-idx {args.trace_idx} out of range (0–{len(all_traces)-1})")

    trace = all_traces[args.trace_idx]
    trace_id = trace.get("metadata", {}).get("trace_id", f"trace_{args.trace_idx}")
    n_steps = len(trace["steps"])

    if not DEFS_PATH.exists():
        sys.exit(f"ERROR: Definitions not found at {DEFS_PATH}")
    definitions = DEFS_PATH.read_text()

    # Output paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = trace_id.replace("/", "_").replace(":", "_")
    fw_tag       = args.framework.lower()
    raw_path     = out_dir / f"stability_raw_{fw_tag}_{safe_id}_{stamp}.csv"
    summary_path = out_dir / f"stability_summary_{fw_tag}_{safe_id}_{stamp}.csv"

    # Judge config
    config = JudgeConfig(
        name=f"stability_{stamp}",
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        definitions_path=str(DEFS_PATH),
        examples_path="",
    )
    judge = LLMJudge(config)

    print(f"\nStability experiment")
    print(f"  Trace       : [{args.trace_idx}] {trace_id}  ({n_steps} steps)")
    print(f"  Repetitions : {args.n}")
    print(f"  Model       : {args.model}  backend={args.backend}  temp={args.temperature}")
    print(f"  Raw CSV     : {raw_path}")
    print(f"  Summary CSV : {summary_path}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    raw_rows: list[dict] = []
    raw_fieldnames = ["run", "fm", "judge", "forced", "semi", "relaxed"]

    with open(raw_path, "w", newline="") as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=raw_fieldnames)
        writer.writeheader()

        for i in range(1, args.n + 1):
            print(f"  Run {i:2d}/{args.n} ...", end=" ", flush=True)
            try:
                run = run_once(judge, trace, definitions, debug=args.debug)
                rows = encode_row(i, run)
                writer.writerows(rows)
                raw_f.flush()
                raw_rows.extend(rows)

                present = sum(1 for m in FAILURE_MODES if run["judge"].get(m, 0))
                retracted = sum(
                    1 for m in FAILURE_MODES
                    if run["semi"].get(m, {}) and run["semi"][m].get("retracted")
                )
                relaxed_hits = sum(1 for v in run["relaxed"].values() if v)
                print(f"judge={present} present  semi_retracted={retracted}  relaxed_hits={relaxed_hits}")
            except Exception as e:
                print(f"ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = compute_stability(raw_rows, args.n)

    summary_fieldnames = [
        "fm",
        "judge_majority", "judge_agreement", "judge_n",
        "forced_majority", "forced_agreement", "forced_n",
        "semi_majority",   "semi_agreement",   "semi_n",
        "relaxed_majority","relaxed_agreement","relaxed_n",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print_rich_table(summary)
    print(f"\nRaw results → {raw_path}")
    print(f"Summary     → {summary_path}")


if __name__ == "__main__":
    main()
