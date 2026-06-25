#!/usr/bin/env python3
"""
run_retraction_experiment.py

Multi-trace stochastic retraction-rate experiment.

Runs run_once() N_REPEATS times on each of 15 selected traces across
MetaGPT, HyperAgent, and ChatDev, recording per-run Forced vs Semi-Forced
step-localization results.

Key metric: rate at which Forced commits a step that Semi-Forced retracts
            (returns NO_STEP_FOUND) — broken down overall / per-FM / per-framework.

Usage:
    python run_retraction_experiment.py            # full run
    python run_retraction_experiment.py --dry-run  # print plan, no API calls
    python run_retraction_experiment.py --analyze results/retraction_experiment/retraction_raw_<ts>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_stability import run_once, encode_row, PARSER_PATHS
from LLM_models_interface.llm_interface import LLMJudge, JudgeConfig, FAILURE_MODES

# ── Experiment config ─────────────────────────────────────────────────────────

DEFS_PATH   = Path("data/prompts/definitions.txt")
RESULTS_DIR = Path("results/retraction_experiment")

TRACE_LIST: list[tuple[str, int]] = [
    # MetaGPT — 6-step traces; trace_idx=0 ties to original stability run
    ("MetaGPT",     0),   # 3.1(g), 3.2
    ("MetaGPT",     6),   # 1.1(g), 2.6, 3.1(g), 3.2   ← original stability trace
    ("MetaGPT",    13),   # 2.5, 3.2, 3.3               ← zero globals
    ("MetaGPT",    15),   # 1.1(g), 1.3, 1.5(g), 2.2, 2.5, 2.6, 3.2
    ("MetaGPT",    17),   # 1.1(g), 1.3, 1.4, 2.1, 2.5, 3.1(g), 3.2
    # HyperAgent — 21–40 step traces, diverse FM sets
    ("HyperAgent",  0),   # 40 steps | 1.1(g), 1.3, 2.2, 2.4, 2.5, 2.6, 3.2
    ("HyperAgent",  1),   # 40 steps | 1.2, 1.3, 2.6, 3.2, 3.3
    ("HyperAgent",  5),   # 40 steps | 1.3, 1.4, 2.6, 3.2
    ("HyperAgent",  9),   # 21 steps | 1.3, 2.6, 3.2     ← shortest
    ("HyperAgent", 19),   # 32 steps | 1.2, 1.3, 2.5, 2.6  ← no 3.x
    # ChatDev — ~64–70 step traces
    ("ChatDev",     0),   # 69 steps | 2.6, 3.3
    ("ChatDev",     1),   # 68 steps | 1.2, 1.3, 1.5(g), 2.2, 2.3, 2.5, 2.6, 3.2
    ("ChatDev",     6),   # 64 steps | 1.3, 2.6, 3.2
    ("ChatDev",     9),   # 64 steps | 1.3, 2.4, 2.5, 2.6, 3.2
    ("ChatDev",    17),   # 66 steps | 1.3, 2.3, 2.6, 3.3
]

N_REPEATS   = 15
MODEL       = "gemini-2.5-flash"
BACKEND     = "genai"
TEMPERATURE = 1.0
SLEEP_S     = 2      # seconds between run_once() calls
MAX_RETRIES = 4
RETRY_BASE  = 15     # initial retry sleep (doubles each attempt)

RAW_COLS = [
    "framework", "trace_idx", "repeat", "fm",
    "judge", "forced", "semi", "relaxed",
]

GLOBAL_FMS = {"1.1", "1.5", "3.1"}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def ckpt_path(fw: str, tidx: int) -> Path:
    return RESULTS_DIR / f"ckpt_{fw.lower()}_{tidx:03d}.csv"


def ckpt_done(fw: str, tidx: int) -> bool:
    p = ckpt_path(fw, tidx)
    if not p.exists():
        return False
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        # complete = N_REPEATS × 14 rows, no FAIL entries
        return len(rows) >= N_REPEATS * len(FAILURE_MODES)
    except Exception:
        return False


# ── Trace loading ─────────────────────────────────────────────────────────────

def load_trace(fw: str, tidx: int) -> dict:
    path = PARSER_PATHS[fw]
    with open(path) as f:
        traces = json.load(f)
    return traces[tidx]


# ── API call with retry + backoff ─────────────────────────────────────────────

def run_with_retry(
    judge: LLMJudge, trace: dict, definitions: str
) -> dict | None:
    delay = RETRY_BASE
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_once(judge, trace, definitions, debug=False)
        except Exception as exc:
            short = str(exc)[:140]
            if attempt < MAX_RETRIES:
                print(f"      [retry {attempt}/{MAX_RETRIES - 1}] {short}  "
                      f"— wait {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                print(f"      [FAIL after {MAX_RETRIES} attempts] {short}", flush=True)
                traceback.print_exc()
                return None


# ── Retraction detection ──────────────────────────────────────────────────────

def is_committed(forced_str: str) -> bool:
    """True when Forced returned a real location (step or global), not skipped/failed."""
    return forced_str not in ("~", "?", "", "FAIL")


def compute_stats(raw_rows: list[dict]) -> dict:
    """
    Compute retraction rates from the raw row list.
    A retraction event = judge=1 AND forced committed AND semi=RETRACTED.
    Denominator = judge=1 AND forced committed (excludes skipped / parse-fail).
    """
    def empty():
        return {"forced_commits": 0, "retractions": 0}

    overall  = empty()
    per_fm   = defaultdict(empty)
    per_fw   = defaultdict(empty)
    per_fw_fm = defaultdict(lambda: defaultdict(empty))

    for row in raw_rows:
        if str(row["judge"]) != "1":
            continue
        if not is_committed(str(row["forced"])):
            continue
        fm = str(row["fm"])
        fw = str(row["framework"])

        for d in (overall, per_fm[fm], per_fw[fw], per_fw_fm[fw][fm]):
            d["forced_commits"] += 1
            if str(row["semi"]) == "RETRACTED":
                d["retractions"] += 1

    def add_rate(d: dict) -> dict:
        fc = d["forced_commits"]
        d["rate"] = round(d["retractions"] / fc, 4) if fc > 0 else None
        return d

    return {
        "overall":   add_rate(overall),
        "per_fm":    {k: add_rate(v) for k, v in sorted(per_fm.items())},
        "per_fw":    {k: add_rate(v) for k, v in sorted(per_fw.items())},
        "per_fw_fm": {fw: {fm: add_rate(v) for fm, v in sorted(fms.items())}
                      for fw, fms in sorted(per_fw_fm.items())},
    }


# ── Summary text ──────────────────────────────────────────────────────────────

def build_summary_text(stats: dict, n_traces: int, n_repeats: int) -> str:
    ov = stats["overall"]
    rate_pct = f"{ov['rate']*100:.1f}%" if ov["rate"] is not None else "N/A"

    top_fm_retractions = sorted(
        [(fm, v) for fm, v in stats["per_fm"].items() if v["rate"] is not None],
        key=lambda x: x[1]["rate"], reverse=True
    )[:5]

    lines = [
        "Forced-commit → Semi-Forced-retraction experiment",
        f"  {n_traces} traces × 3 frameworks  ·  {n_repeats} repeats/trace  ·  {MODEL}  T={TEMPERATURE}",
        "",
        f"Overall retraction rate: {rate_pct}",
        f"  ({ov['retractions']} retractions out of {ov['forced_commits']} forced-commits)",
        "",
        "Per-framework:",
    ]
    for fw, v in stats["per_fw"].items():
        r = f"{v['rate']*100:.1f}%" if v["rate"] is not None else "N/A"
        lines.append(f"  {fw:<14} {r:>7}  ({v['retractions']}/{v['forced_commits']})")

    lines += ["", "Top-5 FMs by retraction rate:"]
    for fm, v in top_fm_retractions:
        r = f"{v['rate']*100:.1f}%"
        lines.append(f"  FM {fm:<5} {r:>7}  ({v['retractions']}/{v['forced_commits']})")

    # thesis sentence
    if ov["rate"] is not None and top_fm_retractions:
        concentrated = ", ".join(fm for fm, v in top_fm_retractions if v["rate"] > ov["rate"])
        lines += [
            "",
            "Thesis sentence:",
            f'  "Across {n_traces} traces and 3 frameworks (MetaGPT, HyperAgent, ChatDev), '
            f"the Forced localizer committed to a step that Semi-Forced subsequently retracted "
            f"in {rate_pct} of cases (n={ov['forced_commits']} forced-commits), concentrated "
            f'in modes {concentrated or "—"}."',
        ]
    return "\n".join(lines)


# ── Analysis-only mode ────────────────────────────────────────────────────────

def run_analyze(raw_csv: Path) -> None:
    with open(raw_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    stats = compute_stats(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_outputs(rows, stats, RESULTS_DIR, ts)


# ── Output writers ────────────────────────────────────────────────────────────

def _write_outputs(
    all_rows: list[dict], stats: dict, out_dir: Path, ts: str
) -> None:
    # 1. Raw CSV
    raw_out = out_dir / f"retraction_raw_{ts}.csv"
    with open(raw_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nRaw data   → {raw_out}")

    # 2. Summary CSV (one row per FM per framework)
    sum_rows = []
    for fw, fms in stats["per_fw_fm"].items():
        for fm, v in fms.items():
            sum_rows.append({
                "framework": fw, "fm": fm,
                "forced_commits": v["forced_commits"],
                "retractions":    v["retractions"],
                "retraction_rate": v["rate"],
            })
    # add overall rows
    for fw, v in stats["per_fw"].items():
        sum_rows.append({
            "framework": fw, "fm": "OVERALL",
            "forced_commits": v["forced_commits"],
            "retractions":    v["retractions"],
            "retraction_rate": v["rate"],
        })
    for fm, v in stats["per_fm"].items():
        sum_rows.append({
            "framework": "ALL", "fm": fm,
            "forced_commits": v["forced_commits"],
            "retractions":    v["retractions"],
            "retraction_rate": v["rate"],
        })
    ov = stats["overall"]
    sum_rows.append({
        "framework": "ALL", "fm": "OVERALL",
        "forced_commits": ov["forced_commits"],
        "retractions":    ov["retractions"],
        "retraction_rate": ov["rate"],
    })

    sum_out = out_dir / f"retraction_summary_{ts}.csv"
    with open(sum_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["framework", "fm", "forced_commits", "retractions", "retraction_rate"]
        )
        writer.writeheader()
        writer.writerows(sum_rows)
    print(f"Summary    → {sum_out}")

    # 3. Text summary
    text = build_summary_text(stats, n_traces=len(TRACE_LIST), n_repeats=N_REPEATS)
    txt_out = out_dir / f"retraction_summary_{ts}.txt"
    txt_out.write_text(text)
    print(f"Text       → {txt_out}")
    print()
    print(text)


# ── Main run ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print plan and exit without making any API calls")
    ap.add_argument("--analyze",  metavar="RAW_CSV",
                    help="Re-run aggregation on an existing raw CSV, skip LLM calls")
    ap.add_argument("--n",        type=int, default=N_REPEATS,
                    help=f"Repeats per trace (default: {N_REPEATS})")
    ap.add_argument("--sleep",    type=float, default=SLEEP_S,
                    help=f"Sleep between run_once() calls in seconds (default: {SLEEP_S})")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_repeats = args.n
    sleep_s   = args.sleep

    if args.analyze:
        run_analyze(Path(args.analyze))
        return

    total = len(TRACE_LIST)
    est_calls = total * n_repeats * 4
    print(f"\n{'='*62}")
    print(f"  Retraction-rate experiment")
    print(f"  {total} traces  ×  {n_repeats} repeats  ×  ~4 calls  = ~{est_calls} API calls")
    print(f"  Model: {MODEL}  backend: {BACKEND}  T={TEMPERATURE}")
    print(f"  Est. cost: ~${est_calls * 0.004:.1f}  |  Est. time: ~{est_calls * (4+sleep_s) / 60:.0f} min")
    print(f"{'='*62}\n")

    if args.dry_run:
        print("Trace plan:")
        for fw, tidx in TRACE_LIST:
            done = "DONE" if ckpt_done(fw, tidx) else "pending"
            print(f"  {fw:<14} trace_idx={tidx}  [{done}]")
        return

    definitions = DEFS_PATH.read_text()
    judge = LLMJudge(JudgeConfig(
        model=MODEL,
        backend=BACKEND,
        temperature=TEMPERATURE,
        definitions_path=str(DEFS_PATH.resolve()),
        examples_path=str(Path("data/prompts/examples.txt").resolve()),
    ))

    total_api_calls = 0
    exp_start = time.time()
    all_rows: list[dict] = []

    for trace_i, (fw, tidx) in enumerate(TRACE_LIST, 1):
        tag = f"[{trace_i}/{total}]  {fw}  trace_idx={tidx}"

        if ckpt_done(fw, tidx):
            print(f"{tag}  — checkpoint complete, loading …")
            with open(ckpt_path(fw, tidx), newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
            continue

        trace = load_trace(fw, tidx)
        n_steps = len(trace["steps"])
        print(f"\n{tag}  ({n_steps} steps)", flush=True)

        trace_rows: list[dict] = []
        for repeat in range(1, n_repeats + 1):
            t0 = time.time()
            result = run_with_retry(judge, trace, definitions)
            elapsed = time.time() - t0

            if result is None:
                # record null rows so the repeat is marked as failed
                for fm in FAILURE_MODES:
                    trace_rows.append({
                        "framework": fw, "trace_idx": tidx, "repeat": repeat,
                        "fm": fm, "judge": "", "forced": "FAIL",
                        "semi": "FAIL", "relaxed": "",
                    })
                print(f"  repeat {repeat:2d}/{n_repeats}  ← FAILED  ({elapsed:.1f}s)", flush=True)
                if repeat < n_repeats:
                    time.sleep(sleep_s)
                continue

            n_present = sum(result["judge"].values())
            api_calls = 4 if n_present > 0 else 2
            total_api_calls += api_calls

            n_retracts = sum(
                1 for fm in FAILURE_MODES
                if result["judge"].get(fm, 0) == 1
                and isinstance(result["semi"].get(fm), dict)
                and result["semi"][fm].get("retracted", False)
            )

            elapsed_str = f"{elapsed:.1f}s"
            elapsed_total = time.time() - exp_start
            calls_left = (total - trace_i) * n_repeats * 4 + (n_repeats - repeat) * 4
            print(
                f"  repeat {repeat:2d}/{n_repeats}  judge={n_present} present  "
                f"retracts={n_retracts}  {elapsed_str}  "
                f"calls={total_api_calls}  ~{calls_left} left",
                flush=True,
            )

            for encoded in encode_row(repeat, result):
                trace_rows.append({
                    "framework": fw,
                    "trace_idx": tidx,
                    "repeat":    encoded["run"],
                    "fm":        encoded["fm"],
                    "judge":     str(encoded["judge"]),
                    "forced":    encoded["forced"],
                    "semi":      encoded["semi"],
                    "relaxed":   encoded["relaxed"],
                })

            if repeat < n_repeats:
                time.sleep(sleep_s)

        # Write checkpoint
        ck = ckpt_path(fw, tidx)
        with open(ck, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_COLS)
            writer.writeheader()
            writer.writerows(trace_rows)
        all_rows.extend(trace_rows)
        print(f"  → checkpoint saved: {ck.name}", flush=True)

    # Final outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = compute_stats(all_rows)
    _write_outputs(all_rows, stats, RESULTS_DIR, ts)

    total_elapsed = time.time() - exp_start
    print(f"\nTotal API calls: {total_api_calls}  |  Wall time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
