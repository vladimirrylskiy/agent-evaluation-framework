#!/usr/bin/env python3
"""
run_feasibility_matrix.py — Lean feasibility study: can a judge reliably localise
each MAST failure mode to a specific step in each MAS framework?

For every existing (framework × failure_mode) cell the script sends ONE Gemini call
that evaluates whether the logging schema of that framework, combined with the
characteristic footprint of the failure mode, makes step-level localisation
Reliable / Partially Reliable / Unreliable.

Global modes (1.1, 1.5, 3.1) are skipped — they are inherently trace-level and
cannot be localised; their status is recorded automatically without an API call.

Usage:
    python run_feasibility_matrix.py                    # dry-run: print plan, no API
    python run_feasibility_matrix.py --run              # execute all API calls
    python run_feasibility_matrix.py --run --preview 10 # cap trace preview at 10 steps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from LLM_models_interface.llm_interface import (
    FAILURE_MODES,
    _call_genai,
    _extract_mode_definition,
)

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

PARSER_FILES: dict[str, Path] = {
    "ChatDev":     ROOT / "parsers/chatdev_parser/chatdev_output_mad.json",
    "AG2":         ROOT / "parsers/ag2_parser/ag2_output_mad.json",
    "AppWorld":    ROOT / "parsers/appworld_parser/appworld_output_mad.json",
    "HyperAgent":  ROOT / "parsers/hyperagent_parser/hyperagent_output_mad.json",
    "MetaGPT":     ROOT / "parsers/metagpt_parser/metagpt_output_mad.json",
    "MagenticOne": ROOT / "parsers/magenticone_parser/magenticone_output_mad.json",
    "OpenManus":   ROOT / "parsers/openmanus_parser/openmanus_output_mad.json",
}

DEFS_PATH  = ROOT / "data" / "prompts" / "definitions.txt"
RESULTS_DIR = ROOT / "results" / "feasibility"
FM_SCAN_DIR = ROOT / "results" / "fm_scan"

# Failure modes that are inherently trace-level — localisation to a single step
# is meaningless by design.  These are skipped and auto-recorded.
GLOBAL_FMS: set[str] = {"1.1", "1.5", "3.1"}

FM_NAMES: dict[str, str] = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No / Incomplete Verification",
    "3.3": "Incorrect Verification",
}

MODEL      = "gemini-2.5-flash"
PROJECT    = "ingka-map-services-dev"
LOCATION   = "europe-west1"
RATE_SLEEP = 2          # seconds between API calls (avoids 429)
MAX_RETRIES = 2


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_traces(framework: str) -> list[dict]:
    path = PARSER_FILES.get(framework)
    if path is None or not path.exists():
        return []
    return json.loads(path.read_text())


def _format_steps_preview(steps: list[dict], max_steps: int) -> tuple[str, int]:
    """Return (formatted_text, total_steps). Truncates to max_steps steps."""
    n_total = len(steps)
    shown   = steps[:max_steps]
    lines   = []
    for step in shown:
        idx     = step.get("metadata", {}).get("step_index", 0)
        agent   = step.get("agent", "Unknown")
        content = str(step.get("content", "")).strip()
        if len(content) > 800:
            content = content[:800] + "\n… [truncated]"
        lines.append(f"[Step {idx}] {agent}:\n{content}")
    text = "\n\n".join(lines)
    if n_total > max_steps:
        text += f"\n\n… [{n_total - max_steps} more steps not shown]"
    return text, n_total


def _latest_fm_scan_raw() -> Path | None:
    """Return the most recent fm_scan_raw CSV, or None."""
    candidates = sorted(FM_SCAN_DIR.glob("fm_scan_raw_*.csv"))
    return candidates[-1] if candidates else None


def _build_representative_map() -> dict[tuple[str, str], int]:
    """
    Scan the most recent fm_scan_raw results to find one trace_idx per
    (framework, fm) cell where the FM was detected (present=1, is_global=0 preferred).
    Falls back to global detections if no step-level detection exists.
    Returns {} if no scan file is available (caller will use trace_idx=0 fallback).
    """
    raw_path = _latest_fm_scan_raw()
    if raw_path is None:
        return {}

    df = pd.read_csv(raw_path)
    df = df[df["present"] == 1].copy()

    rep: dict[tuple[str, str], int] = {}
    # prefer step-level (is_global==0); accept global if no step-level found
    for is_global in [0, 1]:
        subset = df[df["is_global"] == is_global]
        for _, row in subset.iterrows():
            key = (str(row["framework"]), str(row["fm"]))
            if key not in rep:
                rep[key] = int(row["trace_idx"])
    return rep


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a multi-agent systems (MAS) research expert specialising in \
failure-mode analysis and trace auditing.

Your task is to assess whether STEP-LEVEL LOCALISATION of a specific MAST failure mode \
is feasible given the logging schema and message structure of a particular MAS framework.

Definitions of the three feasibility verdicts:

• Reliable
  The log structure makes it unambiguous which step (or narrow interval) triggered \
the failure. Clear agent roles, explicit step delimiters, and rich per-step metadata \
allow an auditor to point to step N with high confidence.

• Partially Reliable
  Localisation is possible but with caveats — e.g. the failure is spread across \
several steps making it hard to single out one, or the framework's logs lack enough \
per-step context to be fully unambiguous without significant interpretation.

• Unreliable
  The log structure makes step-level localisation impossible or highly arbitrary — \
e.g. the failure is a diffuse property of the entire trace, step boundaries are \
absent or unclear, or the framework logs at a granularity that does not support \
pinpointing a single causal step.

Return ONLY a strict JSON object with exactly two fields:
  "feasibility" — one of: "Reliable", "Partially Reliable", "Unreliable"
  "reasoning"   — a 2–3 sentence explanation of your verdict

No markdown, no extra keys, no preamble."""


def build_feasibility_prompt(
    framework: str,
    fm_code: str,
    fm_definition: str,
    trace_preview: str,
    n_steps_total: int,
    n_steps_shown: int,
) -> str:
    return (
        f"FRAMEWORK: {framework}\n"
        f"FAILURE MODE: {fm_code} — {FM_NAMES[fm_code]}\n\n"
        f"FAILURE MODE DEFINITION:\n{fm_definition}\n\n"
        f"REPRESENTATIVE TRACE EXCERPT\n"
        f"(total steps in trace: {n_steps_total}; showing first {n_steps_shown}):\n\n"
        f"{trace_preview}\n\n"
        "─────────────────────────────────────────────\n"
        f"QUESTION:\n"
        f"Given the logging schema shown in this {framework} trace, could an auditor "
        f"reliably and unambiguously identify the EXACT step number (or a narrow "
        f"interval of steps) where failure mode «{fm_code} {FM_NAMES[fm_code]}» "
        f"occurred?\n\n"
        f"Evaluate whether the combination of:\n"
        f"  1. The information density and structure of {framework}'s step logs\n"
        f"  2. The characteristic footprint of failure mode {fm_code} ({FM_NAMES[fm_code]})\n\n"
        f"…makes step-level localisation RELIABLE, PARTIALLY RELIABLE, or UNRELIABLE.\n\n"
        '{"feasibility": ..., "reasoning": ...}'
    )


# ── API call ──────────────────────────────────────────────────────────────────

def _call_with_retry(prompt: str, call_id: str) -> dict:
    """
    Call Gemini with structured JSON output enforcement.
    Retries up to MAX_RETRIES times on 429 / RESOURCE_EXHAUSTED.
    Returns parsed dict with 'feasibility' and 'reasoning'.
    """
    from google.genai import types as genai_types

    delays = [0] + [60 * (i + 1) for i in range(MAX_RETRIES)]
    last_exc: Exception | None = None

    for attempt, delay in enumerate(delays):
        if delay:
            print(f"    rate-limit — waiting {delay}s (retry {attempt})…")
            time.sleep(delay)
        try:
            resp = _call_genai(
                model=MODEL,
                prompt=prompt,
                temperature=0.0,
                trace_id=call_id,
                project=PROJECT,
                location=LOCATION,
                system_prompt=SYSTEM_PROMPT,
            )
            return _parse_json_response(resp.raw_text)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                last_exc = exc
                continue
            raise

    raise last_exc  # type: ignore[misc]


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from model response; return fallback dict on failure."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        data = json.loads(cleaned)
        feasibility = str(data.get("feasibility", "")).strip()
        if feasibility not in {"Reliable", "Partially Reliable", "Unreliable"}:
            raise ValueError(f"Unexpected feasibility value: {feasibility!r}")
        return {
            "feasibility": feasibility,
            "reasoning":   str(data.get("reasoning", "")).strip(),
        }
    except (json.JSONDecodeError, ValueError) as exc:
        # Best-effort extraction from free text
        for label in ("Reliable", "Partially Reliable", "Unreliable"):
            if label.lower() in raw.lower():
                return {"feasibility": label, "reasoning": raw.strip()[:500]}
        return {"feasibility": "PARSE_ERROR", "reasoning": f"{exc} | raw={raw[:300]}"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",     action="store_true",
                        help="Execute API calls (default: dry-run only)")
    parser.add_argument("--preview", type=int, default=15,
                        help="Max steps to include in the trace preview (default: 15)")
    parser.add_argument("--out",     default="feasibility_matrix_results.csv",
                        help="Output CSV filename (saved under results/feasibility/)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load shared resources ─────────────────────────────────────────────────
    definitions = DEFS_PATH.read_text() if DEFS_PATH.exists() else ""
    rep_map     = _build_representative_map()          # (fw, fm) -> trace_idx
    frameworks  = list(PARSER_FILES.keys())

    # Pre-load all parser JSONs once
    traces_by_fw: dict[str, list[dict]] = {
        fw: _load_traces(fw) for fw in frameworks
    }

    # ── Plan: enumerate all 14 × 7 cells ─────────────────────────────────────
    total_cells   = len(FAILURE_MODES) * len(frameworks)
    global_skips  = sum(1 for fm in FAILURE_MODES if fm in GLOBAL_FMS) * len(frameworks)
    no_data_cells = sum(
        1 for fw in frameworks for fm in FAILURE_MODES
        if fm not in GLOBAL_FMS and (fw, fm) not in rep_map and not traces_by_fw[fw]
    )
    api_calls = total_cells - global_skips - no_data_cells

    print(f"\n{'='*60}")
    print(f"  FEASIBILITY MATRIX — {len(FAILURE_MODES)} FMs × {len(frameworks)} frameworks")
    print(f"{'='*60}")
    print(f"  Total cells       : {total_cells}")
    print(f"  Auto-skip (global): {global_skips}  {sorted(GLOBAL_FMS)}")
    print(f"  No data           : {no_data_cells}")
    print(f"  API calls planned : {api_calls}")
    print(f"  Model             : {MODEL}  temperature=0")
    print(f"  Trace preview     : {args.preview} steps")
    print(f"  Mode              : {'EXECUTE' if args.run else 'DRY-RUN (pass --run to execute)'}")
    print(f"{'='*60}\n")

    records: list[dict] = []
    call_n = 0

    for fm in FAILURE_MODES:
        fm_name = FM_NAMES[fm]
        fm_def  = _extract_mode_definition(fm, definitions)

        for fw in frameworks:
            traces    = traces_by_fw[fw]
            cell_key  = (fw, fm)

            # ── Case 1: Global FM → auto-skip ─────────────────────────────
            if fm in GLOBAL_FMS:
                records.append({
                    "framework":   fw,
                    "fm":          fm,
                    "fm_name":     fm_name,
                    "feasibility": "Unreliable (Global)",
                    "reasoning":   "Global behavior cannot be localised to a single step.",
                    "trace_idx":   None,
                    "n_steps":     None,
                    "skipped":     True,
                })
                continue

            # ── Case 2: No data for this cell ─────────────────────────────
            trace_idx = rep_map.get(cell_key)
            if trace_idx is None:
                # Fallback: use first available trace (log structure inspection
                # is still valid even without a confirmed FM instance)
                trace_idx = 0 if traces else None

            if trace_idx is None or not traces or trace_idx >= len(traces):
                records.append({
                    "framework":   fw,
                    "fm":          fm,
                    "fm_name":     fm_name,
                    "feasibility": "N/A - No Data",
                    "reasoning":   "No trace data available for this framework.",
                    "trace_idx":   None,
                    "n_steps":     None,
                    "skipped":     True,
                })
                continue

            # ── Case 3: Active evaluation ─────────────────────────────────
            trace         = traces[trace_idx]
            steps         = trace.get("steps", [])
            preview, n_total = _format_steps_preview(steps, args.preview)

            prompt = build_feasibility_prompt(
                framework     = fw,
                fm_code       = fm,
                fm_definition = fm_def,
                trace_preview = preview,
                n_steps_total = n_total,
                n_steps_shown = min(args.preview, n_total),
            )

            call_n += 1
            tag = f"[{call_n}/{api_calls}] {fw} × FM {fm}"

            if not args.run:
                print(f"  DRY-RUN {tag}")
                records.append({
                    "framework":   fw,
                    "fm":          fm,
                    "fm_name":     fm_name,
                    "feasibility": "DRY-RUN",
                    "reasoning":   "(not executed)",
                    "trace_idx":   trace_idx,
                    "n_steps":     n_total,
                    "skipped":     False,
                })
                continue

            print(f"  → {tag}  (trace_idx={trace_idx}, {n_total} steps)…", end=" ", flush=True)
            try:
                result = _call_with_retry(prompt, call_id=f"feasibility_{fw}_{fm}")
                print(result["feasibility"])
                records.append({
                    "framework":   fw,
                    "fm":          fm,
                    "fm_name":     fm_name,
                    "feasibility": result["feasibility"],
                    "reasoning":   result["reasoning"],
                    "trace_idx":   trace_idx,
                    "n_steps":     n_total,
                    "skipped":     False,
                })
            except Exception as exc:
                print(f"ERROR: {exc}")
                records.append({
                    "framework":   fw,
                    "fm":          fm,
                    "fm_name":     fm_name,
                    "feasibility": "ERROR",
                    "reasoning":   str(exc)[:400],
                    "trace_idx":   trace_idx,
                    "n_steps":     n_total,
                    "skipped":     False,
                })

            if call_n < api_calls:
                time.sleep(RATE_SLEEP)

    # ── Export ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    # Long-form CSV (all detail)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_csv = RESULTS_DIR / f"feasibility_long_{stamp}.csv"
    df.to_csv(long_csv, index=False)
    print(f"\nLong-form  → {long_csv}")

    # Pivot: 14 × 7 feasibility matrix (cell = feasibility label)
    matrix = df.pivot(index="fm", columns="framework", values="feasibility")
    matrix = matrix.reindex(index=FAILURE_MODES, columns=frameworks)

    out_csv = RESULTS_DIR / args.out
    matrix.to_csv(out_csv)
    print(f"Matrix CSV → {out_csv}")

    # Pretty-print matrix to stdout
    print(f"\n{'='*60}")
    print("  FEASIBILITY MATRIX (rows=FM, cols=framework)")
    print(f"{'='*60}")
    col_w = 20
    header = f"{'FM':<6}" + "".join(f"{fw:>{col_w}}" for fw in frameworks)
    print(header)
    print("-" * len(header))
    for fm in FAILURE_MODES:
        row = f"{fm:<6}"
        for fw in frameworks:
            val = matrix.at[fm, fw] if fm in matrix.index else "—"
            val = str(val) if val else "—"
            row += f"{val[:col_w - 1]:>{col_w}}"
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    main()
