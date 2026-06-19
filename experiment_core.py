"""
Shared logic used by both judge_localise_app.py and run_experiments.py.
Import from here — do not duplicate in either consumer.
"""

from __future__ import annotations
import re


# ── Human-label taxonomy mapping ─────────────────────────────────────────────

_PHRASES: list[tuple[str, str]] = [
    # Match on the mode TITLE line only (number stripped, lowercased).
    # Body text is excluded: it can reference other modes and cause false matches.
    # Excluded descriptions (no equivalent in our 14):
    #   "undetected conversation ambiguities" (≠ fail to clarify)
    #   "unbatched repetitive execution"      (≠ step repetition)
    #   "disagreement induced inaction"
    #   "waiting for known information"
    ("incorrect verification",           "3.3"),
    ("lack of result verification",      "3.2"),
    ("lack of critical verification",    "3.2"),
    ("no or incomplete verification",    "3.2"),
    ("incomplete verification",          "3.2"),
    ("ill specified termination",        "3.1"),
    ("premature termination",            "3.1"),
    ("ignoring suggestions from",        "2.5"),
    ("ignored other agents",             "2.5"),
    ("withholding relevant information", "2.4"),
    ("information witholding",           "2.4"),   # typo present in generalizability round
    ("information withholding",          "2.4"),
    ("derailment from task",             "2.3"),
    ("task derailment",                  "2.3"),
    ("fail to elicit clarification",     "2.2"),
    ("fail to ask for",                  "2.2"),
    ("conversation reset",               "2.1"),
    ("inconsistency between reasoning",  "2.6"),
    ("reasoning-action mismatch",        "2.6"),
    ("reasoning action mismatch",        "2.6"),
    ("backtracking interruption",        "1.4"),
    ("loss of conversation history",     "1.4"),
    ("unaware of stopping",              "1.5"),
    ("unaware of termination",           "1.5"),
    ("step repetition",                  "1.3"),
    ("disobey role specification",       "1.2"),
    ("disobey task specification",       "1.1"),
    ("poor task constraint",             "1.1"),
]


def match_fm_description(text: str) -> str | None:
    """Map a human annotation entry to our taxonomy code via title-phrase matching.

    The human-labelled dataset uses inconsistent numbering across annotation rounds.
    This function ignores the number entirely and matches on the mode title (first
    line, number stripped).  Returns None for modes that have no equivalent in our
    14-mode taxonomy.
    """
    first_line = text.split('\n')[0]
    title = re.sub(r'^\d+\.\d+\s*', '', first_line).strip().lower()
    for phrase, code in _PHRASES:
        if phrase in title:
            return code
    return None


def majority_vote(anno: dict) -> int:
    """Return 1 if ≥2 of 3 annotators marked this mode present, else 0."""
    votes = sum([
        1 if anno.get('annotator_1') else 0,
        1 if anno.get('annotator_2') else 0,
        1 if anno.get('annotator_3') else 0,
    ])
    return 1 if votes >= 2 else 0


def build_ground_truth(record: dict) -> dict[str, int]:
    """Build {our_mode_code: 0|1} ground truth from a human-labelled record."""
    gt: dict[str, int] = {}
    for anno in record.get('annotations', []):
        code = match_fm_description(anno.get('failure mode', ''))
        if code is None:
            continue
        gt[code] = majority_vote(anno)
    return gt


# ── Partial-trace convergence ─────────────────────────────────────────────────

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
FRAC_LABELS = {0.25: "25%", 0.50: "50%", 0.75: "75%", 1.00: "100%"}


def compute_convergence(mode: str, verdicts_by_frac: dict[float, dict[str, int]]) -> dict:
    """Compute first-detected / stable-from / stable? for one mode.

    verdicts_by_frac: {fraction: {mode: 0|1}} — must include 1.00 as reference.
    Returns a dict with keys: full_verdict, first_detected, stable_from, stable.
    """
    ref = verdicts_by_frac.get(1.00, {})
    full_verdict = ref.get(mode, -1)

    if full_verdict != 1:
        return {
            "full_verdict": full_verdict,
            "first_detected": None,
            "stable_from": None,
            "stable": None,        # "absent"
        }

    prefix_verdicts = {frac: verdicts_by_frac.get(frac, {}).get(mode, -1) for frac in FRACTIONS}

    first_det = next(
        (FRAC_LABELS[f] for f in FRACTIONS if prefix_verdicts[f] == 1),
        "100%",
    )
    stable_from = next(
        (FRAC_LABELS[FRACTIONS[i]] for i in range(len(FRACTIONS))
         if all(prefix_verdicts[f] == 1 for f in FRACTIONS[i:])),
        "100%",
    )

    return {
        "full_verdict": 1,
        "first_detected": first_det,
        "stable_from": stable_from,
        "stable": first_det == stable_from,
    }


# ── Trace-text builder (used by both panels) ──────────────────────────────────

def steps_to_text(steps: list[dict]) -> str:
    """Concatenate parsed steps into the flat text the judge prompt expects."""
    return "\n".join(
        f"[Step {s.get('metadata', {}).get('step_index', i)}] "
        f"{s.get('agent', 'Unknown')}: {s.get('content', '')}"
        for i, s in enumerate(steps)
    )
