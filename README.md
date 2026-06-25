# LLM-as-Judge Evaluation Framework for MAS Failure Modes

**Vladimir Rylskiy — Individual contribution to the UvA × IKEA Master's thesis**

This repository contains the evaluation pipeline I built for assessing whether an LLM judge can reliably detect and localise failure modes in multi-agent system (MAS) execution traces, using the [MAD dataset](https://huggingface.co/datasets/mcemri/MAD) and the [MAST taxonomy](https://arxiv.org/abs/2503.13657) of 14 failure modes.

The broader thesis project (shared repo: [uagarwal123/IKEA_MAST_thesis](https://github.com/uagarwal123/IKEA_MAST_thesis)) covers data understanding, human annotation, and model comparison across the full team. This repo focuses specifically on:

- A three-regime LLM localisation pipeline (Forced / Semi-Forced / Relaxed)
- Four experiments on judge reliability, FM prevalence, step localisation, and partial-trace detection
- A Streamlit interactive app for trace inspection

---

## Repository Structure

```
agent-evaluation-framework/
│
├── data/
│   ├── MAST-Data/              # MAD dataset (not shipped — download below)
│   └── prompts/
│       ├── definitions.txt     # MAST failure-mode definitions (few-shot context)
│       └── examples.txt        # Annotated trace examples for the judge prompt
│
├── parsers/                    # One parser per MAS framework → unified JSON schema
│   ├── ag2_parser/
│   ├── appworld_parser/
│   ├── chatdev_parser/
│   ├── hyperagent_parser/
│   ├── magenticone_parser/
│   ├── metagpt_parser/
│   └── openmanus_parser/
│
├── experiments/
│   └── stage1_llm_judge/
│       └── run_100_baseline/   # Baseline judge run: 100 traces, 4 models × 2 shot configs
│
├── data_understanding/
│   ├── general_eda/            # Trace/FM distributions, co-occurrence, token lengths
│   └── fm_1_3_analysis/        # FM 1.3 (Step Repetition) deep-dive
│
├── results/                    # All experiment outputs (see Artifacts section)
│   ├── human_validation.csv
│   ├── partial_trace.csv
│   ├── localization_map.csv
│   ├── fm_scan/
│   ├── stability/
│   ├── retraction_experiment/
│   └── feasibility/
│
├── run_stability.py            # Core pipeline: run_once() + encode_row()
├── run_fm_scan.py              # Exp 2: FM prevalence scan across 7 frameworks
├── run_experiments.py          # Exp 3: partial-trace detection + human validation
├── run_retraction_experiment.py # Exp 1: forced-commit vs semi-forced retraction
├── run_feasibility_matrix.py   # Localisability feasibility matrix
├── experiment_core.py          # Shared judge/localiser logic
├── judge_localise_app.py       # Streamlit interactive inspector
├── compute_metrics.py          # Precision / recall / Cohen's κ
└── analyze_results.py          # Aggregation helpers
```

---

## Setup

```bash
git clone https://github.com/vladimirrylskiy/agent-evaluation-framework.git
cd agent-evaluation-framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download the MAD dataset

```python
from huggingface_hub import hf_hub_download
import os, shutil

os.makedirs("data/MAST-Data", exist_ok=True)
for fn in ["MAD_full_dataset.json", "MAD_human_labelled_dataset.json"]:
    p = hf_hub_download(repo_id="mcemri/MAD", filename=fn, repo_type="dataset")
    shutil.copy(os.path.realpath(p), os.path.join("data/MAST-Data", fn))
```

### Authenticate for Vertex AI (Gemini models)

```bash
gcloud auth application-default login
```

---

## The Three-Regime Localisation Pipeline

The core of this framework is `run_once()` in `run_stability.py`. For each trace and each of the 14 MAST failure modes it runs:

| Regime | Behaviour | Purpose |
|---|---|---|
| **Judge** | Binary: is this FM present in the trace? | Gate — skips localisation if absent |
| **Forced** | Must commit to a step number (no abstain allowed) | Measures what the model does under pressure |
| **Semi-Forced** | Can return `NO_STEP_FOUND` or `RETRACT` | Measures whether the forced answer was genuine |
| **Relaxed** | Open scan: return all relevant steps or none | Upper-bound localisation quality |

A **retraction event** occurs when Forced commits to a step but Semi-Forced returns `RETRACTED`. The retraction rate measures forced hallucination.

---

## Experiments

### Experiment 1 — Stochastic Stability & Forced-Commit Retraction Rate

**Script:** `run_retraction_experiment.py`  
**Model:** gemini-2.5-flash, T=1.0 (stochastic)  
**Design:** 15 traces × 3 frameworks (MetaGPT, HyperAgent, ChatDev) × 15 repeats × 14 FMs = 3,150 pipeline runs  
**Checkpoint:** per-trace CSV in `results/retraction_experiment/` — resumable after crash

The experiment measures how often the Forced regime commits to a step that the Semi-Forced regime later retracts, across repeated runs on the same traces.

**Key result:** 17.9% overall retraction rate (143/801 forced-commits, 95% Wilson CI [15.4, 20.7]). The rate is consistent across frameworks (ChatDev 17.5%, HyperAgent 15.3%, MetaGPT 21.3%), showing forced hallucination is not architecture-specific. Highest per-FM retraction rates: FM 2.3 Task Derailment (46.8%), FM 2.4 Information Withholding (45.5%), FM 1.2 Disobey Role Specification (38.2%).

**Run:**
```bash
python run_retraction_experiment.py
python run_retraction_experiment.py --analyze results/retraction_experiment/retraction_raw_<ts>.csv
```

---

### Experiment 2 — FM Prevalence Scan

**Script:** `run_fm_scan.py`  
**Model:** gemini-2.5-flash, T=0.0  
**Design:** 7 frameworks × up to 30 traces each × 14 FMs — judge-only pass (no localisation)

Maps which failure modes appear in which frameworks and at what rate, providing the prior for all downstream experiments.

**Key result:** FM 2.6 (Action-Reasoning Mismatch) and FM 1.1 (Disobey Task Specification) are the most prevalent across frameworks. FM 2.5 (Ignored Agent Input) and FM 2.1 (Conversation Reset) are rare and framework-specific.

**Run:**
```bash
python run_fm_scan.py
```

---

### Experiment 3 — Partial-Trace Detection

**Script:** `run_experiments.py`  
**Model:** gemini-2.5-flash, T=0.0  
**Design:** 19 human-labelled traces × 14 FMs, judge applied to word-level prefixes at 25%, 50%, 75%, and 100% of each trace

Tests whether failure modes can be detected before the trace is complete, and whether the verdict stabilises early or requires the full trace.

**Key result:** FM 2.6 (Action-Reasoning Mismatch) stabilises earliest — 8 of 11 present cases already detected at 25%. FM 3.3 (Incorrect Verification) and FM 1.3 (Step Repetition) require the full trace, with near-zero detection at 25–50%. FM 3.2 shows excess detections at 25% (false positives from partial context).

| FM | n present | stable@25% | stable@50% | stable@75% |
|---|---|---|---|---|
| 2.6 Action-Reasoning Mismatch | 11 | 8 | 9 | 10 |
| 2.1 Conversation Reset | 2 | 2 | 2 | 2 |
| 1.3 Step Repetition | 9 | 2 | 4 | 8 |
| 3.3 Incorrect Verification | 7 | 0 | 0 | 2 |

---

### Pilot — Single-Trace Stability Study

**Script:** `run_stability.py`  
**Model:** gemini-2.5-flash, T=1.0  
**Design:** MetaGPT trace_idx=6, N=30 repeats across all three regimes

A pre-experiment pilot used to validate the pipeline and estimate variance before scaling to the full retraction experiment.

**Run:**
```bash
python run_stability.py --trace-idx 6 --framework metagpt --n-repeats 30
```

---

## Artifacts

All outputs are in `results/`. The files below are the definitive versions.

### Human validation

| File | Description |
|---|---|
| `results/human_validation.csv` | Per (trace, FM): human label, judge label, match — 532 rows across 19 traces |
| `results/llm_vs_human.png` | Agreement chart: judge vs human per FM |
| `results/validation_table.png` | Precision / recall / Cohen's κ table |

### FM prevalence (Experiment 2)

| File | Description |
|---|---|
| `results/fm_scan/fm_scan_dist_20260618_233820.csv` | Per (framework, FM): prevalence %, mean/std step position |
| `results/fm_scan/fm_scan_raw_20260618_233820.csv` | Raw per-trace scan results |
| `results/fm_scan/fm_trace_distribution.png` | FM distribution chart across frameworks |

### Partial-trace detection (Experiment 3)

| File | Description |
|---|---|
| `results/partial_trace.csv` | Per (trace, FM): verdict at 25/50/75/100%, first_detected_pct, stable_from_pct, stable |

### Stochastic stability & retraction (Experiment 1)

| File | Description |
|---|---|
| `results/retraction_experiment/retraction_raw_20260624_002251.csv` | Raw: 3,150 rows (15 traces × 15 repeats × 14 FMs) |
| `results/retraction_experiment/retraction_summary_20260624_002251.csv` | Per (framework, FM): forced_commits, retractions, retraction_rate |
| `results/retraction_experiment/wilson_ci_table.csv` | Wilson 95% CIs on retraction rate per group |
| `results/retraction_experiment/ckpt_*.csv` | Per-trace checkpoints (15 repeats × 14 FMs each) |

### Localisation map

| File | Description |
|---|---|
| `results/localization_map.csv` | Per (trace, FM): result_type — steps / global / none / NO_DECOMPOSITION |
| `results/feasibility/feasibility_matrix_results.csv` | Which FMs are localisable to a step vs. global-only |

---

## Interactive Inspector

A Streamlit app for exploring any trace with the full judge + localise pipeline:

```bash
streamlit run judge_localise_app.py
```

Select a framework and trace index, choose a model and regime, and inspect per-FM verdicts and step attributions interactively.

---

## Dependencies

| Package | Purpose |
|---|---|
| `google-cloud-aiplatform` | Vertex AI / Gemini via `genai` backend |
| `anthropic` | Claude via Vertex AI (`anthropic` backend) |
| `statsmodels` | Wilson score confidence intervals |
| `pandas`, `numpy` | Data manipulation |
| `streamlit` | Interactive trace inspector |
| `huggingface_hub` | MAD dataset download |

See `requirements.txt` for pinned versions.
