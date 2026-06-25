"""
Streamlit app for step-level failure mode localization in multi-agent traces.

Stage 1: Load parsed traces and visualize step-level localization.
Uses parsed JSON from parsers (all 7 MAS frameworks).

⚠️ IMPORTANT: This is a qualitative proof-of-concept. There is NO step-level ground truth in MAD.
All step-level predictions are compared only against the full-trace verdict as a baseline.
"""

import streamlit as st
import json
from pathlib import Path
import sys
import re
from datetime import datetime

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from LLM_models_interface.llm_interface import (
    build_localise_prompt,
    parse_14_modes_with_steps,
    parse_14_modes,
    build_judge_prompt,
    build_subordinate_localise_prompt,
    parse_localized_steps,
    build_forced_batch_prompt, build_semi_forced_batch_prompt,
    parse_batch_localise_response,
    build_relaxed_localise_prompt, parse_relaxed_steps,
    LLMJudge,
    JudgeConfig,
    FAILURE_MODES,
)
from experiment_core import (
    match_fm_description,
    majority_vote,
    build_ground_truth,
    compute_convergence,
    steps_to_text,
    FRACTIONS,
    FRAC_LABELS,
)

# ============================================================================
# UI CONFIG
# ============================================================================

st.set_page_config(
    page_title="Judge-Localise: Step-Level FM Detection",
    layout="wide",
)

st.markdown("""
# Judge-Localise: Step-Level Failure Mode Localization

⚠️ **Important Notice:** This is a **qualitative proof-of-concept**. 
- The MAD dataset contains **trace-level annotations only** — no step-level ground truth exists.
- Step-level predictions are validated qualitatively against full-trace verdicts.
- This is **NOT** a scored evaluation against a gold standard.
""")

# ============================================================================
# SIDEBAR: Load & Configure
# ============================================================================

if "active_panel" not in st.session_state:
    st.session_state.active_panel = None

with st.sidebar:
    st.header("Configuration")
    
    # Framework selection
    PARSER_PATHS = {
        "ChatDev":     Path("parsers/chatdev_parser/chatdev_output_mad.json"),
        "AG2":         Path("parsers/ag2_parser/ag2_output_mad.json"),
        "AppWorld":    Path("parsers/appworld_parser/appworld_output_mad.json"),
        "HyperAgent":  Path("parsers/hyperagent_parser/hyperagent_output_mad.json"),
        "MetaGPT":     Path("parsers/metagpt_parser/metagpt_output_mad.json"),
        "MagenticOne": Path("parsers/magenticone_parser/magenticone_output_mad.json"),
        "OpenManus":   Path("parsers/openmanus_parser/openmanus_output_mad.json"),
    }

    framework = st.selectbox("Select framework", list(PARSER_PATHS.keys()))
    parser_path = PARSER_PATHS[framework]

    if not parser_path.exists():
        st.error(f"Parser output not found: {parser_path}")
        st.stop()

    with open(parser_path, "r", encoding="utf-8") as f:
        all_traces = json.load(f)
    
    st.info(f"Loaded {len(all_traces)} {framework} traces from parser output.")
    
    # Trace selector
    trace_options = []
    for i, t in enumerate(all_traces):
        trace_id = str(t['metadata'].get('trace_id', f'trace_{i}'))
        trace_options.append(f"{i}: {trace_id} ({len(t['steps'])} steps)")

    selected_idx = st.selectbox("Select a trace", range(len(all_traces)),
                                 format_func=lambda i: trace_options[i])

    selected_trace = all_traces[selected_idx]
    
    # Model config for judge
    st.subheader("Judge Model Configuration")
    model = st.selectbox(
        "LLM Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "claude-sonnet-4-6", "claude-haiku-4-5"],
        help="Cheap models preferred for Stage 1 MVP."
    )
    backend = st.selectbox("Backend", ["genai", "anthropic", "ollama"])
    
    # Localization mode
    localization_mode = st.selectbox(
        "Localization Strategy",
        ["Subordinate (Two-Stage, Recommended)", "Naive (Single-Prompt, For Thesis Reproduction)"],
        help="Subordinate: Baseline detects presence, localizer only finds steps for present modes. "
             "Naive: Single prompt tries to both detect and localize (over-detects)."
    )
    
    if st.button("🔍 Run Judge (Full + Localized)", use_container_width=True):
        st.session_state.active_panel = "judge"
    st.divider()
    if st.button("📊 Run Partial-Trace Detection", use_container_width=True):
        st.session_state.active_panel = "partial"
    st.caption("Runs judge at 25 %, 50 %, 75 %, and 100 % prefixes. Shows when each FM first appears and whether the verdict is stable.")
    st.divider()
    if st.button("🔬 Run Slice Detection", use_container_width=True):
        st.session_state.active_panel = "slice"
    st.caption("Run judge on any contiguous window of the trace (e.g. last 50 %, or steps 25 %–75 %).")
    _n_steps_sidebar = len(all_traces[selected_idx]["steps"])
    slice_range = st.slider(
        "Trace window (%)",
        min_value=0, max_value=100,
        value=(50, 100), step=5,
        help=f"Trace has {_n_steps_sidebar} steps. Drag to select start and end %.",
        key="slice_range",
    )
    _s = round(_n_steps_sidebar * slice_range[0] / 100)
    _e = round(_n_steps_sidebar * slice_range[1] / 100)
    st.caption(f"Steps {_s}–{_e} of {_n_steps_sidebar} ({_e - _s} steps)")
    st.divider()
    if st.button("📈 Batch Slice Analysis", use_container_width=True):
        st.session_state.active_panel = "batch_slice"
    st.caption("Run slice detection across N traces and get an FM prevalence table.")
    st.divider()
    if st.button("🧑 Human-Label Validation", use_container_width=True):
        st.session_state.active_panel = "human_val"
    st.caption("Runs the judge on a trace from MAD_human_labelled_dataset.json and compares to its own annotations.")
    st.divider()
    if st.button("🧪 Framework Comparison", use_container_width=True):
        st.session_state.active_panel = "framework_compare"
    st.caption("Compare FORCED / SEMI-FORCED / RELAXED subordinate localizers on the selected trace.")

run_judge = st.session_state.active_panel == "judge"
run_partial = st.session_state.active_panel == "partial"
run_slice = st.session_state.active_panel == "slice"
run_batch_slice = st.session_state.active_panel == "batch_slice"
run_human_val = st.session_state.active_panel == "human_val"
run_framework_compare = st.session_state.active_panel == "framework_compare"

# ============================================================================
# MAIN: Display Trace & Results
# ============================================================================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Trace Steps")
    st.write(f"**Trace ID:** {selected_trace['metadata'].get('trace_id', 'N/A')}")
    st.write(f"**Total steps:** {len(selected_trace['steps'])}")
    st.write(f"**MAS:** {selected_trace['metadata'].get('mas_name', 'N/A')}")
    
    # List steps
    st.write("---")
    for i, step in enumerate(selected_trace['steps']):
        agent = step.get('agent', 'Unknown')
        kind = step.get('kind', 'message')
        content_preview = step.get('content', '')[:80].replace('\n', ' ') + "..."
        
        with st.expander(f"Step {i} | {agent} | {kind}"):
            st.write(step.get('content', ''))

with col2:
    st.subheader("🎯 Judge Results")
    
    if not run_judge and not run_partial and not run_slice and not run_batch_slice and not run_human_val and not run_framework_compare:
        st.info("Configure and click a button in the sidebar to see results.")
    elif run_judge:
        with st.spinner("Running judge (this may take a moment)..."):
            try:
                # Load definitions and examples
                defs_path = Path("data/prompts/definitions.txt")
                examples_path = Path("data/prompts/examples.txt")
                
                definitions = defs_path.read_text() if defs_path.exists() else ""
                examples = examples_path.read_text() if examples_path.exists() else ""
                
                # ====== FULL-TRACE JUDGE ======
                st.write("#### 1️⃣ Full-Trace Verdict (Baseline)")
                
                trace_text = steps_to_text(selected_trace['steps'])
                
                prompt_full = build_judge_prompt(trace_text, definitions, examples)
                
                # Create config and judge
                config = JudgeConfig(
                    name=f"localise_demo_{datetime.now().isoformat()}",
                    model=model,
                    backend=backend,
                    temperature=0.0,
                    definitions_path=str(defs_path),
                    examples_path=str(examples_path),
                )
                judge = LLMJudge(config)
                
                # Call judge (full-trace)
                response_full = judge._dispatch(prompt_full, f"trace_{selected_idx}")
                annotations_full = parse_14_modes(response_full.raw_text)
                
                # Display full-trace results
                for mode in FAILURE_MODES:
                    verdict = "✅ YES" if annotations_full[mode] == 1 else "❌ no"
                    st.write(f"**{mode}**: {verdict}")
                
                st.write("---")
                
                # ====== STEP-LOCALISED JUDGE ======
                st.write("#### 2️⃣ Step-Level Localization")
                st.caption("⚠️ Qualitative only — no gold standard for validation.")
                
                # Determine which mode to use
                use_subordinate = "Subordinate" in localization_mode
                
                if use_subordinate:
                    st.info("📋 **Mode: Two-Stage (Baseline → Subordinate Localizer)**\n"
                            "Baseline decides presence. Localizer finds steps ONLY for modes baseline marked present.")
                    
                    annotations_localise = {}
                    responses_localise = {}
                    for mode in FAILURE_MODES:
                        baseline_verdict = annotations_full[mode]
                        
                        if baseline_verdict == 0:
                            # Baseline said NO → skip localization
                            annotations_localise[mode] = {'present': 0, 'steps': []}
                            responses_localise[mode] = None
                        else:
                            # Baseline said YES → run subordinate localizer
                            # Get mode name from definitions
                            mode_name_match = re.search(rf"{re.escape(mode)}\s+([^\n]+)", definitions)
                            mode_name = mode_name_match.group(1).strip() if mode_name_match else f"Mode {mode}"
                            
                            prompt_sub = build_subordinate_localise_prompt(
                                mode, mode_name, selected_trace['steps'], definitions, examples
                            )
                            response_sub = judge._dispatch(prompt_sub, f"trace_{selected_idx}_{mode}_sub")
                            responses_localise[mode] = response_sub
                            steps_result = parse_localized_steps(response_sub.raw_text)

                            annotations_localise[mode] = {
                                'present': 1,
                                'steps': steps_result if isinstance(steps_result, list) else [steps_result]
                            }
                else:
                    st.info("📋 **Mode: Naive (Single-Prompt)**\n"
                            "Single prompt tries to both detect AND localize (known to over-detect).")
                    
                    # Old naive approach: single prompt for all modes
                    prompt_localise = build_localise_prompt(
                        selected_trace['steps'], definitions, examples
                    )
                    response_localise = judge._dispatch(
                        prompt_localise, f"trace_{selected_idx}_localise"
                    )
                    annotations_localise = parse_14_modes_with_steps(response_localise.raw_text)
                
                # Display step-localized results (works for both modes)
                for mode in FAILURE_MODES:
                    full_verdict = annotations_full[mode]
                    local = annotations_localise.get(mode, {'present': 0, 'steps': []})
                    local_verdict = local['present']
                    steps = local['steps']

                    agreement = "✓" if full_verdict == local_verdict else "✗"

                    baseline_str = "✅ YES" if full_verdict == 1 else "❌ no"

                    if local_verdict == 1:
                        if steps == ['global'] or (isinstance(steps, list) and 'global' in steps):
                            steps_str = "🌍 GLOBAL"
                        elif steps:
                            steps_str = f"Steps {steps}"
                        else:
                            steps_str = "(no steps specified)"
                        st.write(f"**{mode}** {agreement}: Baseline {baseline_str} → Localizer: {steps_str}")
                        # Show failure-mode description in an expander (arrow dropdown)
                        with st.expander(f"{mode} description"):
                            # Try to extract the descriptive paragraph for this mode from `definitions`
                            desc_match = re.search(rf"{re.escape(mode)}\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\d|$)", definitions, re.S)
                            desc = desc_match.group(1).strip() if desc_match else "(no description found)"
                            st.markdown(desc)
                    else:
                        st.write(f"**{mode}** {agreement}: Baseline {baseline_str} → Localizer: (skipped)")
                        with st.expander(f"{mode} description"):
                            desc_match = re.search(rf"{re.escape(mode)}\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\d|$)", definitions, re.S)
                            desc = desc_match.group(1).strip() if desc_match else "(no description found)"
                            st.markdown(desc)
                
                # ====== COMPARISON ======
                st.write("---")
                st.write("#### 📊 Baseline Comparison")
                
                matches = sum(
                    1 for mode in FAILURE_MODES
                    if annotations_full[mode] == annotations_localise[mode]['present']
                )
                total = len(FAILURE_MODES)
                
                st.metric(
                    "Localization Agreement with Baseline",
                    f"{matches}/{total} modes ({100*matches//total}%)"
                )

                # ====== RAW RESPONSES (Debug) ======
                with st.expander("🔧 Debug: Raw LLM Responses"):
                    st.subheader("Full-Trace Response")
                    st.code(response_full.raw_text[:500] + "..." if len(response_full.raw_text) > 500 else response_full.raw_text)
                    
                    st.subheader("Localization Response")
                    # If naive single-response exists, show it. Otherwise show per-mode subordinate responses.
                    if 'response_localise' in locals() and response_localise is not None:
                        txt = response_localise.raw_text
                        st.code(txt[:500] + "..." if len(txt) > 500 else txt)
                    elif 'responses_localise' in locals():
                        # concatenate per-mode responses for inspection
                        parts = []
                        for m, resp in responses_localise.items():
                            if resp is None:
                                continue
                            header = f"--- {m} ---\n"
                            body = resp.raw_text if hasattr(resp, 'raw_text') else str(resp)
                            parts.append(header + body)
                        combined = "\n\n".join(parts) if parts else "(no localization responses captured)"
                        st.code(combined[:2000] + "..." if len(combined) > 2000 else combined)
                    else:
                        st.write("(no localization response available)")
                
                st.success("✅ Judge run complete!")
                
            except Exception as e:
                st.error(f"Error running judge: {e}")
                import traceback
                st.write(traceback.format_exc())

    if run_partial:
        st.markdown("---")
        st.subheader("📊 Partial-Trace Detection")
        st.caption(
            "The judge runs on growing prefixes of the trace (25 %, 50 %, 75 %, 100 %). "
            "The 100 % run is the reference verdict. For each failure mode, the table shows "
            "the smallest prefix at which the verdict converges to — and stays at — the full-trace verdict."
        )
        try:
            LABELS = FRAC_LABELS
            n_steps = len(selected_trace["steps"])
            prefix_sizes = {frac: max(1, round(n_steps * frac)) for frac in FRACTIONS}
            st.caption(
                f"Trace has **{n_steps}** steps → prefix sizes: "
                + ", ".join(f"{LABELS[f]} = {prefix_sizes[f]} steps" for f in FRACTIONS)
            )

            defs_path_p = Path("data/prompts/definitions.txt")
            examples_path_p = Path("data/prompts/examples.txt")
            definitions_p = defs_path_p.read_text() if defs_path_p.exists() else ""
            examples_p = examples_path_p.read_text() if examples_path_p.exists() else ""

            verdicts_by_frac: dict[float, dict[str, int]] = {}
            progress_bar = st.progress(0, text="Starting…")

            for step_i, frac in enumerate(FRACTIONS):
                n = prefix_sizes[frac]
                steps_subset = selected_trace["steps"][:n]
                trace_text_p = steps_to_text(steps_subset)
                progress_bar.progress(
                    step_i / len(FRACTIONS),
                    text=f"Running {LABELS[frac]} prefix ({n}/{n_steps} steps)…",
                )
                config_p = JudgeConfig(
                    name=f"partial_{frac}_{datetime.now().isoformat()}",
                    model=model,
                    backend=backend,
                    temperature=0.0,
                    definitions_path=str(defs_path_p),
                    examples_path=str(examples_path_p),
                )
                judge_p = LLMJudge(config_p)
                prompt_p = build_judge_prompt(trace_text_p, definitions_p, examples_p)
                try:
                    resp_p = judge_p._dispatch(prompt_p, f"trace_{selected_idx}_frac{frac}")
                    verdicts_by_frac[frac] = parse_14_modes(resp_p.raw_text)
                except Exception as e:
                    st.error(f"Error at {LABELS[frac]}: {e}")
                    verdicts_by_frac[frac] = {m: -1 for m in FAILURE_MODES}

            progress_bar.progress(1.0, text="Done.")
            progress_bar.empty()

            rows_partial = []
            present_modes = []

            for mode in FAILURE_MODES:
                conv = compute_convergence(mode, verdicts_by_frac)
                row = {"FM": mode}
                for frac in FRACTIONS:
                    v = verdicts_by_frac.get(frac, {}).get(mode, -1)
                    row[LABELS[frac]] = "✅" if v == 1 else ("❌" if v == 0 else "?")

                if conv["full_verdict"] != 1:
                    row["First detected"] = "n/a"
                    row["Stable from"] = "n/a"
                    row["Stable?"] = "absent"
                else:
                    present_modes.append(mode)
                    row["First detected"] = conv["first_detected"]
                    row["Stable from"] = conv["stable_from"]
                    row["Stable?"] = "stable" if conv["stable"] else "unstable"

                rows_partial.append(row)

            header = "| FM | 25% | 50% | 75% | 100% | First detected | Stable from | Stable? |"
            sep    = "|---|---|---|---|---|---|---|---|"
            lines  = [header, sep]
            for row in rows_partial:
                stability = row["Stable?"]
                stable_flag = "✓" if stability == "stable" else ("✗ unstable" if stability == "unstable" else "—")
                lines.append(
                    f"| {row['FM']} | {row['25%']} | {row['50%']} | {row['75%']} | {row['100%']}"
                    f" | {row['First detected']} | {row['Stable from']} | {stable_flag} |"
                )
            st.markdown("\n".join(lines))

            present_rows = [r for r in rows_partial if r["Stable?"] != "absent"]
            n_early = sum(1 for r in present_rows if r["First detected"] == "25%")
            n_unstable = sum(1 for r in present_rows if r["Stable?"] == "unstable")
            st.write(
                f"**Present modes at 100 %:** {', '.join(present_modes) if present_modes else 'none'}. "
                f"Of these, **{n_early}** first detected at 25 %, **{n_unstable}** unstable across prefixes."
            )

        except Exception as e:
            st.error(f"Partial-trace error: {e}")
            import traceback
            st.code(traceback.format_exc())

    if run_slice:
        st.markdown("---")
        st.subheader("🔬 Slice Detection")
        pct_start, pct_end = st.session_state.get("slice_range", (50, 100))
        n_steps = len(selected_trace["steps"])
        idx_start = round(n_steps * pct_start / 100)
        idx_end   = round(n_steps * pct_end   / 100)
        idx_end   = max(idx_end, idx_start + 1)  # at least 1 step

        st.caption(
            f"Running judge on steps **{idx_start}–{idx_end - 1}** "
            f"({idx_end - idx_start} of {n_steps} steps, "
            f"{pct_start} %–{pct_end} % of trace)."
        )

        try:
            steps_slice = selected_trace["steps"][idx_start:idx_end]
            trace_text_slice = steps_to_text(steps_slice)

            # Tell the model it is seeing a deliberate slice so it can correctly
            # evaluate 3.1 (Premature Termination) — without this note the model
            # has no way to know the trace ends abruptly on purpose vs. naturally.
            slice_note = (
                f"[ANALYSIS CONTEXT: You are evaluating steps {idx_start}–{idx_end - 1} "
                f"of a {n_steps}-step trace ({pct_start}%–{pct_end}% of the full conversation). "
                f"The trace ends at step {idx_end - 1}. "
                f"Treat this endpoint as the actual end of the conversation for your evaluation — "
                f"if the task appears unfinished or incomplete at this point, "
                f"flag Premature Termination (3.1) as present.]\n\n"
            )
            trace_text_slice = slice_note + trace_text_slice

            defs_path_s  = Path("data/prompts/definitions.txt")
            examples_path_s = Path("data/prompts/examples.txt")
            definitions_s = defs_path_s.read_text() if defs_path_s.exists() else ""
            examples_s    = examples_path_s.read_text() if examples_path_s.exists() else ""

            config_s = JudgeConfig(
                name=f"slice_{pct_start}_{pct_end}_{datetime.now().isoformat()}",
                model=model,
                backend=backend,
                temperature=0.0,
                definitions_path=str(defs_path_s),
                examples_path=str(examples_path_s),
            )
            judge_s = LLMJudge(config_s)

            with st.spinner(f"Running judge on {pct_start}%–{pct_end}% slice…"):
                prompt_s = build_judge_prompt(trace_text_slice, definitions_s, examples_s)
                resp_s   = judge_s._dispatch(prompt_s, f"slice_{idx_start}_{idx_end}")
                verdicts_s = parse_14_modes(resp_s.raw_text)

            # FM name lookup
            _fm_names = {
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
                "3.2": "No or Incomplete Verification",
                "3.3": "Incorrect Verification",
            }

            present = [m for m in FAILURE_MODES if verdicts_s.get(m) == 1]

            # Render markdown table
            header = "| FM | Failure Mode | Verdict |"
            sep    = "|:---:|:---|:---:|"
            lines  = [header, sep]
            for m in FAILURE_MODES:
                v = verdicts_s.get(m, -1)
                verdict = "✅ present" if v == 1 else ("❌ absent" if v == 0 else "?")
                lines.append(f"| {m} | {_fm_names.get(m, m)} | {verdict} |")
            st.markdown("\n".join(lines))

            n_present = len(present)
            st.write(
                f"**{n_present} mode(s) detected** in steps {idx_start}–{idx_end - 1} "
                f"({pct_start}%–{pct_end}% of trace): "
                + (", ".join(present) if present else "none")
            )

            with st.expander("🔧 Raw LLM response"):
                st.code(resp_s.raw_text)

            st.success("✅ Slice detection complete!")

        except Exception as e:
            st.error(f"Slice detection error: {e}")
            import traceback
            st.code(traceback.format_exc())

    if run_batch_slice:
        st.markdown("---")
        st.subheader("📈 Batch Slice Analysis")

        _fm_names_batch = {
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
            "3.2": "No or Incomplete Verification",
            "3.3": "Incorrect Verification",
        }

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            n_traces_batch = st.number_input(
                "Number of traces", min_value=1,
                max_value=len(all_traces), value=min(10, len(all_traces)),
                step=1, key="batch_n_traces",
            )
        with col_ctrl2:
            batch_slice_range = st.slider(
                "Trace window (%)", 0, 100, (0, 50), step=5,
                key="batch_slice_range",
            )

        pct_b_start, pct_b_end = batch_slice_range
        st.caption(
            f"Will run judge on the **{pct_b_start}%–{pct_b_end}%** window of "
            f"**{n_traces_batch}** {framework} traces ({n_traces_batch} LLM calls)."
        )

        if st.button("▶ Run Batch", key="run_batch_btn"):
            try:
                defs_path_b   = Path("data/prompts/definitions.txt")
                examples_path_b = Path("data/prompts/examples.txt")
                definitions_b = defs_path_b.read_text() if defs_path_b.exists() else ""
                examples_b    = examples_path_b.read_text() if examples_path_b.exists() else ""

                counts = {m: 0 for m in FAILURE_MODES}
                errors = 0
                progress = st.progress(0, text="Starting…")

                for trace_i, trace in enumerate(all_traces[:n_traces_batch]):
                    n_steps_b = len(trace["steps"])
                    idx_s = round(n_steps_b * pct_b_start / 100)
                    idx_e = max(round(n_steps_b * pct_b_end   / 100), idx_s + 1)

                    steps_b     = trace["steps"][idx_s:idx_e]
                    trace_text_b = steps_to_text(steps_b)

                    slice_note_b = (
                        f"[ANALYSIS CONTEXT: You are evaluating steps {idx_s}–{idx_e - 1} "
                        f"of a {n_steps_b}-step trace ({pct_b_start}%–{pct_b_end}% of the full conversation). "
                        f"The trace ends at step {idx_e - 1}. "
                        f"Treat this endpoint as the actual end of the conversation — "
                        f"if the task appears unfinished at this point, flag Premature Termination (3.1) as present.]\n\n"
                    )
                    trace_text_b = slice_note_b + trace_text_b

                    progress.progress(
                        trace_i / n_traces_batch,
                        text=f"Trace {trace_i + 1}/{n_traces_batch} "
                             f"(steps {idx_s}–{idx_e - 1} of {n_steps_b})…",
                    )

                    try:
                        config_b = JudgeConfig(
                            name=f"batch_{trace_i}_{datetime.now().isoformat()}",
                            model=model, backend=backend, temperature=0.0,
                            definitions_path=str(defs_path_b),
                            examples_path=str(examples_path_b),
                        )
                        judge_b  = LLMJudge(config_b)
                        prompt_b = build_judge_prompt(trace_text_b, definitions_b, examples_b)
                        resp_b   = judge_b._dispatch(prompt_b, f"batch_{trace_i}")
                        verdicts_b = parse_14_modes(resp_b.raw_text)
                        for m in FAILURE_MODES:
                            if verdicts_b.get(m) == 1:
                                counts[m] += 1
                    except Exception as e:
                        st.warning(f"Trace {trace_i} error: {e}")
                        errors += 1

                progress.progress(1.0, text="Done.")
                progress.empty()

                # Results table
                st.write(
                    f"**Results:** {framework} · {n_traces_batch} traces · "
                    f"window {pct_b_start}%–{pct_b_end}%"
                    + (f" · {errors} error(s)" if errors else "")
                )
                header = "| FM | Failure Mode | Count | % of traces |"
                sep    = "|:---:|:---|:---:|:---:|"
                lines  = [header, sep]
                for m in FAILURE_MODES:
                    pct_val = round(100 * counts[m] / n_traces_batch)
                    bar = "█" * (pct_val // 10) + "░" * (10 - pct_val // 10)
                    lines.append(
                        f"| {m} | {_fm_names_batch[m]} | {counts[m]} | {bar} {pct_val}% |"
                    )
                st.markdown("\n".join(lines))
                st.success("✅ Batch complete!")

            except Exception as e:
                st.error(f"Batch error: {e}")
                import traceback
                st.code(traceback.format_exc())

    if run_human_val:
        st.markdown("---")
        st.subheader("🧑 Human-Label Validation")
        st.caption(
            "Loads traces directly from MAD_human_labelled_dataset.json. "
            "Runs the judge on each trace's own text and compares to that record's human annotations "
            "(majority vote of 3 annotators). Same record — valid comparison."
        )
        try:
            human_path_v = Path("data/MAST-Data/MAD_human_labelled_dataset.json")
            if not human_path_v.exists():
                st.error("MAD_human_labelled_dataset.json not found.")
                st.stop()

            with open(human_path_v) as f:
                human_records = json.load(f)

            human_options = [
                f"{i}: {r['mas_name']} trace_id={r['trace_id']} ({r.get('benchmark_name','')})"
                for i, r in enumerate(human_records)
            ]
            selected_human_idx = st.selectbox(
                "Select human-labelled trace", range(len(human_records)),
                format_func=lambda i: human_options[i],
                key="human_val_selector"
            )
            record = human_records[selected_human_idx]

            ground_truth = build_ground_truth(record)

            st.write(f"**MAS:** {record['mas_name']} | **Benchmark:** {record.get('benchmark_name','')} | **trace_id:** {record['trace_id']}")
            st.write(f"**Human ground truth (majority vote):** {sum(ground_truth.values())} modes present out of {len(ground_truth)}")

            if st.button("▶ Run judge on this trace", key="run_human_judge"):
                defs_path_h = Path("data/prompts/definitions.txt")
                examples_path_h = Path("data/prompts/examples.txt")
                definitions_h = defs_path_h.read_text() if defs_path_h.exists() else ""
                examples_h = examples_path_h.read_text() if examples_path_h.exists() else ""

                trace_text_h = record.get('trace', '')
                prompt_h = build_judge_prompt(trace_text_h, definitions_h, examples_h)

                config_h = JudgeConfig(
                    name=f"human_val_{record['mas_name']}_{record['trace_id']}",
                    model=model,
                    backend=backend,
                    temperature=0.0,
                    definitions_path=str(defs_path_h),
                    examples_path=str(examples_path_h),
                )
                judge_h = LLMJudge(config_h)

                with st.spinner("Running judge on human-labelled trace…"):
                    try:
                        resp_h = judge_h._dispatch(prompt_h, f"human_{record['trace_id']}")
                        pred_h = parse_14_modes(resp_h.raw_text)
                    except Exception as e:
                        st.error(f"Judge error: {e}")
                        import traceback; st.code(traceback.format_exc())
                        st.stop()

                agree, total = 0, 0
                lines_h = ["| FM | Human (majority) | Judge | Match |", "|---|---|---|---|"]
                for mode in FAILURE_MODES:
                    human_v = ground_truth.get(mode, -1)
                    judge_v = pred_h.get(mode, -1)
                    if human_v == -1:
                        h_str, j_str, match = "?", "?", "—"
                    else:
                        h_str = "✅" if human_v == 1 else "❌"
                        j_str = "✅" if judge_v == 1 else "❌"
                        match = "✓" if human_v == judge_v else "✗"
                        agree += 1 if human_v == judge_v else 0
                        total += 1
                    lines_h.append(f"| {mode} | {h_str} | {j_str} | {match} |")
                st.markdown("\n".join(lines_h))
                pct = 100 * agree // total if total else 0
                st.metric("Agreement (judge vs human majority)", f"{agree}/{total} modes ({pct}%)")

        except Exception as e:
            st.error(f"Human validation error: {e}")
            import traceback
            st.code(traceback.format_exc())

    if run_framework_compare:
        st.markdown("---")
        st.subheader("🧪 Framework Comparison: Subordinate Localizers")
        st.caption(
            "**4 LLM calls total per trace.** "
            "Shared judge call detects present modes → "
            "**Forced** batch (1 call, all detected modes, must commit to steps) → "
            "**Semi-Forced** batch (1 call, all detected modes, may output `NO_STEP_FOUND`) → "
            "**Relaxed** (1 call, full FM taxonomy, no judge signal, detect + locate simultaneously)."
        )

        steps = selected_trace["steps"]
        n_steps = len(steps)
        defs_path_fw = Path("data/prompts/definitions.txt")
        definitions_fw = defs_path_fw.read_text() if defs_path_fw.exists() else ""

        def _resolve_mode_name(code):
            m = re.search(rf"{re.escape(code)}\s+([^\n:]+)", definitions_fw)
            return m.group(1).strip() if m else f"Mode {code}"

        def _make_judge_fw():
            return LLMJudge(JudgeConfig(
                name=f"fw_{datetime.now().isoformat()}",
                model=model, backend=backend, temperature=0.0,
                definitions_path=str(defs_path_fw), examples_path="",
            ))

        if st.button("▶ Run all 3 frameworks on this trace",
                     key="run_fw_all", use_container_width=True):
            judge_fw = _make_judge_fw()
            trace_text = steps_to_text(steps)
            raw_responses: dict = {}
            bar = st.progress(0.0, text="Starting…")

            # ── Call 1: Shared judge (Forced + Semi-Forced) ───────────────────
            bar.progress(0.05, text="Call 1/4 — Judge: detecting present modes…")
            try:
                p_judge = build_judge_prompt(trace_text, definitions_fw, "")
                r_judge = judge_fw._dispatch(p_judge, "fw_judge")
                judge_verdicts = parse_14_modes(r_judge.raw_text)
                raw_responses["judge"] = {"raw": r_judge.raw_text, "error": None}
            except Exception as e:
                judge_verdicts = {m: 0 for m in FAILURE_MODES}
                raw_responses["judge"] = {"raw": "", "error": str(e)}

            present_modes = [m for m in FAILURE_MODES if judge_verdicts.get(m, 0) == 1]

            # ── Call 2: Forced batch ──────────────────────────────────────────
            bar.progress(0.30, text=f"Call 2/4 — Forced: localizing {len(present_modes)} mode(s)…")
            forced_res: dict = {}
            if present_modes:
                try:
                    p = build_forced_batch_prompt(present_modes, steps, definitions_fw)
                    r = judge_fw._dispatch(p, "forced_batch")
                    forced_res = parse_batch_localise_response(r.raw_text, n_steps)
                    raw_responses["forced"] = {"raw": r.raw_text, "error": None}
                except Exception as e:
                    raw_responses["forced"] = {"raw": "", "error": str(e)}
            else:
                raw_responses["forced"] = {"raw": "(no modes to localize)", "error": None}

            # ── Call 3: Semi-Forced batch ─────────────────────────────────────
            bar.progress(0.55, text=f"Call 3/4 — Semi-Forced: localizing {len(present_modes)} mode(s)…")
            semi_res: dict = {}
            if present_modes:
                try:
                    p = build_semi_forced_batch_prompt(present_modes, steps, definitions_fw)
                    r = judge_fw._dispatch(p, "semi_batch")
                    semi_res = parse_batch_localise_response(r.raw_text, n_steps)
                    raw_responses["semi"] = {"raw": r.raw_text, "error": None}
                except Exception as e:
                    raw_responses["semi"] = {"raw": "", "error": str(e)}
            else:
                raw_responses["semi"] = {"raw": "(no modes to localize)", "error": None}

            # ── Call 4: Relaxed — no judge pre-run ───────────────────────────
            bar.progress(0.80, text="Call 4/4 — Relaxed: detect + locate with full taxonomy…")
            try:
                p_relaxed = build_relaxed_localise_prompt(steps, definitions_fw)
                r_relaxed = judge_fw._dispatch(p_relaxed, "relaxed_full")
                relaxed_parsed = parse_relaxed_steps(r_relaxed.raw_text, n_steps)
                raw_responses["relaxed"] = {"raw": r_relaxed.raw_text, "error": None}
            except Exception as e:
                relaxed_parsed = {}
                raw_responses["relaxed"] = {"raw": "", "error": str(e)}

            bar.progress(1.0, text="Done — 4 calls complete.")
            bar.empty()
            st.session_state.fw_table = {
                "judge_verdicts": judge_verdicts,
                "forced": forced_res,
                "semi": semi_res,
                "relaxed_parsed": relaxed_parsed,
                "raw": raw_responses,
            }

        if "fw_table" in st.session_state:
            import pandas as pd
            fw = st.session_state.fw_table
            judge_verdicts = fw.get("judge_verdicts", {})
            forced_res     = fw["forced"]
            semi_res       = fw["semi"]
            relaxed_parsed = fw.get("relaxed_parsed", {})

            def _fmt_forced(mode):
                if not judge_verdicts.get(mode, 0):
                    return "—"
                v = forced_res.get(mode)
                if v is None:                    return "✓ (no parse)"
                if v["steps"] == "global":       return "GLOBAL"
                if v["steps"]:                   return "Steps " + ", ".join(str(s) for s in v["steps"])
                return "✓ (no parse)"

            def _fmt_semi(mode):
                if not judge_verdicts.get(mode, 0):
                    return "—"
                v = semi_res.get(mode)
                if v is None:                    return "✓ (no parse)"
                if v["retracted"]:               return "✗ retracted"
                if v["steps"] == "global":       return "GLOBAL"
                if v["steps"]:                   return "Steps " + ", ".join(str(s) for s in v["steps"])
                return "✓ (no parse)"

            def _fmt_relaxed(mode):
                hits = sorted(
                    [k for k, v in relaxed_parsed.items() if mode in v],
                    key=lambda k: -1 if k == "global" else int(k),
                )
                if not hits: return "—"
                return ", ".join("GLOBAL" if k == "global" else f"Step {k}" for k in hits)

            rows = []
            for mode in FAILURE_MODES:
                mname = _resolve_mode_name(mode)
                rows.append({
                    "FM": f"{mode}  {mname}",
                    "🔒 Forced":      _fmt_forced(mode),
                    "⚖️ Semi-Forced": _fmt_semi(mode),
                    "🔓 Relaxed":     _fmt_relaxed(mode),
                })

            df = pd.DataFrame(rows).set_index("FM")
            st.dataframe(df, use_container_width=True)

            present = [m for m in FAILURE_MODES if judge_verdicts.get(m, 0)]
            st.caption(
                f"Judge detected **{len(present)}** mode(s) present: "
                + (", ".join(present) if present else "none")
            )

            raw = fw.get("raw", {})
            with st.expander("🔧 Raw Judge response"):
                if raw.get("judge", {}).get("error"):
                    st.warning(raw["judge"]["error"])
                st.code(raw.get("judge", {}).get("raw", ""), language="text")
            with st.expander("🔧 Raw Forced response"):
                if raw.get("forced", {}).get("error"):
                    st.warning(raw["forced"]["error"])
                st.code(raw.get("forced", {}).get("raw", ""), language="text")
            with st.expander("🔧 Raw Semi-Forced response"):
                if raw.get("semi", {}).get("error"):
                    st.warning(raw["semi"]["error"])
                st.code(raw.get("semi", {}).get("raw", ""), language="text")
            with st.expander("🔧 Raw Relaxed response"):
                if raw.get("relaxed", {}).get("error"):
                    st.warning(raw["relaxed"]["error"])
                st.code(raw.get("relaxed", {}).get("raw", ""), language="text")

st.markdown("---")
st.caption(
    "**Stage 1 MVP**: ChatDev only. Stage 2 will add raw-trace input and other frameworks. "
    "Built for RQ C: step-level localization of failure modes."
)
