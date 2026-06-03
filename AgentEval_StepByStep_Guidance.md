# Step-by-Step Guidance — IKEA AgentEval

> **How to use this document — please read first**
>
> This is our working plan for the project. It's a draft to finalise together, not the final word. Please read it, mark it up **directly in here**, and send it back — or, since it's just markdown, drop it straight into the GitHub repo so your edits and comments are versioned along with the code. Once we've worked through your questions and comments, this becomes the document we all work from.
>
> **To leave a comment**, add your note inline, right next to the part it refers to, in this format so I can pick it out later:
>
> > **[Name]** your question, comment, or suggestion here
>
> For example — **[Ivayla]** should the dev slice be stratified by model too, or only by framework?
>
> Optionally tag the type after your name: **Q** question · **?** needs clarification · **+** suggestion or addition · **OK** read and agree. Keep your notes in bold or indented so they stand out from the plan text — anything that visually separates them from the original is fine (or use your editor's comment feature if it has one).
>
> Please **don't delete or rewrite** the original lines — just add beside them, and we'll tidy up together afterwards.

*Do the steps in order; each builds on the last. (Some later steps are still being written.)*

The shape: **Step 0** (foundation, already done — just tidy the outputs) → **Step 1** (build the baseline, everyone together) → **Step 2** (each person branches into a variation / individual RQ).

---

## Running through everything — the cost/value lens

This is not a separate step; it is a lens you apply at every step where a cost/quality trade-off exists (the baseline, the model comparison, partial traces, step exclusion, and so on).

- **Don't collapse results to a single "best."** Place them on the cost/accuracy/speed trade-off (the Pareto / indifference view), built from the tokens and time you log on every run.
- **Express cost relative to the system's own cost where it makes sense** — the judge adds ~X% on top of running the agentic system. That ratio is the economically meaningful figure, not the absolute number.
- **Let the weighting come from the use case.** Take that question to IKEA, so the trade-off points at their real priorities rather than an arbitrary one.

The specifics already appear in Step 0 (the token statistics) and Step 1 (the two statistics and "what better means"); the point here is that the same thinking carries into every later step too.

---

## Step 0 — Foundation (done): make the statistics ordered and reproducible

**Status:** done. The seven parsers, the step-splitting, the schema unification, and the EDA are in place and carry forward unchanged.

**The one thing to tidy:** produce the descriptive statistics in an **ordered, reproducible** way, generated directly by the notebook from top to bottom — not scattered cells. Run-all should emit a clean sequence of labelled tables and figures, saved to `outputs/`, that can drop straight into the thesis/artefact.

The set to standardise:

- Dataset composition — traces per framework and per underlying model (GPT-4o vs Claude), and the FM-present rate.
- Failure-mode prevalence across the 14 modes, with an explicit note of which are too rare to evaluate (FM-2.5 absent, FM-1.2 ~10 traces, FM-2.1 ~1.6%).
- Failure-mode co-occurrence.
- **Token length per framework (median and max).** This is the one that matters most for the project's motivation — it is the "size of the bill" the cheap-model idea attacks. Two things to add around it:
  - **External numbers and the trend.** Beyond your own dataset, find published statistics on how many tokens agentic systems consume and how that is developing over time (it is rising). This establishes that the cost problem is real and growing, not specific to MAD.
  - **Express evaluation cost relative to the system's own cost.** The economically meaningful figure is not the judge's absolute cost but the judge's cost as a fraction of the tokens the agentic system itself spent. A judge that uses ~20% of the tokens the MAS already consumed naively adds ~20% to the cost of running it. That ratio (judge cost / system cost) is what shows economic relevance — a 20% evaluation surcharge and a 200% one are completely different propositions. It depends on the model and trace length, but those are exactly the figures a company weighs when deciding whether evaluation is worth deploying. (You compute the judge side once Step 1 produces token counts; set up the framing here.)
- Step counts per framework, and the step-kind distribution (message / tool_call / tool_result / system).

**Done when:** a single notebook run regenerates every statistic in order, as artefact-ready tables/figures written to `outputs/`.

---

## Step 1 — Build the baseline (everyone, together)

This is the spine. It is shared work: build it once, build it well, and it becomes the core of the artefact.

### Goal — two statistics

1. **Accuracy / F1 on the human-labelled set.** Run the judge over the human-annotated traces and compare its 14-mode output to the human labels — per-FM precision/recall/F1, overall F1, and κ. Do this for **several models**, so you can see which models track human judgement best. (Small set, so report with confidence intervals and don't over-read small gaps.)
2. **Differences per model on the o1-curated set.** Run the same judge over the (sliced) o1-labelled data and compare each model's output to o1's labels — and to each other. This shows how far cheaper models diverge from o1, and how much they disagree among themselves.

Keep the two clearly separate: #1 is agreement with humans (gold, small), #2 is agreement with o1 (silver, large). Different quantities.

**What "better" means here — not necessarily cheaper.** The aim is not automatically to find a cheaper model. Landing on a cheap model that matches o1 would be a great result, but cheap is not the objective in itself — the objective is the trade-off `f(cost, accuracy, speed)`, and which model is "best" depends on the user's priorities, which you don't know in advance. A high-stakes governance check may gladly pay more for recall; a high-volume pre-deployment screen wants cheap and fast. So report where each model sits on that trade-off and let the use case choose — don't pre-commit to "cheapest wins."

And don't treat those weights as unknowable in the abstract — **ask IKEA.** Susana and the Ingka governance team have a concrete deployment context (which workflows, what volume, how expensive a missed failure actually is), and that is exactly what fixes the weights on cost vs accuracy vs speed. Bringing them that question — "what would you trade off, and where?" — turns an abstract trade-off curve into a recommendation aimed at their real use case, which is also what makes the work land for the company.

### How — same prompt, swappable model

- **Same prompt as the paper.** Port the authors' MAST definitions + few-shot block as-is. Do not write your own — if you do, it is no longer a comparison to their pipeline.
- **A config file** holds everything that varies: model name + backend (Ollama / API), parameters (temperature, shots, reasoning on/off), which dataset and slice, the prompt version, and paths. Changing the model = editing the config, not the code.
- **A shared class** is the reusable core everyone builds on and that goes directly into the artefact. It should take a configuration, and expose one main operation: give it a trace, and it returns the 14-mode failure vector together with the usage for that call (tokens in/out and wall-clock time). The model backend — local Ollama or an API — is selected from the config, so the rest of the code never needs to know which model it is talking to.

Everyone works through that one shared object rather than each writing their own version. Cost is derived later from the logged tokens — keep tokens and time as the primitives so open and API models stay comparable.

### Slicing — a compute decision, not a separate step

The o1-curated set is large; running every model over all of it is wasteful while you are still building. Use a fixed, stratified slice for development so the compute stays manageable, and keep that slice frozen so every run is comparable. The human-labelled set is small — run it in full.

### How to work on it together

- **Program the class + config together** (pair or mob on it) so there is one shared implementation, not five.
- **Get it working on one model first** to debug end-to-end, then run the set of models for the two statistics.
- **Split the computation.** Once the class works, divide the models (and/or slice shards) across people and machines — local Ollama on the 256 GB Mac, APIs elsewhere — run in parallel, and **combine** the outputs into one results table keyed by (model, dataset, trace).
- Log tokens and time on every run from the start.

### Done when

- The shared class and config file are committed and anyone can run a model by editing the config.
- A runnable commit produces both statistics — per-model accuracy/F1 (+κ, CIs) on the human set, and per-model agreement-with-o1 + model-vs-model differences on the o1 slice — with tokens and time logged per run.
- It reproduces on my machine from the README — I (Matthias) can re-run everything end to end on my own local machine, a 256 GB Mac that runs the open models locally, so by the end I can verify the whole pipeline on my side. That is your assurance it genuinely works.

---

## Step 2 — Branch into individual RQs (one each)

Once the baseline runs and reproduces, each person takes one of these. Every one is measured as a comparison against the baseline on the same frozen slice, and every one reports through the cost/value lens above. These are starting points — each owner will shape their own into a proper research question.

**A. Parameter sweep and judge consistency.** Systematically vary the judge's settings — temperature, zero- vs few-shot, reasoning on/off — and map how each moves the cost/accuracy/speed trade-off. (The baseline already runs the model roster; this RQ turns those runs into the trade-off surface and adds the settings sweep on top, so it works out both the model and the settings recommendation.) Three things lift it above a plain grid:

- *Interactions, not main effects.* Read the effects per failure mode and against cost: does reasoning-on rescue the semantic modes like 2.6, does few-shot help the rare modes, is the extra accuracy worth the extra tokens? The useful output is a per-mode map of which knob is worth its cost for which kind of failure — which also feeds a routing idea (cheap settings where they suffice, expensive ones only where they pay off).
- *Internal consistency.* Re-run identical inputs several times and measure how much the judge's label vector wobbles between identical calls. A model that is accurate but high-variance is worse for a governance tool than a slightly less accurate but stable one, so report reliability alongside accuracy. There is good literature to build on here — the LLM-as-a-judge reliability / self-consistency work — so this is grounded, not just a measurement.
- *The outcome is a recommendation, not the grid.* The deliverable is the decision rule — "run model X at these settings, because past this point you pay N% more tokens for under one F1 point" — plus the stability ranking. The grid is the evidence; the cost-justified recommendation and the reliability map are the result.

Each setting is a delta against the default-setting baseline, and everything lands as points on the shared trade-off surface.

**B. Failure-mode question extension (enhancing detectability).** *Caveat: this is your proposed idea as I understood it from a short description — I may not have the full picture, so correct me.* The idea is to sharpen the per-failure-mode definitions and examples in the prompt so the judge detects failures better — i.e. enhancing detectability by improving the judge's weakest link, its instructions, rather than only swapping models. That is a genuinely good contribution.

The central risk is **overfitting**, and it is acute because the human (gold) set is tiny (~20 traces): if you keep rewriting the prompt and checking it against those 20, you tune to them and learn nothing that generalises. There is a second trap too — improving agreement with *o1* just means mimicking o1's own mistakes, so o1 is not the target. Tackle it like this:

- **Change from reasoning, not from a metric.** Use the baseline's per-mode results to find where the judge most disagrees with the ground truth, form a hypothesis for why (an ambiguous definition, a missing example type), and make a *small, documented* set of targeted edits — predicting the effect before measuring. A handful of principled changes overfits far less than fifty variants hill-climbed against the same data.
- **Reserve the gold set as a held-out test, used once.** Iterate on the larger o1 slice (or a dev set you curate); keep the human gold set untouched until the final check. If detectability improves on data you never tuned on, it is real.
- **Keep examples disjoint from the eval data.** Any few-shot example you add must come from traces not in any evaluation set (curate from raw traces). A trace is a few-shot exemplar OR a test item, never both — the most common leakage trap.
- **Report mechanistically and with CIs.** "The sharper definition of 2.6 cut these specific reasoning-vs-action confusions on held-out data" survives small N; "agreement went up two points" does not. Read every result under the utility lens too — N extra input tokens per trace for M points of F1.

Optional but valuable: hand-label a handful of extra real traces to enlarge the gold test pool, which directly eases the small-N constraint and reuses your parsers. The comparison throughout is baseline prompt vs extended prompt, same models and slice.

**C. Partial-trace and step-level detection (where does the failure live?).** Start with the honest constraint: there is no step-level ground truth in MAD — labels are trace-level — so every partial or step-level run has to be paired with a full-trace run, and the full-trace judge verdict (the baseline) is the only thing to compare against. The ~20 human-labelled traces carry prose justifications that often point at *where* a failure occurred; use those as the qualitative spot-check.

And state this plainly in the thesis: the step decomposition you have already built is exactly what makes this RQ possible — without a clean split into steps there is nothing to localise to. That chunking is a genuine contribution in its own right, so it should be evaluated and written up, not used silently: how traces are decomposed into steps, whether the decomposition is consistent and sound across the seven frameworks, and whether its granularity affects detection.

With that fixed, the contribution is two-pronged:

- *Assign failure modes to individual steps or step-sequences (the major addition).* MAD says a trace contains FM-X; it never says *where*. A method that attributes a mode to a specific step or a contiguous span — by asking the judge to localise, by feeding single steps or windows in isolation to see which alone trigger the mode, or by finding the minimal span that does — creates the "where" the dataset lacks. (This pairs with RQ D: a step that *alone* triggers a mode and whose *removal* stops detection is strongly implicated — sufficiency from C, necessity from D.) It can't be scored against a step-level gold because none exists, so the deliverable is the *framework* plus qualitative validation against the human justifications and manual inspection — proof-of-concept, honestly caveated.
- *Does more or less input at once reproduce the baseline?* Feed growing prefixes, or the trace in chunks (your "2×50% vs 100%"), and check whether the verdict matches the full-trace judge. This validates whether partial detection is viable at all, and how much context each mode needs before the judge converges on its full-trace answer.

A finding falls out of this: **step-level detection is suitable for some modes and not others.** Reasoning-action mismatch (2.6) and role violations (1.2) are properties of a single step; step repetition (1.3) is a relation between steps; but premature termination (3.1) and unaware-of-termination (1.5) are properties of the whole trace and resist localisation. So part of the result is a map of which modes admit direct step-level detection and which are intrinsically global — and for the step-local ones, a direct detector becomes sensible.

This RQ builds the *framework* for step-level detection. The natural next move — catching the model right after a failing step and steering it back on track — is the runtime-correction idea, worth naming explicitly as **future work** rather than scope here.

Cost caution: step-level and chunked runs multiply judge calls on long traces, so work on a cheap model and a small slice, pair each split run with its full-trace reference, and compare across frameworks by trace fraction rather than absolute step count (AG2 has ~4 steps, AppWorld ~79).

**D. Step exclusion (which steps can you drop, and which carry the signal?).** Same skeleton as C — the full-trace verdict is the reference: remove some steps, re-serialise the trace, re-judge, and compare the verdict and the token count to the full run. It works in two modes, with two payoffs:

- *Rule-based exclusion — the cost lever.* Drop whole classes of steps by metadata and check the verdict survives: all `system`/boilerplate steps (trace 0 was mostly these), echoed or duplicate `tool_result`s, near-identical neighbours (use the step-similarity measure). Cheap, deterministic structural rules — the RegEx/structural angle in its most useful form. The result is deployable: "stripping class X cuts Y% of tokens and moves the verdict on only Z% of traces," reported against the judge-cost-as-%-of-system-cost framing.
- *Ablation — the saliency/necessity lever.* Remove steps and watch which removals *flip* the verdict (present → absent). A step whose removal kills the detection is necessary for it — it carries the signal. This is the necessity counterpart to C's sufficiency: a step that alone triggers a mode (C) and whose removal stops it (D) is strongly implicated, and together they localise the failure with no step-level gold.
- *A null-model simulation — the reliability layer.* On its own, "removing step X changed the verdict" has no reference point. Build one: if the failure signal were spread uniformly across N steps, removing one would cost about 1/N of the evidence, so detection would decay roughly linearly as steps are removed. That linear 1/N curve is the *null*, not the prediction — simulate it properly by removing *random* steps over many resamples to get an empirical decay curve with confidence bands, then compare the real curves to it. The deviation is the finding: decay *slower* than the null means the signal is redundant and spread across many steps (robust detection); decay *faster* or step-like means it is concentrated in a few, and a step whose removal drops detection significantly below the null is genuinely signal-bearing, not chance. That gives the localisation claim significance behind it, per failure mode — it is the random-ablation flavour of feature attribution (the Monte-Carlo-Shapley family), so there is literature to anchor it.

Keep it tractable: don't brute-force leave-one-out on every step (a 79-step AppWorld trace would be 79 judge calls). Test step *classes* for the cost lever; use grouping or bisection for the saliency lever; and for the simulation, bound it hard — a small slice, one cheap model, a fixed grid of removal fractions, and modest resamples — or the call count explodes.

Two caveats. "Unnecessary" is *per mode*, not global — a step irrelevant to 1.3 may be essential for 3.1 — so report exclusion effects per failure mode. And mind the coherence confound: remove steps sloppily and the judge may react to the gaps rather than the missing signal, so drop cleanly and sanity-check. As with C, the cost saving is hard and clean; the localisation claim is softer and validated qualitatively against the human justifications.

**E. Observability-by-design: would structured traces make cheap detection viable?** The most forward-looking of the five — lighter on hard measurement, heavier on argument, proof-of-concept, and a design proposal. Its distinct contribution is a counterfactual the others don't touch: today's traces are messy unstructured logs, which is *why* cheap/regex detection struggles (C and D already show which modes are step-local vs global — that is most of the regex-ability evidence, so E should draw on it, not re-derive it). E asks whether, if agent systems emitted *structured* records at defined points, cheap detection would become viable.

How the case study works:

- Take a small set of real traces with known failures — a handful across a couple of modes, not one.
- Define, *up front*, a structured-logging schema: what an instrumented framework would emit per step (agent, role, declared intent, action, termination flag, verification result). The schema is itself a deliverable — a concrete recommendation to MAS builders.
- Restructure the raw traces into that schema, then run the *same* simple regex/rule detector on the raw version and the restructured version. The delta — regex fails on raw, works on structured — is the result, and it quantifies the value of observability-by-design.

The regex-ability classification rides along as the bridge: for each mode, state how detectable it is as-is versus after restructuring — which modes need structure to become cheaply catchable, and which (semantic ones like 2.6) stay LLM-only no matter how clean the log.

The rigor is all in avoiding circularity — otherwise it is just "my regex works on the format I built for it." So: fix the schema before looking at results and apply it uniformly; use the real failures in the traces, not injected ones; run identical rules on raw and restructured; and ideally have a second person restructure blind, so the restructurer cannot bake in the answer.

This is the normative home of the instrumented-MAS idea, kept to a case study, and its future-work extension is runtime correction — if structured emission makes failures cheaply detectable in real time, you can intervene.

A heavier version — call it **E-heavy** — closes the loop by *running* a MAS rather than retrofitting logs: take one framework and task (ChatDev on the repo's ProgramDev set — 30 coding problems, provided — is the most reproducible), re-run it with slightly modified instructions (a sharper role spec, an explicit termination condition), and measure the failure-mode shift before vs after with your judge. This replicates the paper's own intervention case studies (it reports, e.g., +15.6% for ChatDev from improved role specification) and is the only setup in the project that yields *self-controlled* traces where you know exactly what changed — the closest thing to causal ground truth here. The cost is that running these frameworks is fiddly engineering, so it is a stretch for a strong student or pair, scoped to one framework and one intervention, and gated behind the baseline.

Honest note: as written — the retrofit case study, call it **E-light** — this is the lightest of the five on measurement. The heavier run-a-MAS version above (**E-heavy**) is the way to give the same student more weight if it feels thin.

### Parked for now

- **Full per-failure-mode detector map** — building an actual cheapest-working detector for every mode is likely too much for the time available; the lightweight version (the regex-ability classification) lives in RQ E.

### How the RQs interrelate (own your own, work with others)

Each RQ has a clear owner and stands on its own. The only foundation everyone truly shares is Steps 0 and 1 — the data work and the baseline. Beyond that the RQs are independent, but they fall into families and can benefit from one another, so collaboration is available, not required.

- **Shared foundation (everyone):** Steps 0 and 1 only — the data work and step decomposition, the baseline pipeline, the frozen slice, and cost/time logging. That is the part everyone genuinely shares; the cost/accuracy/speed surface is then the common space results land on. Built once, jointly.
- **"Tune the judge":** A (settings) + B (questions) — two complementary ways to change the same judge; they share the held-out / gold-set discipline.
- **"Where is the failure":** C (sufficiency) + D (necessity + null model) — the tightest pair; one shared step-ablation harness, same human-justification validation.
- **"Trust the judge / labels":** A's consistency + D's null model — reliability/significance kin; shared resampling and confidence-interval machinery, related literature.
- **"Make failures detectable":** B (better questions, prompt-side) + E (better structure, data-side) — two halves of the same goal.
- **Producer → consumer (who can reuse whose work).** Beyond Steps 0–1 nobody is blocked on anyone else, but a few RQs produce something the others can reuse rather than rebuild — benefit, not dependency:
  - *A → everyone.* A's sweep is what builds the cost/accuracy/speed surface and works out which model and settings give the best balance. Everyone branches off the Step-1 baseline, so no one waits on A — but rather than each person re-deciding the judge alone, the others can adopt A's recommended "use this model at these settings" as their default and update later if A finds something better.
  - *C and D → E-light.* Classifying which failure modes are cheaply or structurally detectable is most of E-light's job — but C (which modes are step-local versus whole-trace) and D (which have concentrated versus spread signal) already generate that evidence. E-light reads their results and assembles the regex-ability map from them, rather than re-deriving it from scratch.
  - *E-heavy → the analysis RQs.* If someone runs the MAS to generate fresh, self-controlled traces, those become a resource everyone else can reuse: a new test set to check the judge on unseen data, or new material with a known intervention to localise on.
- **Shared scarce resource:** the ~20-trace gold set underpins B's test and the baseline's human statistic — so expanding it by hand-labelling is a shared win, best done together.

Takeaway: own one RQ end-to-end — each of you has your own question and your own thesis — but build the shared harnesses (pipeline, ablation, resampling, gold set) together. And because the RQs feed one another, you can cross-cite each other's work later: your thesis can build on a teammate's result, and theirs on yours. So working alone on your own RQ and continuing to work as a group or in pairs are not in tension — the structure rewards both.

---

*Draft for refinement — we add and adjust steps as you sign them off.*
