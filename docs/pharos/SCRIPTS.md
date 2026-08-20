# Code and scripts inventory

Everything that has to be written or adapted, with what it consumes and produces,
and where it should live.

Two placement rules, applied throughout:

- **Anything that touches the pool schema or the task registry belongs inside the
  framework** — it is coupled to conventions that change there.
- **Anything that only transforms a third-party file into our schema belongs in a
  separate ingestion toolkit.** These are the pieces most likely to be thrown away
  when a source is dropped, and keeping them out of the framework means dropping a
  source is a delete rather than an edit.

The split matters because roughly half of this list is per-source glue with a short
half-life, and the other half is durable machinery.

Status column: **new** = written from scratch, **adapt** = copy-and-adjust from a
named source, **extend** = an additive change to an existing framework file.

---

## A. Ingestion

Separate toolkit. None of these import framework internals; they emit pool rows.

| Script | Status | In → Out | Notes |
|---|---|---|---|
| `fetch_dataset.py` | new | HF id → local parquet/jsonl + a manifest | Must check size before downloading — a repo rule. Records id, revision, licence and row count into the manifest so provenance is captured at the point of entry rather than reconstructed later |
| `verl_to_pool.py` | new | verl-convention rows → pool rows | **The biggest single reuse win.** Several sources share the verl row format: `data_source`, `prompt`, `reward_model` (JSON *string*), `extra_info` (JSON *string*). One converter with a per-`data_source` routing table serves all of them |
| `pivot_to_pool.py` | adapt | Responses-API pivot rows → pool rows | Adapt from the pivot converters shipped with the Gym skill. Carries `tools` and `expected_action` through |
| `strip_bundled_traces.py` | new | rows with pre-generated responses → prompt-only rows | Several sources bundle a response and often a reasoning trace. We regenerate, so these must not reach the model — but they must be **retained in a provenance column, not dropped**, per the never-silently-drop rule. Keeping them also lets us compare our teacher against the source's |
| `messages_normalise.py` | new | assorted prompt shapes → `messages` | Handles the four shapes seen in practice: a bare string, `{instruction, input}`, an OpenAI-style message list, and a chat template already rendered to text. Must end on a user turn or the row is rejected |

### Per-source adapters

One function per source, not one file. Transformation numbers refer to the standard
set in `DATASETS.md`.

| Adapter | Source | Transformations | Notes |
|---|---|---|---|
| `adapt_nemotron_math` | Nemotron-SFT-Math-v3 | T2 T4 T5 T8–T11 | **Per-row licence** must be preserved, not blanket-applied. Strip the source model's tool scaffolding |
| `adapt_skywork_or1` | Skywork-OR1-RL-Data | T2 T5 T8–T11 | Keep the three probe pass-rate columns as a curriculum signal |
| `adapt_wildsci` | WildSci | T2 T5 T8–T11 | Question plus formatted options into one user turn; keep the voting columns as a trust filter |
| `adapt_oda_fin` | ODA-Fin-RL-12k | T1 T2 T5 T8–T11 | `reward_model.style` is `model` — routes to the judge, not to a string match |
| `adapt_verl_generic` | the verl aggregate, and any future verl source | T1 T2 T5 T6 T8–T11 | Per-`data_source` routing table; filter out the execution-gated prefixes |
| `adapt_nemotron_if` | Nemotron-RL-instruction_following | T2 T5–T11 | The differential test target for the IF verifier |
| `adapt_sysbench_cfbench` | SysBench, CFBench | T2 T3 T5 T6 T8–T11 + rubric mapping | **Multi-turn** — constraints attach to the final user turn. Routes each instruction id by whether our registry has it (~24%) or the judge takes it (~76%). Judge questions arrive unweighted |
| `adapt_terminal_pivot` | Terminal-Pivot | T2 T5 T6 T8–T11 | Text-in/text-out; no tools. Splits on pivot index: index 0 carries a standalone task, deeper rows are single-step only |
| `adapt_tool_pivot` | the three tool-call pivots | T1 T5 T6 T8–T12 | Carries tools and the expected action |

Deferred until their tier opens: `adapt_acecode`, `adapt_klear_code`,
`adapt_deepcoder`, `adapt_competitive_coding`, `adapt_recursive_tasks`.

Not to be written at all unless the licence clears: `adapt_ifevalcode`.

### Two small verifiers the port will not give us for free

- **A multiple-choice exact-match verifier.** Two domains of the verl aggregate,
  28,883 rows between them, are not wired into the upstream dispatcher. The
  verifier is trivial but it has to be written.
- **A rubric-weighting policy** for the judge questions on SysBench and CFBench.
  They arrive unweighted, and the reward arithmetic is weight-based. A flat weight
  is a legitimate choice; it should be a recorded one.
- **An instruction-id router.** Given an id, decide checker / judge / reject, from
  a table rather than a hardcoded list — three quarters of the ids on the constraint
  datasets are outside our registry, and any new source will add more.

### One conversion worth its own entry

`scenario_to_conversation.py` — the conversational tool-use pivots carry a
self-contained policy system prompt and opening customer message. Gym's
conversational tool-use simulation environment can consume that content to produce
**fresh multi-turn trajectories** rather than single-step comparisons. This is the
only path in the catalogue from imported pivot data to a renewable source that does
not require building new infrastructure, and it is worth a script of its own.

## B. Verification

Inside the framework, under the verifiers package.

| Script | Status | Notes |
|---|---|---|
| `verifiers/instruction_following/` | adapt | The multilingual IF checker. Reward path removed; emits evidence with per-instruction `pass` / `fail` / `not_applicable`. Plan 01 |
| `verifiers/tool_call.py` | adapt | Single-step tool-argument comparator, re-expressed as evidence. Source is pure standard-library Python and drags in nothing. Plan 04 |
| `verifiers/tool_call_lenient.py` | adapt | The coding-agent variant: tool-name category equivalence, path-suffix matching, sequence similarity. Worth having separately — strict matching punishes paraphrase rather than error |
| `verifiers/terminal_action.py` | new | JSON-schema validation of a command batch plus similarity against the expert keystrokes. Small, and unlocks a whole dataset at T0 |
| `run_verifiers.py` | extend | One dispatch branch per new `checks` key, plus its entry in the pass/fail histogram |
| `verifiers/__init__.py` | extend | The hand-maintained export list is the registry |

## C. Generation

Inside the framework.

| Script | Status | Notes |
|---|---|---|
| `generate.py --tools-field` | extend | Attach a row's tool schemas to the request. Mirrors the existing messages-field flag. `--extra-body` cannot substitute — it is per-run, tools are per-row. Must return a non-empty expected-field list or the run is a silent no-op |
| `gen_<family>_test.sh` | new | One per new dataset family, throwaway output path, small row count. Additive by default — never edit a production launcher |
| `regenerate_with_teacher.sh` | new | Orchestrates a teacher swap: new suffix, empty-response count, per-track judging subsets, pass-rate recompute. Plan 05 |

## D. Synthesis

Inside the framework — these are task-registry artefacts, not standalone scripts.

| Artefact | Status | Notes |
|---|---|---|
| 4 task registry entries + templates | adapt | Decompose, classify, evolve, extract-keywords. Converted to the house ` ```json ` + Pydantic convention. Plan 02 |
| 4 Pydantic models | new | Two behaviours must survive the conversion from the current hand-written parsers: an unparseable constraint label defaults to *rubric*, and keywords are filtered to surface forms present in the reference |
| `sample_constraints.py` | adapt | Draws programmatic constraints from the eligibility matrix, honours the conflict table, fills literals from the language profile. Port of the flywheel's constraint builder with the length-calibration bug fixed |
| `evolve_round.sh` | new | One flywheel round: evolve → generate → tag → route. Iterative, so it is a new script rather than an edit to the phase-based pipeline |

## E. Quality assurance and diagnostics

Mostly separate toolkit; these are what make a silent failure visible.

| Script | Status | Notes |
|---|---|---|
| `validate_pool_rows.py` | adapt | Port of the row validator: language in the verifiable set, id/kwargs length agreement, no ineligible constraint, no conflicting pair, rubrics bound to their constraint, non-empty prompt. Run at every stage boundary |
| `validate_pivot_dataset.py` | adapt | Copy from the Gym skill. Row shape, expected-action schema, agent-ref alignment |
| `instruction_id_inventory.py` | new | Enumerate distinct instruction ids across a dataset and diff against our registry. **Write this first** — it is what sizes the constraint-dataset work, and the answer is currently unknown |
| `pass_rate_report.py` | adapt | Per-language and per-source difficulty distribution. The primary diagnostic: a distribution crushed toward 0 or 1 relative to its peers means an unsatisfiable or vacuous constraint, not a model result |
| `count_empty_generations.py` | new | Counts empty raw responses per label after every generation phase. A truncated thinking-model response is indistinguishable from a refusal, and this is the standing check for it |
| `compare_teacher_versions.py` | new | Per-row grade deltas and pass-rate band migration between two suffixes. Means hide ranking changes; this is what makes a teacher swap a measurement rather than an assumption |

## F. Deferred — execution and containers

Not to be written until the tier is opened. Listed so the shape is known.

| Script | Notes |
|---|---|
| `verifiers/code_tests.py` | Run candidate code against test cases, return evidence. Needs a hard per-case wall clock, an address-space limit, and a killed-process path that returns evidence rather than raising |
| `sandbox_runner.py` | Process or container isolation. Nothing in the framework provides this today |
| `normalise_execution_evidence.py` | Strip timings, addresses, traceback paths and iteration order out of rendered evidence. Non-optional: the evidence text feeds a judge prompt and then judge-SFT data, so non-determinism would make identical responses produce different training rows |
| container orchestration | Per-task image, setup and teardown around each episode, resource limits, reproducible task state |

---

## Write order

1. `instruction_id_inventory.py` — cheap, and it sizes the largest unknown.
2. `fetch_dataset.py`, `messages_normalise.py`, `strip_bundled_traces.py` — nothing
   moves without ingestion.
3. `validate_pool_rows.py` — before any adapter, so the first adapter is checked
   from its first row rather than after the first pool build.
4. `verifiers/instruction_following/` and the IF adapters — the first end-to-end
   slice.
5. `verl_to_pool.py` — unlocks several sources at once.
6. `count_empty_generations.py`, `pass_rate_report.py` — before the first
   full-scale generation, not after it.
7. Tool support: `--tools-field`, the comparators, `pivot_to_pool.py`,
   `validate_pivot_dataset.py`.
8. Synthesis artefacts.
9. `compare_teacher_versions.py` and `regenerate_with_teacher.sh`.

The QA scripts sit deliberately early. Every failure mode recorded in the handover
notes is a silent one — a uniformly null column, a vacuously complete resumption,
an empty generation that looks like a refusal — and each of these scripts is the
thing that makes one of them visible.
