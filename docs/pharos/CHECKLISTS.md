# Checklists

Execution steps for both plans. Every phase ends in a state that can be reviewed
and abandoned without leaving the repos worse.

Standing rules that apply to every box below, from both `AGENTS.md` files and the
runbook:

- Test on 10–100 rows first, **always to a throwaway output path** — `--limit-rows`
  at a real output path destroys that file (G8).
- Update `data_team/AGENTS.md` and `training_data_acquisition/AGENTS.md` in the
  **same session** as any script added or changed. Call out schema or file-naming
  changes explicitly. Never cite line numbers.
- Additive by default: a new `*_test.sh` rather than an edit to a production
  script. A script that is *broken* may be fixed in place, stated plainly.
- Code style: minimal, neutral, no narration. The existing verifier modules are the
  reference.
- Never silently drop dataset columns. On a name collision, rename the incoming one.

---

## Phase 0 — before any code

- [ ] **D1 signed off** — IFEval accepted as new scope, with a target slice size
- [ ] **D3 answered** — production will invoke `run_verifiers.py` for IF rows.
      *If no, stop and reconsider the rubric-only route*
- [ ] **D2 chosen** — language coverage; default is ship 22, `not_applicable` for 8
- [ ] **D5 chosen** — subpackage layout and provenance
- [ ] Confirm `regex` may be added as a dependency, and that `numpy<2` is
      acceptable in the environment `run_verifiers.py` runs in
- [ ] Confirm the fastText LID model can be resolved on the target machine
      (env var, local copy, or `ftlid`) — it cannot be committed

---

## Phase 1 — the verifier (Plan 01)

### 1a. Port

- [ ] Create `verifiers/instruction_following/` and copy the core across verbatim
- [ ] Write `PROVENANCE.md`: source repo, commit, re-sync recipe, "do not edit in
      place"
- [ ] **Remove the reward path.** `verify()` / `verify_one()` are not re-exported.
      §8.4: never return a reward
- [ ] Write `adapter.py::verify_instructions` returning the evidence dict, with
      `status: pass | fail | not_applicable` per instruction
- [ ] Map ineligible constraints to `not_applicable`, never `fail`
- [ ] Write `render(result) -> str`, one bullet per instruction
- [ ] Reuse `strip_fence()` from `schema_formats.py`; do not reimplement
- [ ] Blanket-guard the adapter so no input shape can raise
- [ ] Extend `verifiers/__init__.py` import and `__all__`

### 1b. Tests

- [ ] Port the 62 differential tests
- [ ] Add Greek cases for every eligible instruction
- [ ] Add one `not_applicable` case per gap language
- [ ] Add a `render()` golden per status — this text is judge-visible and feeds
      judge-SFT downstream
- [ ] Assert the adapter never raises on malformed `kwargs`, missing keys, empty
      response, wrong-type values

### 1c. Bridge

- [ ] `run_verifiers.py`: `if "ifeval" in checks:` branch
- [ ] Append `ifeval:pass` / `ifeval:fail` to the `kinds` histogram
- [ ] Confirm rows with no applicable checks still get the `NO_CHECKS` sentinel
      rather than an empty string
- [ ] Run over ~50 real generated rows; **read the rendered output by eye** before
      trusting the histogram

### 1d. Sign-off gate

- [ ] Rendered evidence is legible and states language-inapplicability clearly
- [ ] No instruction reports `fail` for a language where it is not meaningful
- [ ] Both `AGENTS.md` updated
- [ ] Report written to `acquisition/reports/` — the owner reads reports, not
      terminals

---

## Phase 2 — synthesis, programmatic only (Plan 02 phase 1)

### 2a. Tasks

- [ ] Add 4 registry entries: `if_decompose`, `if_classify_constraints`,
      `if_evolve`, `if_extract_keywords`
- [ ] Write the 4 templates with ` ```json ` / ` ``` ` parsing tags and doubled
      braces in every JSON example
- [ ] Add Pydantic models: `IFDecomposition`, `IFConstraintLabels`, `IFEvolution`,
      `IFKeywords`
- [ ] Preserve the two parser behaviours as model/post-validation logic:
      unparseable label ⇒ `rubric`; keywords filtered to surface forms present in
      the reference
- [ ] Verify every `prompt_template` path resolves — three registry entries already
      dangle (`verify_abstention`, `weighted_rubrics`, `fusion_of_n_2turn`). A
      one-line audit over the whole registry is a cheap additive contribution
- [ ] Map `language` to a **column**; `--language` alone does not satisfy it (G4)
- [ ] Confirm each task returns a non-empty expected-field list (G19)

### 2b. First run

- [ ] `--preflight-rows 3 --limit-rows 20`, throwaway output path
- [ ] **Check no parsed column is uniformly null** — that means `parsing_tags`
      disagree with the template, and it fails silently (G5)
- [ ] Count empty `raw_response_*` (G2)
- [ ] Inspect 10 decompositions by eye: are the constraints in the target language,
      untranslated?

### 2c. Constraint construction

- [ ] Sample programmatic constraints from the eligibility matrix only
- [ ] Honour the conflict table, including the multilingual pairs
- [ ] Draw literals from the language profile: quote pair, postscript marker,
      section word, letter inventory
- [ ] Derive `verification_meta.answer_format` from the instruction spec; never
      hardcode (G18)
- [ ] Calibrate length constraints with a word count, not `len()` of a string
- [ ] Run `validate_row` before `make_row`

### 2d. Pool integration

- [ ] `adapters.py::adapt_<source>` with the new `verification_type` and `checks`
- [ ] `expected_response` non-empty on every row, or `make_row` drops it
- [ ] `translatable = "0"` as a **string** (G10)
- [ ] `pass_rate = -1.0`, `pass_rate_n = -1` — unknown until the generation pass
- [ ] `build_pool.py::source_specs()` entry + `POOL_CAPS`
- [ ] `dedupe_pool_ids.py` clean; ids unique
- [ ] Both `AGENTS.md` updated, schema addition called out explicitly

### 2e. Greek end-to-end

- [ ] One language only. Generate → 8 generations → verify → judge
- [ ] Read the pass-rate distribution. **A distribution crushed toward 0 or 1
      means an unsatisfiable or vacuous constraint**, i.e. a missing row in
      `profile.yaml` — fix before scaling
- [ ] Inspect ~20 full rollouts by eye. Green tests are not sufficient
- [ ] Report to `acquisition/reports/`

### 2f. Scale

- [ ] Extend to the remaining covered languages
- [ ] Compare pass-rate distributions **across** languages; an outlier is a data
      bug, not a model signal
- [ ] Confirm within-group reward variance is non-trivial — a prompt every rollout
      passes or fails teaches nothing under GRPO

---

## Phase 3 — rubric measurement (Plan 02 phase 2)

Off the training path. The point is to decide D4 with evidence.

- [ ] Generate rubric criteria for a sample of already-synthesised IF rows, using
      `weighted_rubrics_dialogues`
- [ ] Supply **genuinely different models** as candidates, not several samples from
      one — rubric quality is bounded by candidate diversity
- [ ] Grade with `grade_weighted_rubrics_turn`; merge with `merge_rubric_scores.py`
- [ ] Score over `weighted_rubrics_turn<N>`, never `rubrics_turn<N>` — the latter is
      unweighted and silently gives a different reward
- [ ] Average over scored rows only; `-1.0` means unscored, not zero
- [ ] Keep "skipped, no response" separate from "judge output unparsable"
- [ ] **The measurement:** does the rubric reward separate rollouts that the
      programmatic checks score identically? Report the joint distribution, not two
      marginals
- [ ] Recommend on D4 from that result. If the overlap is near-total, say so and
      stop — a hybrid track that adds cost without variance is worse than either half

---

## Phase 4 — hybrid (Plan 02 phase 3)

Only after D4 is signed off.

- [ ] Implement the chosen option (B: new track + composed reward, or D: rubric
      grades as evidence into `{check_results}`)
- [ ] If B: schema change called out explicitly in both `AGENTS.md`, and every
      consumer audited — `run_verifiers.py`, the judging phase, the merge, the
      selection notebook
- [ ] If D: confirm the extra rubric-judge pass is budgeted; it is materially more
      expensive per call than the verify judge
- [ ] Re-measure reward variance after composition
- [ ] Report to `acquisition/reports/`
