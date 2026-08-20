# Plan 02 — synthesising IF prompts with programmatic checks and rubrics

Generates instruction-following prompts whose constraints split into two kinds:
those a program can check, and those a rubric judge must assess. Both are produced
from the same decomposition, so the split is a property of the constraint rather
than of the source.

Depends on Plan 01 for the verification side.

## 1. What already exists and must not be rebuilt

Three parts of the flywheel described in `docs/SYNTHESIS_PIPELINE.md` are already
present in Pharos, better than the original:

| Flywheel stage | Already in Pharos |
|---|---|
| W5 difficulty tagging (k rollouts, pass-rate) | **2 teacher + 6 student generations per prompt, all graded** — an in-house pass-rate-out-of-8, which is exactly the difficulty signal, measured on the actual student |
| difficulty routing | `notebooks/select_rl_pools.ipynb` §7 — the owner's surface. Do not automate it |
| rubric creation, grading, reward | mechanism 3, complete: `weighted_rubrics_dialogues`, `grade_weighted_rubrics_turn`, `merge_rubric_scores.py` |

The alignment on difficulty is close to exact. The flywheel routes on pass-rate
because GRPO needs within-group reward variance — a prompt every rollout solves,
or none does, teaches nothing. `AGENT_RUNBOOK.md` §0.1 gives the same reason for
the 8-generation design.

**So the contribution is narrower than the flywheel:** constraint synthesis
(decompose, classify, build) and evolution. Everything downstream is Pharos's.

## 2. Pipeline

```
seed prompts (existing pool sources, or a new source)
   │
   ├─ if_decompose            task → {task_description, constraints[], input}
   ├─ if_classify_constraints task → programmatic | rubric, per constraint
   │
   ├── programmatic branch → sampled from the eligibility matrix, never invented
   │      instruction_ids + kwargs → checks {"ifeval": …}
   │
   └── rubric branch → weighted_rubrics_dialogues (existing task)
                       → rubrics [{criterion, weight, type, turn_idx}]
   │
   └─ if_evolve (optional, iterative) → adds one constraint per pass
   │
   ↓ adapters.make_row → pool row
   ↓ existing 8-generation pass → in-house pass rate
   ↓ notebook §7 selection
```

The programmatic branch **samples from the verified taxonomy rather than asking a
model to invent constraints**. That is what makes everything sampled scoreable by
construction, and it is why the eligibility matrix is shared with Plan 01 instead
of duplicated: a constraint that synthesis can emit is exactly a constraint that
verification can score.

## 3. Tasks to register

Four new entries in `generation_prompt_templates.py` plus their templates. Shapes
are ported from `synthesis/prompts.py`, **rewritten to house convention**.

| Task | `required_inputs` | Output field / model |
|---|---|---|
| `if_decompose` | `dialogue_history, language` | `if_decomposition` / `IFDecomposition` |
| `if_classify_constraints` | `dialogue_history, language, constraints` | `if_constraint_labels` / `IFConstraintLabels` |
| `if_evolve` | `instruction, language` | `if_evolution` / `IFEvolution` |
| `if_extract_keywords` | `instruction, reference_response, language` | `if_keywords` / `IFKeywords` |

### The conversion that has to happen

The templates in `synthesis/prompts.py` emit line-anchored `#field:` markers and
are read by hand-written parsers in `synthesis/parsing.py`. Pharos parses with a
literal `str.find` over a two-element `parsing_tags` pair, and every verify and
rubric task uses ` ```json … ``` ` plus a Pydantic model.

**Convert to the house form.** This deletes `synthesis/parsing.py` entirely in
favour of Pydantic validation, which is a simplification, not a compromise — but
two behaviours from the parsers must survive as model or post-validation logic,
because they are load-bearing:

- **An unparseable constraint label defaults to `rubric`, not `programmatic`.** A
  constraint wrongly marked programmatic becomes an unenforceable reward signal;
  wrongly marked rubric only costs a judge call.
- **Keywords must be surface forms present in the reference response.** Matching is
  literal, so a keyword the model would have to inflect is never found. In Greek,
  Polish, Czech and Finnish that is the common case, not the exception.

G5 applies with force here: `parsing_tags` disagreeing with what the template
instructs produces a **uniformly null column while `raw_response` fills normally**.
Check the parsed column is not uniformly null before scaling past the smoke run.

### Template conventions

- `{{` and `}}` for literal braces in the JSON example, or `KeyError` at render.
- Meta-prompt in English, generated content in the target language, with an
  explicit instruction not to translate the extracted text. Without it models
  translate constraints into English, which silently severs the link between a
  constraint and the instruction it came from.
- `language` needs a **column**, not just `--language` (G4). Map it.
- Never leave a template 0 bytes — that renders an empty prompt and silently skips
  every row (G12's `prompt_translation_evolution.txt`).

## 4. The hybrid question — needs a decision

`VERIFICATION_ONBOARDING.md` §1: *"Three reward tracks, never mixed inside one
training run."* `track` selects the grading mechanism, and the pool schema encodes
the separation — `rubrics` is `""` on constraint rows, `expected_response` is `""`
on rubric rows.

A row carrying both programmatic checks and rubric criteria is therefore a genuine
departure. Four ways to land it:

| | Approach | Schema change | Reward | Verdict |
|---|---|---|---|---|
| **A** | `track: constraint`, populate `rubrics`, reward from `verify_constraints` only | none | 1–10 → (s−1)/9 | rubrics carried but unused at reward time — not really hybrid |
| **B** | new `track: if_hybrid`, composed reward | **yes** | composition of both | honest, but touches a live schema and every consumer |
| **C** | two rows, same prompt, different tracks | none | separate | **blocked** — dedupe is on `prompt_hash`, so the second row is dropped |
| **D** | `track: constraint`; grade rubrics first, feed the grades into `{check_results}` as evidence | none | single 1–10 | framework-compatible, but costs a rubric-judge pass per row |

Option C is worth stating explicitly because it looks like the obvious answer and
is silently impossible: `build_pool.py` dedupes on `prompt_hash`, never `uuid`, so
two rows sharing a prompt collapse to one.

**Recommended staging:**

- **Phase 1 — option A.** Ship constraint-track synthesis with programmatic checks
  only. No schema change, no new invariant, no additional sign-off beyond D1. This
  is where the measurable value is, and it is the part the eligibility matrix makes
  safe.
- **Phase 2 — measure.** Generate rubric criteria for the same rows and grade them
  separately, off the training path. Check whether the rubric signal adds reward
  variance the programmatic checks do not already capture. If it does not, stop:
  a hybrid track that adds cost without variance is worse than either half.
- **Phase 3 — option B or D, with sign-off** (`DECISIONS.md` D4), chosen on what
  phase 2 measured.

Phase 2 is the part worth insisting on. §10.1a records that existing rubrics were
generated from candidates including an older DeepSeek Flash V4, and that rubric
quality is bounded by candidate diversity — *"if the candidates agree, or are all
weak, the criteria that emerge are correspondingly generic and will not produce
reward variance."* For IF prompts, where the programmatic checks already capture
the explicit requirements, generic rubric criteria are the likely outcome. Measure
before committing to the composition.

## 5. Rubric branch specifics

Where a constraint is classified `rubric`, use the existing creation task rather
than a new one. Two things follow from §10:

- **Add criteria through the creation template, never by editing the grading
  template.** The rubric judge is a separately finetuned Qwen3.6-35B-A3B tuned
  against that prompt shape; changing block names, the output schema or the rule
  numbering shifts the distribution it was tuned on.
- **`weighted_rubrics_dialogues` is response-grounded**, mining differences between
  several candidate responses. Feeding it one model's samples produces flat
  criteria. If phase 2 runs, supply genuinely different models.

The creation template already mandates a weight-9–10 `hard_rule` for any negative
user constraint ("do not use bullet points"). That overlaps with what the
programmatic checker covers, and the overlap is the thing phase 2 measures: if the
rubric restates a check, it adds cost and no variance.

## 6. Orchestration

Evolution is iterative — generate, tag, route, evolve, repeat — and the existing
pipeline is phase-based. Per rule 2 this is a **new** script, not an edit to a
running one: `scripts/gen_if_synthesis_test.sh` first, promoted only once it has
run clean.

Gotchas that apply directly:

- `--checkpoint-every-rows == --batch-size` (G7); checkpoints only happen between
  batches.
- `--limit-rows N` at a real output path **destroys that file** (G8). Carve a copy
  with `head_parquet.py`.
- Count empty `raw_response_*` after every generation phase (G2). On a thinking
  model a too-small `--max-tokens` returns `content: null` and loses the thinking
  with it, which is indistinguishable from a refusal.
- Each new task must return a non-empty expected-field list, or every row looks
  complete and the run is a silent no-op (G19).
- Adding a required input to an existing task adds a mandatory column everywhere
  that task runs (G21). Prefer new tasks over extending `verify_constraints`.

## 7. Row mapping

`synthesis/schema.py::Row` → the 19-column pool schema:

| `Row` | Pool column | Note |
|---|---|---|
| `id` | `id` | `"<source>::<key>"`; uniqueness enforced by `dedupe_pool_ids.py` |
| `language` | `language` | ISO code |
| `prompt` | `messages` (final user turn) → `dialogue_history` | must be a **rendered JSON string** (G3) |
| `prompt_wo_programmatic` | — | drop; Pharos's judge reads `dialogue_history`, and the constraint is in `expected_response` |
| `instruction_id_list` + `kwargs` | `checks` → `{"ifeval": {...}}` | JSON string |
| — | `expected_response` | human-readable constraint list; **must be non-empty** |
| `rubrics` | `rubrics` | phase 2+ only; `""` in phase 1 |
| `pass_rate` | `pass_rate` | `-1.0` for synthesised rows — unknown until the 8-generation pass. Never `None` (G10) |
| `round` | `source_meta.evolution_round` | provenance, kept per rule 4 |

`validate_row` from `synthesis/schema.py` ports as a pre-`make_row` gate. Its
checks that matter most in this framework: language in the verifiable set,
`instruction_id_list`/`kwargs` length agreement, no ineligible constraint, no
conflicting pair, and rubrics bound to their constraint rather than held in a
parallel list.

## 8. What to run first

One language, end to end, before anything else. **Greek** — it is the priority
language, it has the sharpest measured defects, and its script gate makes LID
failures impossible to confuse with checker failures.

Read the per-language pass-rate distribution before scaling. A language whose
distribution is crushed toward 0 or 1 relative to the others has an unsatisfiable
or vacuous constraint, which means a missing row in `profile.yaml` — far cheaper to
find at one language than at 22.
