# Plan 05 — regenerating existing pools with a newer teacher

Existing frozen pools were generated and judged with DS V4 Flash 0731. The rubric
lists in those pools were produced earlier still, by an older DeepSeek Flash V4.
This plan covers swapping in a newer teacher without invalidating data silently.

The framework already supports regeneration. What it does not do is tell you
**which derived columns a teacher change invalidates**, and several of them fail in
ways that look like a model result rather than a stale artefact.

## 1. What a teacher change invalidates

| Artefact | Produced by | Survives a teacher swap? |
|---|---|---|
| `response_student_*` / `raw_response_student_*` | the student | **yes** — unaffected |
| `response_teacher_*` / `raw_response_teacher_*` | the teacher | **no** — this is the point of the exercise |
| `verification_grade*` | the judge, which **is the teacher model** on the verifiable tracks | **no** — every 1–10 grade was produced by the old model |
| `check_results_*` | programmatic verifiers | **no**, but only because they are computed from the teacher's response; the verifier itself is unchanged |
| `rubric_grade_*` / `score_turn<N>` | the separately finetuned Qwen3.6-35B-A3B rubric judge | **yes** — a teacher swap does not touch the rubric judge |
| `rubrics` / `weighted_rubrics_turn<N>` | an older creator model | **yes technically, but see §4** |
| in-house pass-rate-out-of-8 | 2 teacher + 6 student generations | **no** — 2 of the 8 change, so the difficulty label moves |

The asymmetry worth internalising: **the rubric track is largely insulated from a
teacher change and the verifiable tracks are not**, because on the verifiable
tracks the teacher and the judge are the same model.

## 2. Column strategy

Regenerate into a **new suffix**, never over the old one.

```
response_teacher_1        → response_teacher_v5_1
verification_grade_teacher_1 → verification_grade_teacher_v5_1
```

Three reasons, all of them rules rather than preferences:

- "Never silently drop dataset columns… on a name collision, rename the incoming
  column — never overwrite."
- Keeping both lets the teacher swap be **measured** rather than assumed. The
  interesting number is not the new pass rate, it is the per-row delta.
- Resumption keys on non-empty output fields. Reusing a suffix makes every row
  look complete and the run a **silent no-op** unless `--force-regenerate` is
  passed — the same failure shape as G19.

If disk pressure makes carrying both copies impossible, drop the old columns in a
separate, explicit step after the comparison, and record it in `AGENTS.md` as a
schema change.

## 3. Order of operations

1. **Snapshot.** Note the pool file, its row count and the column list. `G22`: the
   generation wrapper reuses an existing output copy (`[[ -f $out ]] || cp`), so a
   stale pre-copy silently pins old pool content — delete matching generated files
   after any pool rebuild.
2. **Teacher rollouts** into the new suffix, `--save-raw-response`.
3. **Count empty `raw_response_*`.** A too-small `--max-tokens` on a thinking model
   returns `content: null` and loses the thinking with it (G2), which is
   indistinguishable from a refusal. Do this before anything downstream consumes
   the column.
4. **`split_raw_responses.py`** if only raw exists.
5. **Programmatic checks** — `run_verifiers.py --labels <new label>`.
6. **Judging** with the new judge, per-track subsets so the 5th-input asymmetry
   does not hard-error (G21).
7. **Recompute the in-house pass rate** over the new 2+6 mix.
8. **Compare, then decide.** Not before.

## 4. The rubric trap

Rubric *grades* survive a teacher change. Rubric *lists* are a different matter,
and regenerating them is far more expensive than it looks.

Adding or changing criteria changes `sum(weight)`, and the reward is
`sum(weight where satisfied == "YES") / sum(weight)`. So:

- **every previously computed `score_turn<N>` on that row becomes invalid**, and
- a stored `satisfied` array is then length-mismatched against the new rubric list,
  which the merge treats as **missing, not zero** — the row silently keeps its
  `-1.0` sentinel and drops out of every mean.

There is also no merge plumbing for `rubric_calibration` output: nothing folds
`additional_weighted_rubrics` back into `weighted_rubrics_turn<N>` or recomputes
`total_weight`. Anyone regenerating rubrics writes that step themselves.

**Recommendation: do not regenerate rubric lists as part of a teacher swap.** Treat
it as a separate project with its own re-grading budget. The one argument for
doing it is that existing rubrics encode the failure modes of a previous-generation
teacher, which may not be the current student's failure modes — that is a real
concern, but it is a rubric-quality question, not a teacher-swap question.

## 5. What to measure

The point of keeping both suffixes is the comparison. Report:

- **Per-row grade delta**, not just the two means. A teacher swap that moves the
  mean by 0.02 while moving individual rows by ±0.4 has changed the ranking, and
  the mean hides it.
- **Pass-rate migration across the difficulty bands.** Rows leaving `[0.1, 0.9]`
  in either direction stop being useful for GRPO — a prompt every rollout solves,
  or none does, gives zero advantage.
- **Empty-response rate** per label, old vs new. A newer model with different
  thinking behaviour can silently need a larger `--max-tokens`.
- **Refusal and language-drift rate**, if the new teacher is not the same family.
  Wrong-language output is capped at 5 by the judge ladder, so it shows up as a
  grade shift rather than as an obvious failure.

Report to `reports/`, not to a terminal.

## 6. Regeneration provenance

The flat-column path records **no** model provenance: `verification_grade_teacher_1`
does not say which model produced it. Only `--generations-field` carries
`model, temperature, top_p, backend, system_prompt`, and the judging phases do not
use it.

After two or three teacher generations, nothing in the file says which column came
from which model. Suffix naming is the current answer and it is a convention, not
a guarantee.

**Cheap fix, worth doing before the first swap:** write a sidecar manifest per pool
recording, per suffix, the model name, sampling params, system prompt key, and the
date of the run. A JSON file beside the parquet costs nothing and makes the second
swap tractable. Alternatively, adopt `--generations-field` for teacher rollouts and
accept that it leaves the column fast path.

## 7. Regenerating rows that we synthesised

For pools produced by Plan 02, one extra step. Synthesised rows carry
`pass_rate = -1.0` until a generation pass measures them, and their difficulty
routing depends on that measurement. A teacher swap therefore re-opens the routing
decision for exactly those rows.

Re-run the routing diagnostic per language after regeneration. A language whose
distribution shifts much more than the others under a teacher swap is worth
inspecting: it usually means the constraint mix in that language was marginal
already.
