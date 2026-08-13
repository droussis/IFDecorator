# Multilingual data synthesis with the IFDecorator flywheel

How to run the IFDecorator pipeline for Latin/Cyrillic/Greek languages, and where
it changes shape when soft constraints become LLM-judged rubrics.

Companion to [`multilingual_if/README.md`](../multilingual_if/README.md), which
covers the verifier itself. This document covers the *pipeline*.

---

## 1. The pipeline as it stands

```
preprocess/          v0_raw ──► v1_seed          seed collection, English-only
enhance/w0.1, w0.2   v1_seed ──► high_quality    langdetect + dedup + length + LLM gate
enhance/w1           ──► v2_decomposed           task / constraints / input
enhance/w2           ──► v3_classified           each constraint tagged hard | soft
enhance/w3           ──► v4_checklist            checklist per hard-typed constraint
enhance/w5           ──► v5_difficult            k rollouts, pass_rate, easy/hard split
enhance/w4           ──► round_n                 evolve easy pool, add constraints
postprocess/         ──► verl_format             gather, filter, convert
```

Rounds 1–5 loop w4 → w5. `pass_rate` is the difficulty label, and it routes every
row: `> threshold_easy` goes back to w4 to be made harder, otherwise it lands in
`hard_pool`, which becomes the training set.

**The single most important consequence:** `pass_rate` is computed with the same
checkers used at RL time. Every verifier defect corrupts the *curriculum* before
any training happens, and w4 compounds it across five rounds. An unsatisfiable
constraint pushes a row into `hard_pool` for the wrong reason; a vacuous one
pushes it into `easy_pool`, where w4 evolves it further. Land the verifier patches
before generating data.

---

## 2. Changes required for multilingual synthesis

### 2.1 Remove the English-only gates

Three independent `langdetect(...) == "en"` filters drop everything else:

| Location | What it does |
|---|---|
| `modules/preprocess/data_preprocess.py:52` | drops rows whose prompt or response is not English |
| `modules/enhance/w0.1_quality_filter.py:41` | same, on the prompt |
| `modules/utils.py:283` (`is_en_lang`) | shared helper |

Replace with a check against the target language set, using
`multilingual_if.lid.check_language` rather than langdetect — it is deterministic,
which matters here for the same reason it matters at RL time.

### 2.2 Gate constraint sampling on eligibility

`w4_evol.py:770` samples uniformly over the whole taxonomy:

```python
hard_keys = [key for key in cons_dict.keys()]
```

That is where an ineligible constraint enters the data. Replace with:

```python
from multilingual_if import eligible_ids, conflicts
hard_keys = [k for k in cons_dict if k in set(eligible_ids(row_lang))]
```

and extend `INSTRUCTION_CONFLICTS` with `conflicts()` — the upstream conflict table
is English-shaped and does not know that, for example, `response_language` and
`constrained_response` cannot coexist.

Note `all_capital` × `response_language` is deliberately **not** a conflict:
lowercasing before LID removes the interaction. Without that step an all-caps
French answer scores `ca @ 0.170` and fails a constraint it satisfied.

### 2.3 Fill args from the language profile

The `localize` rows have correct semantics and English defaults:

```python
from multilingual_if import localized_kwargs, alphabet, quote_pair

kwargs.update(localized_kwargs(inst_id, row_lang))   # postscript marker, section word
letter = random.choice(alphabet(row_lang))           # never a-z for Cyrillic/Greek
```

`startend:end_checker` needs its phrase authored in the target language — there is
no sensible default, and an English closing phrase contradicts a
`response_language` constraint on the same prompt.

### 2.4 Fix the length calibration

`w4_evol.py:591` and `:602`:

```python
least_num_words = len(seed_data["response"]) * random.uniform(0.5, 1.2) // 50 * 50
```

`len()` of a string is a **character** count being used as a **word** threshold.
This is a pre-existing English bug (~4–5× too large), but multilingually it is
worse: chars-per-word runs 5.1 (en) to 8.0 (fi), so the same code produces a
systematically harder constraint for agglutinative languages. Difficulty would
then correlate with *language* rather than with the instruction, which corrupts
the w5 routing.

```python
from multilingual_if import count_words, chars_per_word
base = count_words(seed_data["response"], row_lang)
```

### 2.5 Keyword sourcing

`w4_evol.get_keyword()` prompts an LLM and falls back to
`instructions_util.WORD_LIST`, an English word list. Supply per-language fallbacks.

Given that stemming was dropped (see G7 in the verifier README), also prefer
sampling keywords that already appear as surface forms in the seed response —
matching is exact, so a keyword the model must inflect will not be found.

---

## 3. Soft constraints become rubrics

This is the structural change to the pipeline.

### 3.1 What the current path does

`w2` tags each extracted constraint `hard` or `soft`. `w3` then builds a checklist
for the `hard`-typed ones, and `check_instruction.test_instruction_following_llm`
judges with a two-level prompt: an overall judge, then a per-constraint checklist
judge. Both parse `**Final Verification:** <YES/NO>`.

Two things to know before replacing it:

1. **There is an off-by-position bug in the existing path.** `w3` appends
   checklists only for `hard`-typed constraints, but both judges zip against the
   *full* constraints list:
   ```python
   zip(list_constraint, list_of_checklist)   # check_instruction.py:136, cif.py:226
   ```
   Any soft constraint appearing before a hard one shifts every checklist onto the
   wrong constraint. If you are replacing this path wholesale, the bug leaves with
   it — but do not port the zip.

2. **The naming is inverted from what you would expect.** `hard_constraints_checklist`
   holds checklists for LLM-judged natural-language constraints. The
   programmatically verifiable IFEval-style ones live in `instruction_id_list` /
   `kwargs`. Do not let the field name mislead the rubric mapping.

### 3.2 The seam

Replace `w2`'s soft branch and all of `w3` with rubric generation. The hard branch
— `instruction_id_list` and `kwargs` — is untouched and continues to be verified
programmatically by `multilingual_if`.

```
w1 decompose ──► constraints[]
                    │
      ┌─────────────┴─────────────┐
      │                           │
  programmatic                  rubric
  (IFEval-G ids)                (your generator)
      │                           │
  multilingual_if.verify      your LLM-as-Judge
      │                           │
      └─────────────┬─────────────┘
                    ▼
              combined reward
```

What w5 needs from the rubric side is only a boolean (or a score plus a threshold)
per row, to fold into `pass_rate`. Everything else is yours.

### 3.3 Row schema

Keep the programmatic fields exactly as they are — the Gym resources server and
the VERL converter both read them — and add the rubric block alongside:

```jsonc
{
  "prompt": "…",
  "prompt_wo_hard_constraints": "…",   // what the judge sees
  "language": "el",                    // NEW: required, drives eligibility + verify
  "instruction_id_list": ["length_constraints:number_words"],
  "kwargs": [{"relation": "at least", "num_words": 120}],
  "rubrics": [                         // NEW: replaces formatted_ins.hard_constraints_checklist
    {
      "constraint": "Adopt the persona of a museum curator",
      "rubric": "…your generated rubric…",
      "judge": "your-judge-id",
      "weight": 1.0
    }
  ]
}
```

`language` is not optional. Both `eligible_ids()` and `verify()` need it, and a
row without it silently falls back to `en`, which reintroduces exactly the
defects this work removes.

### 3.4 Judging in the target language

The existing judge prompts (`check_instruction.py:13`, `:52`) are English and
close with `**Final Verification:** <YES/NO>` — note the instruction "conclude
with final verdict using strict formatting **in English**" in the checklist
prompt. Whatever your rubric judge does, decide explicitly whether the *judge
prompt* is in English while the *content* is in the target language, or both are
localized. This changes what difficulty means, so make it a recorded decision
rather than an accident.

---

## 4. Order of operations

1. Land the verifier patches (`multilingual_if`) — before generating anything.
2. Remove the three English gates; add `language` to every row at preprocess time.
3. Gate w4 sampling on `eligible_ids(lang)`; extend the conflict table.
4. Fix the length calibration.
5. Swap the soft-constraint branch for rubric generation.
6. Run one round end-to-end on a **single** non-English language and inspect
   rollouts before scaling to the full set. `pass_rate` distributions per language
   are the diagnostic: if one language's distribution is shifted hard toward 0 or
   1, a constraint is unsatisfiable or vacuous there, and the profile needs a row.

Per the repo quality bar: green unit tests are not enough for environment or
agent changes — run real rollouts and inspect agent and verifier behaviour.

---

## 5. Known-broken things in this repo, unrelated to language

These will stop the pipeline before any multilingual concern does:

| Where | What |
|---|---|
| everywhere | `from logs.logger import logger` — there is no `logs/` package in the repo |
| `modules/utils.py:36` | `unified_judge_parse` has no `return` in the non-strict branch, so the default path returns `None`. Every loose judgement is falsy. The correct implementation exists in `recipe/reward/cif.py:103` |
| `modules/preprocess/run_preprocess.sh:60` | invokes `modules.data_fetch.data_preprocess`; the module is `modules.preprocess.data_preprocess` |
| `modules/postprocess/gather.py`, `trans2verl_format.py` | import `config` and `utils`, neither of which exists here, and hardcode a `/cpfs01/...` cluster path |
| `modules/postprocess/run_postprocess.sh` | runs `post_filter.py` before `trans2verl_format.py`, but `post_filter` reads `reward_model.ground_truth.complex_instruction_soft`, which only exists *after* conversion |
| `modules/preprocess/data_preprocess.py:82` | `random.choices` samples with replacement, so the seed set can contain duplicates |
