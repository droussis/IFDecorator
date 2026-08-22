# Multilingual IF data synthesis — the complete pipeline

The IFDecorator flywheel, documented end to end and adapted for our case:
Latin/Cyrillic/Greek languages, rubrics instead of soft constraints, our own
LLM-as-Judge, and our own orchestration library.

Every stage below gives inputs, outputs, a minimal data example, the prompt
template, and the checks that gate it. Templates live in
[`synthesis/prompts.py`](../synthesis/prompts.py) as importable constants;
parsers in [`synthesis/parsing.py`](../synthesis/parsing.py); row schema and
validation in [`synthesis/schema.py`](../synthesis/schema.py).

Verification is [`multilingual_if/`](../multilingual_if/README.md) and is shared
with RL — that is the point of it. **The constraint set we sample from and the
constraint set we can score must be the same set**, or difficulty is measured
against one thing and rewarded against another.

---

## 0. What changed from upstream, and why

| Upstream | Ours | Reason |
|---|---|---|
| English-only (3 langdetect gates) | 24 languages, Latin/Cyrillic/Greek | the actual goal |
| soft constraints → LLM checklist | rubric constraints → our judge | we have a better judge and a rubric generator |
| `hard`/`soft` naming | `programmatic`/`rubric` | upstream's `hard_constraints_checklist` held the *judged* ones; the naming actively misleads |
| checklists zipped positionally | rubric carries its constraint | upstream mis-paired checklists whenever a soft constraint preceded a hard one |
| 2 LLM calls per merge | deterministic string comparison | largest avoidable cost in the flywheel |
| 1 classification call per constraint | 1 call per row | ~4× fewer calls at the same quality |
| judge includes a safety check | safety removed | it conflated refusal with instruction-following, so difficulty became partly a safety signal |
| `pass_rate` from unpatched checkers | from `multilingual_if` | otherwise the curriculum is corrupted before training starts |

Two conventions run through everything:

**Meta-prompts in English, content in the target language.** The generator is
instructed in English; the instruction it builds and the constraint text it emits
are in the target language. This decouples meta-task competence from
target-language competence and keeps parse markers stable across all 24
languages. It is a recorded decision — localizing the meta-prompts is a
legitimate alternative, but it changes what difficulty means and must be measured
before adopting.

**`language` is a required field on every row from stage 0 onward.** Not
optional, not inferred. A row without it falls back to English in verification,
which silently reintroduces every defect the multilingual work removed.

---

## 1. Pipeline shape

```
                    ┌──────────────────────────────────────────┐
                    │  seed collection        (§2)             │
                    │  v0_raw ──► v1_seed                      │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  W0  quality gate       (§3)             │
                    │  dedup, length, language, clarity        │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  W1  decompose          (§4)             │
                    │  task / constraints / input              │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  W2  classify           (§5)             │
                    │  programmatic | rubric                   │
                    └────────┬────────────────────┬────────────┘
                             ▼                    ▼
              ┌──────────────────────┐  ┌──────────────────────┐
              │ W3a programmatic     │  │ W3b rubric gen  (§6) │
              │ args from taxonomy   │  │ your generator       │
              └──────────┬───────────┘  └──────────┬───────────┘
                         └─────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │  W5  difficulty tagging (§8)             │
                    │  k rollouts ──► pass_rate ──► routing    │
                    └───────┬──────────────────────┬───────────┘
                            │ easy                 │ hard
                            ▼                      ▼
                    ┌───────────────┐      ┌──────────────┐
                    │ W4 evolve (§7)│      │ training set │
                    │ ──► back to W5│      │   (§9, §10)  │
                    └───────────────┘      └──────────────┘
```

Rounds 1–5 loop W4 → W5. **`pass_rate` is the difficulty label and it routes
every row**, so every verifier defect corrupts the curriculum before any training
happens, and W4 compounds it across five rounds.

---

## 2. Seed collection

**In:** raw instruction/response corpora, one file per source.
**Out:** `v1_seed/*.jsonl`, one row per task.

Upstream sampled 350k rows across 8 English sources with per-source quotas
(`data_preprocess.py:322`) and kept only the first turn of each conversation.
Keep that shape; change what is selected.

**Changes:**
- Drop the `langdetect(...) == "en"` filter (`data_preprocess.py:52`). Replace
  with `multilingual_if.lid.check_language(text, target)` against the language
  set — deterministic, unlike langdetect.
- Stamp `language` on every row here. This is the only stage where it is cheap.
- Fix the sampling: upstream uses `random.choices` (`data_preprocess.py:82`),
  which samples **with replacement**, so the seed set contains duplicates.

**Row out:**
```json
{"id": "oasst2_8814", "language": "de", "source": "oasst2",
 "prompt": "Erkläre mir, wie ein Verbrennungsmotor funktioniert.",
 "seed_response": "Ein Verbrennungsmotor wandelt chemische Energie …"}
```

**Checks:** language stamped and in `supported_languages()`; prompt non-empty;
no duplicate `(prompt, response)` pairs.

---

## 3. W0 — quality gate

**In:** `v1_seed`. **Out:** `v1_seed/high_quality.jsonl` + a stats file.

Four filters, in cost order — cheapest first, so the LLM call sees the fewest rows:

| # | Filter | Implementation | Note |
|---|---|---|---|
| 1 | language | `lid.check_language` | replaces the langdetect gate |
| 2 | dedup | SentenceBERT cosine > 0.9 | use a **multilingual** encoder (`paraphrase-multilingual-MiniLM-L12-v2`); upstream's `all-MiniLM-L6-v2` is English-only and will cluster unrelated non-English prompts |
| 3 | length | tokenizer, ≤ 8192 tokens | use the tokenizer of the model you will train — token counts per character differ sharply across languages |
| 4 | clarity | `prompts.QUALITY_GATE` | one LLM call |

Upstream split these across `w0.1` and `w0.2`, which are ~95% duplicated files
where each defines functions the other uses. Collapse them.

**Prompt:** `prompts.QUALITY_GATE` — clarity and language in one call.
**Parse:** `parsing.parse_verdict`.

> `modules/utils.py:36::unified_judge_parse` has **no `return` in its non-strict
> branch**, so the default path returns `None` and every loose judgement in the
> pipeline is falsy. Use `parsing.parse_verdict`.

**Checks:** log per-filter drop counts. A language whose drop rate is far from
the others means either a bad seed corpus or a mis-tuned dedup encoder.

---

## 4. W1 — decompose

**In:** high-quality seeds. **Out:** each row gains `decomposition`.

**Prompt:** `prompts.DECOMPOSE` — few-shot examples rendered in the target
language, output markers fixed in English, and an explicit instruction **not to
translate** the extracted text. Without that, models translate constraints into
English and silently break the link to the instruction they came from.

**Parse:** `parsing.parse_decomposition`.

> Upstream's parser (`w1_decompose.py:83`) appends *any* line starting with `-`
> to `constraints` once a task description has been seen — including bullets
> under `#Input:`. Ours tracks section membership.

**Row gains:**
```json
{"decomposition": {
   "task_description": "Erkläre die Funktionsweise eines Verbrennungsmotors.",
   "constraints": ["Verwende höchstens 200 Wörter",
                   "Schreibe in einem sachlichen Ton"],
   "input_text": null}}
```

**Checks:** constraints in the target language, not English (`lid` on the joined
constraint text catches a translating generator immediately); task description
non-empty.

---

## 5. W2 — classify constraints

**In:** decomposed rows. **Out:** a `programmatic | rubric` label per constraint.

**Prompt:** `prompts.CLASSIFY_CONSTRAINTS`. Two changes from upstream: the whole
list is classified in **one** call (upstream looped per constraint,
`w2_classify_constraints.py:108`), and "programmatic" is grounded in a concrete
capability list rather than left to the model's imagination — a constraint
classified programmatic that no checker implements is dead weight.

**Parse:** `parsing.parse_classification` — unparseable ⇒ `rubric`. That
direction is deliberate: a constraint wrongly marked programmatic becomes an
unenforceable reward signal; wrongly marked rubric merely costs a judge call.

**Checks:** the programmatic/rubric ratio per language. A language skewing hard
toward `rubric` usually means the decomposition is producing vaguer constraints
there, which is worth fixing upstream of here rather than accepting.

---

## 6. W3 — the two branches

### 6a. Programmatic constraints

Not generated by a model — **sampled from the verified taxonomy**, so everything
sampled is scoreable by construction.

```python
from multilingual_if import eligible_ids, localized_kwargs, alphabet, conflicts

ids = eligible_ids(row.language)                      # 53 of 54
kwargs = localized_kwargs(instruction_id, row.language)
letter = rng.choice(alphabet(row.language))           # never a-z for Cyrillic/Greek
```

Rules:
- Sample only from `eligible_ids(lang)`.
- Honour `INSTRUCTION_CONFLICTS` **plus** `conflicts()` — the upstream table is
  English-shaped and does not know that `response_language` and
  `constrained_response` cannot coexist.
- Draw literals from the language profile: postscript marker, section word, quote
  pair, letter inventory.
- `startend:end_checker` needs its phrase authored in the target language. There
  is no sensible default, and an English closing phrase contradicts a
  `response_language` constraint on the same prompt.
- **Length constraints:** calibrate with `count_words`, never `len()`.

> `w4_evol.py:591` computes the word threshold as
> `len(seed_data["response"]) * uniform(0.5, 1.2)` — `len()` of a *string* is a
> **character** count used as a **word** threshold. It is ~4–5× too large in
> English, and chars-per-word ranges 5.1 (en) to 8.0 (fi), so uncorrected it makes
> difficulty correlate with *language* rather than with the instruction.

Constraint text is appended to the prompt; `prompt_wo_programmatic` keeps the
version without it, and **that** is what the judge sees. Otherwise programmatic
constraints are scored twice.

### 6b. Rubric constraints — the seam

This is where your generator and judge replace upstream's checklist step.

`prompts.RUBRIC_GENERATION` and `prompts.JUDGE_RUBRIC` are **reference only** —
present so the pipeline runs end to end without your components, and so the
contract is written down. The contract is:

```python
Rubric(constraint="Schreibe in einem sachlichen Ton",
       criteria=["Does the response avoid exclamations and rhetorical questions?",
                 "Is the register consistent throughout?"],
       pass_rule="ALL", judge="our-judge-v2", weight=1.0)
```

What the pipeline needs back from your judge is **one boolean per rubric**, to
fold into `pass_rate`. Everything else is yours.

> **Do not reintroduce a positional zip.** Upstream generated checklists only for
> `hard`-typed constraints but zipped them against the *full* constraint list
> (`check_instruction.py:136`, `recipe/reward/cif.py:226`), so any soft constraint
> appearing earlier shifted every checklist onto the wrong constraint. Keeping the
> constraint text inside the rubric object makes that class of bug impossible.

**Decision to record:** is the judge prompt in English while content is in the
target language, or both localized? We default to the former, consistent with the
rest of the pipeline. Make it explicit, because it changes what difficulty means.

---

## 7. W4 — evolution

**In:** the previous round's `easy_pool`. **Out:** evolved instructions.

Two phases, as upstream.

**(a) Add a rubric constraint.** `prompts.render_evolution(instruction, language,
distribution, rng)` — assembled per call, with categories, types and rules
shuffled, optionally weighted by inverse observed frequency. The randomization is
load-bearing: a fixed template collapses the distribution onto whichever category
the model prefers. Pass running counts from previous rounds to actively correct
an imbalance.

Attempts scale with the round: `randint(1 + n//2, n)`, so later rounds stack more
constraints — that is the difficulty ramp.

**Parse:** `parsing.parse_evolution`.

> Upstream (`w4_evol.py:278`) splits the whole response on `#`, so any `#` inside
> the generated instruction — a markdown heading, a hex colour — corrupts every
> subsequent field. Ours anchors on line-initial field markers.

**Merge:** `matching.merge_evolved` — replaces upstream's two LLM containment
calls per attempt with NFC-normalized comparison. Use
`matching.verdict()` when you want the three-way answer and only call
`prompts.CONTAINMENT_CHECK` for the ambiguous band.

**(b) Add programmatic constraints.** `randint(n, 3n)` drawn as in §6a.

Then re-gate with `prompts.QUALITY_GATE`. Upstream split the result into
`reasonable`/`unreasonable` and fed **both** back into W5 (the latter via
`--input_file_extra`) — keep that; the unreasonable pool is recall, not garbage.

**Checks:** `validate_row` after every evolution; track the rate of
`merge_evolved`'s third branch (instruction dropped) — a high rate means the
evolution prompt is failing, not that the data is hard.

---

## 8. W5 — difficulty tagging

**In:** W3/W4 output. **Out:** `pass_rate` per row, routed into pools.

For each row, k rollouts (upstream: 8) at temperature 1.0 from the policy model,
each judged by:

1. **Programmatic** — `multilingual_if.verify(ids, kwargs, response, lang)`.
2. **Intent** — `prompts.JUDGE_INTENT` against `prompt_wo_programmatic`.
3. **Rubrics** — your judge, one verdict per rubric.

A rollout passes when all three pass. `pass_rate = passes / k`.

**Routing** — `schema.route_by_difficulty`:

| pass_rate | pool | fate |
|---|---|---|
| `> 0.5` | easy | back to W4 |
| `(0, 0.5]` | hard | **the training set** |
| `= 0` | too_hard | discarded |

We discard only rows nothing ever solved, rather than upstream's `< 0.032`
senior-model threshold (disabled in the shipped script anyway). A pass_rate of
exactly 0 is as likely to mean "unsatisfiable constraint" as "genuinely hard" —
the profile exists to make the former impossible, and `pass_rate_report` is how
you confirm it.

**Checks — this is the most important instrumentation in the pipeline:**

```python
from synthesis.schema import pass_rate_report
report = pass_rate_report(tagged_rows)
```

Compare `frac_zero` and `frac_one` across languages. A language with an outlying
`frac_zero` has an unsatisfiable constraint; an outlying `frac_one` has a vacuous
one. Either means a missing row in `profile.yaml`. **Check this before scaling
past one language** — it is far cheaper than discovering it after five rounds.

Also keep a `chosen`/`rejected` rollout per row as upstream does; they cost
nothing extra and give you SFT/DPO data alongside the RL set.

---

## 9. Postprocess

1. **Gather** `hard_pool` across rounds 0–5, stamping `round`.
2. **Domain filter** — `prompts.DOMAIN_FILTER`, keep only `Other`. Math/Code/
   Reasoning carry a correctness signal that competes with the formatting signal
   being trained. Classify from the instruction alone; upstream generated a fresh
   response first, doubling the cost.
3. **Convert** to the training format.

> Upstream's `run_postprocess.sh` runs `post_filter.py` **before**
> `trans2verl_format.py`, but `post_filter` reads
> `reward_model.ground_truth.complex_instruction_soft` — a field that only exists
> *after* conversion. The documented order cannot work.

Hold out the test split **by seed id**, not by row: after five evolution rounds
many rows share a seed, and splitting by row leaks near-duplicates across the
boundary. Upstream took the last 512 rows (`trans2verl_format.py:13`).

---

## 10. Handoff to RL

Rows carry everything the Gym server needs:

```json
{"id": "oasst2_8814_r3", "language": "de",
 "prompt": "Erkläre … Verwende höchstens 200 Wörter. Schreibe alles in Kleinbuchstaben.",
 "prompt_wo_programmatic": "Erkläre … Verwende höchstens 200 Wörter.",
 "instruction_id_list": ["change_case:english_lowercase"],
 "kwargs": [{}],
 "rubrics": [{"constraint": "Schreibe in einem sachlichen Ton", "criteria": [...],
              "pass_rule": "ALL", "judge": "our-judge-v2", "weight": 1.0}],
 "pass_rate": 0.25, "round": 3}
```

The Gym server
(`resources_servers/multilingual_instruction_following`) reads `language`,
`instruction_id_list` and `kwargs`. Rubrics are scored by your judge and combined
outside it.

`strict_eligibility` defaults to raising rather than scoring 0, so a constraint
that escaped the synthesis gate surfaces as a data bug rather than as a model
failure. Keep it on for data you generate.

---

## 11. Order of operations

1. Land the verifier (done — `multilingual_if`).
2. Stamp `language` at seed collection; swap the three langdetect gates.
3. Gate W4 sampling on `eligible_ids(lang)`; extend the conflict table.
4. Fix the length calibration (§6a).
5. Wire your rubric generator and judge into the §6b seam.
6. **Run one round end-to-end on a single non-English language.** Inspect
   rollouts and `pass_rate_report` before touching the other 23.
7. Scale.

Green unit tests are not sufficient here — the failure mode is a plausible-looking
row with a silently wrong constraint, and only real rollouts surface it.

---

## 12. Inherited breakage

Unrelated to language; these stop the pipeline before any multilingual concern does.

| Where | What |
|---|---|
| everywhere | `from logs.logger import logger` — there is no `logs/` package in the repo |
| `modules/utils.py:36` | `unified_judge_parse` returns `None` on its default path |
| `modules/preprocess/run_preprocess.sh:60` | invokes `modules.data_fetch.data_preprocess`; the module is `modules.preprocess.data_preprocess` |
| `modules/postprocess/gather.py`, `trans2verl_format.py` | import `config` and `utils`, neither of which exists here; hardcode a `/cpfs01/...` path |
| `modules/postprocess/run_postprocess.sh` | stage order cannot work (§9) |
| `modules/preprocess/data_preprocess.py:82` | `random.choices` samples with replacement |
| `modules/enhance/w4_evol.py:532` | branches on `xml_format`, which is not in the taxonomy (dead, harmless) |
| `modules/enhance/w0.1`, `w0.2` | ~95% duplicated; each defines functions only the other uses |
