# Plan 01 — programmatic instruction-following verification

Adds an IFEval-style checker as a fourth programmatic verifier alongside
`verify_schema` and `verify_string_match`, corrected for the 30 target languages.

## 0. Status: this is scope EXTENSION, not gap-fill

`AGENT_RUNBOOK.md` §1.3 decision 7 excludes IFEval, InverseIFEval and
MultiChallenge, and §1.3 decisions are marked "do not relitigate". **Nothing here
should be built before explicit sign-off** (`DECISIONS.md` D1). The exclusion's
stated reason — "need framework changes (tools in the payload) or have
rubric-shaped goldens" — does not apply to this proposal: it needs no framework
change and no new pool column.

## 1. What this reuses

`multilingual_if/` in the IFDecorator fork: the 54-instruction IFEval-G registry
with eight corrections for Latin/Cyrillic/Greek, `profile.yaml` as the
per-constraint eligibility matrix, calibrated fastText LID, and 62 differential
tests. Every correction is backed by a measurement recorded in `profile.yaml`.

The corrections matter here because §13.1 already anticipates them: *"Most IFEval
checkers are English-centric… a naive tokeniser or `.istitle()` will emit false
FAILs that the judge then weighs as evidence."* That is exactly what was measured:

| Defect | Effect in a non-English pool |
|---|---|
| `count_words` uses `\w+`, which excludes combining marks | the same Polish sentence counts 14 or 18 words depending on Unicode normalisation form |
| `[a-zA-Z]` character classes | match **zero** characters in Greek and Bulgarian, so "at most N lowercase words" passes for free |
| `count_sentences` loads `punkt/english.pickle` | a three-question Greek paragraph counts as one sentence — Greek marks questions with `;` |
| `change_case:english_lowercase` conjoins `langdetect == "en"` | unsatisfiable outside English, though `islower()` is already Unicode-correct |
| language checks return `True` on `LangDetectException` | an emoji-only response satisfies "respond entirely in Bulgarian" |

A silent false FAIL is worse here than elsewhere, because §8's contract routes it
into `{check_results}` as *evidence*, and the judge is instructed to weigh it.

## 2. Language coverage — the first thing to decide

`build_pool.py::TARGET_LANGUAGES` has 30 entries. `multilingual_if` verifies 24,
of which 22 are in the target set.

**Covered (22):** `en el bg cs da de et fi fr hu it lt lv nl pl pt ro sk es sv uk mt`

Greek, the priority language, is covered and is the strongest case in the set: it
is the only target using the Greek script, so a Unicode script gate resolves it
outright without the LID model.

**Gap (8):** `hr is nb sl sr tr mk sq`

| Language | Status |
|---|---|
| `hr`, `sl`, `sr` | **measured unreliable.** lid.176 carries a separate `sh` (Serbo-Croatian) label, so hr/bs/sr/sh split the probability mass and none clears a usable margin; Slovene is misread as Serbian on generic prose |
| `nb` | **label mismatch** — lid.176 emits `no` for Bokmål, never `nb`. A straight pass-through of the pool's language code fails every row |
| `tr` | untested, and the one language where Python's default casing is known wrong (dotted/dotless I), which affects every case-based check |
| `is`, `mk`, `sq` | never tested. `mk` is Cyrillic and shares a confusable group with `bg` |

Three options, in `DECISIONS.md` D2. The default recommendation is to ship the 22
and emit `not_applicable` for the rest, which is safe under the §8.4 contract and
degrades to "no evidence" rather than to wrong evidence.

## 3. The contract adaptation — the one thing that must not be copied verbatim

`multilingual_if.verify()` returns `{"reward": float, …}`. **The §8.4 contract
forbids that**: *"Never raise, never rewrite the response, never return a reward."*
`verify_constraints.txt` states the same invariant to the judge.

The port therefore exposes evidence only. `verify()` and `verify_one()` are not
re-exported; the reward path stays behind in the RL-side repo where it belongs.

```python
verify_instructions(response, instruction_ids, kwargs, language) -> {
    "check": "ifeval",
    "language": str,
    "results": [
        {"instruction_id": str,
         "status": "pass" | "fail" | "not_applicable",
         "detail": str},
        ...
    ],
    "n_applicable": int,
    "passed": bool,        # all APPLICABLE instructions passed; False when none apply
    "errors": [str],
}
```

Three rules follow from the contract and from §13.1:

1. **Ineligible constraints report `not_applicable`, never `fail`.** §13.1 asks for
   exactly this — *"Prefer emitting `not applicable in <language>` over
   `passed: False`"* — and `profile.yaml` already decides it per
   (constraint, language) from measurement rather than from a guess.
2. **Never raise.** Every failure becomes an `errors` entry. `multilingual_if`
   already fails closed; the wrapper adds a blanket guard so a malformed `kwargs`
   entry cannot take down a batch.
3. **Ship `render(result) -> str`**, one bullet per instruction. Bullets are what
   the judge reads.

Rendered form, matching the existing verifiers' style:

```
- instruction-following checks (el): 2/3 passed
- length_constraints:number_words: PASSED
- language:response_language: PASSED
- change_case:english_capital: NOT APPLICABLE (no cased script in this language)
```

`strip_fence()` is reused from `schema_formats.py` rather than reimplemented: an
unrequested code fence is already an instruction-following failure and the judge
prompt caps it at 6.

## 4. Files

New, under `data_team/generation_multilingual/verifiers/`:

```
instruction_following/
├── __init__.py            # exports verify_instructions, render
├── adapter.py             # the §8.4 evidence wrapper
├── checks.py              # vendored from multilingual_if, reward path removed
├── eligibility.py         # vendored
├── textops.py             # vendored
├── lid.py                 # vendored
├── setup_lid.py           # vendored
├── profile.yaml           # vendored — the eligibility matrix
└── PROVENANCE.md          # source, commit, re-sync recipe
```

A subpackage rather than a flat module: `verifiers/` is currently three files of
~230 lines total, and this is larger than all of them together. Keeping it in one
directory leaves `verifiers/` readable and makes the vendored boundary obvious.

Changed:

| File | Change |
|---|---|
| `verifiers/__init__.py` | import `verify_instructions`, extend `__all__` — this hand-maintained list is the registry |
| `acquisition/run_verifiers.py` | `if "ifeval" in checks:` branch; append `ifeval:pass` / `ifeval:fail` to `kinds` |
| `acquisition/adapters.py` | `adapt_<source>` emitting the new `verification_type` and `checks` |
| `acquisition/build_pool.py` | `source_specs()` entry + `POOL_CAPS` |
| both `AGENTS.md` | same session, per rule 9 |

No new pool column. No change to `generate.py`. No change to any judge template.

## 5. Pool row shape

```jsonc
{
  "track": "constraint",
  "verification_type": "instruction_following",
  "task_family": "instruction_following",
  "language": "el",
  "expected_response": "[\"Respond in Greek only\", \"Use at least 120 words\"]",
  "checks": "{\"ifeval\": {\"instruction_ids\": [\"language:response_language\", \"length_constraints:number_words\"], \"kwargs\": [{\"language\": \"el\"}, {\"relation\": \"at least\", \"num_words\": 120}], \"language\": \"el\"}}",
  "verification_meta": "{\"answer_format\": \"…derived…\", \"verifier_type\": \"instruction_following\", \"n_instructions\": 2}",
  "translatable": "0"
}
```

Points that follow from §7 and the runbook:

- **`expected_response` must be non-empty** or `make_row` drops the row. It carries
  the human-readable constraint list — that is what the judge sees in
  `<required-constraint>` — mirroring citation's `["[ref:3]"]`.
- **`checks` carries `language`.** Unlike `schema`, whose `schema_str` is folded in
  from `expected_response` at run time, the instruction kwargs are not
  reconstructible from anything else, so they live in `checks`.
- **`verification_meta.answer_format` is derived** from the instruction spec, never
  hardcoded — G18's lesson, which was wrong on 64% of science rows.
- **`translatable` is `"0"`** (string, per G10). Constraint-track rows are never
  translated anyway per §1.3, and instruction kwargs are language-bound.
- **`track` stays `constraint`.** `verify_constraints` already scores both
  "does the response satisfy the constraint" and "is it also a good answer" — the
  compliant-but-vacuous failure mode, capped at 3, which is precisely the IFEval
  gaming pattern.

## 6. Dependencies

| Package | Status |
|---|---|
| `regex` | **new**; needed for `\p{Lu}`-class Unicode properties |
| `fasttext` | already used for language detection |
| `numpy<2` | **constraint, not a new package.** `fasttext` 0.9.3 calls `np.array(probs, copy=False)`, which raises on numpy 2 — at `predict()` time, not import, so an unpinned environment fails mid-run |
| `PyYAML` | already a verifier dependency; `profile.yaml` needs it |

The fastText LID model (`lid.176.ftz`, ~938 KB) is **CC-BY-SA and must not be
committed** to an Apache-licensed repo. `setup_lid.py` resolves it from an env var,
a local copy, the `ftlid` PyPI package (which vendors it verbatim), or a direct
download — in that order. The `ftlid` path matters where `dl.fbaipublicfiles.com`
is blocked by an egress proxy.

## 7. The blocker that decides whether any of this runs

§14.4: the production judging phase maps `check_results:checks` — the raw check
*specs*, not executed *results* — and never invokes `run_verifiers.py`. The full
`generate → split_raw_responses → run_verifiers → judge` chain exists only in
`smoke_test_verifiable.sh`.

**If that stays, this verifier is inert in production**: the judge would be told
which instructions *would have been* checked, never what was found. Everything in
this plan is cheap; that decision is what determines whether it is worth doing at
all. `DECISIONS.md` D3.

## 8. Testing

Port the 62 differential tests. They are not smoke tests — each pins a defect that
unpatched IFEval-G gets wrong — and three of them caught real bugs during
authoring. Add on top:

- Greek-priority cases for every eligible instruction, since Greek is the priority
  language and has the sharpest measured defects (the `;` question mark, the
  script gate).
- One case per gap language asserting `not_applicable`, so a later coverage
  extension has to update the test deliberately.
- A `render()` golden per status, since its output is judge-visible text and feeds
  judge-SFT data downstream. Determinism matters here for the same reason §13.2
  raises it for test execution.
- Smoke against 20 rows of a real pool with a throwaway output path, per rule 7.

## 9. Language coverage table

Generated from `multilingual_if.supported_languages()` against
`build_pool.py::TARGET_LANGUAGES`.

| | Languages |
|---|---|
| **Verifiable (22)** | `en el bg cs da de et fi fr hu it lt lv nl pl pt ro sk es sv uk mt` |
| **Gap — measured unreliable (3)** | `hr sl sr` |
| **Gap — never tested (5)** | `is nb tr mk sq` |
| Verified but not a Pharos target (2) | `ru ca` |

`nb` is a label mismatch rather than a coverage gap and may be cheap to close.
