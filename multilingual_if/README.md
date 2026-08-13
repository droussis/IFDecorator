# multilingual_if

Multilingual instruction-following verification for **Latin, Cyrillic and Greek**
scripts, built on the IFEval-G checkers (`verifiable-instructions`, which is
`open_instruct/IFEvalG` — the same code NeMo Gym's `instruction_following`
resources server runs).

It exists because the upstream checkers encode English assumptions that fail
silently on other languages. Not loudly, with an exception — silently, by
returning the wrong boolean. Under RLVR that is either a constant-zero reward on
a constraint the model cannot satisfy, or free reward on a constraint it never
satisfied.

## Three consumers, one source of truth

| Consumer | Entry point |
|---|---|
| Synthetic data generation | `eligible_ids(lang)`, `localized_kwargs()`, `alphabet()`, `quote_pair()`, `chars_per_word()` |
| RL verification | `verify(ids, kwargs, response, lang)` |
| The IFDecorator flywheel | both, plus `conflicts()` for constraint sampling |

Keeping them on one profile is the point. A constraint that gets sampled but
cannot be scored — or scored but never sampled — is exactly the drift this
prevents.

## Install

```bash
pip install -e .            # from this directory
python -m multilingual_if.setup_lid   # resolves lid.176.ftz, prints the path
```

Two dependency gotchas, both hit immediately in practice:

- **`fasttext` 0.9.3 is incompatible with `numpy>=2`** — `predict()` raises
  `ValueError` from `np.array(probs, copy=False)`. Pin `numpy<2` or use the
  `fasttext-numpy2` fork.
- **The LID model is not vendored here.** fastText's pretrained models are
  CC-BY-SA, incompatible with Apache-2.0. `setup_lid.py` resolves it from
  `$MULTILINGUAL_IF_LID_MODEL`, a local copy, the `ftlid` PyPI package (which
  vendors `lid.176.ftz` verbatim — useful where `dl.fbaipublicfiles.com` is
  blocked by an egress proxy), or a direct download, in that order.

## Use

### Verification (RL, and as a second opinion against your own validator)

```python
from multilingual_if import verify

verify(
    instruction_id_list=["length_constraints:number_words", "language:response_language"],
    kwargs_list=[{"relation": "at least", "num_words": 100}, {"language": "el"}],
    response=model_output,
    lang="el",
    grading_mode="binary",  # or "fraction"
)
# -> {"reward": 1.0, "follow_all_instructions": True,
#     "follow_instruction_list": [True, True], "language": "el"}
```

`verify_one` raises `ValueError` if an id is ineligible for the language. That is
deliberate: an ineligible constraint reaching the verifier means synthesis was not
gated, and you want that loud at the boundary rather than silently scored as 0.
Everything else fails closed.

### Synthesis

```python
from multilingual_if import eligible_ids, localized_kwargs, alphabet, conflicts

ids = eligible_ids("ru")  # 53 of 54
localized_kwargs("detectable_content:postscript", "es")  # {"postscript_marker": "P.D."}
alphabet("uk")  # Ukrainian order, incl. ґ є і ї
conflicts()  # pairs to add to INSTRUCTION_CONFLICTS
```

## What the profile encodes

`profile.yaml` classifies all 54 IFEval-G ids and records the measurement behind
every decision. Current split: **29 ok, 4 localize, 20 patch, 1 exclude**.

Eight global patches do most of the work; `checks.py` applies them by swapping
three helpers in `instructions_util` and overriding six classes, so the divergence
from upstream stays small enough to re-merge.

| | Fix | Why it matters |
|---|---|---|
| **G1** | NFC-normalize before counting | Python's `\w` excludes combining marks, so decomposed text shatters. Measured: the same Polish sentence counts as 14 words (NFC) or 18 (NFD). Model output form is not guaranteed. |
| **G2** | Unicode classes (`\p{Ll}`, `\p{L}`) | `[a-zA-Z]` matches **zero** characters in Cyrillic and Greek, so "at most N lowercase words" passes for free. Latin leaks too — Polish `koty śpią` counts 6 of 8 letters. |
| **G3** | Per-language sentence segmentation | `count_sentences` loads `punkt/english.pickle` by name. Greek marks questions with `;` not `?`, so a 3-question Greek paragraph segments as 1 sentence. |
| **G4** | Unicode word tokenization | Keeps hyphens and apostrophes intact (`aujourd'hui`, `'s-Gravenhage`). |
| **G5** | fastText LID replaces langdetect | langdetect calls `"cats are great."` Catalan (40/40) and never seeds `DetectorFactory`. |
| **G6** | Fail closed | Upstream returns `True` from the `LangDetectException` path, so an emoji-only response satisfies "respond entirely in Bulgarian". |
| **G7** | *Dropped by decision* | Stemming was judged not worth the complexity. Keyword ids ship with upstream surface-form matching. **See the residual risk below.** |
| **G8** | `casefold()` over `lower()` | Turkish dotted/dotless I; Greek final sigma breaks round-trip `text == text.lower()` comparisons. |

### Accepted residual risk (G7)

Keyword matching is surface-form, as upstream. In morphologically rich languages
this is wrong in both directions, and it is measured:

```
forbidden_words evasion:  ru ban "кошка" → "Кошки спят, кошкам хорошо."  PASSES
                          pl ban "kot"   → "Koty śpią, kotom dobrze."    PASSES
                          el ban "γάτα"  → "Οι γάτες ... στις γάτες"     PASSES
existence freebie:        de "Hund" satisfied by "Hundefutter"
                          fi "talo" satisfied by "Taloudessa"
```

This exists in English too but is the exception there; in Slavic, Baltic, Finnic,
Greek and German it is the default case. **Treat `keywords:forbidden_words` as a
weak constraint**, and do not stack several on one prompt expecting a conjunctive
difficulty increase. The tripwire section of `profile.yaml` defines monitors for
these — they leak by design, so watch the *trend* across checkpoints rather than
the absolute rate.

## Language identification

`lid.py` is a validated drop-in for `ResponseLanguageChecker`. On the supported
set (`en de fr es it pt nl pl cs sk hu ro fi et lv lt sv da bg uk ru el ca mt`):

| | |
|---|---|
| True positives | **48/48** |
| False positives, all cross-language pairs | **0/552** |
| Degenerate inputs rejected | **6/6** |
| Determinism | 200/200 identical |

Every constant is calibrated, not guessed:

- **`confidence_floor = 0.40`** — correct predictions bottom out at 0.481 (`sk`);
  incorrect ones top out at 0.357. `0.50` measurably rejects correct Slovak,
  Croatian and Galician.
- **`min_chars = 40`** — at 15 chars, accuracy is 38/48 with **8 wrong predictions
  above the floor**. Confidence is not calibrated on short text, so the floor does
  not protect you; the length gate is doing separate work.
- **Lowercase before `predict()`** — the biggest surprise in testing. All-caps
  French scores `ca @ 0.170` (wrong); lowercased it scores `fr @ 0.980`. Without
  this, `change_case:all_capital` and `language:response_language` on the same
  prompt silently conflict. With it, they do not need to be declared as conflicts.
- **The script gate runs first** — deterministic, free, 34/34 agreement, and it
  resolves Greek outright, since `el` is the only in-scope language using its
  script.

**Do not use as reward targets:** `ga` (generic prose → `sk` @ 0.22), `nn`
(→ `no`; note lid.176 labels Bokmål `no`, not `nb`), `sl` (→ `sr` @ 0.39),
`bs`/`hr`/`sr` (lid.176 carries a separate `sh` label, so four labels split the
mass and none clears a useful margin — use `collapse_confusables=True` and accept
one `bcs` target, or leave them out).

Groups that testing cleared, contrary to expectation: `cs`/`sk`, `bg`/`mk`,
`es`/`gl`/`ca`/`pt`, `nl`/`af` all resolve correctly.

One deliberate limitation: below `min_chars` the pipeline trusts the script gate
alone, so a short Bulgarian answer satisfies "respond in Russian". Rejecting all
short answers would penalise legitimately terse correct ones. Pairing
`response_language` with a minimum word count closes the window.

## Tests

```bash
pytest multilingual_if/tests/ -q     # 47 tests
```

They are differential, not smoke tests — each pins a defect that upstream gets
wrong. Spot-check against unpatched IFEval-G:

| case | upstream | patched |
|---|---|---|
| `count:lowercase_counting` N=0, Russian prose | `True` (vacuous) | `False` |
| `letters:letter_counting` ≥20, Russian prose | `False` (unsatisfiable) | `True` |
| `change_case:english_lowercase`, lowercase Russian | `False` | `True` |
| `startend:quotation`, `«Погода была хорошая.»` | `False` | `True` |
| `language:response_language` bg, `🙂🙂🙂` | `True` (fail-open) | `False` |
