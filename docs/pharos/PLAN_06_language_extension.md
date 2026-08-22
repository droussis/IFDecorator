# Plan 06 — adding a language to the verifiable set

A repeatable procedure. Verification and synthesis share one eligibility matrix, so
adding a language means answering, per constraint, whether it is meaningful there —
and answering it by measurement rather than by assumption.

Today: 24 verified, 22 of them in the target set. Gap: `hr is nb sl sr tr mk sq`.

## The procedure

### 1. Script and profile

Determine the language's script (Latin / Cyrillic / Greek — nothing else is in
scope) and add a language-profile entry: quote pair, postscript marker, section
word, alphabet, and characters-per-word.

Characters-per-word is not cosmetic: it calibrates length constraints, and the
range across the current set runs 5.1 (English) to 8.0 (Finnish). Getting it wrong
makes difficulty correlate with language rather than with the instruction.

### 2. Language identification

Two measurements, on a parallel corpus of at least 30 sentences of *generic prose*
— no proper nouns. Named-entity-rich text flatters LID badly: Irish scored 0.81 on
a "capital of" sentence set and collapsed to Slovak at 0.22 on generic prose.

- **Script gate.** Does the dominant Unicode script identify the language
  unambiguously among in-scope targets? For Greek the answer is yes, and it needs
  no model at all.
- **fastText accuracy and confidence.** Record top-1 accuracy and the minimum
  confidence over correct predictions. Confirm the language's label exists in
  lid.176 and matches the pool's code — Norwegian Bokmål is `no` upstream, not
  `nb`, and a straight pass-through fails every row.

Then check the **confusable set**: does the new language pull probability mass from
one already supported, or vice versa? Re-run the existing set's accuracy with the
new language enabled. Adding a confusable neighbour can degrade a language that
was previously fine.

Reject the language if correct-prediction confidence overlaps the incorrect-
prediction band. That is what disqualified Croatian, Slovene and Serbian: lid.176
carries a separate Serbo-Croatian label, so four labels split the mass and none
clears a usable margin.

### 3. Per-constraint eligibility

Walk the 54 constraints and classify each as ok / needs-a-literal / ineligible.
Most follow from script, but check these explicitly:

| Constraint family | What to check |
|---|---|
| case constraints | is the script bicameral? Turkish additionally needs a casefold review — Python's default casing is wrong for the dotted/dotless I, which affects every case-based check |
| letter counting and frequency | supply the language's letter inventory; never `a-z` |
| sentence counting | terminator set. Greek marks questions with `;`; check for any language-specific terminator or abbreviation convention |
| word counting | confirm space-delimited and that NFC normalisation makes counts stable |
| quotation | the language's quote pair, in addition to ASCII |
| postscript / section markers | the conventional local form |
| alphabet-order constraints | Cyrillic ordering is language-specific — Ukrainian has letters Russian does not |

An honest `not_applicable` is always better than a guessed `pass`. Under the
verifier contract the evidence goes to a judge that is instructed to weigh it, so
wrong evidence is worse than absent evidence.

### 4. Tests

Add to the differential suite:

- NFC/NFD invariance for word counting.
- Sentence counting with the language's terminators and at least three common
  abbreviations.
- One case per ineligible constraint asserting `not_applicable`, so a later change
  has to update the test deliberately.
- LID acceptance on correct prose, rejection on a confusable neighbour, and
  rejection of degenerate input.

### 5. Promote

Add to the supported list, regenerate the coverage table, and run one synthesis
round end to end in that language before adding it to a production pool. Read the
per-language pass-rate distribution against the established languages: an outlier
`frac_zero` or `frac_one` means a constraint that is unsatisfiable or vacuous
there, i.e. the eligibility walk in step 3 missed something.

## Cost

Roughly half a day per language for a clean case — most of it is the parallel
corpus and the eligibility walk, both of which are judgement rather than code.

The gap languages are not equal:

| Language | Expected |
|---|---|
| `nb` | possibly an afternoon — it may be only the `no` label mapping |
| `is`, `sq` | ordinary Latin-script cases, no known obstacle |
| `mk` | Cyrillic; check the confusable set against `bg` and `sr` first, since that is where it will fail if it fails |
| `tr` | needs the casefold review before any case-based constraint can be trusted |
| `hr`, `sl`, `sr` | **may be unachievable with lid.176.** The `sh` label is the obstacle and it is upstream. Options are collapsing them to one BCS target, or a different LID model — a larger change than adding a language |

Recommendation: take `nb`, `is`, `sq`, `mk` first — four of the eight for a few
days' work — and treat `tr` and the BCS group as separate, scoped pieces.
