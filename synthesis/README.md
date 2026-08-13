# synthesis

Reusable pieces of the multilingual instruction-following data synthesis
pipeline, adapted from the IFDecorator flywheel.

**Orchestration is deliberately not here** — that lives in our own synthesis
library. What this package provides is the part worth sharing across it, the
flywheel, and RL:

| module | what |
|---|---|
| `prompts.py` | every stage's prompt template, multilingual-adapted, with the delta from upstream documented per template |
| `parsing.py` | parsers for those outputs, each replacing an upstream parser with a known defect |
| `matching.py` | deterministic containment, replacing two LLM calls per evolution attempt |
| `schema.py` | row schema, stage validation, difficulty routing, per-language diagnostics |

Verification is [`multilingual_if`](../multilingual_if/README.md), shared with
RL. The full stage-by-stage guide is
[docs/SYNTHESIS_PIPELINE.md](../docs/SYNTHESIS_PIPELINE.md).

## Use

```python
from synthesis import prompts, parsing, matching
from synthesis.schema import (
    Row,
    Rubric,
    validate_row,
    route_by_difficulty,
    pass_rate_report,
)

# W1 - decompose, content in the target language
query = prompts.render(
    prompts.DECOMPOSE,
    "el",
    prompt=seed_prompt,
    example_prompt=...,
    example_task=...,
    example_constraints=...,
    example_input=...,
)
decomposition = parsing.parse_decomposition(await generate(query))

# W2 - classify the whole list in one call
labels = parsing.parse_classification(
    await generate(...), expected=len(decomposition.constraints)
)

# W4 - evolution, re-randomized per call
query = prompts.render_evolution(
    instruction, "el", distribution=running_counts, rng=rng
)
evolution = parsing.parse_evolution(await generate(query))
merged = matching.merge_evolved(
    instruction,
    evolution.enhanced_instruction,
    evolution.constraint,
    decomposition.input_text,
)

# every stage boundary
validate_row(row, stage="w4")

# W5 - route, then read the diagnostic before scaling
pools = route_by_difficulty(tagged_rows)
report = pass_rate_report(tagged_rows)
```

## Two conventions

**Meta-prompts in English, content in the target language.** The generator is
instructed in English; what it writes is in the target language. This decouples
meta-task competence from target-language competence and keeps the parse markers
stable across all 24 languages. Localizing the meta-prompts too is a legitimate
alternative — it changes what difficulty means, so measure it before adopting.

**`language` is required on every row.** `prompts.render()` raises on a language
`multilingual_if` cannot verify, rather than defaulting: a row nothing can score
should never be generated in the first place.

## What each replacement fixes

Every parser here exists because the upstream one is wrong in a way that produces
plausible-looking rows rather than a visible error:

| replaces | defect |
|---|---|
| `utils.py::unified_judge_parse` | no `return` in the non-strict branch — the default path returns `None`, so every loose judgement is falsy |
| `w1_decompose.py::process_llm_response` | appends any `-` line to constraints once a task description is seen, including bullets under `#Input:` |
| `w4_evol.py::parse_evol_response` | splits the whole response on `#`, so a markdown heading or hex colour inside the generated instruction corrupts every later field |
| `w4_evol.py::get_keyword` | `eval()` on a regex capture, falling back to an English word list |
| `w4_evol.py::merge_evol` | two LLM calls per attempt to answer a string-comparison question |
| `w4_evol.py:147` weighting | `random.choices` samples with replacement, so a down-weighted constraint type can vanish from the prompt entirely — the opposite of what the weighting intends |
| checklist zip | checklists generated only for one class of constraint, then zipped against the full list |

The last one is fixed structurally rather than by care: `Rubric` carries its
`constraint`, so there is nothing to mis-zip. Do not introduce a parallel-list
representation of rubrics.

## Tests

```bash
pytest synthesis/tests/ -q     # 51 tests
```

Differential, not smoke tests — each pins a specific upstream defect. Three of
them caught real bugs in this package during authoring (verdict ordering, input
bullet accumulation, and the weighted-shuffle type loss), so they are load-bearing.
