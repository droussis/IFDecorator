# Provenance

Vendored from NVIDIA NeMo Gym, at commit `049986e` of the fork.

| File | Source | Modified? |
|---|---|---|
| `comparator.py` | `resources_servers/single_step_tool_use_with_argument_comparison/common/verification_utils.py` | **verbatim** |
| `validate.py` | `.claude/skills/nemo-gym-pivot-datasets/scripts/validate_pivot_dataset.py` | **verbatim** |
| `actions.py` | `.../common/response_utils.py` | **yes** — decoupled from the framework's response type so it accepts plain dicts. Precedence rules and batch behaviour unchanged |
| `evidence.py` | — | **ours.** The contract adaptation |

The comparator was chosen for vendoring because it depends on nothing but the
standard library and pydantic. Copying it costs one dependency we already have and
removes a framework dependency we do not want.

## Why `evidence.py` exists

Gym's verifier returns a `reward` float. A programmatic verifier in the target
framework must not — it returns evidence, and the judge decides the reward. The
translation loses nothing: the comparator's reward categories are already
human-readable sentences describing exactly why a comparison failed, which tells a
judge more than a scalar does.

## Re-syncing

```bash
cp <gym>/resources_servers/single_step_tool_use_with_argument_comparison/common/verification_utils.py \
   pivot_tools/comparator.py
cp <gym>/.claude/skills/nemo-gym-pivot-datasets/scripts/validate_pivot_dataset.py \
   pivot_tools/validate.py
```

`actions.py` and `evidence.py` are hand-maintained; re-read them against upstream
rather than overwriting. Run the tests after any re-sync — several of them pin the
upstream behaviours we depend on (tool calls taking precedence over text, multiple
calls becoming a batch), so an upstream change that alters those will show up as a
failure rather than as a silent behaviour change.

## Not vendored, and why

- The **lenient coding-agent comparator** (tool-name category equivalence,
  path-suffix matching, sequence similarity, diff-size shaping) lives in its own
  Gym server. Worth copying when the SWE pivots are onboarded — strict matching
  penalises paraphrase rather than error on a coding tool vocabulary — but it is
  not needed for the conversational or function-calling pivots.
- The **four reference converters** shipped with the Gym skill carry
  dataset-specific assumptions and are meant to be borrowed from, not run. They
  stay where they are; the adapters that replace them belong with the ingestion
  toolkit.
