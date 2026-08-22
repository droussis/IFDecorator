# Plan 04 — tool-calling support

What the framework needs in order to generate and grade tool-use data. The
recommendation is to adopt **single-step tool grading first**, because it maps onto
the existing one-call generation model with a small, bounded change, and it covers
every `*-Pivot-v1` dataset without a container or a tool-execution loop.

## 1. The contract

NeMo Gym carries tool schemas **on the dataset row**, hand-authored, not derived
from live introspection. A row's `responses_create_params` validates against a
strict copy of OpenAI's Responses API request type (`extra="forbid"`):

```json
{
  "input": [
    {"role": "developer", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "tools": [
    {"type": "function",
     "name": "get_weather",
     "description": "...",
     "parameters": {"type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false},
     "strict": true}
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

A model turn that calls a tool returns:

```json
{"type": "function_call", "call_id": "...", "name": "get_weather",
 "arguments": "{\"city\": \"sf\"}"}
```

`arguments` is a **JSON string**, not an object — a detail that has to survive
every conversion, because comparison decodes it.

A tool result is fed back as:

```json
{"type": "function_call_output", "call_id": "...", "output": "<string>"}
```

appended to `input` for the next turn. The loop is stateless: each turn re-sends
the whole accumulated `input`, so an agent needs no memory between turns.

Tool *execution* in Gym is a plain HTTP POST to the resources server at a route
named after the tool — `get_weather` ⇒ `POST /get_weather` with the decoded
arguments as the body. There is no manifest and no MCP requirement; MCP exposure is
an optional layer over the same routes. **We do not need it.**

## 2. Two modes, and why to take the cheap one first

| | Single-step | Live loop |
|---|---|---|
| Model calls per row | exactly one | many, until the episode ends |
| Tools executed | none | yes |
| Container | no | not necessarily, but sometimes |
| Grading | compare the produced action against a stored `expected_action` | verify the episode outcome |
| Fit with the current generation entrypoint | close — one call, one parse | poor — needs a real agent harness |

Gym's own single-step agent is about 90 lines and makes exactly one model call. The
change to our generation entrypoint is correspondingly small:

**Add a `--tools-field <column>` flag**, mirroring the existing `--messages-field`.
Where `--messages-field` continues a conversation from a column, `--tools-field`
attaches that row's tool schemas to the request. The sampling, batching, retry,
checkpointing and resumption machinery is unchanged.

`--extra-body` cannot substitute: it is per-run, and tools are per-row.

Constraints that carry over unchanged:
- the new mode must return a non-empty expected-field list, or every row looks
  complete and the run is a silent no-op;
- `--tool-choice` should be `auto` for pivot rows — `required` can push some
  inference engines into structured-decoding paths;
- rows skipped for empty inputs must be carried through, never dropped.

## 3. Grading a single step

The comparator in Gym's single-step server is **pure Python with no framework
dependency** — standard library only. That makes it the most directly portable
asset in this plan.

What it does:

- tool **name** must match (mandatory);
- **arguments** compared recursively by JSON type: exact for numbers, lists and
  objects (with a float epsilon), and **Jaccard word overlap** for strings, gated
  by a `word_count_similarity_threshold`;
- **parallel calls** matched by bipartite maximum matching, with
  `binary_strict` / `fractional` / `f1` scoring modes and `allow_subset` /
  `allow_superset` gates.

A second, deliberately looser variant exists for coding-agent tool vocabularies:
tool-name *category* equivalence (a patch-applying tool and an editor tool treated
as interchangeable), path-suffix matching, and sequence-similarity thresholds
instead of exact match. Worth copying alongside the strict one — a coding agent has
many ways to express the same edit, and strict matching punishes paraphrase rather
than error.

### The contract adaptation, again

Gym's `verify()` returns a `reward` float. **Our verifier contract forbids that** —
programmatic verifiers return evidence, never a verdict. The port therefore
re-expresses the comparison as evidence:

```
- expected tool: apply_patch — MATCHED
- argument `path`: exact match
- argument `content`: string similarity 0.34 (threshold 0.60) — BELOW THRESHOLD
- no unexpected additional tool calls
```

This is the same adaptation as the instruction-following verifier, and here it is
arguably an improvement on the original: a bare similarity scalar tells the judge
much less than a per-argument breakdown, and the judge is the thing deciding the
reward.

## 4. What to copy

**Already vendored** into `pivot_tools/` in this repo — the comparator verbatim, the
action extractor decoupled from the framework's response type, the pivot-row
validator verbatim, and an evidence wrapper that re-expresses a comparison as
evidence rather than a reward. 32 tests, including replaying real Gym pivot rows
(both the single and the parallel tool-call shapes) against themselves.

Still to copy when the SWE pivots are onboarded:

| From | Gives | Drags in |
|---|---|---|
| `swe_pivot`'s app | the lenient coding-agent variant: tool-name category equivalence, path-suffix matching, sequence similarity | nothing extra |
| the Responses-API tool and tool-call types from `nemo_gym/openai_utils.py` | strict typing for `tools`, `function_call`, `function_call_output` | the `openai` SDK's type module — typing only, no network |
| the simplest agent's app | the full tool-execution loop, for when a live loop is actually needed | the Gym agent base class and its trajectory bookkeeping, both strippable |
| the pivot skill's four reference converters | worked trajectory→pivot conversions; borrow from rather than run, they carry dataset-specific assumptions | nothing |

Explicitly **not** worth copying for this purpose: the MCP auto-exposure layer, the
sandbox module, and the observability machinery in the model-server base class.

## 5. Pool schema impact

Tools are **input**, not verification metadata, so they cannot ride in `checks` or
`verification_meta` — they have to reach the request payload.

| Field | Where |
|---|---|
| tool schemas | **a new `tools` column** (JSON string), or a JSON string inside a new column carrying the whole `responses_create_params` |
| expected action | `expected_response` — it must be non-empty anyway, and this is what the judge renders as the required behaviour |
| comparator knobs | `checks`, e.g. `{"tool_call": {"word_count_similarity_threshold": 0.1, "mode": "strict"}}` |
| conversation prefix | `messages` / `dialogue_history` as today |

This is the plan's one **schema change**, and it needs sign-off (D6). The
alternative — packing tools into an existing JSON column — avoids the migration but
puts input data in a column named for verification, which will mislead every later
reader. Prefer the honest column.

## 6. Sequencing

1. **`--tools-field`** on the generation entrypoint, plus a smoke run proving tool
   calls come back and parse.
2. **Port the comparator** as an evidence-emitting verifier; test it against the
   Gym example rows, which give known-good inputs and expected outcomes.
3. **Pivot adapters** for the `*-Pivot-v1` datasets, validated with the shipped
   validator before anything downstream sees them.
4. **Measure**, then decide whether a live loop is needed at all. Several tool-use
   datasets grade fine single-step; the loop is only required where the reward
   depends on what the tools actually returned.

## 7. The honest limit of single-step

Single-step grading lets us **consume** pivots. It does not let us **produce** them
from our own rollouts, because a pivot's provenance is a trajectory generated
inside a live environment.

So T1 buys the imported datasets and a verifier we own. A renewable supply of our
own pivots — in our languages, against our student — still needs the environment
work. Worth stating plainly now, because it decides whether these datasets are a
one-off import or the start of a pipeline.
