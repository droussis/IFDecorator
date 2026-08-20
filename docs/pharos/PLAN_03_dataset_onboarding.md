# Plan 03 — onboarding the target datasets

Twelve datasets. They do not need twelve pieces of work — they need **four
capabilities**, and each dataset is a combination of them. Sequencing by capability
rather than by dataset is what keeps this tractable, because the expensive
capability (containers) is needed by far fewer rows than it first appears.

> Per-dataset schemas, sizes, licences and the mapping to existing NeMo Gym
> resources servers are in `DATASETS.md`. This file is the strategy; that file is
> the reference.

## 0. Two findings to absorb before reading the rest

**These datasets buy task diversity, not language coverage.** Eleven of the twelve
are English-only. The twelfth, CFBench, is multilingual in Arabic, Hindi, Chinese,
Japanese and Korean — every one outside the Latin/Cyrillic/Greek scope by design.
Greek and the other European targets still have to come from our own synthesis and
from language extension. "Onboard twelve datasets" and "add languages" read like
one workstream and are two.

**Licensing is a real constraint here, not a formality.** One dataset has no
licence at all — no tag on the card and no LICENSE file upstream, so all rights
reserved and copy-and-adjust is unavailable. One is share-alike. One is a composite
of about thirty upstream sources that the card says to check individually. And two
Gym configs disagree with the live cards about what the licence even is. Route
these before any copying; two of them would otherwise be caught late.

Details per dataset in `DATASETS.md`.

## 1. Capability tiers

| Tier | Capability | Framework change | Blocked by |
|---|---|---|---|
| **T0** | single-turn text, graded programmatically / by judge / by rubric | **none** — this is what the framework already does | nothing |
| **T1** | tool schemas in the request payload, single decision step graded against an expected action | moderate: a `tools` field on the row, and one new verifier | Plan 04 |
| **T2** | code execution against test cases | large: sandboxing, timeouts, resource limits, determinism | see §4 |
| **T3** | containerised multi-turn agentic environments | very large: images, orchestration, per-task setup and teardown | deferred |

The load-bearing observation is at T1. **A "pivot" dataset is a single-step slice
of an agentic trajectory**, and a single step can be graded by comparing the
produced tool call against a stored expected one. No container, no live tool
execution, no episode loop. That moves several datasets which look like T3 down to
T1 — with the caveat in §5.

Two corrections that research produced, both in our favour:

- **The terminal pivot dataset is T0, not T1.** Its action is a JSON blob emitted
  as assistant *text*, not a function call, and the prior terminal history is
  already flattened into the prompt. It carries no tool schemas at all. It needs no
  framework change whatsoever.
- **One dataset is partially T0.** The verl-format aggregate can be filtered by its
  routing key: its maths, logic, MCQ, IFEval and IFBench domains are rule-based and
  execution-free, and only its code and simulation domains need a sandbox. Onboard
  the slice, not the whole.

Result: **five of the twelve need no framework change at all**, three need only
tool payload support, and just four need execution or containers.

## 2. Sequencing

**Phase A — T0.** No framework change. Instruction-following and constraint-style
datasets land on the existing `constraint` track using Plan 01's verifier. This is
also where the multilingual work pays off directly, since these are the datasets
whose constraints are language-bound.

**Phase B — T1.** Add tool payload support (Plan 04) and the single-step
expected-action verifier. All four `*-Pivot-v1` datasets become usable, and the
same verifier serves future pivots generated from our own rollouts.

**Phase C — T2, only if justified.** Code execution. Do not start this because a
dataset is on the list; start it because a measurement says the reward signal is
worth the sandbox. §4.

**Phase D — T3.** Deferred. §5 records what it would cost so the deferral is a
decision rather than an omission.

## 3. The pivot format, and why it matters most

NeMo Gym's pivot contract, from the shipped skill and validator:

```json
{
  "responses_create_params": {
    "input": [],
    "tools": [],
    "parallel_tool_calls": false,
    "tool_choice": "auto"
  },
  "expected_action": {},
  "agent_ref": {"type": "responses_api_agents", "name": "<agent block name>"}
}
```

- `input` is the model-call prefix *before* the pivot action.
- `tools` is the full tool list available at that state.
- `expected_action` is **singular** — one `function_call` (with `name` and
  JSON-string `arguments`) or one `message`. A source turn containing more than one
  tool call is filtered out, not split.
- `tool_choice` is `"auto"`; `"required"` can push some inference engines into
  structured-decoding paths.

Grading is argument comparison, with `word_count_similarity_threshold` as the main
knob for string arguments — set to 0.1 in the shipped tool-call configs and 0.0 in
at least one pivot config, i.e. it is tuned per dataset rather than global.

### Three sources agree on how to select pivots

The pivot skill says: profile candidates with **at least 8 local rollouts**, keep
mixed-reward candidates, discard all-pass and all-fail groups, and drop the easiest
first when data is abundant.

That is the same rule as the Pharos 8-generation design (2 teacher + 6 student,
"difficulty on our own student rather than by prior") and the same rule as the
flywheel's pass-rate routing. All three exist because GRPO's advantage is
group-relative: a prompt every rollout solves, or none does, contributes nothing.

**Consequence: no new difficulty machinery is needed for pivots.** The existing
8-generation pass *is* the pivot profiler. This is the single biggest saving in
this plan.

### Portable assets

Shipped with the Gym skill and worth copying rather than reimplementing:

| Asset | What it gives |
|---|---|
| `scripts/validate_pivot_dataset.py` | row-shape, expected-action and agent-ref validation; optional validation against the Gym Pydantic models |
| `scripts/reference/chat_messages_to_pivot_dataset_reference.py` | chat-completion messages → pivot rows |
| `scripts/reference/conversational_messages_to_pivot_dataset_reference.py` | conversational trajectories → pivot rows, with reasoning and provenance handling |
| `scripts/reference/tool_messages_to_pivot_dataset_reference.py` | tool-use message rows → pivot rows |
| `scripts/reference/generic_pivot_dataset_reference.py` | generic source rows → pivot rows |

The skill's own warning applies: these are examples carrying dataset-specific
assumptions, to borrow from rather than run unchanged.

## 4. Code execution — the T2 decision

Nothing in either Pharos repo executes code. A search for subprocess, exec, pytest,
sandbox, firejail, nsjail, bwrap, setrlimit or signal-based timeouts across the
generation and acquisition trees returns a single hit, and it is an HTTP timeout.
This is greenfield, and it is a different *kind* of thing from every existing
verifier, all of which are pure functions over strings.

Five concerns, none of which the framework currently solves:

1. **Isolation.** Verifiers run in-process, in the same environment as the rest of
   the pipeline. Options are a short-lived container per batch, `nsjail`/`bwrap`,
   or a dedicated host. None exists today.
2. **Timeouts and resource limits.** One infinite loop in millions of generations
   wedges a phase. Needs a hard per-case wall clock, an address-space limit, and a
   killed-process path that returns evidence rather than raising — the verifier
   contract forbids raising.
3. **Determinism.** Evidence text feeds a judge prompt and then judge-SFT data.
   Timings, memory addresses, set iteration order, traceback paths and seeds would
   make identical responses produce different training rows. They must be
   normalised out of the rendered evidence.
4. **Cost model.** Throughput budgeting counts GPU-seconds. Test execution is CPU
   time competing with the generation client, and needs separate budgeting.
5. **Streaming.** The verifier bridge materialises the whole table. Adding per-row
   subprocess spawning to a multi-GB generation file makes a batched rewrite close
   to mandatory.

**Recommendation: do not open T2 on the strength of dataset availability.** Run
Phase A and B first, then ask whether execution-based reward adds signal the
existing mechanisms do not. If it does, scope it as its own project with the five
concerns above as its acceptance criteria.

## 5. Containers — the T3 shape

Deferred, but the shape should be known before anything in Phase B is designed so
that Phase B does not accidentally foreclose it.

A containerised agentic environment needs, at minimum: a per-task image (or a base
image plus a per-task setup script), orchestration to start and tear it down around
each episode, a mechanism for the agent to issue commands into it and read results
back, per-episode wall-clock and resource limits, and a way to reproduce a specific
task state for debugging. The Gym servers in this area carry their own
orchestration and, in at least one case, an Apptainer configuration — worth reading
before committing, not before deferring.

**The caveat on the T1 reframing:** grading a stored pivot without a container is
sound, but the pivot's *provenance* is a trajectory that was produced inside one.
We can consume someone else's pivots at T1. Producing our *own* pivots from our own
agentic rollouts still needs the environment. So T1 buys consumption, not
generation — worth being precise about, because it determines whether the pivot
datasets are a one-off import or a renewable source.

## 6. What "make them ours" means concretely

For each dataset: a row adapter into the pool schema, a verifier or a mapping to an
existing one, and a licence check on both the data and any borrowed verifier code.

The framework's own conventions decide most of the rest — the pool schema's
sentinel rules, the non-empty `expected_response` invariant, `prompt_hash` dedupe,
and the rule that a verifier returns evidence rather than a verdict. A dataset is
"ours" when it lands on that schema and is scored by our own copy of the verifier,
with provenance recorded in `source_meta`.

Where an upstream project's verifier is worth borrowing, copy and adjust rather
than depend — consistent with how the instruction-following verifier is being
brought across — and record the source, commit and licence alongside it.
