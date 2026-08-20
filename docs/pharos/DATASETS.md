# Dataset catalogue

Nineteen datasets we intend to generate against. Strategy is in
`PLAN_03_dataset_onboarding.md`; the scripts each needs are in `SCRIPTS.md`.

## How to read this catalogue

Every dataset here is one we want to **generate against** — take its prompts, run
our own teacher and student, and keep the result. What differs is what we can do
with the output, and that is decided by what the row carries besides the prompt.

### Three regeneration classes

| Class | The row carries | We can produce | Feeds |
|---|---|---|---|
| **Full** | a self-contained prompt **and** a target that survives regeneration — a golden answer, unit tests, a constraint spec, or a rubric | complete fresh trajectories, filtered by the target | RL, **SFT/GFT distillation**, MOPD |
| **Pivot** | a trajectory *prefix* and one `expected_action` | one action per row, compared against the expected one | single-step local RL; single-step behaviour cloning |
| **Blocked** | a target we cannot evaluate — needs code execution, a container, or is licence-encumbered | nothing usable yet | nothing until the blocker clears |

The distinction that matters most is **Full vs Pivot**, and it is not about task
difficulty — it is about whether the target outlives the source model's rollout.

A golden answer is model-independent: our teacher's trajectory can be scored
against it exactly as the source model's was. An `expected_action` is not — it is
one specific model's chosen step at one specific state, and the state itself came
from that model's earlier choices. Regenerating from a pivot prefix gives an action
at a state our model would not necessarily have reached.

So pivots are **imported RL data**, not a renewable source. Producing our own
pivots means running our own agent in the original environment, which is the
container work.

### Why "Full" is the valuable column

Pharos's algorithms want different things:

- **GRPO/RLVR** needs a scalar reward per rollout and depends on within-group
  variance — a prompt every rollout solves, or none does, teaches nothing.
- **MOPD** needs **no reward at all**, so prompts routed only to it cost zero judge
  calls.
- **GFT** consumes trajectories, with scores needed only for its subset.

A Full-class dataset feeds all three from one generation pass. A Pivot-class
dataset feeds only the first, and only in its single-step form. That ratio, not the
row count, is the right way to compare two sources.

### One nuance on pivots and SFT

A pivot row can still produce supervised data — a single tool call at a given
state is a legitimate SFT target, and behaviour cloning of tool syntax is a real
use. What it cannot produce is a *trajectory*: no multi-step reasoning to distil,
no long-horizon credit assignment. Treat pivot-derived SFT as teaching call
formatting, not task competence.

### The standard transformations

Nearly every source needs some subset of these. Naming them once keeps the
per-dataset recipes short, and it is also the specification for the ingestion
scripts.

| # | Transformation | Why |
|---|---|---|
| T1 | **JSON-decode nested string columns** | The verl convention stores `reward_model` and `extra_info` as JSON *strings*, not structs. Decode, lift the fields we need, re-encode only what the pool schema wants as a string |
| T2 | **Normalise the prompt to `messages`** | Sources arrive as a bare string, `{instruction, input}`, an OpenAI message list, or an already-rendered chat template. All become a message list ending on a user turn; a row that does not end on a user turn is rejected |
| T3 | **Render `dialogue_history`** | The judge reads a rendered JSON *string*, not a list. A message list passed through becomes a Python repr and corrupts the prompt |
| T4 | **Separate bundled generations from the prompt** | Several sources ship a response and often a reasoning trace. These must not reach the model when we regenerate — but they are retained in a provenance column rather than dropped, and they double as a baseline to compare our teacher against |
| T5 | **Lift the target into `expected_response`** | Must be non-empty on any non-rubric row or the row is dropped at build time. Structured targets are JSON-dumped; this field is also what the judge renders as the required behaviour |
| T6 | **Build `checks`** | The JSON spec naming which programmatic verifiers to run, keyed by verifier name, with that verifier's kwargs as the value |
| T7 | **Derive `answer_format`** | From the extraction regex or the constraint spec — never hardcoded. Hardcoding it was wrong on the majority of rows in one earlier source |
| T8 | **Set sentinels, never null** | `-1.0`, `-1`, `""` in typed columns |
| T9 | **Compute `prompt_hash` and a unique `id`** | The dedupe key is the hash, never a uuid. Note that two rows sharing a prompt collapse to one, which forecloses some designs |
| T10 | **Stamp `language`** | Required, not optional. Absent it, verification silently falls back to English |
| T11 | **Set `translatable`** | As a string. Constraint-bound rows are never translatable — schema keys, regexes and markers are language-bound |
| T12 | **Carry `tools`** | Tool schemas are input, not verification metadata, so they need their own column and must reach the request payload |

### Overlap is significant and must be handled at ingestion

Several of these are aggregations of each other. Tracing the lineages:

```
guru-RL-v1.5 ──┬── Skywork-OR1 ──┬── NuminaMath
               │                 └── DeepScaleR
               ├── DAPO-Math
               ├── LeetCode / TACO / LiveCodeBench ── the DeepCoder–rllm corpus
               ├── WebInstruct-Verified
               └── Nemotron STEM subset

rlvr-guru-raw-data-extended ── the same Guru line, a different packaging

KlearReasoner-CodeSub ──┐
                        ├── both from the DeepCoder–rllm Codeforces corpus
deepcoder-gold-standard ┘

Nemotron-SFT-Math-v3 ── Nemotron-Math-v2 ── OpenMathReasoning (AoPS) + Math StackExchange
```

Three consequences.

**Do not ingest an aggregate and its constituents as separate sources.** Taking
both a Guru packaging and Skywork-OR1 means the same maths problems arrive twice
under different ids and different metadata.

**The prompt-hash dedupe handles this, but only if everything lands in one pool.**
Deduplication is on a hash of the normalised prompt, never a uuid, so identical
problems collapse — but the surviving row is whichever arrived first, and it keeps
*its* metadata. Ingest in a deliberate order, best-annotated source first, or the
dedupe silently picks the worse copy.

**Two Guru packagings appear on our list.** They should be treated as one source
with a choice of packaging, not two datasets. The one whose row count and schema
can actually be verified is the one to take.

### Scale needs a decision before ingestion, not after

The maths set alone is 3.6M rows and roughly 144 GB. Between it, the two Guru
packagings and Skywork-OR1 there is far more maths than any realistic training mix
wants, and generating 8 rollouts per prompt makes the cost of over-ingesting
concrete rather than notional.

Cap at the source-spec level, stratified, before generation — the pool builder
already supports per-source and per-stratum caps. The interesting question is not
how many rows exist but how many *difficulty-appropriate* rows we want per domain,
and several of these sources ship probe pass rates that can seed that decision
before we spend a single rollout.

---

## The catalogue

Ready = usable with what we have or are already building. Blocked = waiting on a
capability or a licence decision.

| # | Dataset | Rows | Licence | Target | Class | Effort | Status |
|---|---|---|---|---|---|---|---|
| 1 | nvidia/Nemotron-SFT-Math-v3 | 3.64M | CC-BY-4.0 / CC-BY-SA-4.0 **per row** | boxed answer | full | low | **ready** |
| 2 | PrimeIntellect/Skywork-OR1-RL-Data | 105.1K | Apache-2.0 | maths answer | full | low | **ready** |
| 3 | JustinTX/WildSci | 56.8K | CC-BY-4.0 | MCQ letter | full | low | **ready** |
| 4 | OpenDataArena/ODA-Fin-RL-12k | 12.4K | Apache-2.0 (22 mixed upstream) | short answer, judge-graded | full | low | **ready** |
| 5 | nvidia/Nemotron-RL-instruction_following | 46,391 | `odc-by` | IF constraint spec | full | low | **ready** |
| 6 | nvidia/Nemotron-RL-SysBench-v1 | 1–10K, unconfirmed | CC-BY-4.0 | **IF spec + rubric**, multi-turn | full | med | **ready** |
| 7 | nvidia/Nemotron-RL-CFBench-v1 | 1,105 | CC-BY-4.0 | **IF spec + rubric**, multi-turn | full | med | **ready** |
| 8 | nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 | 31.1K | CC-BY-4.0 | expected command batch | pivot | med | **ready** |
| 9 | nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 | 97.0K | CC-BY-4.0 | expected tool call **+ reusable scenario** | pivot → full | med | needs tools |
| 10 | nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 | 50.3K | CC-BY-4.0 | expected tool call | pivot | med | needs tools |
| 11 | nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 | 9,600 | CC-BY-4.0 | expected tool call | pivot | med | needs tools |
| 12 | IFM/guru-RL-v1.5 | ~108.8K **unverified** | MIT | mixed by domain | full | high | partial — verify first |
| 13 | AmanPriyanshu/rlvr-guru-raw-data-extended | 150,000 + 221,332 **confirmed** | ODC-BY over ~30 sources | mixed by domain | full | low for the 82.9% | **124,316 rows ready** |
| 14 | TIGER-Lab/AceCode-V2-122K | 122.6K | Apache-2.0 data, MIT code | Python asserts | full | med | blocked: execution |
| 15 | Kwai-Klear/KlearReasoner-CodeSub-15K | 15.0K | Apache-2.0 | stdin/stdout tests | full | med | blocked: execution |
| 16 | PrimeIntellect/deepcoder-gold-standard-solutions | 16.3K | **unclear** | stdin/stdout tests | full | med | blocked: execution + licence |
| 17 | nvidia/Nemotron-RL-coding-competitive_coding | 16.1K | **CC-BY-SA-4.0** | stdin/stdout tests | full | med | blocked: execution |
| 18 | Zhongzhi1228/Recursive-Task-Synthesis | 37,484 | CC-BY-4.0 | container test script, **not in the parquet** | full | high | blocked: containers |
| 19 | Multilingual-Multimodal-NLP/IfEvalCode-Instruct | 3.0K | **none, data and code** | tests + generated assertion fn | full | high | blocked: licence |

**Eight are ready now**, plus 124,316 rows of the verl aggregate that a single
`data_source` prefix filter separates out — so nine sources in practice. Three more
need only tool payload support. Six are blocked on a capability or a licence.

Counting rows rather than datasets changes the picture: the ready set is dominated
by maths, and the execution blocker gates roughly 170K code rows across four
sources — which is what makes the sandbox decision worth taking on its own merits
rather than per dataset.

**None of the nineteen advances the multilingual goal.** Every one is English-only
except two, and those two are bilingual with Chinese or multilingual in scripts
deliberately out of our scope. Language coverage comes from our own synthesis and
from translation, not from this catalogue.

---

# Group A — full regeneration, no execution needed

Ready now. Prompt plus a target that survives regeneration, gradeable by machinery
we either have or are building. **These feed RL, SFT/GFT and MOPD from one
generation pass.**

## nvidia/Nemotron-SFT-Math-v3

3.64M rows, ~144 GB. **Per-row licence** — CC-BY-4.0 on the forum-derived rows,
CC-BY-SA-4.0 on the StackExchange-derived ones, carried in a `license` column.

```jsonc
{"uuid": "...", "problem": "...",
 "messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "...", "reasoning_content": "..."}],
 "expected_answer": "...", "data_source": "AoPS | StackExchange-Math",
 "license": "cc-by-4.0", "tool_usage": "with|without Python TIR", "tools": [...]}
```

Bundles full chain-of-thought from DeepSeek-V3.2-Speciale, and tool-integrated
reasoning trajectories from DeepSeek-V3.2. **That is exactly what we regenerate**,
so the bundled traces are provenance, not input — and a useful baseline to measure
our teacher against.

Target is `expected_answer`, a boxed maths answer. No execution: the TIR rows used
a Python tool at generation time, but grading is answer equivalence.

Preprocessing: T2, T4, T5, T8, T9, T10, T11. Note the per-row licence must be
**preserved per row** — blanket-licensing the set would be wrong. Also strip the
system and tool scaffolding that references the source model's own tool format.

Scale is the real issue: at 3.6M rows this is larger than the rest of the
catalogue combined. Cap and stratify at the source spec.

## PrimeIntellect/Skywork-OR1-RL-Data

105.1K rows, Apache-2.0. The maths split only — the code split was excluded
upstream.

```jsonc
{"question": "...", "answer": "...",
 "info": {"data_source": "train-math-numinamath1.5_olympiads", "index": 0},
 "difficulty_1_5b": 0, "difficulty_7b": 3, "difficulty_32b": 6}
```

No bundled responses. Target is `answer`, rule-based maths equivalence.

**The three `difficulty_*` columns are pass rates from probe rollouts** at 1.5B,
7B and 32B. That is a ready-made curriculum signal, and it can seed selection
*before* we spend rollouts — though our own measurement against our own student
remains the one that decides.

Preprocessing: T2, T5, T8, T9, T10, T11. Low effort; the verl envelope was already
flattened upstream.

## JustinTX/WildSci

56.8K rows, CC-BY-4.0 — the cleanest licence in the catalogue.

```jsonc
{"paper_id": "...", "discipline": "physics", "question": "...",
 "options": {"A": "...", "B": "...", "...": "up to J"},
 "answer": "A", "rationale": "...", "voting_type": "majority_aligned",
 "voting_answers": ["A", "A", "B"]}
```

Science MCQs synthesised from Nature Communications papers — original content, not
a repackaging. Target is the answer letter.

The bundled `rationale` is synthetic and of uneven quality; discard it. The
`voting_*` columns record an ensemble's agreement and are worth keeping as a
**filter**: low-consensus items are where the ground truth is least trustworthy.

Preprocessing: T2 (question plus formatted options into one user turn), T5, T8,
T9, T10, T11.

## OpenDataArena/ODA-Fin-RL-12k

12.2K train + 200 test, Apache-2.0 on the curation — but the underlying pool
aggregates 22 finance sources with their own licences. Verify per `data_source`
before commercial use.

```jsonc
{"data_source": "takala/financial_phrasebank", "ability": "finance",
 "prompt": [{"role": "user", "content": "Analyze the sentiment..."}],
 "reward_model": {"ground_truth": "neutral", "style": "model"},
 "extra_info": {"id": "...", "task": "...", "token_length": 42}}
```

Canonical verl row convention. Targets are deliberately short — at most about 16
tokens — chosen so a judge can grade them reliably.

**`reward_model.style` is `"model"`, not `"rule"`** — this dataset expects an LLM
judge, not string matching. That routes it straight onto our existing judge
mechanism rather than needing a programmatic verifier. Domain diversity is its
main value: nothing else in the catalogue is finance.

Preprocessing: T1, T2, T5, T8, T9, T10, T11. `extra_info.task` is a good
stratification key.

## nvidia/Nemotron-RL-instruction_following

**46,391 rows**, confirmed independently by the dataset card and by metrics
computed locally from the downloaded file. Single-turn throughout. Licence is
`odc-by` — the prompts derive from a corpus under that licence, and the Apache-2.0
label in the Gym config is simply stale for the data.

```jsonc
{"id": 17616,
 "responses_create_params": {"input": [...], "tools": [], "parallel_tool_calls": false},
 "verifier_metadata": {"instruction_id_list": ["paragraphs:paragraphs",
                                               "length_constraints:number_words"],
                       "prompt": "...", "kwargs": [{...}, {...}],
                       "grading_mode": "binary"}}
```

Scored by the same instruction registry our own checker is built on. **The direct
in-place test for the IF verifier**: our corrected checker should reproduce the
existing English results and differ only where a documented correction applies.

Preprocessing: T2, T5, T6, T7, T8, T9, T10, T11.

## nvidia/Nemotron-RL-SysBench-v1 and nvidia/Nemotron-RL-CFBench-v1

The two most on-point datasets in the catalogue: **each row already carries both
mechanisms**, an IFEval-style instruction list and a list of natural-language
YES/NO judge questions.

```jsonc
{"id": "...",
 "agent_ref": {"type": "responses_api_agents", "name": "turing_vif_simple_agent"},
 "instructions": [{"instruction_id": "keywords:existence", "uid": "...",
                   "is_misalignment_check": false}],
 "llm_judge": [{"content": "Are these results listed in order of severity?",
                "uid": "...", "is_misalignment_check": false}],
 "responses_create_params": {"input": [{"role": "system", "content": "..."},
                                       {"role": "user", "content": "..."}]},
 "tools": []}
```

CFBench is 1,105 rows; SysBench's exact count is unresolved. Both are **genuinely
multi-turn** — confirmed by sampling role sequences, two to nine prior turns of
realistic conversation before a final constrained user turn. The constraints attach
to that last turn. An earlier reading of these as single-turn was wrong.

Neither has a verifier in the Gym checkout — the referenced agent is not present —
so the verifier is ours to write.

### The instruction-id inventory, and what it implies

This was the main unknown. Stratified sampling of the raw files gives:

| | Distinct ids sampled | In our registry | **Outside it** |
|---|---:|---:|---:|
| SysBench | 25 | 11 (44%) | **14 (56%)** |
| CFBench | 44 | 11 (25%) | **33 (75%)** |
| Union | 50 | 12 (24%) | **38 (76%)** |

The gap is not a tail of variants — it is **whole families that IFEval never had**:

- `situation:*` — task_specific, audience_alignment, role_based, environment_setting, temporal_context, emotional_alignment
- `stylistic:*` — tone_formality, voice, politeness, sensory_detail, emotive_adjectives, sentence_tone_consistency
- `linguistic:*` — grammatical_mood, pragmatic_context, speech_act, syntactic_pattern
- plus further `detectable_format:*`, `change_case:*` and `length_constraints:*` variants

A full scan would likely surface 60–100+ distinct ids. **Implementing checkers for
all of them is a multi-week effort**, and worse, most of them should not be
programmatic at all: "tone formality" and "emotional alignment" are judgement calls
wearing an instruction-id costume.

**The datasets already answer this.** Each row carries `llm_judge` items alongside
`instructions`. So the ~76% we cannot check programmatically route to the rubric
mechanism, and the ~24% we can route to the checker. That is not a workaround — it
is the hybrid design arriving from the data rather than from us, and it is a strong
independent argument for the split. See D8.

### Two flags

`is_misalignment_check` is true on only about 2% of sampled instructions. It appears
to mark adversarial constraints — testing whether a model over-applies a rule
against conflicting context — but **no scoring code exists locally to confirm the
semantics**. Scoring a trap as an ordinary criterion would invert its purpose.

CFBench rows carry `used_in: ["ultra_v3"]`. **Check for evaluation contamination
before regenerating against it** — a training set built from an eval slice is worse
than no training set.

Preprocessing: T2, T3, T5, T6, T8, T9, T10, T11, plus mapping `llm_judge` items
onto rubric criteria — they arrive unweighted, so a weighting policy is needed.

## nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1

31.1K rows, CC-BY-4.0. Despite the name, **this is not a pivot in the sense that
matters** — it needs no tool payload and no container.

```jsonc
{"uuid": "...", "task_name": "...",
 "responses_create_params": {"input": [{"role": "...", "content": "..."}]},
 "expected_answer": "{\"commands\": [{\"keystrokes\": \"...\"}], \"is_task_complete\": false}",
 "metadata": {"harness": "terminus_1", "pivot_agent_turn_index": 3,
              "total_source_agent_turns": 11}}
```

The action is a JSON blob emitted as assistant **text**, and the terminal history
is already flattened into the prompt. Gradeable by JSON-schema validation plus
similarity against the expert keystrokes.

31,111 rows, drawn from 630 containerised terminal tasks and 2,716 verifier-passing
trajectories — a median of about 45 pivots per task.

It is still a single-step slice, so it is **Pivot class for regeneration purposes**
— one action per row, not a trajectory. But it needs no framework change to score,
which makes it the cheapest agentic-flavoured data available.

**Only rows at pivot index 0 carry a standalone task.** Deeper pivots depend on the
specific container state their trajectory produced, which we do not have. So full
regeneration is available for at most the ~2,716 trajectory-opening rows, and the
other ~28K are single-step RL only.

---

# Group B — pivots: imported RL data, not a renewable source

Three tool-call pivot datasets, all CC-BY-4.0, all consumed by the same comparator:
SWE 50.3K, Conversational Tool Use 97.0K, Function Calling 9,600. Note both the SWE
and conversational cards state larger figures than the live files carry — trust the
files.

```jsonc
{"trajectory_id": "...", "info": {"turn": 4, "step": 9, "depth": 2},
 "responses_create_params": {
   "input": [...],
   "tools": [{"type": "function", "strict": true, "name": "...",
              "description": "...", "parameters": {...}}],
   "parallel_tool_calls": false},
 "expected_action": {"type": "function_call", "name": "...", "arguments": "{...}"},
 "agent_ref": {...}}
```

The SWE set additionally carries `ref_patch` and `ref_message`; the conversational
set carries `scenario` and per-model reward statistics.

Scoring is exact tool-name match, then recursive argument comparison — dict keys
must match, list lengths must match, floats within an epsilon, strings by Jaccard
word overlap against a threshold. Parallel calls are matched bipartitely.

**No container, no execution, no trajectory replay.** The expert ground truth was
extracted at data-build time. What they need is tool schemas in the request
payload, which is the one framework change.

**What they cannot do**, as pivots, is give us fresh trajectories. The prefix is one
model's path through a state space; regenerating from it yields an action at a state
our model would not necessarily have reached.

### But two of the three retain a reusable original task

This is the useful nuance, and it differs per dataset:

| | Original task present? | Path to full regeneration |
|---|---|---|
| **Conversational Tool Use** | yes — the policy system prompt plus the opening customer message, self-contained | **Strongest candidate.** Gym already ships a conversational tool-use simulation environment with domain, policy and scenario generation plus user and tool simulators. It can consume this policy/scenario content directly and produce fresh multi-turn trajectories — no new infrastructure |
| **SWE** | yes — the original GitHub issue is the first user turn | Possible, but grading fresh trajectories needs a SWE-bench-style harness: repo checkout, patch application, test suite. A genuine capability gap, not a data-prep step |
| **Function Calling** | **unverified** — structurally analogous to the other two, but not confirmed this pass | Spot-check before relying on it |

So the conversational set is worth treating as two datasets in one: a pivot slice
usable immediately, and a scenario corpus that an existing simulator can turn into
renewable multi-turn data. That is the single best regeneration opportunity in the
whole catalogue.

Preprocessing: T1, T5 (the expected action into the target field), T6 (comparator
knobs), T8, T9, T10, T11, T12 (tools into their own column). Validate every row
with the pivot validator before anything downstream sees it.

---

# Group C — blocked

Not usable until a blocker clears. Listed with what specifically blocks them, so
the blocker rather than the dataset is what gets decided.

## Blocked on code execution

| Dataset | Rows | Licence | Target |
|---|---|---|---|
| Kwai-Klear/KlearReasoner-CodeSub-15K | 15.0K | Apache-2.0 | stdin/stdout tests in `reward_model.ground_truth` |
| PrimeIntellect/deepcoder-gold-standard-solutions | 16.3K | **unclear — flag** | stdin/stdout tests in `verification_info` |
| TIGER-Lab/AceCode-V2-122K | 122.6K | Apache-2.0 data, MIT code | list of Python `assert` statements |
| nvidia/Nemotron-RL-coding-competitive_coding | 16.1K | **cc-by-sa-4.0** | stdin/stdout test cases |

All four are otherwise full-regeneration-capable: self-contained prompt, target
that survives regeneration, no bundled response worth keeping. **They convert from
blocked to ready the moment a sandbox exists**, which makes the execution decision
worth roughly 170K rows across four sources rather than one.

Two are from the same DeepCoder lineage and will overlap substantially. The
gold-standard set carries reference solutions that upstream explicitly does not
guarantee pass their own tests — useful for sanity-checking a verifier, not as
imitation targets.

AceCode is the cleanest candidate to build against first: permissive licence on
both data and code, and a small, readable upstream verifier using a process pool
with per-test timeouts.

## Blocked on containers

**Zhongzhi1228/Recursive-Task-Synthesis** — 37,484 rows, CC-BY-4.0 with a real
LICENSE file in the repo. Each row carries an instruction, task metadata, a
reference `solve.sh` and a per-task Dockerfile — **all four are in the parquet**,
contrary to an earlier reading.

What is *not* there is the verifier: the task metadata's verifier section carries
only a timeout, and the actual pass/fail logic lives inside the packaged tar
shards. So the row tells you how to build the environment and what a correct
solution looks like, but not how correctness is decided. No execution-free path
exists.

The one borrowable thing is its task-metadata convention (difficulty, category,
tags, time estimates) as a template for our own.

## Blocked on licensing

**Multilingual-Multimodal-NLP/IfEvalCode-Instruct** — 3,000 rows across 8
programming languages. **Neither the data nor the code carries a licence** — no tag
in the card metadata, and no LICENSE file and no licence text in the upstream repo,
both confirmed by direct inspection. All rights reserved by default, so
copy-and-adjust is unavailable; only a clean-room reimplementation from the paper.

Despite the name and a bilingual upstream benchmark, this derivative appears to be
**English-only** — five samples spanning the full row range all carried English.

Worth understanding anyway, because its design is a genuine alternative to ours.
Each row carries two checks: functional correctness, and a **per-example generated
Python assertion function** that inspects the produced code as text.

- We generate constraints from a verified taxonomy, so everything sampled is
  scoreable by construction.
- They generate a bespoke checker per example — far more expressive, able to
  express constraints no taxonomy anticipates — at the cost that the checker is
  untrusted generated code which must be executed.

Its results structure is also, already, close to an evidence dict: two booleans
plus per-check program, stdout, stderr, exit code and status. Useful confirmation
that the shape we chose is the natural one.

---

# Group D — aggregates, take the execution-free slice

Two packagings of the same Guru line appear on the list. They are one source with a
choice of packaging, not two datasets.

| | IFM/guru-RL-v1.5 | AmanPriyanshu/rlvr-guru-raw-data-extended |
|---|---|---|
| Licence (data) | MIT | ODC-BY over ~30 upstream sources |
| Licence (verifier code) | — | Apache-2.0 (the Reasoning360 verl fork) |
| Rows | ~108.8K claimed, **unverified** — the card and the file layout disagree | **150,000 train / 221,332 test, confirmed** |
| Schema verified against real rows | no — the viewer cannot read its layout | yes |

### The execution-free slice is 82.9%, and we know exactly which rows

The second packaging's `data_source` values were enumerated against the actual
parquet files, and the row counts sum to exactly the stated 150,000:

| Domain | Rows | Scorer | Execution |
|---|---:|---|---|
| `math__deepscaler_preview` | 19,142 | regex answer match | no |
| `math__merged_deduped_dapo_or1_dataset` | 19,142 | regex answer match | no |
| `stem__web` (WebInstruct-Verified) | 21,701 | LLM judge | no |
| `stem__medmcqa_train` | 19,142 | **not wired upstream** — needs a trivial MC exact-match verifier | no |
| `stem__commonsenseqa_train` | 9,741 | **not wired upstream** — same | no |
| `simulation__codeio` | 12,117 | JSON/string compare | no |
| `simulation__barc` | 3,398 | grid compare | no |
| `simulation__arcagi1` / `arcagi2` | 297 / 653 | grid compare | no |
| `logic__graph_logical_dataset` | 8,004 | rule-based | no |
| `logic__ordering_puzzle_dataset` | 8,000 | rule-based | no |
| `logic__zebra_puzzle_dataset` | 80 | rule-based | no |
| `table__multihier` | 2,899 | rule-based | no |
| `codegen__primeintellect` | 11,273 | sandboxed execution | **yes** |
| `codegen__taco` | 11,052 | sandboxed execution | **yes** |
| `codegen__leetcode2k` | 2,386 | sandboxed execution | **yes** |
| `codegen__livecodebench` | 599 | sandboxed execution | **yes** |
| `codegen__mbpp` | 374 | sandboxed execution | **yes** |

**Execution-free: 124,316 rows (82.9%). Execution-gated: 25,684 (17.1%), all
`codegen__*`.** The filter is a single prefix test on `data_source`.

The test split additionally carries IFEval, IFBench and LiveBench slices, all
rule-based — directly relevant to the instruction-following work and a second
independent check on our own checker.

Two small pieces of work fall out: the medical and commonsense multiple-choice
domains (28,883 rows between them) are **not wired into the upstream dispatcher**
at all, so they need a multiple-choice exact-match verifier written. It is trivial,
but it will not arrive for free with the port.

**Recommendation: take the ODC-BY packaging.** It is the one whose row counts and
schema were verified against real files, and the filter boundary is exact. The MIT
licence on the other is attractive but its row count could not be confirmed and its
layout defeats the viewer — verify it before preferring it on licence grounds
alone.

The composite licensing is the real caution: the card says to check roughly thirty
upstream sources individually, which is a per-source review rather than one grant.

The genuinely valuable artefact in this line is not the data but the verifier: the
IFEval and IFBench sub-scorers in the Reasoning360 fork are Apache-2.0,
execution-free, and a direct port of the same upstream our own checker derives
from. Two independent ports should agree on English rows — where they disagree, one
has a bug.

Neither packaging should be ingested alongside Skywork-OR1 or the DeepCoder family
without dedupe; the maths and code domains overlap both.
