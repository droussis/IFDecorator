# Dataset reference

Per-dataset schema, mechanism and mapping. Strategy is in
`PLAN_03_dataset_onboarding.md`; tiers referenced here are defined there.

Sources: the Hugging Face dataset cards, and — where a NeMo Gym resources server
consumes the dataset — that server's code, which is treated as ground truth for row
shape and scoring.

## Summary

| Dataset | Tier | Mechanism | Tools | Exec | Gym server |
|---|---|---|---|---|---|
| Nemotron-RL-instruction_following | T0 | programmatic (IFEval registry) | no | no | `instruction_following` |
| Nemotron-RL-SysBench-v1 | T0 | **hybrid: programmatic + YES/NO rubric** | no | no | none |
| Nemotron-RL-CFBench-v1 | T0 | **hybrid: programmatic + YES/NO rubric** | no | no | none |
| Nemotron-RL-Agentic-Terminal-Pivot-v1 | T0 | schema validation + similarity, optional judge | no | no | `terminus_judge` [inferred] |
| Nemotron-RL-Agentic-SWE-Pivot-v1 | T1 | tool-argument comparison | yes | no | `single_step_tool_use_with_argument_comparison` |
| Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 | T1 | tool-argument comparison | yes | no | same |
| Nemotron-RL-Agentic-Function-Calling-Pivot-v1 | T1 | tool-argument comparison | yes | no | same |
| Nemotron-RL-coding-competitive_coding | T2 | test execution | no | **yes** | `code_gen` |

**Four of the eight NVIDIA datasets need no framework change at all.** Only one
needs code execution.

---

## The two most valuable: SysBench and CFBench

These are the closest thing to what we are trying to build, already in dataset
form. Each row carries **both** mechanisms side by side:

```jsonc
{
  "id": "...",
  "agent_ref": {"type": "responses_api_agents", "name": "turing_vif_simple_agent"},
  "instructions": [
    {"instruction_id": "keywords:existence", "uid": "...", "source": "...",
     "is_misalignment_check": false, "...": "kwargs inline"}
  ],
  "llm_judge": [
    {"content": "Are these analysis results listed in order of severity?",
     "uid": "...", "source": "...", "is_misalignment_check": false}
  ],
  "responses_create_params": {"input": [{"role": "system", "content": "..."},
                                        {"role": "user", "content": "..."}]},
  "tools": []
}
```

`instructions` is an IFEval-style id list. `llm_judge` is a list of
natural-language YES/NO questions — structurally a rubric without weights.

Four things follow.

1. **This validates the hybrid design and gives us a ready-made evaluation of it.**
   We do not have to invent the shape or bootstrap it; we have to map it.
2. **Neither has a Gym server in this checkout.** The agent they reference,
   `turing_vif_simple_agent`, is not present. So the verifier is ours to write —
   which is what "make them ours" means anyway, and it is Plan 01 plus the rubric
   mechanism, not new machinery.
3. **CFBench's multilingual coverage does not overlap ours.** Its languages are
   English, Arabic, Hindi, Chinese, Japanese and Korean — every non-English one is
   in a script we deliberately excluded. Only the English slice is verifiable by
   our checker today, and extending to the rest is not a language-profile addition
   but a re-opening of the script decision. Worth knowing before anyone plans
   around "CFBench is multilingual".
4. **New instruction families.** `stylistic:tone_formality` appears among CFBench's
   instruction ids and has no counterpart in the 54-instruction registry. Expect a
   tail of ids that need either an implementation or an honest `not_applicable`.
   Enumerate the full id set across both datasets before estimating.

`is_misalignment_check` on both instruction and judge items looks like a
deliberate-trap marker analogous to a reward-hacking tripwire. Confirm its
semantics before scoring — a trap item scored as an ordinary criterion would
invert its purpose. [inference]

---

## Nemotron-RL-instruction_following

`odc-by` on the HF card. Roughly 78 MB of JSONL.

```jsonc
{"id": 17616,
 "responses_create_params": {"input": [...], "tools": [], "parallel_tool_calls": false},
 "verifier_metadata": {"instruction_id_list": ["paragraphs:paragraphs",
                                               "length_constraints:number_words"],
                       "prompt": "...", "kwargs": [{...}, {...}],
                       "grading_mode": "binary"}}
```

Scored by the same `verifiable_instructions` registry our own verifier is built on,
reward `all(...)` or the pass fraction. **This is the direct in-place test for Plan
01**: our corrected checker should reproduce the existing English results and
differ only where a correction applies.

**Licence flag:** the HF card says `odc-by`; the Gym server's README footer says
"Data: Apache 2.0". Resolve against the card before redistributing.

---

## Nemotron-RL-Agentic-Terminal-Pivot-v1

31.1K rows, `cc-by-4.0`. The surprise of the set: **no `tools` field**.

```jsonc
{"uuid": "...", "task_name": "...", "tool_name": "...",
 "responses_create_params": {"input": [{"role": "...", "content": "..."}]},
 "expected_answer": "{\"commands\": [{\"keystrokes\": \"...\"}], \"is_task_complete\": false}",
 "metadata": {"harness": "terminus_1", "teacher_model": "...",
              "source_trajectory_uid": "...", "pivot_agent_turn_index": 3,
              "total_source_agent_turns": 11}}
```

The action is a **JSON blob emitted as assistant text**, not a function call, and
the prior terminal history is already flattened into the prompt. Scoring is
JSON-schema validation against a harness schema plus sequence similarity on the
concatenated keystrokes, with an optional LLM-judge equivalence path.

So this is a **text-in, text-out task**: T0, not T3. It needs no tool payload, no
container, and no agent loop. It is the cheapest agentic-flavoured data we can
onboard and it belongs in Phase A.

The Gym server match (`terminus_judge`) is inferred from schema fit — no local
config wires this repo, and that server's README says its data is not yet publicly
released. Verify before relying on it.

---

## The three tool-call pivots

SWE (50.3K), Conversational Tool Use (97.0K), Function Calling (9.6K), all
`cc-by-4.0`, all consumed by the same server and the same comparator.

```jsonc
{"trajectory_id": "...",
 "info": {"turn": 4, "step": 9, "depth": 2},
 "responses_create_params": {
   "input": [...],
   "tools": [{"type": "function", "strict": true, "name": "...",
              "description": "...", "parameters": {...}}],
   "parallel_tool_calls": false},
 "expected_action": {"type": "function_call", "name": "...", "arguments": "{...}"},
 "agent_ref": {...}}
```

SWE additionally carries `ref_patch` and `ref_message`; Conversational Tool Use
carries `scenario`, `num_unique_actions` and per-model reward statistics.

Scoring: exact tool-name match, then recursive argument comparison — dict keys must
match, list lengths must match, floats within an epsilon, strings by Jaccard word
overlap against a configurable threshold. Parallel calls are matched bipartitely.

**No container, no execution, no trajectory replay** — the expert ground truth was
extracted at data-build time. These are T1 purely because tools must reach the
request payload.

A second, looser comparator exists for SWE specifically (tool-name category
equivalence, path-suffix matching, sequence similarity, diff-size shaping). Worth
having, because a coding agent has many valid ways to express one edit and strict
matching penalises paraphrase rather than error.

`pass_rate` columns are present on the pivot datasets — the upstream difficulty
signal. Useful for ranking, but our own measurement against our own student is the
one that decides selection.

---

## Nemotron-RL-coding-competitive_coding

16.1K rows. The only T2 dataset in the NVIDIA set.

```jsonc
{"responses_create_params": {"input": [{"role": "user", "content": "..."}]},
 "verifier_metadata": {"unit_tests": {"inputs": ["..."], "outputs": ["..."]}},
 "hash_id": "...", "dataset": "...", "source": "..."}
```

LiveCodeBench-style stdin/stdout test cases. The Gym server extracts code from the
response and runs it against the tests through a Ray-distributed harness with a
timeout; reward is 1.0 only if every case passes. A sibling server for the same
problem class requires a Docker-based sandbox.

**Not portable without adding execution.** It is the concrete instance of the T2
decision in Plan 03 §4.

**Licence flag:** the HF card says `cc-by-sa-4.0`; the Gym config labels it Apache
2.0. Share-alike has real redistribution consequences — resolve before use.

---

## Open items

- **Two licence mismatches** between Gym config metadata and the live HF cards
  (`instruction_following`, `competitive_coding`). The card is the safer authority.
- **SysBench and CFBench mechanisms are inferred from row schema**, not read off a
  verifier, because no server for them exists in this checkout.
- **SysBench multi-turn status unconfirmed** — the card is tagged multi-turn; the
  sampled row was a single system + user turn.
- **Full instruction-id inventory not yet taken** across SysBench and CFBench.
  Do this before estimating Phase A: it determines how much of Plan 01's registry
  is reused and how much is new.

---

# Non-NVIDIA datasets

| Dataset | Tier | Mechanism | Exec | Licence of the verifier code |
|---|---|---|---|---|
| rlvr-guru-raw-data-extended | **T0 for part of it**, T2 for the rest | domain-routed; the IFEval/IFBench slice is rule-based | partial | Apache-2.0 |
| AceCode-V2-122K | T2 | Python asserts → pass rate | yes | MIT |
| IfEvalCode-Instruct | T2 | unit tests + a generated per-example assertion function | yes | **none — see below** |
| Recursive-Task-Synthesis | T3 | Docker-sandboxed terminal task, private verifier | yes | Apache-2.0 (framework) |

## Two findings that change the plan

### None of the twelve advances the multilingual goal

Every non-NVIDIA dataset is English-only. The one multilingual NVIDIA dataset,
CFBench, covers Arabic, Hindi, Chinese, Japanese and Korean — all outside the
Latin/Cyrillic/Greek scope by design.

**So these datasets buy task diversity, not language coverage.** Greek and the
other European targets still have to come from our own synthesis and from language
extension. Worth stating explicitly, because "onboard twelve datasets" and "add
languages" read like one workstream and are two.

### Licences are a real constraint here, not a formality

| Dataset | Issue |
|---|---|
| **IfEvalCode-Instruct** | **No licence at all.** No tag on the dataset card, and no LICENSE file in the upstream code repo — which means all rights reserved by default. Copy-and-adjust is not available; a clean-room reimplementation from the paper would be, at higher cost |
| competitive_coding | `cc-by-sa-4.0` on the card. Share-alike has downstream consequences |
| instruction_following | `odc-by` on the card, Apache 2.0 claimed in the Gym README |
| rlvr-guru | ODC-BY in aggregate, but the card says to check roughly thirty upstream sources individually. Composite licensing, not one grant |
| AceCode-V2 | MIT code, Apache-2.0 data — the cleanest of the set |
| Recursive-Task-Synthesis | CC-BY-4.0 data, Apache-2.0 framework — clean, but the verifier is not in the parquet |

Route these through whoever owns licence decisions before any copying. Two of them
would be caught late otherwise.

## rlvr-guru-raw-data-extended

150K train / 221K test, in verl row format:

```jsonc
{"data_source": "codegen__leetcode2k",
 "prompt": [{"role": "user", "content": "..."}],
 "ability": "...",
 "reward_model": "{\"ground_truth\": ..., \"style\": \"rule\"}",
 "extra_info": "{\"dataset\": ..., \"difficulty\": ..., \"reference\": ...}"}
```

A re-serialisation of about thirty benchmarks behind one schema, with
`data_source` as the routing key. The verifier lives in the Reasoning360 fork of
verl, under per-domain reward-score modules.

**The valuable part is a slice, not the whole.** The `ifeval` and `ifbench`
sub-scorers are Apache-2.0, execution-free, and a direct port of Google's original
instruction-following evaluation — the same lineage as our own checker. The test
split alone carries roughly 95K IFBench and 541 IFEval rows.

Two consequences:

1. **Filter by `data_source` and onboard the rule-based domains at T0.** Math,
   logic, MCQ, IFEval and IFBench need no execution. Only the code and simulation
   domains do.
2. **The IFEval/IFBench scorer is the single most reusable artefact across all
   twelve datasets** — permissively licensed, execution-free, and aligned with the
   work already done. It is also a useful cross-check: two independent ports of the
   same upstream should agree on English rows, and where they disagree, one of them
   has a bug.

## AceCode-V2-122K

122,603 rows, Apache-2.0 data, MIT code.

```jsonc
{"id": "...", "question": "...",
 "tests": ["assert filter_palindrome_anagrams([...]) == [...]", "..."],
 "source": "oss | evol | stack_python"}
```

Reward is the fraction of asserts that pass, per sample. The upstream verifier uses
a process pool with per-test timeouts rather than containers.

Python-only, English-only. **The cleanest T2 candidate we have** — small,
permissively licensed, and a good reference for timeout and isolation handling if
and when execution is opened. Not usable before then.

## IfEvalCode-Instruct

3,000 rows across 8 programming languages (Python, Java, C++, C#, TypeScript,
JavaScript, PHP, Shell).

```jsonc
{"question": "...", "programming_language": "javascript", "language": "en",
 "check_correctness": "<test harness in the target language>",
 "check_instruction": "<a Python function that regex-inspects the generated code>",
 "eval_results": {"if_correct": true, "if_instruction": true,
                  "if_correct_logs": {"program": "...", "stdout": "...",
                                      "stderr": "...", "exit_code": 0,
                                      "status": "..."},
                  "if_instruction_logs": {"...": "..."}}}
```

Two independent programmatic checks per row: functional correctness, and
instruction-following. **The instruction check is the interesting part** — it is
not a fixed taxonomy of instruction ids but a **per-example generated Python
assertion function** that inspects the produced code as text.

That is a genuinely different hybrid design from ours, and worth understanding even
though the dataset is unusable as-is:

- **We generate constraints from a verified taxonomy** so that everything sampled
  is scoreable by construction.
- **They generate a bespoke checker per example**, which is far more expressive —
  it can express constraints no taxonomy anticipates — at the cost of the checker
  itself being untrusted generated code that has to be executed.

Its `eval_results` shape is also, structurally, already an evidence dict — two
booleans plus per-check program, stdout, stderr, exit code and status. That is
close to what our verifier contract asks for, and a useful confirmation of the
shape.

**Blocked by licensing.** All rights reserved by default, so neither the harness
nor the data can be copied. If the generated-checker pattern is wanted, it has to
come from the paper, not the repo.

## Recursive-Task-Synthesis

37,484 rows, CC-BY-4.0 data, Apache-2.0 framework (Harbor).

Each row carries an instruction, a `task_toml` of metadata, a reference `solve.sh`,
and a per-task Dockerfile. The actual verifier — a test script plus a state check —
**is not in the parquet at all**; it lives in the tar shards and runs inside the
container against the final workspace state.

Full T3: a Docker sandbox per task is mandatory and there is no execution-free path.
The one execution-free thing worth borrowing is the `task_toml` metadata convention
(difficulty, category, tags, time estimates) as a template for our own task
metadata.
