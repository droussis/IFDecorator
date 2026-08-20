# Decisions needed before code is written

Five items. D1 and D3 block Plan 01 outright; the rest shape it.

---

## D1 — Sign-off on IFEval as new scope

**Owner decision. Blocks everything.**

`AGENT_RUNBOOK.md` §1.3 decision 7 excludes IFEval, InverseIFEval and
MultiChallenge, and §1.3 decisions are marked "do not relitigate". The stated
reason is "need framework changes (tools in the payload) or have rubric-shaped
goldens".

Neither applies to this proposal: no framework change, no new pool column, and the
goldens are programmatic rather than rubric-shaped. But the exclusion is explicit
and this is an extension of scope, not a correction of an error.

**Asking for:** approval to add `verification_type: "instruction_following"` on the
`constraint` track, and a target size for the slice.

---

## D2 — Language coverage for the 8 gap languages

**Owner decision. Shapes Plan 01's scope.**

22 of the 30 `TARGET_LANGUAGES` are verifiable today. The gap is
`hr is nb sl sr tr mk sq`.

| Option | Cost | Risk |
|---|---|---|
| **A. Ship 22, emit `not_applicable` for the rest** *(recommended)* | none | those 8 get no IF rows; the pool is thinner there |
| **B. Extend coverage first** | measurement work per language; `hr`/`sl`/`sr` may be unfixable, since lid.176's separate `sh` label splits the mass four ways | delays the slice |
| **C. Ship all 30 unverified** | none up front | **do not.** A false FAIL enters `{check_results}` as evidence the judge is instructed to weigh. Wrong evidence is worse than no evidence |

Option A degrades to "no evidence", which the framework already handles: rows with
no applicable checks get the `NO_CHECKS` sentinel rather than an empty string, so
`--skip-empty-fields` never drops them.

Two cheap partial fixes regardless of choice:

- **`nb` is a label mismatch, not a coverage gap.** lid.176 emits `no` for Bokmål.
  A code-mapping entry may be all Norwegian needs — worth an hour to check.
- **`tr` needs a casefold review before it can be trusted at all.** Python's default
  casing is wrong for the dotted/dotless I, which affects every case-based check.

---

## D3 — Will `run_verifiers.py` ever run in production?

**Owner decision. Determines whether Plan 01 is worth building.**

Per §14.4, the production judging phase maps `check_results:checks` — the raw check
*specs*, not executed *results* — and never invokes `run_verifiers.py`. The full
`generate → split_raw_responses → run_verifiers → judge` chain exists only in
`smoke_test_verifiable.sh`, with the note "nobody runs programmatic checks
mid-vacation".

If that is permanent, an IF verifier is **inert**: the judge is told which
instructions would have been checked, never what was found, and the entire value
of a corrected multilingual checker is lost.

**Asking for:** confirmation that the production chain will invoke `run_verifiers.py`
for IF rows. If not, Plan 01 should not be built — the honest alternative is to
express IF constraints as rubric `hard_rule` items instead, which the framework
already supports and which needs no new code.

This is the single most consequential question in either plan.

---

## D4 — How a hybrid row is graded

**Owner decision. Blocks Plan 02 phase 3 only; phases 1–2 proceed without it.**

§1 states the three reward tracks are "never mixed inside one training run". A row
carrying both programmatic checks and rubric criteria departs from that. Options A
/ B / D are tabulated in `PLAN_02_hybrid_synthesis.md` §4; option C (two rows) is
blocked by `prompt_hash` dedupe.

**Recommendation:** do not decide this yet. Ship phase 1 (option A, no schema
change), then measure whether rubric criteria add reward variance the programmatic
checks do not already capture. For IF prompts the overlap is likely to be large —
the rubric creation template already mandates a weight-9–10 `hard_rule` for
negative user constraints, which is exactly what the programmatic checker covers.

A hybrid track that adds a rubric-judge pass per row without adding variance costs
real money and buys nothing.

---

## D5 — Where the vendored verifier lives, and how it stays in sync

**Technical, low stakes, but decide before writing.**

The owner's preference is copy-and-adjust over an external dependency. Two
consequences:

1. **`verifiers/instruction_following/` is a subpackage**, not a flat module.
   `verifiers/` is three files of ~230 lines total; the vendored core is larger
   than all of them. A directory keeps `verifiers/` readable and marks the
   boundary.
2. **`PROVENANCE.md` records source, commit and a re-sync recipe**, following the
   pattern already used for vendored code elsewhere in the org's repos.

The one thing that must not drift is `profile.yaml`: it is the shared contract
between synthesis and verification, and if the copies diverge, synthesis emits
constraints verification cannot score. Keeping one file, copied verbatim, is the
whole mechanism.

**Not vendorable:** the fastText LID model is CC-BY-SA and cannot be committed to
an Apache-licensed repo. `setup_lid.py` resolves it at startup instead.
