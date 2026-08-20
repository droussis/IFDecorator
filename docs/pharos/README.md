# Pharos integration plans

Plans for folding the instruction-following verification and synthesis work in this
repo into the Pharos data-generation libraries.

**These files are written for `data_team/plans_summaries/`.** They are staged here
because `data_team` and `training_data_acquisition` are not reachable from the
session that produced them. Copy them across unchanged; nothing in them depends on
this repo's layout.

| File | Purpose |
|---|---|
| `PLAN_01_if_verification.md` | Mechanism 1 — programmatic IF verification |
| `PLAN_02_hybrid_synthesis.md` | Generating IF prompts with programmatic checks *and* rubrics |
| `CHECKLISTS.md` | Tickable execution steps for both plans |
| `DECISIONS.md` | What needs owner sign-off before code is written |

Written against `VERIFICATION_ONBOARDING.md`. Section references (§8.4, G21, …)
are to that document. Per `data_team/AGENTS.md`, no line numbers are cited: files
and symbols only.

## Source material in this repo

| Here | Role |
|---|---|
| `multilingual_if/` | 54 IFEval-G checkers corrected for Latin/Cyrillic/Greek, with `profile.yaml` as the per-constraint/per-language eligibility matrix |
| `multilingual_if/lid.py` | calibrated fastText language identification |
| `synthesis/prompts.py` | the flywheel's prompt templates, multilingual-adapted |
| `synthesis/schema.py` | row schema, stage validation, difficulty routing |
| `docs/SYNTHESIS_PIPELINE.md` | the full flywheel, stage by stage |

The plans assume the **copy-and-adjust** approach rather than an external
dependency, per the owner's preference.
