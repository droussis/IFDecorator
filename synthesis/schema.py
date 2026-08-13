# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row schema and per-stage validation.

These are the checks that would have caught the defects found in the original
pipeline. Run `validate_row` at every stage boundary: the flywheel loops five
times, so a malformed row that survives one stage is amplified by the next four,
and `pass_rate` will look like a model signal rather than a data bug.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from multilingual_if import conflicts, eligible, supported_languages


class SchemaError(ValueError):
    """A row violates the contract. Always fatal - never score a row that fails."""


@dataclass
class Rubric:
    """A judged constraint and its criteria, bound together.

    The binding is the point. Upstream kept checklists in a bare list zipped
    positionally against the full constraint list, so a judged constraint
    appearing before a programmatic one shifted every checklist onto the wrong
    constraint (check_instruction.py:136, recipe/reward/cif.py:226). Never
    reintroduce a structure that allows a positional zip.
    """

    constraint: str
    criteria: list[str] = field(default_factory=list)
    pass_rule: str = "ALL"
    judge: str | None = None
    weight: float = 1.0


@dataclass
class Row:
    """One synthesized instruction-following task."""

    id: str
    language: str
    prompt: str
    prompt_wo_programmatic: str
    instruction_id_list: list[str] = field(default_factory=list)
    kwargs: list[dict[str, Any]] = field(default_factory=list)
    rubrics: list[Rubric] = field(default_factory=list)
    seed_response: str | None = None
    pass_rate: float | None = None
    round: int = 0
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Row":
        payload = dict(data)
        payload["rubrics"] = [
            Rubric(**r) if isinstance(r, dict) else r
            for r in payload.get("rubrics", [])
        ]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def validate_row(row: Row, stage: str = "unknown") -> None:
    """Raise SchemaError on anything that would corrupt downstream stages."""
    problems: list[str] = []

    if not row.language:
        problems.append(
            "`language` is missing. It is required, not optional: without it "
            "verification silently falls back to English, which reintroduces every "
            "defect the multilingual work removed."
        )
    elif row.language not in supported_languages():
        problems.append(
            f"language {row.language!r} is outside the verifiable set {supported_languages()}. "
            "Synthesizing it produces rows nothing can score."
        )

    if len(row.instruction_id_list) != len(row.kwargs):
        problems.append(
            f"instruction_id_list has {len(row.instruction_id_list)} entries but kwargs has "
            f"{len(row.kwargs)}; they are positional and must match."
        )

    if row.language in supported_languages():
        ineligible = [
            i for i in row.instruction_id_list if not eligible(i, row.language)
        ]
        if ineligible:
            problems.append(
                f"{ineligible} are not eligible for {row.language!r}. An ineligible constraint is "
                "either unsatisfiable (constant-zero reward) or vacuous (free reward); both are "
                "worse than omitting it."
            )

    forbidden = {tuple(sorted(pair)) for pair in conflicts()}
    present = set(row.instruction_id_list)
    for pair in forbidden:
        if set(pair) <= present:
            problems.append(
                f"{pair} are mutually contradictory and must not appear on one prompt."
            )

    if len(set(row.instruction_id_list)) != len(row.instruction_id_list):
        problems.append("duplicate entries in instruction_id_list.")

    for rubric in row.rubrics:
        if not rubric.constraint:
            problems.append(
                "a rubric has no `constraint`; rubrics must stay bound to their constraint."
            )
        if not rubric.criteria:
            problems.append(f"rubric for {rubric.constraint!r} has no criteria.")
        if rubric.pass_rule not in ("ALL", "MAJORITY"):
            problems.append(
                f"rubric pass_rule {rubric.pass_rule!r} must be ALL or MAJORITY."
            )

    if row.pass_rate is not None and not 0.0 <= row.pass_rate <= 1.0:
        problems.append(f"pass_rate {row.pass_rate} is outside [0, 1].")

    if not row.prompt or not row.prompt.strip():
        problems.append("`prompt` is empty.")

    if row.instruction_id_list and row.prompt_wo_programmatic == row.prompt:
        problems.append(
            "`prompt_wo_programmatic` equals `prompt` despite programmatic constraints being "
            "present. The judge must not see the programmatic constraints, or they are scored twice."
        )

    if problems:
        raise SchemaError(
            f"row {row.id!r} failed validation at stage {stage!r}:\n  - "
            + "\n  - ".join(problems)
        )


def route_by_difficulty(
    rows: list[Row],
    threshold_easy: float = 0.5,
    threshold_too_hard: float = 0.0,
) -> dict[str, list[Row]]:
    """Split tagged rows into easy / hard / too_hard pools.

    `hard` is the training set. `easy` goes back through evolution. `too_hard` is
    discarded - by default only rows nothing ever solved, since a pass_rate of
    exactly 0 across k rollouts is as likely to mean "unsatisfiable constraint" as
    "genuinely hard", and the profile exists to make the former impossible.
    """
    pools: dict[str, list[Row]] = {"easy": [], "hard": [], "too_hard": []}
    for row in rows:
        if row.pass_rate is None:
            raise SchemaError(
                f"row {row.id!r} has no pass_rate; run difficulty tagging before routing."
            )
        if row.pass_rate <= threshold_too_hard:
            pools["too_hard"].append(row)
        elif row.pass_rate > threshold_easy:
            pools["easy"].append(row)
        else:
            pools["hard"].append(row)
    return pools


def pass_rate_report(rows: list[Row]) -> dict[str, dict[str, float]]:
    """Per-language pass_rate summary - the primary diagnostic after a round.

    A language whose distribution is crushed toward 0 or 1 relative to the others
    signals a constraint that is unsatisfiable or vacuous there, i.e. a missing
    row in the profile. Check this before scaling past one language.
    """
    by_language: dict[str, list[float]] = {}
    for row in rows:
        if row.pass_rate is not None:
            by_language.setdefault(row.language, []).append(row.pass_rate)

    report = {}
    for language, rates in sorted(by_language.items()):
        ordered = sorted(rates)
        report[language] = {
            "n": float(len(ordered)),
            "mean": sum(ordered) / len(ordered),
            "median": ordered[len(ordered) // 2],
            "frac_zero": sum(1 for r in ordered if r == 0.0) / len(ordered),
            "frac_one": sum(1 for r in ordered if r == 1.0) / len(ordered),
        }
    return report
