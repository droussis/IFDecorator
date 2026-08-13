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
"""Parsers for the generator outputs produced by `prompts.py`.

Each one replaces an upstream parser that has a documented defect. The defects
matter more than they look: a parser that silently returns the wrong thing
produces plausible-looking rows that poison the curriculum, and there is no stage
downstream that would catch it.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from dataclasses import dataclass, field


_VERDICT_MARKERS = ("final verification", "final verdict", "final_ver")


def parse_verdict(response: str | None, strict: bool = False) -> bool:
    """Parse a `**Final Verification:** YES/NO` tail into a boolean.

    Replaces modules/utils.py::unified_judge_parse, which has no `return` in its
    non-strict branch and therefore returns None on the default path - making
    every loose judgement in the pipeline falsy. The working implementation lives
    in recipe/reward/cif.py:103; this is that logic, hardened.

    Fails closed: an unparseable verdict is False, never True.
    """
    if not response:
        return False

    if strict:
        return bool(
            re.search(
                r"\*{0,2}final[ _]verification:?\*{0,2}\s*[\"'<*]*\s*yes",
                response,
                re.IGNORECASE,
            )
        )

    text = response.lower()
    for marker in _VERDICT_MARKERS:
        if marker in text:
            tail = text.rsplit(marker, 1)[-1]
            # Take whichever of yes/no comes FIRST after the marker, and only
            # within a short window. Checking "yes" before "no" would let trailing
            # commentary ("...NO. the response says yes to the question") flip a
            # NO verdict. Word boundaries matter too: "no" is a substring of
            # "not", "none" and "know".
            match = re.search(r"\b(yes|no)\b", tail[:40])
            if match:
                return match.group(1) == "yes"
            break
    return False


@dataclass
class Decomposition:
    task_description: str | None = None
    constraints: list[str] = field(default_factory=list)
    input_text: str | None = None


def parse_decomposition(content: str) -> Decomposition:
    """Parse the W1 decompose output.

    Replaces w1_decompose.py::process_llm_response, which appends ANY line
    starting with "-" to `constraints` as long as a task description has been
    seen - including bullets that appear under `#Input:`. Section membership is
    tracked explicitly here instead.
    """
    result = Decomposition()
    section = None

    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("#task description:"):
            section = "task"
            result.task_description = line.split(":", 1)[1].strip() or None
        elif lowered.startswith("#constraints:"):
            section = "constraints"
            inline = line.split(":", 1)[1].strip()
            if inline and inline.lower() != "null":
                result.constraints.append(inline)
        elif lowered.startswith("#input:"):
            section = "input"
            inline = line.split(":", 1)[1].strip()
            result.input_text = None if inline.lower() == "null" else (inline or None)
        elif line.startswith(("-", "*", "•")) and section == "constraints":
            constraint = line.lstrip("-*• ").strip()
            if constraint and constraint.lower() != "null":
                result.constraints.append(constraint)
        elif section == "input":
            # Accumulate continuation lines whether or not `#Input:` had inline
            # content, and whether or not they are bulleted. Input blocks are
            # frequently multi-line and frequently contain bullets - which is
            # exactly the case upstream mis-parsed as constraints.
            result.input_text = (
                line if result.input_text is None else f"{result.input_text}\n{line}"
            )

    return result


def parse_classification(content: str, expected: int) -> list[str]:
    """Parse the W2 output into per-constraint labels.

    Returns "rubric" for anything unparseable. That direction is deliberate: a
    constraint wrongly marked programmatic becomes an unenforceable reward
    signal, while one wrongly marked rubric only costs a judge call.
    """
    labels: dict[int, str] = {}
    for match in re.finditer(
        r"#\s*(\d+)\s*:\s*(programmatic|rubric)", content or "", re.IGNORECASE
    ):
        labels[int(match.group(1))] = match.group(2).lower()
    return [labels.get(i + 1, "rubric") for i in range(expected)]


@dataclass
class Evolution:
    rationale: str | None
    constraint_type: tuple[str, str] | None
    constraint: str | None
    enhanced_instruction: str | None

    def is_usable(self) -> bool:
        return bool(self.constraint and self.enhanced_instruction)


def parse_evolution(response: str) -> Evolution | None:
    """Parse the W4 evolution output.

    Replaces w4_evol.py::parse_evol_response, which splits the whole response on
    "#" - so any "#" inside the generated instruction (a markdown heading, a C
    preprocessor line, a hex colour) corrupts every subsequent field. Anchors on
    line-initial field markers instead.

    Returns None when the required fields are absent, so the caller can retry.
    """
    if not response:
        return None

    fields: dict[str, str] = {}
    current = None
    for raw_line in response.splitlines():
        match = re.match(
            r"^\s*#\s*(rationale|constraint_type|constraint|enhanced_instruction)\s*:\s*(.*)$",
            raw_line,
            re.IGNORECASE,
        )
        if match:
            current = match.group(1).lower()
            fields[current] = match.group(2).strip()
        elif current:
            fields[current] = (fields[current] + "\n" + raw_line).strip()

    constraint_type = None
    if "constraint_type" in fields:
        parts = [p.strip() for p in fields["constraint_type"].split(",")]
        if len(parts) == 2:
            constraint_type = (parts[0], parts[1])

    evolution = Evolution(
        rationale=fields.get("rationale"),
        constraint_type=constraint_type,
        constraint=fields.get("constraint") or None,
        enhanced_instruction=fields.get("enhanced_instruction") or None,
    )
    return evolution if evolution.is_usable() else None


def parse_keywords(response: str, reference: str | None = None) -> list[str]:
    """Parse the keyword-extraction output.

    Upstream used `eval()` on a regex capture and fell back to an English word
    list. This uses ast.literal_eval, and - when a reference response is given -
    drops any keyword that does not actually occur in it. That filter is what
    makes the keyword usable: matching is surface-form, so a keyword absent from
    the reference is one the model would have to inflect, and it would never match.
    """
    if not response:
        return []

    candidates: list[str] = []
    match = re.search(
        r"\*{0,2}keywords:?\*{0,2}\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL
    )
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, (list, tuple)):
                candidates = [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            candidates = []

    if not candidates:
        tail = re.split(r"\*{0,2}keywords:?\*{0,2}", response, flags=re.IGNORECASE)[-1]
        candidates = [w for w in re.split(r"[,\n]", tail) if w.strip()]

    cleaned = []
    for candidate in candidates:
        word = (
            unicodedata.normalize("NFC", candidate.strip().strip("'\"[]").split()[0])
            if candidate.strip()
            else ""
        )
        if not word:
            continue
        if reference and word not in unicodedata.normalize("NFC", reference):
            continue
        cleaned.append(word)
    return cleaned


@dataclass
class Rubric:
    constraint: str
    criteria: list[str] = field(default_factory=list)
    pass_rule: str = "ALL"


def parse_rubric(content: str, constraint: str) -> Rubric:
    """Parse the reference rubric generator's output.

    The constraint is carried INSIDE the returned object on purpose. Upstream
    kept checklists in a bare list that was later zipped positionally against the
    full constraint list, which silently paired checklists with the wrong
    constraints. Keeping them bound removes that failure mode structurally.
    """
    criteria = [
        m.group(1).strip()
        for m in re.finditer(
            r"^\s*#\s*criterion_\d+\s*:\s*(.+)$", content or "", re.MULTILINE
        )
    ]
    rule_match = re.search(
        r"^\s*#\s*pass_rule\s*:\s*(ALL|MAJORITY)",
        content or "",
        re.MULTILINE | re.IGNORECASE,
    )
    return Rubric(
        constraint=constraint,
        criteria=criteria,
        pass_rule=(rule_match.group(1).upper() if rule_match else "ALL"),
    )


def parse_domain(content: str) -> str:
    """Parse the domain filter output. Unrecognized output is 'Other', i.e. kept."""
    text = (content or "").strip().lower()
    for label in ("math problem", "code task", "reasoning task"):
        if text.startswith(label) or f"\n{label}" in text[:200]:
            return label.title()
    return "Other"
