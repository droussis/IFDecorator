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
"""Deterministic containment checks, replacing per-merge LLM calls.

Upstream `merge_evol` (w4_evol.py:367) spends two LLM calls per evolution attempt
asking "does text1 contain text2" - once for the original input block, once for
the instruction. With 5 rounds and multiple attempts per row that is the single
largest avoidable cost in the flywheel, and it is a question a string comparison
answers correctly almost every time.

Use `contains()`. Fall back to the CONTAINMENT_CHECK prompt only for the
ambiguous band, which `verdict()` identifies explicitly.
"""

from __future__ import annotations

import difflib
import unicodedata

import regex


_WHITESPACE = regex.compile(r"\s+")
_PUNCT = regex.compile(r"[\p{P}\p{S}]+")

# Above this, the needle is present. Below AMBIGUOUS_LOW, it is absent. Between
# the two, ask the model. The band is intentionally narrow: widening it trades
# cost for a decision the string comparison was already making correctly.
CONFIDENT_HIGH = 0.92
AMBIGUOUS_LOW = 0.60


def normalize(text: str, drop_punctuation: bool = False) -> str:
    """NFC, casefold, collapse whitespace. Optionally strip punctuation.

    NFC matters here for the same reason it matters in the verifier: the
    generator may echo text in a different normalization form than the seed, and
    a decomposed copy of an identical string will not compare equal.
    """
    text = unicodedata.normalize("NFC", text or "").casefold()
    if drop_punctuation:
        text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def similarity(haystack: str, needle: str) -> float:
    """Best-matching-block ratio of `needle` against `haystack`, in [0, 1]."""
    haystack_n = normalize(haystack, drop_punctuation=True)
    needle_n = normalize(needle, drop_punctuation=True)
    if not needle_n:
        return 1.0
    if not haystack_n:
        return 0.0
    if needle_n in haystack_n:
        return 1.0
    matcher = difflib.SequenceMatcher(None, haystack_n, needle_n, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return min(1.0, matched / len(needle_n))


def contains(haystack: str, needle: str) -> bool:
    """True if `needle` is present in `haystack`, allowing minor rewording."""
    return similarity(haystack, needle) >= CONFIDENT_HIGH


def verdict(haystack: str, needle: str) -> tuple[bool | None, float]:
    """Three-way containment: True, False, or None meaning "ask the model".

    Returns (decision, score). Only when decision is None is an LLM call worth
    making - route those to prompts.CONTAINMENT_CHECK.
    """
    score = similarity(haystack, needle)
    if score >= CONFIDENT_HIGH:
        return True, score
    if score < AMBIGUOUS_LOW:
        return False, score
    return None, score


def merge_evolved(
    original_instruction: str,
    evolved_instruction: str,
    constraint: str,
    original_input: str | None,
) -> str | None:
    """Reassemble an evolved instruction, restoring the input block if dropped.

    Port of w4_evol.py::merge_evol with the LLM containment calls replaced. The
    three branches are upstream's, and the ordering matters:

      * generator kept both instruction and input -> use its output as-is
      * generator kept the instruction but dropped the input block -> re-append it
      * generator dropped the instruction -> fall back to concatenation, which is
        semantically poor; upstream logged a warning here and so should you,
        because a high rate of this branch means the evolution prompt is failing

    Returns None when there is no constraint to apply, matching upstream.
    """
    if not constraint:
        return None

    original_input = original_input or ""
    has_input = not original_input.strip() or contains(
        evolved_instruction, original_input
    )
    has_instruction = contains(evolved_instruction, original_instruction)

    if has_input and has_instruction:
        return evolved_instruction
    if has_instruction:
        return f"{evolved_instruction}\n{original_input}".rstrip()
    return f"{original_instruction} {constraint}\n{original_input}".rstrip()
