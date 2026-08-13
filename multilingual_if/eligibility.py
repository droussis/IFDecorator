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
"""Reads profile.yaml and answers the two questions the profile exists to answer.

Synthesis:    which constraint ids may I sample for language X, and with which args?
Verification: may I score this id for language X at all?

Keeping both consumers on one file is the point - a constraint that is sampled but
not scoreable (or scoreable but never sampled) is exactly the drift this prevents.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml


PROFILE_PATH = pathlib.Path(__file__).parent / "profile.yaml"


@functools.lru_cache(maxsize=1)
def profile(path: str | None = None) -> dict[str, Any]:
    with open(path or PROFILE_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def supported_languages() -> list[str]:
    """Languages fastText LID resolves reliably. Measured 48/48 / 0-552 FP.

    Sampling response_language outside this set injects verifier noise straight
    into the reward, so synthesis and verification both gate on it.
    """
    return list(profile()["language_identification"]["supported"])


def language_profile(lang: str) -> dict[str, Any]:
    profiles = profile()["language_profiles"]
    if lang not in profiles:
        raise KeyError(f"no language profile for {lang!r}; known: {sorted(k for k in profiles if k != 'alphabets')}")
    return profiles[lang]


def status(instruction_id: str) -> str:
    entry = profile()["instructions"].get(instruction_id)
    if entry is None:
        raise KeyError(f"{instruction_id!r} is not in profile.yaml")
    return entry["status"]


def eligible(instruction_id: str, lang: str) -> bool:
    """False for ids that are degenerate in `lang`.

    A constraint that cannot be satisfied yields a constant-zero reward; one that
    is satisfied for free yields a constant-one reward. Both are worse than not
    sampling the constraint at all, and under IFDecorator's conjunctive reward a
    single unsatisfiable id zeroes the whole sample.
    """
    if instruction_id not in profile()["instructions"]:
        return False
    if status(instruction_id) == "exclude":
        return False
    return not (instruction_id == "language:response_language" and lang not in supported_languages())


def eligible_ids(lang: str) -> list[str]:
    return sorted(i for i in profile()["instructions"] if eligible(i, lang))


def localized_kwargs(instruction_id: str, lang: str) -> dict[str, Any]:
    """Args that must come from the language profile rather than an English default.

    These are the `localize` rows: the checker is correct, but its default literal
    is English, so a correct answer in the target language fails a constraint it
    actually satisfied.
    """
    prof = language_profile(lang)
    if instruction_id == "detectable_content:postscript":
        return {"postscript_marker": prof["postscript_marker"]}
    if instruction_id == "detectable_format:multiple_sections":
        return {"section_spliter": prof["section_word"]}
    if instruction_id == "startend:quotation":
        return {}  # no kwarg upstream; the checker itself needs the quote pair
    return {}


def quote_pair(lang: str) -> tuple[str, str]:
    pair = language_profile(lang)["quote_pair"]
    return pair[0], pair[1]


def alphabet(lang: str) -> str:
    """Letter inventory for letter-frequency and alphabet-order constraints.

    Cyrillic ordering is language-specific, so this is not a shared range.
    """
    alphabets = profile()["language_profiles"]["alphabets"]
    script = language_profile(lang)["script"]
    if script == "greek":
        return alphabets["greek"]
    if script == "cyrillic":
        return alphabets.get(f"cyrillic_{lang}", alphabets["cyrillic_ru"])
    return alphabets["latin_base"]


def conflicts() -> list[tuple[str, str]]:
    """Pairs that are contradictory only in a multilingual setting.

    INSTRUCTION_CONFLICTS upstream is English-shaped and does not contain these.
    Note that all_capital x response_language is deliberately NOT here: lowercasing
    before LID removes that interaction (measured - without it, an all-caps French
    answer scores ca @ 0.170 and fails).
    """
    return [tuple(pair) for pair in profile()["synthesis"]["conflicts_to_add"]["pairs"]]


def chars_per_word(lang: str) -> float:
    """For calibrating length constraints against a reference response.

    w4_evol.py:591 currently uses len(response) - a CHARACTER count - as a WORD
    threshold. Ranges from 5.1 (en) to 8.0 (fi), so leaving it uncorrected makes
    difficulty correlate with language rather than with the instruction.
    """
    return language_profile(lang)["chars_per_word"]
