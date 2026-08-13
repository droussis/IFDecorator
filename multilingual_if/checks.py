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
"""Multilingual verification over the IFEval-G checkers.

Design note - why patching rather than forking:

IFEval-G is 54 classes and ~2700 lines. Forking it means owning a permanent
divergence from open-instruct / verifiable-instructions, and re-merging by hand
every time upstream moves. Instead:

  * three module-level helpers in `instructions_util` are swapped for
    locale-aware equivalents. Most length/position/count checkers call through
    them, so ~12 ids are fixed transparently with no per-class code.
  * six classes whose ASCII assumptions live inline in their own `check_following`
    are overridden individually. That is the entire remaining surface.

Total divergence: three function swaps and six small subclasses. Upstream can move
freely underneath it.
"""

from __future__ import annotations

import contextvars
from typing import Any

import regex

from . import eligibility, textops


# The upstream helpers take no `lang`, so the active language rides a contextvar
# rather than threading a parameter through 54 classes we do not own.
_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("multilingual_if_lang", default="en")

_installed = False


def _install() -> None:
    """Swap the three English-only helpers in instructions_util. Idempotent."""
    global _installed
    if _installed:
        return

    from verifiable_instructions import instructions_util

    instructions_util.count_words = lambda text: textops.count_words(text, _LANG.get())
    instructions_util.count_sentences = lambda text: textops.count_sentences(text, _LANG.get())
    instructions_util.split_into_sentences = lambda text: textops.split_into_sentences(text, _LANG.get())
    _installed = True


# --- the six classes whose assumptions are inline -------------------------------

_LOWER_WORD = regex.compile(r"\b\p{Ll}+\b")
_ANY_LETTER = regex.compile(r"\p{L}")

# Overrides take the raw kwargs rather than a built upstream instruction, so an
# overridden id never calls upstream build_description(). That is not just tidier
# - it is required. Upstream LANGUAGE_CODES has 30 entries and 13 of our 24
# supported languages are absent from it (nl cs sk hu ro et lv lt sv da el ca mt),
# so build_description(language="el") raises KeyError before verification runs.


def _check_lowercase_counting(kwargs: dict[str, Any], value: str, lang: str) -> bool:
    """count:lowercase_counting - upstream \\b[a-z]+\\b matches 0 in Cyrillic/Greek."""
    return len(_LOWER_WORD.findall(value)) <= int(kwargs["N"])


def _check_letter_counting(kwargs: dict[str, Any], value: str, lang: str) -> bool:
    """letters:letter_counting - upstream [a-zA-Z] matches 0 in Cyrillic/Greek and
    undercounts accented Latin (measured: Polish 6 of 8)."""
    count = len(_ANY_LETTER.findall(value))
    threshold = int(kwargs["N"])
    return count < threshold if kwargs.get("relation") == "less than" else count >= threshold


def _check_all_lowercase(kwargs: dict[str, Any], value: str, lang: str) -> bool:
    """change_case:english_lowercase -> all_lowercase.

    Upstream is `value.islower() and langdetect.detect(value) == "en"`. islower()
    is already Unicode-correct (measured True for lowercase ru/el/pl/de); only the
    hardcoded English conjunct makes this unsatisfiable outside English. Language
    belongs to language:response_language, not here.
    """
    return value.islower()


def _check_all_capital(kwargs: dict[str, Any], value: str, lang: str) -> bool:
    """change_case:english_capital -> all_capital. Same reasoning as above.

    Deliberately isupper() and not `value == value.upper()`: Greek all-caps
    orthography drops accents, and a round-trip comparison would reject it.
    """
    return value.isupper()


def _check_response_language(kwargs: dict[str, Any], value: str, lang: str) -> bool:
    from . import lid

    target = kwargs.get("language") or lang
    return lid.check_language(value, target)


def _check_quotation(kwargs: dict[str, Any], value: str, lang: str) -> bool:
    """startend:quotation - upstream accepts only ASCII double quotes, so a correct
    answer using the language's own quote marks fails a constraint it satisfied."""
    value = value.strip()
    if len(value) < 2:
        return False
    pairs = [('"', '"'), eligibility.quote_pair(lang)]
    return any(value.startswith(open_q.strip()) and value.endswith(close_q.strip()) for open_q, close_q in pairs)


_OVERRIDES = {
    "count:lowercase_counting": _check_lowercase_counting,
    "letters:letter_counting": _check_letter_counting,
    "change_case:english_lowercase": _check_all_lowercase,
    "change_case:english_capital": _check_all_capital,
    "language:response_language": _check_response_language,
    "startend:quotation": _check_quotation,
}


def verify_one(instruction_id: str, kwargs: dict[str, Any] | None, response: str, lang: str = "en") -> bool:
    """Score a single constraint. Never raises - an unscoreable constraint is False.

    Fails closed on purpose: an exception path that returns True is exactly how
    upstream lets an emoji-only response satisfy "respond entirely in Bulgarian".
    """
    if not eligibility.eligible(instruction_id, lang):
        raise ValueError(
            f"{instruction_id!r} is not eligible for {lang!r} - it should not have been sampled. "
            f"Gate synthesis on eligibility.eligible_ids(lang)."
        )

    _install()

    token = _LANG.set(lang)
    try:
        response = textops.nfc(response or "")
        if not response.strip():
            return False

        clean_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}

        override = _OVERRIDES.get(instruction_id)
        if override is not None:
            return bool(override(clean_kwargs, response, lang))

        from verifiable_instructions import instructions_registry

        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        instruction.build_description(**clean_kwargs)
        return bool(instruction.check_following(response))
    except Exception:  # noqa: BLE001 - deliberate: see the fail-closed note above
        return False
    finally:
        _LANG.reset(token)


def verify(
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any] | None],
    response: str,
    lang: str = "en",
    grading_mode: str = "binary",
) -> dict[str, Any]:
    """Score a full constraint list. Mirrors the Gym resources-server contract."""
    follow = [
        verify_one(iid, kw, response, lang)
        for iid, kw in zip(instruction_id_list, kwargs_list or [{}] * len(instruction_id_list))
    ]
    if grading_mode == "binary":
        reward = float(all(follow))
    elif grading_mode == "fraction":
        reward = float(sum(follow) / len(follow)) if follow else 0.0
    else:
        raise ValueError(f"Invalid grading_mode: {grading_mode}")
    return {
        "reward": reward,
        "follow_all_instructions": all(follow),
        "follow_instruction_list": follow,
        "language": lang,
    }
