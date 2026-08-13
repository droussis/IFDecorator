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
"""fastText language identification for `language:response_language`.

Replaces langdetect in the IFEval-G ResponseLanguageChecker. Every constant here
was calibrated against lid.176.ftz on a 34-language EU parallel corpus; see
ifeval_lcg_profile.yaml -> language_identification for the measurements.

Scope: Latin, Cyrillic and Greek scripts.

Install notes (both are real, both bite):
  * fasttext 0.9.3 is incompatible with numpy >= 2 - `np.array(probs, copy=False)`
    raises ValueError. Pin numpy<2 or use the fasttext-numpy2 fork.
  * model.predict() raises on embedded newlines
    ("predict processes one line at a time"). _prepare() handles this.
"""

from __future__ import annotations

import functools
import re
import unicodedata

import fasttext
import regex

from .setup_lid import ensure_lid_model


# --- calibrated constants ---------------------------------------------------

# Measured: on SUPPORTED below, correct predictions bottom out at 0.481 (sk) and
# there were no incorrect predictions at all (48/48). 0.50 would reject Slovak.
CONFIDENCE_FLOOR = 0.40

# Measured accuracy vs truncation length (48 samples, lowercased pipeline):
#   15 chars 38/48 (8 wrong ABOVE the floor - confidence is not calibrated on
#            short text, so the floor does not protect you here)
#   25 chars 44/48 (1 wrong above floor)
#   40 chars 47/48 (0 wrong above floor)
#   60 chars 48/48
MIN_CHARS = 40

# Fraction of sentences that must be on-target for multi-sentence responses.
SEGMENT_RATIO = 0.80

# Languages that lid.176 resolves reliably in this script set.
# Measured 48/48 across a named-entity set and a generic-prose set.
SUPPORTED = frozenset(
    [
        "en",
        "de",
        "fr",
        "es",
        "it",
        "pt",
        "nl",
        "pl",
        "cs",
        "sk",
        "hu",
        "ro",
        "fi",
        "et",
        "lv",
        "lt",
        "sv",
        "da",
        "bg",
        "uk",
        "ru",
        "el",
        "ca",
        "mt",
    ]
)

# Do NOT use these as response_language targets - measured failures:
#   ga  Irish        generic prose -> sk @ 0.22   (named-entity text -> ga @ 0.81)
#   nn  Nynorsk      generic prose -> no @ 0.29
#   sl  Slovene      generic prose -> sr @ 0.39, correct label only 0.265
#   bs  Bosnian      -> hr on BOTH sets (0.31, 0.37); never wins
#   hr  Croatian     wins but weakly (0.50, 0.53) against bs/sr/sh
#   sr  Serbian      named-entity text -> sh @ 0.31
UNRELIABLE = frozenset(["ga", "nn", "sl", "bs", "hr", "sr", "sh"])

# lid.176 carries a separate `sh` (Serbo-Croatian) label, so hr/bs/sr/sh split
# probability four ways and none of them clears a useful margin. Collapsing them
# took set A from 32/34 to 34/34 and set B from 30/34 to 32/34. Use this only if
# you accept one BCS reward target rather than three; otherwise leave them out.
COLLAPSE = {
    "hr": "bcs",
    "bs": "bcs",
    "sr": "bcs",
    "sh": "bcs",
    "no": "nor",
    "nn": "nor",  # lid.176 labels Bokmal `no`, not `nb`
}

SCRIPT_OF_LANG = {
    "el": "greek",
    "bg": "cyrillic",
    "uk": "cyrillic",
    "ru": "cyrillic",
    "mk": "cyrillic",
}  # everything else in scope is latin

_SCRIPT_PATTERNS = {
    "greek": regex.compile(r"\p{Greek}"),
    "cyrillic": regex.compile(r"\p{Cyrillic}"),
    "latin": regex.compile(r"\p{Latin}"),
}

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")


@functools.lru_cache(maxsize=2)
def _model(path: str | None):
    return fasttext.load_model(path or ensure_lid_model())


def _strip_non_prose(text: str) -> str:
    """Remove spans that carry no language signal.

    Measured impact is small but one-directional: a Greek answer with a Python
    block still scores el @ 0.956 (vs 0.989 clean), while a code-only response
    scores en @ 0.188 - below the floor, so it is rejected either way.
    """
    for pattern in (_FENCED_CODE, _INLINE_CODE, _URL, _EMAIL):
        text = pattern.sub(" ", text)
    return text


def _prepare(text: str) -> str:
    """Normalize, de-newline and lowercase before predict().

    Lowercasing is not cosmetic. Measured on all-caps responses:
        fr  raw UPPER -> ca @ 0.170  (WRONG)  ->  lowered -> fr @ 0.980
        ru  raw UPPER -> ru @ 0.668           ->  lowered -> ru @ 0.997
        pl  raw UPPER -> pl @ 0.773           ->  lowered -> pl @ 0.993
        de  raw UPPER -> de @ 0.856           ->  lowered -> de @ 0.970
    Without this, combining `change_case:all_capital` with
    `language:response_language` on one prompt makes the language constraint
    fail on a correct answer - the two constraints would silently conflict.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.lower().strip()


def dominant_script(text: str) -> str | None:
    counts = {name: len(pat.findall(text)) for name, pat in _SCRIPT_PATTERNS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def _canonical(lang: str, collapse: bool) -> str:
    return COLLAPSE.get(lang, lang) if collapse else lang


def check_language(
    text: str,
    target: str,
    *,
    model_path: str | None = None,
    collapse_confusables: bool = False,
    segment: bool = True,
) -> bool:
    """True iff `text` is in `target`. Fails closed on every ambiguity.

    Order matters: the script gate runs first because it is deterministic, free,
    and on this corpus it agreed with the expected script 34/34. Greek needs
    nothing else - el is the only in-scope language using the Greek script.
    """
    if not text or not text.strip():
        return False  # G6: never "count as followed"

    prose = _strip_non_prose(text)
    prepared = _prepare(prose)
    if not prepared:
        return False

    # 1. script gate - deterministic, catches the gross failures for free
    expected_script = SCRIPT_OF_LANG.get(target, "latin")
    if dominant_script(prepared) != expected_script:
        return False

    # 2. below MIN_CHARS the model is confidently wrong, so trust only the gate.
    #
    #    KNOWN LIMITATION, deliberate: this accepts a short response that is in the
    #    right SCRIPT but the wrong language - a 20-character Bulgarian answer will
    #    satisfy "respond in Russian". The alternative, rejecting everything short,
    #    penalises legitimately terse correct answers. Greek is unaffected (it owns
    #    its script outright). If you pair response_language with a minimum word
    #    count on the same prompt, the window closes on its own.
    if len(prepared) < MIN_CHARS:
        return True

    model = _model(model_path)
    want = _canonical(target, collapse_confusables)

    def _predict(chunk: str) -> tuple[str, float]:
        """Return (canonical label, mass assigned to that canonical label).

        In collapse mode the group members must be summed, not compared
        individually: hr/bs/sr/sh split the probability four ways, so Bosnian
        text peaks at sr @ 0.357 and no single label clears CONFIDENCE_FLOOR
        even though the group collectively owns ~0.78 of the mass.
        """
        k = 8 if collapse_confusables else 1
        labels, probs = model.predict(chunk, k=k)
        mass: dict[str, float] = {}
        for label, prob in zip(labels, probs):
            key = _canonical(label.replace("__label__", ""), collapse_confusables)
            mass[key] = mass.get(key, 0.0) + float(prob)
        best = max(mass, key=mass.get)
        return best, mass[best]

    label, confidence = _predict(prepared)
    if label != want or confidence < CONFIDENCE_FLOOR:
        return False

    # 3. segment vote - stops one stray English clause sinking a correct answer,
    #    and stops a mostly-English answer passing on one on-target sentence
    if segment:
        chunks = [c.strip() for c in re.split(r"(?<=[.!?;])\s+", prepared) if len(c.strip()) >= MIN_CHARS]
        if len(chunks) >= 3:
            hits = sum(1 for c in chunks if _canonical(_predict(c)[0], collapse_confusables) == want)
            if hits / len(chunks) < SEGMENT_RATIO:
                return False

    return True
