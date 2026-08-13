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
"""Locale-aware text primitives for Latin / Cyrillic / Greek.

These replace the English-only helpers in IFEval-G's `instructions_util`:

    count_words          nltk RegexpTokenizer(r"\\w+")     -> Unicode word regex over NFC
    count_sentences      nltk punkt/english.pickle          -> per-script terminators
    split_into_sentences hardcoded English abbreviations    -> per-language abbreviations

Every constant here is justified by a measurement recorded in profile.yaml.
"""

from __future__ import annotations

import functools
import unicodedata

import regex


# A word is a letter/digit run that may carry combining marks, internal hyphens
# and apostrophes. Combining marks are the whole point: Python's \w excludes
# categories Mn/Mc, so on decomposed text "bilqiyan"-type Indic and even Polish
# "zazolc" shatter into fragments. Measured: pl NFD 3 words -> 8.
_WORD = regex.compile(r"[\p{L}\p{N}][\p{L}\p{M}\p{N}\p{Pd}'’]*")

# Sentence terminators differ by script. Greek marks questions with ";"
# (U+003B or U+037E), never "?" - measured failure: a 3-question Greek
# paragraph segmented as 1 sentence under the shipped splitter.
_TERMINATORS = {
    "latin": ".!?…",
    "cyrillic": ".!?…",
    "greek": ".!;;…",
}

# The Greek ano teleia is a semicolon, NOT a full stop. Listed so nobody
# "helpfully" adds it to the terminator set later.
_NEVER_TERMINATOR = "··"

SCRIPT_OF_LANG = {
    "el": "greek",
    "bg": "cyrillic",
    "uk": "cyrillic",
    "ru": "cyrillic",
    "mk": "cyrillic",
    "sr": "cyrillic",
}

# Abbreviations that must not end a sentence. Deliberately short - this is the
# long tail, and over-fitting it reproduces the English-only mistake in reverse.
_ABBREVIATIONS = {
    "en": ["mr", "mrs", "ms", "dr", "prof", "st", "inc", "ltd", "jr", "sr", "co", "e.g", "i.e", "vs"],
    "de": ["dr", "prof", "hr", "fr", "nr", "bzw", "ca", "usw", "z.b", "u.a", "d.h", "evtl"],
    "fr": ["m", "mme", "mlle", "dr", "pr", "cf", "env", "p.ex", "c.-a-d"],
    "es": ["sr", "sra", "srta", "dr", "dra", "ej", "etc", "p.ej", "ee.uu"],
    "it": ["sig", "sig.ra", "dott", "prof", "ecc", "es", "av"],
    "pt": ["sr", "sra", "dr", "dra", "prof", "etc", "ex"],
    "nl": ["dhr", "mevr", "dr", "prof", "bijv", "enz", "o.a", "z.o.z"],
    "pl": ["dr", "prof", "mgr", "inz", "np", "itd", "itp", "tzn", "ok"],
    "cs": ["dr", "prof", "ing", "mgr", "napr", "atd", "tj", "cca"],
    "sk": ["dr", "prof", "ing", "mgr", "napr", "atd", "tj", "cca"],
    "hu": ["dr", "prof", "pl", "stb", "ill", "kb"],
    "ro": ["dl", "dna", "dr", "prof", "ex", "etc", "sa"],
    "fi": ["esim", "jne", "ns", "mm", "ks", "vrt", "tri", "prof"],
    "et": ["nt", "jne", "vt", "dr", "prof"],
    "lv": ["piem", "utt", "dr", "prof", "u.c"],
    "lt": ["pvz", "ir pan", "dr", "prof", "t.y"],
    "sv": ["bl.a", "t.ex", "dvs", "osv", "dr", "prof"],
    "da": ["bl.a", "f.eks", "dvs", "osv", "dr", "prof"],
    "ru": ["т.е", "т.д", "т.п", "др", "проф", "напр", "г", "гг", "рис"],
    "uk": ["т.д", "т.п", "напр", "проф", "р", "рис"],
    "bg": ["напр", "проф", "д-р", "т.е", "т.н"],
    "el": ["κ", "κα", "κοс", "π.χ", "κ.λπ", "κ.α", "δρ", "καθ"],
}


def nfc(text: str) -> str:
    """Canonical composition. Run this before ANY counting.

    Model output normalization form is not guaranteed, and the difference is not
    cosmetic: the same Polish sentence counts as 3 or 8 words depending on form.
    """
    return unicodedata.normalize("NFC", text)


def script_of(lang: str) -> str:
    return SCRIPT_OF_LANG.get(lang, "latin")


def count_words(text: str, lang: str = "en") -> int:
    """Unicode-aware word count. `lang` is accepted for symmetry; L/C/G all
    tokenize the same way once combining marks are handled."""
    return len(_WORD.findall(nfc(text)))


def words(text: str, lang: str = "en") -> list[str]:
    return _WORD.findall(nfc(text))


@functools.lru_cache(maxsize=64)
def _splitter(lang: str):
    script = script_of(lang)
    terms = regex.escape(_TERMINATORS[script])
    abbrevs = _ABBREVIATIONS.get(lang, [])
    # A terminator ends a sentence when followed by whitespace and the token
    # before it is not a known abbreviation and not a single initial.
    abbrev_alt = "|".join(regex.escape(a) for a in abbrevs) or r"(?!)"
    pattern = (
        r"(?<!\b(?:" + abbrev_alt + r"))"  # not a known abbreviation
        r"(?<!\b\p{Lu})"  # not a single-letter initial
        r"[" + terms + r"]+[\"'»”’)\]]*\s+"
    )
    return regex.compile(pattern, regex.IGNORECASE)


def split_into_sentences(text: str, lang: str = "en") -> list[str]:
    text = nfc(text).strip()
    if not text:
        return []
    parts = _splitter(lang).split(text)
    return [p.strip() for p in parts if p and p.strip()]


def count_sentences(text: str, lang: str = "en") -> int:
    return len(split_into_sentences(text, lang))
