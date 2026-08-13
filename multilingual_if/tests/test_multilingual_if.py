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
"""Each test pins a specific defect documented in profile.yaml.

Every assertion here fails against unpatched IFEval-G. If one of these starts
passing for the wrong reason, the patch it guards has been reverted.
"""

from __future__ import annotations

import unicodedata

import pytest

from multilingual_if import checks, eligibility, textops

PROSE = {
    "en": "The weather has been unusually warm this week, so many people decided to eat outside.",
    "de": "Das Wetter war diese Woche ungewöhnlich warm, deshalb haben viele Leute draußen gegessen.",
    "fr": "Le temps a été inhabituellement chaud cette semaine, donc beaucoup de gens ont mangé dehors.",
    "pl": "Pogoda w tym tygodniu była niezwykle ciepła, więc wiele osób postanowiło zjeść na zewnątrz.",
    "ru": "Погода на этой неделе была необычно тёплой, поэтому многие решили поесть на улице.",
    "el": "Ο καιρός ήταν ασυνήθιστα ζεστός αυτή την εβδομάδα, γι' αυτό πολλοί αποφάσισαν να φάνε έξω.",
    "bg": "Времето тази седмица беше необичайно топло, затова много хора решиха да ядат навън.",
}

pytest.importorskip("verifiable_instructions", reason="verifiable-instructions not installed")


# --- G1: NFC ------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["pl", "fr", "el", "ru", "de"])
def test_word_count_is_normalization_invariant(lang):
    """NFD input must not inflate the count. Unpatched: pl 3 words -> 8."""
    text = PROSE[lang]
    nfc_count = textops.count_words(unicodedata.normalize("NFC", text), lang)
    nfd_count = textops.count_words(unicodedata.normalize("NFD", text), lang)
    assert nfc_count == nfd_count == len(text.split())


def test_verify_is_normalization_invariant():
    text = PROSE["pl"]
    args = (["length_constraints:number_words"], [{"relation": "at least", "num_words": 14}])
    nfc_result = checks.verify(*args, unicodedata.normalize("NFC", text), lang="pl")
    nfd_result = checks.verify(*args, unicodedata.normalize("NFD", text), lang="pl")
    assert nfc_result["reward"] == nfd_result["reward"] == 1.0


# --- G3: sentence segmentation ------------------------------------------------


def test_greek_question_mark_terminates_a_sentence():
    """Greek uses ';' not '?'. Unpatched punkt/english returns 1."""
    assert textops.count_sentences("Τι κάνεις; Είμαι καλά; Εσύ;", "el") == 3


def test_greek_ano_teleia_does_not_terminate_a_sentence():
    """'·' is a semicolon, not a full stop."""
    assert textops.count_sentences("Πήγα στην αγορά· αγόρασα ψωμί. Μετά γύρισα σπίτι.", "el") == 2


@pytest.mark.parametrize(
    "lang,text,expected",
    [
        ("en", "Dr. Smith went home. He slept.", 2),
        ("de", "Dr. Schmidt ging nach Hause. Er schlief. Es war z.B. spät.", 3),
        ("ru", "Он пришёл домой. Было поздно, т.е. около полуночи. Он уснул.", 3),
        ("pl", "Poszedł do domu. Było późno, np. o północy. Zasnął.", 3),
    ],
)
def test_abbreviations_do_not_split_sentences(lang, text, expected):
    assert textops.count_sentences(text, lang) == expected


# --- G2: Unicode character classes --------------------------------------------


@pytest.mark.parametrize("lang", ["ru", "el", "bg"])
def test_lowercase_counting_is_not_vacuous_in_non_latin(lang):
    """Unpatched \\b[a-z]+\\b matches 0, so 'at most 0 lowercase words' passes free."""
    result = checks.verify(["count:lowercase_counting"], [{"N": 0}], PROSE[lang], lang=lang)
    assert result["reward"] == 0.0


@pytest.mark.parametrize("lang", ["ru", "el"])
def test_letter_counting_sees_non_latin_letters(lang):
    """Unpatched [a-zA-Z] matches 0, making 'at least N letters' unsatisfiable."""
    result = checks.verify(["letters:letter_counting"], [{"N": 20, "relation": "at least"}], PROSE[lang], lang=lang)
    assert result["reward"] == 1.0


def test_letter_counting_counts_accented_latin():
    """Unpatched [a-zA-Z] undercounts Polish diacritics (measured 6 of 8)."""
    text = "koty śpią"
    result = checks.verify(["letters:letter_counting"], [{"N": 8, "relation": "at least"}], text, lang="pl")
    assert result["reward"] == 1.0


# --- case constraints: English gate removed -----------------------------------


@pytest.mark.parametrize("lang", ["ru", "el", "pl", "de"])
def test_all_lowercase_is_satisfiable_outside_english(lang):
    """Unpatched requires langdetect == 'en', so this is unsatisfiable elsewhere."""
    result = checks.verify(["change_case:english_lowercase"], [{}], PROSE[lang].lower(), lang=lang)
    assert result["reward"] == 1.0


@pytest.mark.parametrize("lang", ["ru", "el", "pl", "de"])
def test_all_lowercase_still_rejects_mixed_case(lang):
    result = checks.verify(["change_case:english_lowercase"], [{}], PROSE[lang], lang=lang)
    assert result["reward"] == 0.0


@pytest.mark.parametrize("lang", ["ru", "el", "pl"])
def test_all_capital_is_satisfiable_outside_english(lang):
    result = checks.verify(["change_case:english_capital"], [{}], PROSE[lang].upper(), lang=lang)
    assert result["reward"] == 1.0


# --- startend:quotation localization ------------------------------------------


@pytest.mark.parametrize(
    "lang,text",
    [
        ("de", "„Das Wetter war schön und wir gingen hinaus.“"),
        ("fr", "« Le temps était beau et nous sommes sortis. »"),
        ("ru", "«Погода была хорошая, и мы вышли на улицу.»"),
        ("el", "«Ο καιρός ήταν ωραίος και βγήκαμε έξω.»"),
    ],
)
def test_quotation_accepts_native_quote_marks(lang, text):
    """Unpatched accepts only ASCII '"', failing a constraint the answer satisfied."""
    assert checks.verify(["startend:quotation"], [{}], text, lang=lang)["reward"] == 1.0


def test_quotation_still_accepts_ascii_quotes():
    assert checks.verify(["startend:quotation"], [{}], '"So ist es."', lang="de")["reward"] == 1.0


def test_quotation_rejects_unquoted_text():
    assert checks.verify(["startend:quotation"], [{}], PROSE["de"], lang="de")["reward"] == 0.0


# --- eligibility gate ---------------------------------------------------------


def test_constrained_response_is_excluded_everywhere():
    assert not eligibility.eligible("detectable_format:constrained_response", "de")
    with pytest.raises(ValueError, match="not eligible"):
        checks.verify_one("detectable_format:constrained_response", {}, "My answer is yes.", lang="de")


def test_response_language_gated_on_supported_set():
    assert eligibility.eligible("language:response_language", "de")
    # Irish: measured to collapse to sk @ 0.22 on generic prose
    assert not eligibility.eligible("language:response_language", "ga")


def test_every_profiled_id_exists_upstream():
    from verifiable_instructions import instructions_registry

    profiled = set(eligibility.profile()["instructions"])
    assert profiled == set(instructions_registry.INSTRUCTION_DICT)


def test_eligible_ids_excludes_only_the_documented_row():
    ids = eligibility.eligible_ids("ru")
    assert "detectable_format:constrained_response" not in ids
    assert len(ids) == 53


# --- fail-closed --------------------------------------------------------------


@pytest.mark.parametrize("response", ["", "   ", "🙂🙂🙂", "... !!! ???"])
def test_degenerate_responses_never_earn_reward(response):
    """Upstream returns True from the LangDetectException path for these."""
    result = checks.verify(["language:response_language"], [{"language": "bg"}], response, lang="bg")
    assert result["reward"] == 0.0


def test_unknown_kwargs_fail_closed_rather_than_raising():
    result = checks.verify(
        ["length_constraints:number_words"], [{"relation": "sideways", "num_words": 5}], PROSE["de"], lang="de"
    )
    assert result["reward"] == 0.0


# --- localization helpers used by synthesis -----------------------------------


def test_localized_kwargs_are_language_specific():
    assert eligibility.localized_kwargs("detectable_content:postscript", "es") == {"postscript_marker": "P.D."}
    assert eligibility.localized_kwargs("detectable_format:multiple_sections", "ru") == {"section_spliter": "Раздел"}


def test_alphabet_is_script_and_language_specific():
    assert eligibility.alphabet("el").startswith("αβγ")
    assert eligibility.alphabet("ru").startswith("абв")
    assert "ґ" in eligibility.alphabet("uk")  # Ukrainian-only letter
    assert "ґ" not in eligibility.alphabet("ru")
    assert eligibility.alphabet("pl") == "abcdefghijklmnopqrstuvwxyz"


def test_chars_per_word_spread_is_real():
    """The w4 length calibration must not treat fi and en alike."""
    assert eligibility.chars_per_word("fi") > eligibility.chars_per_word("en") * 1.4


# --- upstream LANGUAGE_CODES gap ----------------------------------------------


@pytest.mark.parametrize("lang", ["el", "cs", "sk", "hu", "ro", "nl", "sv", "da", "et", "lv", "lt", "ca", "mt"])
def test_languages_absent_from_upstream_language_codes_still_verify(lang):
    """13 of the 24 supported languages are missing from upstream LANGUAGE_CODES.

    Calling upstream build_description(language=...) for them raises KeyError, so
    response_language must be handled without touching upstream instruction state.
    Regression guard: this failed for every language here before the override
    refactor, and the original test suite missed it by only exercising `bg`.
    """
    from verifiable_instructions.instructions_util import LANGUAGE_CODES

    assert lang not in LANGUAGE_CODES, f"{lang} is now upstream; this test needs updating"
    result = checks.verify(["language:response_language"], [{"language": lang}], PROSE.get(lang, ""), lang=lang)
    assert result["reward"] in (0.0, 1.0)  # must not raise


def test_greek_response_language_accepts_greek():
    text = "Ο καιρός ήταν ασυνήθιστα ζεστός αυτή την εβδομάδα, γι' αυτό πολλοί αποφάσισαν να φάνε έξω."
    assert checks.verify(["language:response_language"], [{"language": "el"}], text, lang="el")["reward"] == 1.0


def test_greek_response_language_rejects_other_scripts():
    assert checks.verify(["language:response_language"], [{"language": "el"}], PROSE["de"], lang="el")["reward"] == 0.0
