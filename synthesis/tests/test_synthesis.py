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
"""Each test pins a defect in the upstream pipeline that this package fixes."""

from __future__ import annotations

import random
import unicodedata

import pytest

from synthesis import matching, parsing, prompts
from synthesis.schema import (
    Row,
    Rubric,
    SchemaError,
    pass_rate_report,
    route_by_difficulty,
    validate_row,
)


# --- parse_verdict: replaces the None-returning unified_judge_parse -----------


@pytest.mark.parametrize(
    "response,expected",
    [
        ("analysis...\n**Final Verification:** YES", True),
        ("analysis...\n**Final Verification:** NO", False),
        ("Final Verification: yes", True),
        ("**Final Verdict:** YES", True),
        ("#final_ver: YES", True),
        ("**Final Verification:** **YES**", True),
        ("**Final Verification:** <<YES>>", True),
        # The upstream loose branch searched the whole tail for "yes", so trailing
        # commentary after a NO verdict flipped the result.
        ("**Final Verification:** NO\nthe response says yes to the question", False),
        (None, False),
        ("", False),
        ("no verdict marker at all", False),
    ],
)
def test_parse_verdict(response, expected):
    assert parsing.parse_verdict(response) is expected


def test_parse_verdict_never_returns_none():
    """upstream unified_judge_parse returns None on the default path."""
    for response in ("YES", "garbage", "", None, "**Final Verification:** MAYBE"):
        assert isinstance(parsing.parse_verdict(response), bool)


# --- parse_decomposition: bullets under #Input must not become constraints ----


def test_decomposition_does_not_absorb_input_bullets():
    content = """#Task Description: Fasse den Text zusammen.
#Constraints:
- Höchstens 200 Wörter
- Sachlicher Ton
#Input:
- Dies ist Eingabetext mit einem Aufzählungszeichen
- Und noch eine Zeile
"""
    result = parsing.parse_decomposition(content)
    assert result.constraints == ["Höchstens 200 Wörter", "Sachlicher Ton"]
    assert "Eingabetext" in result.input_text


def test_decomposition_handles_null_sections():
    result = parsing.parse_decomposition(
        "#Task Description: Do it.\n#Constraints: NULL\n#Input: NULL"
    )
    assert result.constraints == []
    assert result.input_text is None


def test_decomposition_inline_constraint_is_kept():
    result = parsing.parse_decomposition(
        "#Task Description: T\n#Constraints: Genau drei Sätze\n#Input: NULL"
    )
    assert result.constraints == ["Genau drei Sätze"]


# --- parse_classification -----------------------------------------------------


def test_classification_parses_labels_in_order():
    assert parsing.parse_classification(
        "#1: programmatic\n#2: rubric\n#3: programmatic", 3
    ) == [
        "programmatic",
        "rubric",
        "programmatic",
    ]


def test_classification_defaults_unparseable_to_rubric():
    """Fail toward the judge, not toward an unenforceable reward signal."""
    assert parsing.parse_classification("#1: programmatic", 3) == [
        "programmatic",
        "rubric",
        "rubric",
    ]
    assert parsing.parse_classification("total garbage", 2) == ["rubric", "rubric"]


# --- parse_evolution: '#' inside the instruction must not corrupt fields ------


def test_evolution_survives_hash_in_generated_instruction():
    """Upstream splits the whole response on '#', so this loses every later field."""
    response = """#rationale: Adds a persona for specificity
#constraint_type: Situation, Role-based
#constraint: Antworte als Museumskurator
#enhanced_instruction: Erkläre den Motor.
# Überschrift
Nutze die Farbe #FF0000 als Beispiel.
"""
    evolution = parsing.parse_evolution(response)
    assert evolution is not None
    assert evolution.constraint == "Antworte als Museumskurator"
    assert evolution.constraint_type == ("Situation", "Role-based")
    assert "#FF0000" in evolution.enhanced_instruction
    assert "Überschrift" in evolution.enhanced_instruction


def test_evolution_returns_none_when_unusable():
    assert parsing.parse_evolution("#rationale: only this") is None
    assert parsing.parse_evolution("") is None


# --- parse_keywords: must be surface forms present in the reference -----------


def test_keywords_filtered_to_forms_present_in_reference():
    """Matching is surface-form, so a keyword absent from the reference is useless."""
    response = (
        "**Thinking Process:** ...\n**Keywords:** ['Motor', 'Kolben', 'Zylinderkopf']"
    )
    reference = "Der Motor treibt den Kolben an."
    assert parsing.parse_keywords(response, reference) == ["Motor", "Kolben"]


def test_keywords_without_reference_are_all_kept():
    response = "**Keywords:** ['κινητήρας', 'έμβολο']"
    assert parsing.parse_keywords(response) == ["κινητήρας", "έμβολο"]


def test_keywords_does_not_eval_arbitrary_code():
    """Upstream called eval() on a regex capture."""
    assert parsing.parse_keywords("**Keywords:** __import__('os').system('true')") != [
        "__import__"
    ]


# --- matching: replaces the per-merge LLM containment calls -------------------


def test_contains_is_normalization_insensitive():
    text = "Erkläre die Funktionsweise eines Verbrennungsmotors"
    assert matching.contains(
        unicodedata.normalize("NFD", text), unicodedata.normalize("NFC", text)
    )


def test_contains_tolerates_whitespace_and_case():
    assert matching.contains(
        "Der   MOTOR\ntreibt den Kolben an.", "der motor treibt den kolben an"
    )


def test_contains_rejects_unrelated_text():
    assert not matching.contains(
        "Der Motor treibt den Kolben an.", "Die Katze schläft den ganzen Tag im Garten."
    )


def test_verdict_three_way():
    decision, score = matching.verdict(
        "Der Motor treibt den Kolben an.", "Der Motor treibt den Kolben an."
    )
    assert decision is True and score == 1.0
    decision, _ = matching.verdict(
        "Der Motor treibt den Kolben an.", "Völlig anderer Satz über Katzen und Gärten."
    )
    assert decision is False


def test_merge_restores_dropped_input_block():
    merged = matching.merge_evolved(
        original_instruction="Fasse den Text zusammen.",
        evolved_instruction="Fasse den Text als Museumskurator zusammen.",
        constraint="als Museumskurator",
        original_input="Der Text lautet: Ein Motor ist eine Maschine.",
    )
    assert "Ein Motor ist eine Maschine" in merged


def test_merge_keeps_evolved_text_when_everything_survived():
    evolved = "Fasse den Text zusammen. Der Text lautet: Ein Motor ist eine Maschine."
    merged = matching.merge_evolved(
        "Fasse den Text zusammen.",
        evolved,
        "kurz",
        "Der Text lautet: Ein Motor ist eine Maschine.",
    )
    assert merged == evolved


def test_merge_returns_none_without_constraint():
    assert matching.merge_evolved("a", "b", "", None) is None


# --- prompts ------------------------------------------------------------------


def test_render_rejects_unverifiable_language():
    """A language nothing can verify must not be synthesizable."""
    with pytest.raises(KeyError, match="no language name"):
        prompts.render(prompts.QUALITY_GATE, "ga", prompt="x")


def test_render_resolves_language_name():
    rendered = prompts.render(prompts.QUALITY_GATE, "el", prompt="κάτι")
    assert "Greek" in rendered and "κάτι" in rendered


def test_every_supported_language_has_a_name():
    from multilingual_if import supported_languages

    assert set(supported_languages()) <= set(prompts.LANGUAGE_NAMES)


def test_evolution_prompt_is_randomized_per_call():
    """A fixed template collapses the constraint distribution."""
    a = prompts.render_evolution("Erkläre den Motor.", "de", rng=random.Random(1))
    b = prompts.render_evolution("Erkläre den Motor.", "de", rng=random.Random(2))
    assert a != b


def test_evolution_prompt_names_the_target_language():
    rendered = prompts.render_evolution(
        "Erkläre den Motor.", "de", rng=random.Random(0)
    )
    assert rendered.count("German") >= 3
    assert "Erkläre den Motor." in rendered


def test_evolution_distribution_weighting_is_accepted():
    rendered = prompts.render_evolution(
        "Erkläre den Motor.",
        "de",
        distribution={"style_constraints": {"Tonal": 50}},
        rng=random.Random(0),
    )
    assert "Tonal" in rendered


# --- schema validation --------------------------------------------------------


def _row(**overrides) -> Row:
    base = dict(
        id="r1",
        language="de",
        prompt="Erkläre den Motor. Schreibe alles in Kleinbuchstaben.",
        prompt_wo_programmatic="Erkläre den Motor.",
        instruction_id_list=["change_case:english_lowercase"],
        kwargs=[{}],
    )
    base.update(overrides)
    return Row(**base)


def test_valid_row_passes():
    validate_row(_row(), stage="test")


def test_missing_language_is_fatal():
    with pytest.raises(SchemaError, match="`language` is missing"):
        validate_row(_row(language=""))


def test_unverifiable_language_is_fatal():
    with pytest.raises(SchemaError, match="outside the verifiable set"):
        validate_row(_row(language="ga"))


def test_ineligible_constraint_is_fatal():
    with pytest.raises(SchemaError, match="not eligible"):
        validate_row(
            _row(
                instruction_id_list=["detectable_format:constrained_response"],
                kwargs=[{}],
            )
        )


def test_kwargs_length_mismatch_is_fatal():
    with pytest.raises(SchemaError, match="positional and must match"):
        validate_row(_row(kwargs=[]))


def test_conflicting_constraints_are_fatal():
    with pytest.raises(SchemaError, match="mutually contradictory"):
        validate_row(
            _row(
                language="el",
                instruction_id_list=[
                    "language:response_language",
                    "startend:end_checker",
                ],
                kwargs=[{"language": "el"}, {"end_phrase": "x"}],
            )
        )


def test_judge_must_not_see_programmatic_constraints():
    with pytest.raises(SchemaError, match="scored twice"):
        validate_row(
            _row(
                prompt_wo_programmatic="Erkläre den Motor. Schreibe alles in Kleinbuchstaben."
            )
        )


def test_rubric_without_criteria_is_fatal():
    with pytest.raises(SchemaError, match="no criteria"):
        validate_row(_row(rubrics=[Rubric(constraint="Sachlicher Ton")]))


def test_rubric_without_constraint_is_fatal():
    """Guards the structural fix for upstream's positional zip bug."""
    with pytest.raises(SchemaError, match="bound to their constraint"):
        validate_row(_row(rubrics=[Rubric(constraint="", criteria=["c"])]))


def test_duplicate_instruction_ids_are_fatal():
    with pytest.raises(SchemaError, match="duplicate"):
        validate_row(
            _row(instruction_id_list=["punctuation:no_comma"] * 2, kwargs=[{}, {}])
        )


def test_pass_rate_out_of_range_is_fatal():
    with pytest.raises(SchemaError, match="outside"):
        validate_row(_row(pass_rate=1.5))


def test_row_roundtrips_through_dict():
    row = _row(rubrics=[Rubric(constraint="Sachlicher Ton", criteria=["a?"])])
    restored = Row.from_dict(row.to_dict())
    assert restored.rubrics[0].constraint == "Sachlicher Ton"
    assert restored.language == "de"


# --- routing and reporting ----------------------------------------------------


def test_routing_splits_pools():
    rows = [
        _row(id=str(i), pass_rate=p) for i, p in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])
    ]
    pools = route_by_difficulty(rows)
    assert [r.pass_rate for r in pools["too_hard"]] == [0.0]
    assert [r.pass_rate for r in pools["hard"]] == [0.25, 0.5]
    assert [r.pass_rate for r in pools["easy"]] == [0.75, 1.0]


def test_routing_requires_pass_rate():
    with pytest.raises(SchemaError, match="no pass_rate"):
        route_by_difficulty([_row()])


def test_pass_rate_report_surfaces_a_degenerate_language():
    """The diagnostic that catches an unsatisfiable constraint in one language."""
    rows = [_row(id=f"de{i}", language="de", pass_rate=0.5) for i in range(4)]
    rows += [_row(id=f"el{i}", language="el", pass_rate=0.0) for i in range(4)]
    report = pass_rate_report(rows)
    assert report["el"]["frac_zero"] == 1.0
    assert report["de"]["frac_zero"] == 0.0
    assert report["el"]["mean"] < report["de"]["mean"]


def test_evolution_weighting_never_drops_a_type():
    """Weighted ordering must not lose types.

    Upstream used random.choices (sampling WITH replacement), so a heavily
    down-weighted type could vanish from the prompt entirely - meaning the model
    could never select the very type the weighting was meant to promote, and the
    imbalance it was correcting would get worse, not better.
    """
    heavy = {"style_constraints": {"Tonal": 500, "Structural": 400}}
    for seed in range(25):
        rendered = prompts.render_evolution(
            "x", "de", distribution=heavy, rng=random.Random(seed)
        )
        for expected in ("Tonal", "Structural", "Creative"):
            assert expected in rendered, f"seed {seed} dropped {expected}"
