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
"""Tests for the vendored comparator and our evidence wrapper.

The comparator itself is upstream code and is exercised here mainly to prove the
vendoring is intact and importable without the framework. The evidence wrapper is
ours, and the contract it must satisfy - never raise, never return a reward - is
what most of these assert.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pivot_tools import evidence, extract_action, verify_tool_call


def _tool_call(name: str, arguments: dict) -> list[dict]:
    return [{"type": "function_call", "name": name, "arguments": json.dumps(arguments)}]


def _message(text: str) -> list[dict]:
    return [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    ]


# --- the vendored pieces import and work without the framework ----------------


def test_comparator_imports_without_nemo_gym():
    import sys

    from pivot_tools import comparator  # noqa: F401

    assert "nemo_gym" not in sys.modules


def test_extract_action_prefers_tool_calls_over_text():
    """Upstream precedence: a response that both narrates and calls is judged on the call."""
    items = _message("I will look that up.") + _tool_call("get_weather", {"city": "sf"})
    action = extract_action(items)
    assert action.type == "function_call"
    assert action.name == "get_weather"


def test_extract_action_batches_multiple_calls():
    items = _tool_call("a", {"x": 1}) + _tool_call("b", {"y": 2})
    action = extract_action(items)
    assert action.type == "function_call_batch"
    assert len(action.calls) == 2


def test_extract_action_accepts_a_response_object():
    """Must work with either a bare list or something carrying `.output`."""

    class Response:
        output = _tool_call("get_weather", {"city": "sf"})

    assert extract_action(Response()).name == "get_weather"


def test_extract_action_returns_none_on_empty_output():
    assert extract_action([]) is None


# --- the evidence contract ----------------------------------------------------


def test_matching_tool_call_passes():
    result = verify_tool_call(
        _tool_call("get_weather", {"city": "sf"}),
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "sf"}'},
    )
    assert result["passed"] is True
    assert result["check"] == "tool_call"
    assert result["errors"] == []


def test_wrong_tool_name_fails_with_a_reason():
    result = verify_tool_call(
        _tool_call("get_time", {"city": "sf"}),
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "sf"}'},
    )
    assert result["passed"] is False
    assert "not the expected tool" in result["category"]


def test_wrong_argument_value_fails():
    result = verify_tool_call(
        _tool_call("get_weather", {"city": "berlin"}),
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "sf"}'},
    )
    assert result["passed"] is False


def test_no_action_found_is_reported_not_raised():
    result = verify_tool_call(
        [], {"type": "function_call", "name": "get_weather", "arguments": "{}"}
    )
    assert result["passed"] is False
    assert "No tool call or chat message" in result["category"]


def test_expected_message_matches_a_message():
    result = verify_tool_call(_message("hello there"), {"type": "message", "content": "hello there"})
    assert result["passed"] is True


def test_tool_call_when_a_message_was_expected_fails():
    result = verify_tool_call(_tool_call("get_weather", {}), {"type": "message", "content": "hi"})
    assert result["passed"] is False
    assert "chat message was expected" in result["category"]


def test_expected_action_accepts_a_json_string():
    """Pool rows carry the expected action as a JSON string."""
    result = verify_tool_call(
        _tool_call("get_weather", {"city": "sf"}),
        json.dumps({"type": "function_call", "name": "get_weather", "arguments": '{"city": "sf"}'}),
    )
    assert result["passed"] is True


# --- never raises, never returns a reward -------------------------------------


@pytest.mark.parametrize(
    "expected",
    [None, "", "not json", {"type": "nonsense"}, {}, 42, [1, 2, 3]],
)
def test_malformed_expected_action_never_raises(expected):
    result = verify_tool_call(_tool_call("x", {}), expected)
    assert result["passed"] is False
    assert isinstance(result["errors"], list)


@pytest.mark.parametrize("output_items", [None, [], "garbage", [{"type": "unknown"}]])
def test_malformed_output_never_raises(output_items):
    result = verify_tool_call(output_items, {"type": "function_call", "name": "a", "arguments": "{}"})
    assert result["passed"] is False


def test_undecodable_arguments_are_reported_not_raised():
    items = [{"type": "function_call", "name": "get_weather", "arguments": "{not json"}]
    result = verify_tool_call(
        items, {"type": "function_call", "name": "get_weather", "arguments": "{}"}
    )
    assert result["passed"] is False


def test_result_never_contains_a_reward():
    """The verifier contract forbids returning a reward; the judge decides that."""
    result = verify_tool_call(
        _tool_call("get_weather", {"city": "sf"}),
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "sf"}'},
    )
    assert "reward" not in result
    assert "score" not in result
    assert not any(isinstance(v, float) for v in result.values())


def test_required_evidence_keys_are_always_present():
    for expected in [None, {"type": "function_call", "name": "a", "arguments": "{}"}]:
        result = verify_tool_call(_tool_call("a", {}), expected)
        assert {"check", "passed", "errors"} <= set(result)


# --- render -------------------------------------------------------------------


def test_render_states_the_verdict_and_the_reason():
    result = verify_tool_call(
        _tool_call("get_time", {}),
        {"type": "function_call", "name": "get_weather", "arguments": "{}"},
    )
    text = evidence.render(result)
    assert text.startswith("- tool-call check: FAILED")
    assert "get_weather" in text and "get_time" in text


def test_render_of_a_pass_is_short():
    result = verify_tool_call(
        _tool_call("get_weather", {"city": "sf"}),
        {"type": "function_call", "name": "get_weather", "arguments": '{"city": "sf"}'},
    )
    text = evidence.render(result)
    assert "PASSED" in text
    assert len(text.splitlines()) <= 5


def test_render_never_raises_on_a_partial_result():
    assert isinstance(evidence.render({"passed": False}), str)


# --- integration against real NeMo Gym example rows ---------------------------

GYM_DATA = Path(
    "/home/user/Gym/resources_servers/single_step_tool_use_with_argument_comparison/data"
)


def _load(name: str) -> list[dict]:
    path = GYM_DATA / name
    if not path.exists():
        pytest.skip(f"NeMo Gym checkout not available at {GYM_DATA}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _replay(expected: dict) -> list[dict]:
    """Turn an expected action back into model output that should match it exactly."""
    if expected["type"] == "function_call":
        return [
            {"type": "function_call", "name": expected["name"], "arguments": expected["arguments"]}
        ]
    if expected["type"] == "function_call_batch":
        return [
            {"type": "function_call", "name": c["name"], "arguments": c["arguments"]}
            for c in expected["calls"]
        ]
    return [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": expected["content"]}],
        }
    ]


@pytest.mark.parametrize("filename", ["example.jsonl", "parallel_example.jsonl"])
def test_expected_actions_from_real_rows_match_themselves(filename):
    """Replaying a row's own expected action as the model output must always pass.

    This is the strongest available check that the vendored comparator behaves as
    upstream does: it runs against real dataset rows, including the parallel
    tool-call shape, rather than against fixtures we wrote.
    """
    rows = _load(filename)
    assert rows, f"{filename} is empty"

    for index, row in enumerate(rows):
        expected = row["expected_action"]
        config = {"word_count_similarity_threshold": 0.1}
        if expected["type"] == "function_call_batch":
            config["parallel_tool_call_rewarding"] = True

        result = verify_tool_call(_replay(expected), expected, config)
        assert result["passed"], f"row {index} of {filename}: {result['category']} {result['errors']}"


def test_a_wrong_tool_against_a_real_row_fails_legibly():
    row = _load("example.jsonl")[0]
    result = verify_tool_call(
        [{"type": "function_call", "name": "definitely_not_the_tool", "arguments": "{}"}],
        row["expected_action"],
    )
    assert result["passed"] is False
    rendered = evidence.render(result)
    assert "FAILED" in rendered
    assert "definitely_not_the_tool" in rendered
