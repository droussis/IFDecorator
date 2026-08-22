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
"""Evidence-emitting wrapper around the single-step tool-call comparator.

The comparator returns a `reward` float. A programmatic verifier in our framework
must not: it returns evidence, and the judge decides the reward. This module is
that adaptation, and it is the only file here that is ours rather than vendored.

The translation loses nothing. The comparator's `StepRewardCategory` is already a
human-readable sentence describing exactly why a comparison failed, which is more
useful to a judge than the scalar was - "the tool in a tool call is not the
expected tool" tells it something a 0.0 does not.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .actions import extract_action
from .comparator import (
    ActionComparator,
    ExpectedAction,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
    get_tool_calls,
)


CHECK_NAME = "tool_call"

# Matches the shipped tool-call configs. Pivot datasets tune this per dataset - one
# shipped pivot config sets it to 0.0 - so it belongs in the row's `checks` spec
# rather than being fixed here.
DEFAULT_WORD_COUNT_SIMILARITY_THRESHOLD = 0.1

_PASS_CATEGORIES = frozenset(
    {
        StepRewardCategory.EXPECTED_TOOL_CALL,
        StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND,
    }
)


def _parse_expected(expected: Any) -> Optional[ExpectedAction]:
    """Accept an expected action as a dict, a JSON string, or an already-built model."""
    if expected is None:
        return None
    if isinstance(expected, (MessageAction, FunctionCallAction, FunctionCallBatchAction)):
        return expected
    if isinstance(expected, str):
        expected = json.loads(expected)
    if not isinstance(expected, dict):
        return None

    action_type = expected.get("type")
    if action_type == "message":
        return MessageAction(**expected)
    if action_type == "function_call":
        return FunctionCallAction(**expected)
    if action_type == "function_call_batch":
        return FunctionCallBatchAction(**expected)
    return None


def _describe(action: Optional[ExpectedAction]) -> str:
    if action is None:
        return "none"
    if isinstance(action, MessageAction):
        text = action.content or ""
        return f'message "{text[:60]}{"..." if len(text) > 60 else ""}"'
    calls = get_tool_calls(action)
    return ", ".join(call.name for call in calls) or "none"


def verify_tool_call(
    output_items: Any,
    expected_action: Any,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compare a produced action against the expected one and return EVIDENCE.

    Never raises and never returns a reward. Every failure mode - unparseable
    expected action, malformed arguments, missing output - becomes an `errors`
    entry with `passed` False.
    """
    result: dict[str, Any] = {
        "check": CHECK_NAME,
        "passed": False,
        "category": None,
        "expected": None,
        "actual": None,
        "errors": [],
    }

    try:
        expected = _parse_expected(expected_action)
        if expected is None:
            result["errors"].append("expected action missing or unrecognised")
            return result
        result["expected"] = _describe(expected)

        actual = extract_action(output_items)
        result["actual"] = _describe(actual)
        if actual is None:
            result["category"] = str(StepRewardCategory.NO_ACTION_FOUND)
            return result

        settings = dict(config or {})
        settings.setdefault(
            "word_count_similarity_threshold", DEFAULT_WORD_COUNT_SIMILARITY_THRESHOLD
        )
        comparator = ActionComparator(config=ToolCallComparatorConfig(**settings))

        comparison = comparator.compare_action(expected_action=expected, actual_action=actual)
        result["category"] = str(comparison.category)
        result["passed"] = comparison.category in _PASS_CATEGORIES
    except Exception as exc:  # noqa: BLE001 - deliberate: a verifier never raises
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    return result


def render(result: dict[str, Any]) -> str:
    """Short bullet block for the judge prompt. Bullets are what the judge reads."""
    verdict = "PASSED" if result.get("passed") else "FAILED"
    lines = [f"- tool-call check: {verdict}"]

    expected, actual = result.get("expected"), result.get("actual")
    if expected is not None:
        lines.append(f"- expected action: {expected}")
    if actual is not None:
        lines.append(f"- actual action: {actual}")
    if result.get("category"):
        lines.append(f"- {result['category']}")
    for error in result.get("errors", []):
        lines.append(f"- error: {error}")

    return "\n".join(lines)
