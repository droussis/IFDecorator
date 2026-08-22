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
"""Normalise a model response into the canonical action shape.

Adapted from NeMo Gym so it accepts plain output items - a list of dicts, or any
objects with the same attributes - instead of the framework's response type. That
is the only change: the precedence rules and the batch behaviour are upstream's.
"""

from __future__ import annotations

from typing import Any, Optional

from .comparator import (
    ExpectedAction,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
)


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def extract_action(output_items: Any) -> Optional[ExpectedAction]:
    """Normalize model output into the canonical action shape that dataset rows use.

    Tool calls take precedence over assistant text, so a response that both narrates
    and calls tools is judged on the calls. Several tool calls in one response become
    a batch, which the comparator then matches without regard to emission order.

    `output_items` may be a response object with an `output` attribute, or the list
    of output items directly.
    """
    items = _get(output_items, "output", output_items) or []

    tool_calls: list[FunctionCallAction] = []
    assistant_text: Optional[str] = None

    for output_item in items:
        item_type = _get(output_item, "type")

        if item_type == "function_call":
            tool_calls.append(
                FunctionCallAction(
                    type="function_call",
                    name=_get(output_item, "name"),
                    arguments=_get(output_item, "arguments"),
                )
            )

        elif item_type == "message" and _get(output_item, "role") == "assistant" and assistant_text is None:
            for content_item in _get(output_item, "content", []) or []:
                if _get(content_item, "type") == "output_text":
                    assistant_text = _get(content_item, "text")
                    break

    if len(tool_calls) == 1:
        return tool_calls[0]

    if tool_calls:
        return FunctionCallBatchAction(type="function_call_batch", calls=tool_calls)

    if assistant_text is not None:
        return MessageAction(type="message", content=assistant_text)

    return None
