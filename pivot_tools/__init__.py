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
"""Single-step tool-call grading, vendored from NeMo Gym.

    comparator.py  the comparator, verbatim - stdlib and pydantic only
    actions.py     normalises model output into the canonical action shape
    validate.py    pivot-dataset row validator, verbatim
    evidence.py    OURS - re-expresses a comparison as evidence, not a reward

See PROVENANCE.md for sources and the re-sync recipe.
"""

from .actions import extract_action
from .comparator import (
    ActionComparator,
    ExpectedAction,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
    ToolCallComparatorConfig,
)
from .evidence import render, verify_tool_call


__all__ = [
    "verify_tool_call",
    "render",
    "extract_action",
    "ActionComparator",
    "ToolCallComparatorConfig",
    "ExpectedAction",
    "MessageAction",
    "FunctionCallAction",
    "FunctionCallBatchAction",
]
