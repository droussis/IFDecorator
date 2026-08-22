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
"""Reusable pieces of the multilingual IF synthesis pipeline.

Orchestration is deliberately absent - that lives in our own synthesis library.
What is here is the part worth sharing: the prompt templates, the parsers for
their outputs, deterministic containment matching, and the row schema with the
validation that catches the defects the original pipeline shipped with.

See docs/SYNTHESIS_PIPELINE.md for the complete stage-by-stage guide.
"""

from . import matching, parsing, prompts, schema
from .schema import (
    Row,
    Rubric,
    SchemaError,
    pass_rate_report,
    route_by_difficulty,
    validate_row,
)


__all__ = [
    "prompts",
    "parsing",
    "matching",
    "schema",
    "Row",
    "Rubric",
    "SchemaError",
    "validate_row",
    "route_by_difficulty",
    "pass_rate_report",
]
