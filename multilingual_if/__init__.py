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
"""Multilingual instruction-following verification for Latin / Cyrillic / Greek.

Three consumers, one source of truth:

    synthetic data generation   eligibility.eligible_ids(lang) to decide what to
                                sample; localized_kwargs / alphabet / quote_pair
                                to fill args in the target language
    RL verification             checks.verify(ids, kwargs, response, lang)
    the IFDecorator flywheel    both of the above, plus chars_per_word for the
                                length-constraint calibration in w4

See README.md for the integration guide and profile.yaml for the per-constraint
matrix and the measurements behind every constant.
"""

from .checks import verify, verify_one
from .eligibility import (
    alphabet,
    chars_per_word,
    conflicts,
    eligible,
    eligible_ids,
    language_profile,
    localized_kwargs,
    quote_pair,
    status,
    supported_languages,
)
from .textops import count_sentences, count_words, nfc, split_into_sentences


__all__ = [
    "alphabet",
    "chars_per_word",
    "conflicts",
    "count_sentences",
    "count_words",
    "eligible",
    "eligible_ids",
    "language_profile",
    "localized_kwargs",
    "nfc",
    "quote_pair",
    "split_into_sentences",
    "status",
    "supported_languages",
    "verify",
    "verify_one",
]
