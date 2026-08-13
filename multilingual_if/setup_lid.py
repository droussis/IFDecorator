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
"""Locates fastText's lid.176.ftz, following the setup_ifbench.py pattern.

The model is NOT vendored in this repo on purpose: fastText's pretrained LID
models are released under CC-BY-SA, which is incompatible with Apache-2.0 (see
Gym's AGENTS.md, "Do not introduce licenses incompatible with Apache-2.0"). It is
fetched or located at setup time instead.

Resolution order:
  1. $MULTILINGUAL_IF_LID_MODEL, if set
  2. a previously resolved copy in this directory
  3. the `ftlid` PyPI package, which vendors lid.176.ftz verbatim
     (ftlid/lid.176.ftz, 938,013 bytes) - useful where dl.fbaipublicfiles.com is
     unreachable, as it is behind many corporate egress proxies
  4. a direct download from dl.fbaipublicfiles.com
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys


MODEL_NAME = "lid.176.ftz"
MODEL_URL = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{MODEL_NAME}"
EXPECTED_SIZE = 938_013
SERVER_DIR = pathlib.Path(__file__).parent
LOCAL_MODEL = SERVER_DIR / MODEL_NAME


def _from_ftlid() -> pathlib.Path | None:
    spec = importlib.util.find_spec("ftlid")
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = pathlib.Path(next(iter(spec.submodule_search_locations))) / MODEL_NAME
    return candidate if candidate.exists() else None


def ensure_lid_model() -> str:
    """Return a path to lid.176.ftz, acquiring it if needed."""
    override = os.environ.get("MULTILINGUAL_IF_LID_MODEL")
    if override:
        if not pathlib.Path(override).exists():
            raise FileNotFoundError(f"MULTILINGUAL_IF_LID_MODEL={override!r} does not exist")
        return override

    if LOCAL_MODEL.exists():
        return str(LOCAL_MODEL)

    vendored = _from_ftlid()
    if vendored is not None:
        return str(vendored)

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ftlid"], check=True)
    except subprocess.CalledProcessError:
        pass
    else:
        importlib.invalidate_caches()
        vendored = _from_ftlid()
        if vendored is not None:
            return str(vendored)

    if shutil.which("curl"):
        subprocess.run(["curl", "-sSL", "-o", str(LOCAL_MODEL), MODEL_URL], check=True)
        if LOCAL_MODEL.exists() and LOCAL_MODEL.stat().st_size > 0:
            return str(LOCAL_MODEL)

    raise RuntimeError(
        f"Could not obtain {MODEL_NAME}. Install the `ftlid` package (which vendors it), "
        f"download it from {MODEL_URL}, or set MULTILINGUAL_IF_LID_MODEL to an existing copy."
    )


if __name__ == "__main__":
    print(ensure_lid_model())
