#!/bin/bash
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
  virtualenv -p python3 .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  PYTHON_BIN="python3"
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m unittest discover -s flowracle -p "*_test.py" -v
