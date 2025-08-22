#!/bin/bash
# Copyright 2025 The Google Research Authors.
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


virtualenv -p python3 ./venv
source ./venv/bin/activate

# Installs scenic library.
git clone https://github.com/google-research/scenic.git
cd scenic
pip3 install .
cd ../

pip3 install -r invariant_slot_attention/requirements.txt

python -m invariant_slot_attention.main --config invariant_slot_attention/configs/tetrominoes/equiv_transl.py --workdir tmp/
