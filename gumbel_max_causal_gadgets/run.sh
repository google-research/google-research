# Copyright 2024 The Google Research Authors.
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

#!/bin/bash
# Make sure that the gadgets can be trained without errors.

set -e
set -x

virtualenv -p python3 .venv_gumbel_max_causal_gadgets
source .venv_gumbel_max_causal_gadgets/bin/activate
pip install --upgrade pip
pip install -r gumbel_max_causal_gadgets/requirements.txt

python -m unittest gumbel_max_causal_gadgets/gadget_test.py
