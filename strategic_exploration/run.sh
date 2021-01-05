# Copyright 2021 The Google Research Authors.
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
set -e
set -x

virtualenv -p python3.5 venv
source venv/bin/activate

export HRL_DATA="data/"
pip install -r strategic_exploration/requirements.txt
python -m strategic_exploration.main strategic_exploration/configs/debug-systematic.txt strategic_exploration/configs/config-mixins/extra_repeat.txt
