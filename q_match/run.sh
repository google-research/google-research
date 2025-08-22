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

virtualenv q_match_env
source q_match_env/bin/activate

pip install -r q_match/requirements.txt
python -m q_match.scripts.train \
-algo=q_match_pretext+supervised_training \
--dataset=example \
--eval_task=B \
--pretext_epochs=1 \
--supervised_epochs=1 \
--num_trials=1
