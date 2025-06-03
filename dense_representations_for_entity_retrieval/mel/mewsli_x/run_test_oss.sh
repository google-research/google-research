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


# Runs test on the open-sourced modules stored under mewsli_x/.
#
# NOTE: This script should be run from the parent directory of
# dense_representations_for_entity_retrieval/.

set -eux
D="dense_representations_for_entity_retrieval"
REQUIREMENTS="${D}/mel/wikinews_extractor/requirements.txt"

# Set up virtual environment.
virtualenv -p python3 ./testenv
source ./testenv/bin/activate
pip install -r ${REQUIREMENTS}

python -m "${D}.mel.mewsli_x.io_util_test"
python -m "${D}.mel.mewsli_x.schema_test"
python -m "${D}.mel.mewsli_x.restore_text_test"
