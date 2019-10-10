# Copyright 2019 The Google Research Authors.
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

TEMP_DIR=$(mktemp -d --tmpdir=${TMPDIR})

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install tensorflow-gpu
pip install tensorflow-hub
pip install git+https://github.com/google-research/language.git
pip install -r property_linking/src/requirements.txt

python3 -m property_linking.src.property_linker --root_dir=property_linking/data --num_epochs=10

rm -r $TEMP_DIR
