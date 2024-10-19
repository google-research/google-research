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

# creating a virtual environment and installing requirements
python3 -m venv env
source env/bin/activate
pip3 install -r requirments.txt

# Downloading and preprocessing GoEmotions dataset
python3 preprocess.py --corpus emotions

# Replicating the experiment on Joy
python3 run_annotators_modeling.py --corpus emotions --label joy