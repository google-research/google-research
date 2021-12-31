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
# Note: this script must be run from the git repository root (google_research/).
set -e
set -x

# Create and activate a new virtual environment.
python3 -m venv env
source ./env/bin/activate

# Install necessary dependencies.
pip install -r flare_removal/requirements.txt

# The following command should execute without missing dependencies. However,
# it's expected to fail unless you provide appropriate model and data paths in
# additional command-line arguments (or change the default argument values in
# the source file). See README.md for more details.
python3 -m flare_removal.python.remove_flare
