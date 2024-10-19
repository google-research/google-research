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
#
# Execute this file in the root, i.e., one folder above `vct`.
#
set -e
set -x

python -m venv vct_env
source ./vct_env/bin/activate

pip install -r vct/requirements.txt
python -m vct.src.models_test

# We won't land here if there are errors due to set -x.
echo "*** Success!"
