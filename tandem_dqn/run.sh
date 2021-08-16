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
# Note that to run this on the classic control and ALE environments you need to
# obtain the gin files for Dopamine JAX agents:
# github.com/google/dopamine/tree/master/dopamine/jax/agents/dqn/configs
#
# The MinAtar config files are included with this library.
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

cd ..
pip install -r tandem_dqn/requirements.txt
python3 -m tandem_dqn.train \
  --base_dir=/tmp/tandem_dqn \
  --gin_files=tandem_dqn/configs/dqn_asterix.gin \
  --gin_bindings="TandemRunner.suite='minatar'"
