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
cd wavelets_repr/swvox/swvox
rm -rf ../build ../swvox.egg-info
rm -rf __pycache__/ swvox-test.cpython-38-x86_64-linux-gnu.so CMakeFiles CMakeCache.txt
cmake csrc -DCMAKE_PREFIX_PATH=$HOME/anaconda3/envs/plenoxels/lib/python3.8/site-packages/torch
make -j32
cd wavelets_repr/swvox
pip uninstall -y swvox
pip install .
