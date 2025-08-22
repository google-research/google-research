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

python3 -m venv ugsl
source ugsl/bin/activate

pip3 install -r ugsl/requirements.txt

# We have used tfgnn for Graph Neural Networks operations. To get the latest
# features of the library, you should install it from the source using this link
# https://github.com/tensorflow/gnn#installation-from-source.

# Fill in appropriate data path
python3 -m ugsl.main