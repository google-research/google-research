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

virtualenv -p python3 .
source ./bin/activate

pip install -r streetview_contrails_dataset/persistence/requirements.txt

# Note: even though 'protobuf' is listed in the requirements.txt,
# `protoc` did not appear to get added to the $PATH.
# If that happens to you too,
# consider `sudo apt-get install protobuf-compiler` before running this script.
protoc --experimental_allow_proto3_optional streetview_contrails_dataset/persistence/streetview_dataset.proto --python_out=.

python -m streetview_contrails_dataset.persistence.read_protos_test
