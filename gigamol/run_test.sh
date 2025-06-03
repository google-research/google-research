# To run this script, please make sure you have virtualenv and bazel installed.
# Follow the instructions https://docs.bazel.build/versions/4.0.0/install.html
# to install bazel, and run "pip install virtualenv" to install virtualenv.

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

# Use a virtual environment.
virtualenv -p python3 .
source ./bin/activate

pip install -r gigamol/requirements.txt

# Note: even though 'protobuf' is listed in the requirements.txt,
# `protoc` did not appear to get added to the $PATH.
# If that happens to you too,
# consider `sudo apt-get install protobuf-compiler` before running this script.
protoc --experimental_allow_proto3_optional gigamol/molecule_graph_proto/molecule_graph.proto --python_out=.

python3 -m gigamol.molecule_graph_proto.molecule_graph_test
cd gigamol
python3 configure.py
bazel build molecule_graph_parsing_ops/cc:molecule_graph_parsing_ops.so --experimental_repo_remote_exec
cd ..
python3 -m gigamol.molecule_graph_parsing_ops.py.molecule_graph_parsing_ops_test
