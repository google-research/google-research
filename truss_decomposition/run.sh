# Copyright 2022 The Google Research Authors.
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

# Install gflags development files if not already present.
[ -d /usr/include/gflags/ ] || sudo apt -y install libgflags-dev

# For travis: libomp.so is not found otherwise.
export LD_LIBRARY_PATH=/usr/local/clang/lib:$LD_LIBRARY_PATH

cd "$(readlink -f "$(dirname "$0")")"

echo "Compiling..."

./compile.sh

echo "Computing exact truss decomposition"

./truss_decomposition_parallel < clique10.nde

echo "Computing approximate truss decomposition"

./td_approx_external < clique10.nde

echo "Computing approximate truss decomposition and switching to exact algorithm"

./td_approx_external --edge_limit 10000 --exact_exe ./truss_decomposition_parallel < clique10.nde
