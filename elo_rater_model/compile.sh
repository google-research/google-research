# Copyright 2023 The Google Research Authors.
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

cd "$(dirname "$0")"

[ -d abseil-cpp ] || git clone https://github.com/abseil/abseil-cpp.git -b lts_2024_01_16
if ! [ -d eigen ]
then
  git clone https://gitlab.com/libeigen/eigen
  (cd eigen && git checkout b2814d53a707f699e5c56f565847e7020654efc2)
fi
[ -d highway ] || git clone https://github.com/google/highway -b 1.0.7
if ! [ -d csv-parser ]
then
  git clone https://github.com/AriaFallah/csv-parser
  (cd csv-parser && git checkout 4965c9f320d157c15bc1f5a6243de116a4caf101)
fi

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=On ..
make -j $(nproc)
