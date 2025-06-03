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


# Run this script (once) to download SyGuS benchmarks.

set -e

dir=$(dirname "$(readlink -f "$0")")
cd "${dir}"

echo "Downloading SyGuS benchmarks..."
mkdir -p sygus_benchmarks
cd sygus_benchmarks
svn checkout https://github.com/SyGuS-Org/benchmarks/trunk/comp/2019/PBE_SLIA_Track/euphony
svn checkout https://github.com/ellisk42/ec/trunk/PBE_Strings_Track
cd ..
