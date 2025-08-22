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


make CXX=clang++
phi=1 k=1 T=1000 verbose=0 ./exposure_design_main <<EOF
div1 out1 1
div1 out2 0.5
div2 out2 2
div3 out1 0.001
div3 out1 100
div4 out4 1
div5 out4 1
div6 out5 1
EOF
