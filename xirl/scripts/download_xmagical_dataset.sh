# coding=utf-8
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

mkdir -p /tmp/xirl/datasets/
cd /tmp/xirl/datasets/
ID="1f_8FdWtUqc-_heAOUSN-KdGgeJK6AjOW"
gdown https://drive.google.com/uc?id=$ID
unzip xirl_corl.zip
rm xirl_corl.zip
mv xirl_corl xmagical
# Final location: /tmp/xirl/datasets/xmagical
