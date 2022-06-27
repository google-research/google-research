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

# coding=utf-8
#!/bin/bash
#
# This script downloads the X-MAGICAL demonstration dataset and stores it in
# /tmp/xirl/datasets/xmagical.
set -e
set -x

# GDrive file ID.
ID="1VdMRYu0Y-ep_vq28hW2n0UZaow2iaW1i"

mkdir -p /tmp/xirl/datasets/
cd /tmp/xirl/datasets/
gdown https://drive.google.com/uc?id=$ID
unzip xmagical.zip && rm xmagical.zip
