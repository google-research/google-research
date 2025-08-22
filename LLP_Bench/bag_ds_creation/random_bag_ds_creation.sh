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


for bag_size in 64 128 256 512
do
  for split in 0 1 2 3 4
  do
    python3 bag_ds_creation/random_bag_ds_creation.py --bag_size=$bag_size --split=$split --which_dataset=criteo_ctr
  done
done

for bag_size in 64 128 256 512
do
  for split in 0 1 2 3 4
  do
    python3 bag_ds_creation/random_bag_ds_creation.py --bag_size=$bag_size --split=$split --which_dataset=criteo_sscl
  done
done