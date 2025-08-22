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



raw_dataset_dir=$1
data_dir=$2

for dataset in mnist cifar10 domainnet fmow amazon_review otto
do
  python -m active_selective_prediction.build_datasets --gpu 0 --dataset $dataset --data-dir $data_dir --raw-dataset-dir $raw_dataset_dir
done
