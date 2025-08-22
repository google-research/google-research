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


python3 data_preprocessing/preprocess.py --which_dataset=criteo_ctr
python3 data_preprocessing/preprocess.py --which_dataset=criteo_sscl

#script to generate analysis
python3 bag_ds_analysis/autoint_embedding_generator.py
python3 bag_ds_analysis/sscl_embedding_generator.py
source bag_ds_analysis/label_analysis.sh
source bag_ds_analysis/feature_analysis.sh

#script to generate feature bag datasets
source bag_ds_creation/feature_bag_ds_creation.sh

#script to generate random bag datasets
source bag_ds_creation/random_bag_ds_creation.sh

#script to generate fixed size feature bag datasets
source bag_ds_creation/fixed_size_feature_bag_ds_creation.sh

#script to generate mean map vectors for all datasets
source mean_map_comp/mean_map.sh

#script to train all baselines on feature bags datasets
source model_training/feature_bags_train.sh

#script to train all baselines on random bags datasets
source model_training/random_bags_train.sh

#script to train all baselines on fixed size feature bags datasets
source model_training/fixed_size_feature_bags_train.sh

#to collect and process all the results
python3 bag_ds_analysis/result_collector.py
