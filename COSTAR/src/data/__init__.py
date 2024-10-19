# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Dataset."""

import src.data.cancer_sim
import src.data.dataset_collection
import src.data.m5
import src.data.mimic_iii

SyntheticCancerDatasetCollection = (
    src.data.cancer_sim.SyntheticCancerDatasetCollection
)
RealDatasetCollection = src.data.dataset_collection.RealDatasetCollection
SyntheticDatasetCollection = (
    src.data.dataset_collection.SynteticDatasetCollection
)
M5RealDatasetCollection = src.data.m5.M5RealDatasetCollection
MIMIC3RealDatasetCollection = src.data.mimic_iii.MIMIC3RealDatasetCollection
MIMIC3SyntheticDatasetAgeDomainCollection = (
    src.data.mimic_iii.MIMIC3SyntheticDatasetAgeDomainCollection
)
MIMIC3SyntheticDatasetCollection = (
    src.data.mimic_iii.MIMIC3SyntheticDatasetCollection
)
