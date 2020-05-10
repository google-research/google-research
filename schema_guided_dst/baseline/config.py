# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Code for configuring the model depending on the dataset."""
import collections

# Config object that contains the following info:
# file_ranges: the file ranges of train, dev, and test set.
# max_num_cat_slot: Maximum allowed number of categorical trackable slots for a
# service.
# max_num_noncat_slot: Maximum allowed number of non-categorical trackable slots
# for a service.
# max_num_value_per_cat_slot: Maximum allowed number of values per categorical
# trackable slot.
# max_num_intent: Maximum allowed number of intents for a service.
DatasetConfig = collections.namedtuple("DatasetConfig", [
    "file_ranges", "max_num_cat_slot", "max_num_noncat_slot",
    "max_num_value_per_cat_slot", "max_num_intent"
])

DATASET_CONFIG = {
    "dstc8_single_domain":
        DatasetConfig(
            file_ranges={
                "train": range(1, 44),
                "dev": range(1, 8),
                "test": range(1, 12)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4),
    "dstc8_multi_domain":
        DatasetConfig(
            file_ranges={
                "train": range(44, 128),
                "dev": range(8, 21),
                "test": range(12, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4),
    "dstc8_all":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4),
    "multiwoz21_all":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1)
}
