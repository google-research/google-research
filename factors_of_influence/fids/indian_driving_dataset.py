# coding=utf-8
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

"""Defines IDD (Indian Driving Dataset), including the MSeg version.

URL: https://idd.insaan.iiit.ac.in/dataset/details/

Paper:
IDD: A Dataset for Exploring Problems of Autonomous Navigation
in Unconstrained Environments. Girish Varma, Anbumani Subramanian,
Anoop Namboodiri, Manmohan Chandraker, and C V Jawahar. In WACV, 2019.

Note: The detection dataset (also available for download) has roughly 4 times
more annotated images available than the segmentation dataset.
"""

from factors_of_influence.fids import mseg_base

IDD = mseg_base.MSegBase(
    mseg_name='Indian Driving Dataset (IDD)',
    mseg_original_name='idd-40',
    mseg_base_name='idd-39',
    mseg_dirname='IDD/IDD_Segmentation/',
    mseg_train_dataset=True,
    mseg_segmentation_background_labels=[
        'unlabeled', 'out of roi', 'train', 'ego vehicle',
        'rectification border', 'license plate'
    ],  # These classes are either not present in the official label set, but
    # yet are present in the annotation, or are annotated as not for training.
    # See: https://idd.insaan.iiit.ac.in/dataset/details/
    mseg_use_mapping_for_mseg_segmentation=True,
    )
