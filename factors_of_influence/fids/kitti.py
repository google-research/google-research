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

"""Defines KITTI segmentation, including the MSeg version.

KITTI has a wide setup of dataset configurations aimed for benchmarking
different modalities (flow, depth, segmentation, detection, ...), see:

http://www.cvlibs.net/datasets/kitti/

Paper:
Vision meets Robotics: The KITTI Dataset.
Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun.
 International Journal of Robotics Research (IJRR), 2013
"""


from factors_of_influence.fids import mseg_base

KITTISeg = mseg_base.MSegBase(
    mseg_name='KITTI Segmentation',
    mseg_original_name='kitti-34',
    mseg_base_name='kitti-19',
    mseg_dirname='KITTI/',
    mseg_train_dataset=False,
    mseg_segmentation_background_labels=['unlabeled', 'out of roi'],
    )
