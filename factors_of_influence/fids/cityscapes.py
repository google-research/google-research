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

"""Imports CityScapes, only the MSeg version.

URL: https://www.cityscapes-dataset.com/downloads/

Paper:
The cityscapes dataset for semantic urban scene understanding.
M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
U. Franke, S. Roth, and B. Schiele.  In CVPR, 2016.

City Scapes contains additional modalities / ground-truths, including:
 - disparity (depth)
 - coarse segmentation
 - person boundingboxes.
"""


from factors_of_influence.fids import mseg_base

CityScapes = mseg_base.MSegBase(
    mseg_name='CityScapes',
    mseg_original_name='cityscapes-34',
    mseg_base_name='cityscapes-19',
    mseg_dirname='Cityscapes/',
    mseg_train_dataset=True,
    mseg_segmentation_background_labels=['unlabeled', 'out of roi'],
    )
