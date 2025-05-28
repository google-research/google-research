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

"""Defines WildDash-V1, including the MSeg version.

URL: https://wilddash.cc/

Paper:
Wilddash - creating hazard-aware benchmarks.
O. Zendel, K. Honauer, M. Murschitz, D. Steininger, and G. Fernandez Dominguez.
In ECCV, 2018.
"""


from factors_of_influence.fids import mseg_base

WildDash19 = mseg_base.MSegBase(
    mseg_name='WildDashDataset',
    mseg_original_name='wilddash-34',
    mseg_base_name='wilddash-19',
    mseg_dirname='WildDash/',
    mseg_train_dataset=False,
    mseg_segmentation_background_labels=['unlabeled', 'out of roi'],
    )
