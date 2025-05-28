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

"""Defines ScanNet, including the MSeg version.

URL: http://www.scan-net.org/

Paper:
ScanNet: Richly-annotated 3d reconstructions of indoor scenes.
A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner.
In CVPR, 2017.
"""


from factors_of_influence.fids import mseg_base

ScanNet20 = mseg_base.MSegBase(
    mseg_name='ScanNet',
    mseg_original_name='scannet-41',
    mseg_base_name='scannet-20',
    mseg_dirname='ScanNet/scannet_frames_25k/',
    mseg_train_dataset=False,
    )
