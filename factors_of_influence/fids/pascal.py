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

"""Defines the Pascal VOC 2012 and Pascal Context datasets.

Pascal VOC 2012:
URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
Paper:
- The PASCAL Visual Object Classes Challenge: A Retrospective.
  Everingham, M., Eslami, S. M. A., Van Gool, L., Williams, C. K. I.,
  Winn, J. and Zisserman, A. In International Journal of Computer Vision, 2015.

Pascal Context:
URL: https://www.cs.stanford.edu/~roozbeh/pascal-context/
Paper:
- The Role of Context for Object Detection and Semantic Segmentation in the Wild
  Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee,
  Sanja Fidler, Raquel Urtasun, Alan Yuille. In CVPR, 2014.
"""


from factors_of_influence.fids import mseg_base

PascalContext = mseg_base.MSegBase(
    mseg_name='Pascal Context',
    mseg_original_name='pascal-context-460',
    mseg_base_name='pascal-context-60',
    mseg_dirname='PASCAL_Context/',
    mseg_train_dataset=False,
    mseg_segmentation_background_labels=[
        'background', 'unlabeled', 'Unlabeled'
    ],
    mseg_use_mapping_for_mseg_segmentation=True,
)

PascalVOC = mseg_base.MSegBase(
    mseg_name='Pascal VOC2012',
    mseg_original_name='voc2012',
    mseg_base_name='voc2012',
    mseg_dirname='PASCAL_VOC_2012/',
    mseg_train_dataset=False,
    mseg_segmentation_background_labels=['background'],
    mseg_use_mapping_for_mseg_segmentation=True,
    )
