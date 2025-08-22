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

"""Provides / defines data directories throughout the package."""

MSEG_ROOT_DIR = './data/mseg/'
MSEG_MASTER_FILE_PATH = f'{MSEG_ROOT_DIR}/MSeg_master.tsv'

ADE20K_IMAGES_DIR = f'{MSEG_ROOT_DIR}/after_remapping/ADE20K/ADE20K_2016_07_26/images/'

BDD_100K_DIR = './data/bdd100k/'
BDD_100K_PAN_SEG = f'{BDD_100K_DIR}/pan_seg/'

COCO_ANNOTATION_DIR = './data/coco/annotations/'
COCO_PANOPTIC_DIR = f'{MSEG_ROOT_DIR}/after_remapping/COCOPanoptic/annotations/'

SUNRGBD_DEPTH_DIR = './data/sunrgbd/depth/'

ISAID_DATASET_DIR = './data/isaid/'
ISPRS_DATASET_DIR = './data/isprs/'

STANFORD_DOGS_DIR = './data/stanford_dogs'

SUIM_DATASET_DIR = './data/suim'

UNDERWATER_TRASH_DIR = './data/underwater_trash'

VGALLERY_DIR = './data/vgallery/'

VKITTI2_DIR = './data/vkitti2/'
