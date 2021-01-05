# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Central location for all constants related to MLPerf SSD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==============================================================================
# == Model =====================================================================
# ==============================================================================
IMAGE_SIZE = 300

# TODO(taylorrobie): MLPerf uses 80, but COCO documents 90. (RetinaNet uses 90)
# Update(taylorrobie): Labels > 81 show up in the pipeline. This will need to
#                      be resolved.
NUM_CLASSES = 81  # Including "no class". Not all COCO classes are used.

# Note: Zero is special. (Background class) CLASS_INV_MAP[0] must be zero.
CLASS_INV_MAP = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90)
_MAP = {j: i for i, j in enumerate(CLASS_INV_MAP)}
CLASS_MAP = tuple(_MAP.get(i, -1) for i in range(max(CLASS_INV_MAP) + 1))

NUM_SSD_BOXES = 8732

RESNET_DEPTH = 34

"""SSD specific"""
MIN_LEVEL = 3
MAX_LEVEL = 8

FEATURE_SIZES = (38, 19, 10, 5, 3, 1)
STEPS = (8, 16, 32, 64, 100, 300)

# https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
SCALES = (21, 45, 99, 153, 207, 261, 315)
ASPECT_RATIOS = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
NUM_DEFAULTS = (4, 6, 6, 6, 4, 4)
NUM_DEFAULTS_BY_LEVEL = {3: 4, 4: 6, 5: 6, 6: 6, 7: 4, 8: 4}
SCALE_XY = 0.1
SCALE_HW = 0.2
BOX_CODER_SCALES = (1 / SCALE_XY, 1 / SCALE_XY, 1 / SCALE_HW, 1 / SCALE_HW)
MATCH_THRESHOLD = 0.5

# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

# SSD Cropping
NUM_CROP_PASSES = 50
CROP_MIN_IOU_CHOICES = (0, 0.1, 0.3, 0.5, 0.7, 0.9)
P_NO_CROP_PER_PASS = 1 / (len(CROP_MIN_IOU_CHOICES) + 1)

# Hard example mining
NEGS_PER_POSITIVE = 3

# Batch normalization
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-4


# ==============================================================================
# == Optimizer =================================================================
# ==============================================================================
LEARNING_RATE_SCHEDULE = (
    (0, 1e-3),
    (160000, 1e-4),
    (200000, 1e-5),
)
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


# ==============================================================================
# == Keys ======================================================================
# ==============================================================================
BOXES = "boxes"
CLASSES = "classes"
NUM_MATCHED_BOXES = "num_matched_boxes"
IMAGE = "image"
SOURCE_ID = "source_id"
RAW_SHAPE = "raw_shape"
PRED_BOXES = "pred_boxes"
PRED_SCORES = "pred_scores"


# ==============================================================================
# == Evaluation ================================================================
# ==============================================================================

# Note: This is based on a batch size of 32
#   https://github.com/mlperf/reference/blob/master/single_stage_detector/ssd/train.py#L21-L37
CHECKPOINT_FREQUENCY = 20000
MAX_NUM_EVAL_BOXES = 200
OVERLAP_CRITERIA = 0.5  # Used for nonmax supression
MIN_SCORE = 0.05  # Minimum score to be considered during evaluation.
DUMMY_SCORE = -1e5  # If no boxes are matched.

ANNOTATION_FILE = "annotations/instances_val2017.json"
COCO_NUM_TRAIN_IMAGES = 118287
COCO_NUM_VAL_IMAGES = 4952
