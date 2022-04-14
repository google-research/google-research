# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Constants used in dual encoder SMITH model."""

# Constants related to training mode.
# There are three different modes for the model training:
# Joint_train: the loss includes both masked LM losses and the text matching
# loss.
# Pretrain: the loss includes masked word LM loss and masked sentence LM loss.
# The masked sentence LM loss only applies to the dual encoder SMITH model.
# Finetune: the loss includes the text matching loss.
TRAIN_MODE_FINETUNE = "finetune"
TRAIN_MODE_PRETRAIN = "pretrain"
TRAIN_MODE_JOINT_TRAIN = "joint_train"

# Constants related to model name.
MODEL_NAME_SMITH_DUAL_ENCODER = "smith_dual_encoder"

# Constants related to final document representation combining method.
DOC_COMBINE_NORMAL = "normal"
DOC_COMBINE_SUM_CONCAT = "sum_concat"
DOC_COMBINE_MEAN_CONCAT = "mean_concat"
DOC_COMBINE_ATTENTION = "attention"

# Constants related to human rating aggregation methhod.
RATING_AGG_MEAN = "mean"
RATING_AGG_MAJORITY_VOTE = "majority_vote"
