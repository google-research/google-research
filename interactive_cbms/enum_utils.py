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

"""Useful enum definitions."""

import enum


class Dataset(str, enum.Enum):
  CUB = 'cub'
  CHEXPERT = 'chexpert'
  OAI = 'oai'


class Arch(str, enum.Enum):
  X_TO_C = 'XtoC'
  C_TO_Y = 'CtoY'
  X_TO_C_TO_Y = 'XtoCtoY'
  X_TO_C_TO_Y_SIGMOID = 'XtoCtoY_sigmoid'
  X_TO_Y = 'XtoY'


class BottleneckType(str, enum.Enum):
  INDEPENDENT = 'independent'
  JOINT = 'joint'
  JOINT_SIGMOID = 'joint_sigmoid'


class InterventionPolicy(str, enum.Enum):
  GLOBAL_RANDOM = 'global_random'
  INSTANCE_RANDOM = 'instance_random'
  GLOBAL_GREEDY = 'global_greedy'
  INSTANCE_GREEDY = 'instance_greedy'
  COOP = 'coop'


class InterventionFormat(str, enum.Enum):
  PROBS = 'probs'
  LOGITS = 'logits'
  BINARY = 'binary'


class Metric(str, enum.Enum):
  """Enum for metrics used in Greedy and CooP policies."""
  MEAN_REC_RANK = 'mean_rec_rank'
  AUC = 'auc'
  CAT_AUC = 'cat_auc'
  BINARY_XENT = 'binary_xent'
  CAT_XENT = 'cat_xent'
  CONCEPT_ENTROPY = 'concept_entropy'
  CONCEPT_CONFIDENCE = 'concept_confidence'
  LABEL_ENTROPY_CHANGE = 'label_entropy_change'
  LABEL_ENTROPY_DECREASE = 'label_entropy_decrease'
  LABEL_CONFIDENCE_CHANGE = 'label_confidence_change'
  LABEL_CONFIDENCE_INCREASE = 'label_confidence_increase'
  LABEL_KLD = 'label_kld'
  LABEL_ENTROPY_CHANGEV2 = 'label_entropy_changev2'
  LABEL_ENTROPY_DECREASEV2 = 'label_entropy_decreasev2'
  LABEL_CONFIDENCE_CHANGEV2 = 'label_confidence_changev2'
  LABEL_CONFIDENCE_INCREASEV2 = 'label_confidence_increasev2'


class Checkpoint(str, enum.Enum):
  TRAINLOSS = 'trainloss'
  CONCEPT = 'concept'
  CLASS = 'class'








