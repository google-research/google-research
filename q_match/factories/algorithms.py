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

"""Algorithms Factory.

Maps a name to a pair of pretext and supervised trainign algorithms.
"""

# pylint: disable=g-bad-import-order
from q_match.algorithms.dummy_pretext_training import DummyPretextTraining
from q_match.algorithms.npair_imix_pretext_training import NPairiMixPretextTraining
from q_match.algorithms.supervised_training import SupervisedTraining
from q_match.algorithms.tabnet_pretext_training import TabnetPretextTraining
from q_match.algorithms.vime_pretext_training import VimePretextTraining
from q_match.algorithms.q_match_pretext import QMatchPretextTraining
from q_match.algorithms.dino_pretext_training import DinoPretextTraining
from q_match.algorithms.simsiam_pretext_training import SimSiamPretextTraining
from q_match.algorithms.vicreg_pretext_training import VICRegPretextTraining
from q_match.algorithms.simclr_pretext_training import SimCLRPretextTraining


NAME_TO_ALGOS = {'supervised_training':
                     (None, SupervisedTraining),
                 'dummy_pretext+supervised_training':
                     (DummyPretextTraining, SupervisedTraining),
                 'vime_pretext':
                     (VimePretextTraining, None),
                 'vime_pretext+supervised_training':
                     (VimePretextTraining, SupervisedTraining),
                 'tabnet_pretext+supervised_training':
                     (TabnetPretextTraining, SupervisedTraining),
                 'npair_imix_pretext+supervised_training':
                     (NPairiMixPretextTraining, SupervisedTraining),
                 'q_match_pretext+supervised_training':
                     (QMatchPretextTraining, SupervisedTraining),
                 'dino_pretext+supervised_training':
                     (DinoPretextTraining, SupervisedTraining),
                 'simsiam_pretext+supervised_training':
                     (SimSiamPretextTraining, SupervisedTraining),
                 'vicreg_pretext+supervised_training':
                     (VICRegPretextTraining, SupervisedTraining),
                 'simclr_pretext+supervised_training':
                     (SimCLRPretextTraining, SupervisedTraining),
                 }


def get_algorithms(algo_name):
  """Returns training and eval model pair."""
  if algo_name not in NAME_TO_ALGOS:
    raise ValueError('%s not supported yet.' % algo_name)
  return NAME_TO_ALGOS[algo_name]
