# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""List all available algorithms."""

from tcc.algos.alignment import Alignment
from tcc.algos.alignment_sal_tcn import AlignmentSaLTCN
from tcc.algos.classification import Classification
from tcc.algos.sal import SaL
from tcc.algos.tcn import TCN

ALGO_NAME_TO_ALGO_CLASS = {
    'alignment': Alignment,
    'sal': SaL,
    'classification': Classification,
    'tcn': TCN,
    'alignment_sal_tcn': AlignmentSaLTCN,
}


def get_algo(algo_name):
  """Returns training algo."""
  if algo_name not in ALGO_NAME_TO_ALGO_CLASS.keys():
    raise ValueError('%s not supported yet.' % algo_name)
  algo = ALGO_NAME_TO_ALGO_CLASS[algo_name]
  return algo()
