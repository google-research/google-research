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

"""Model pruning implementation in tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from model_pruning.python.layers.rnn_cells import MaskedBasicLSTMCell
from model_pruning.python.layers.rnn_cells import MaskedLSTMCell
from model_pruning.python.pruning import apply_mask
from model_pruning.python.pruning import get_masked_weights
from model_pruning.python.pruning import get_masks
from model_pruning.python.pruning import get_pruning_hparams
from model_pruning.python.pruning import get_thresholds
from model_pruning.python.pruning import get_weight_sparsity
from model_pruning.python.pruning import get_weights
from model_pruning.python.pruning import Pruning
from model_pruning.python.pruning_interface import apply_pruning
from model_pruning.python.pruning_interface import get_pruning_update
from model_pruning.python.strip_pruning_vars_lib import graph_def_from_checkpoint
from model_pruning.python.strip_pruning_vars_lib import strip_pruning_vars_fn

# pylint: enable=unused-import
