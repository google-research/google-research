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

# Lint as: python3
"""Helper functions for pruning functionalities."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v1 as tf

from model_pruning.python import pruning


def apply_pruning(pruning_obj,
                  pruning_hparams,
                  weight_params_fn, weight_init_obj, lstmobj,
                  wm_pc, dtype):
  """Apply pruning to a weight matrix.

  Args:
    pruning_obj: a Pruning object;
    pruning_hparams: a Pruning hparams object;
    weight_params_fn: functional handle to create model parameters;
    weight_init_obj: a weight initialization object;
    lstmobj: a LSTM cell object in the lingvo package;
    wm_pc: weight matrix;
    dtype: data type of the weight matrix.

  Returns:
    pruning_obj as passed in or a compression_obj.
  """
  # Pruning options that corresponds to the pruning operations in model_pruning.
  if pruning_hparams.prune_option in [
      'weight', 'first_order_gradient', 'second_order_gradient']:
    mask_pc = weight_params_fn(wm_pc.shape, weight_init_obj.Constant(1.0),
                               dtype)
    threshold_pc = weight_params_fn([], weight_init_obj.Constant(0.0),
                                    tf.float32)
    lstmobj.CreateVariable('mask', mask_pc, theta_fn=None, trainable=False)
    lstmobj.CreateVariable(
        'threshold', threshold_pc, theta_fn=None, trainable=False)
    if lstmobj.vars.mask not in tf.get_collection(pruning.MASK_COLLECTION):
      tf.add_to_collection(pruning.WEIGHT_COLLECTION, lstmobj.vars.wm)
      tf.add_to_collection(pruning.MASK_COLLECTION, lstmobj.vars.mask)
      tf.add_to_collection(pruning.THRESHOLD_COLLECTION, lstmobj.vars.threshold)
    return pruning_obj
  else:  # TODO(wanxin): add model_compression options.
    return pruning_obj


def get_pruning_update(pruning_obj, pruning_hparams):
  """Return pruning mask update op.

  Args:
    pruning_obj: a Pruning object;
    pruning_hparams: a Pruning hparams object.

  Returns:
    a mask_update_op if the prune_option of the pruning_obj is 'weight',
    'first_order_gradient', or 'second_order_gradient'.

  Raises:
    NotImplementedError if the prune_option of the pruning_obj is not 'weight',
    'first_order_gradient', or 'second_order_gradient'.
  """
  if pruning_hparams.prune_option in [
      'weight', 'first_order_gradient', 'second_order_gradient']:
    return pruning_obj.conditional_mask_update_op()
  else:
    raise NotImplementedError()
