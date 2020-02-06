# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Helper functions for applying Pruning / Compression to tensors.

Example:

compression_obj = get_matrix_compression_object(hparams, global_step)

# matrix_tensor = tf.Variable(...)
compressed_tensor = apply_matrix_compression(compression_obj,
                                             matrix_tensor)
# compressed_tensor can be used in place of matrix_tensor when constructing
# graphs.

update_op = get_matrix_compression_update_op(matrix_compression_obj,
                                             hparams)
# update_op should be called during training. One way to do this can be to group
# the two ops, and used the grouped op in place of train_op.

train_op = tf.group(train_op, update_op)
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v1 as tf

from graph_compression.compression_lib import compression_wrapper_py2 as compression_wrapper
from model_pruning.python import pruning


def get_matrix_compression_object(hparams,
                                  global_step=None,
                                  sparsity=None):
  """Returns a pruning/compression object.

  Args:
    hparams: Pruning spec as defined in pruing.py;
    global_step: A tensorflow variable that is used for scheduling
    pruning/compression;
    sparsity: A tensorflow scalar variable storing the sparsity.

  Returns:
    A Pruning or compression_lib.compression_op.ApplyCompression object.
  """
  if not global_step:
    global_step = tf.cast(tf.train.get_global_step(), tf.int32)
  if hparams.prune_option in [
      'weight', 'first_order_gradient', 'second_order_gradient']:
    return pruning.Pruning(hparams, global_step, sparsity)
  else:
    return compression_wrapper.get_apply_compression(hparams,
                                                     global_step=global_step)


def apply_matrix_compression(matrix_compression_obj,
                             weight,
                             scope=''):
  """Apply pruning/compression to a weight tensor.

  For pruning, this is equivalent to apply_mask; for compression, this is
  equivalent to apply_compression.

  Args:
    matrix_compression_obj: A Pruning or
      compression_lib.compression_op.ApplyCompression object;
    weight: input weight tensor;
    scope: the current variable scope. Defaults to ''.

  Returns:
    A TF node that represents the masked weight tensor if pruning_indicator is
    True, and the compressed version of weight tensor if pruning_indicator is
    False.
  """
  if isinstance(matrix_compression_obj, pruning.Pruning):
    prune_option = matrix_compression_obj.matrix_compression_spec.prune_option
    return pruning.apply_mask(x=weight, scope=scope, prune_option=prune_option)
  else:
    return matrix_compression_obj.apply_compression(weight, scope)


def apply_pruning(pruning_obj,
                  pruning_hparams,
                  weight_params_fn, weight_init_obj, lstmobj,
                  wm_pc, dtype):
  """Apply pruning to an LSTM cell.

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

  Note: clients are encouraged to use get_matrix_compression_update_op instead,
    which has the same functionality as this function, but supports compression
    too.

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


def get_matrix_compression_update_op(matrix_compression_obj, hparams):
  """Return pruning/compression update op.

  For pruning, this returns a contional_mask_update_op; for compression, this
  returns an ApplyCompression.all_update_op.

  Args:
    matrix_compression_obj: a Pruning or a compression_lib.ApplyCompression
      object;
    hparams: a Pruning tf.HParams object.

  Returns:
    a mask_update_op if the prune_option of the pruning_obj is 'weight',
    'first_order_gradient', or 'second_order_gradient'; or an
    ApplyCompression.all_update_op otherwise.

  Raises:
    NotImplementedError if the prune_option of the pruning_obj is not 'weight',
    'first_order_gradient', or 'second_order_gradient' and update_option is not
    0; in this case, the compression should be applied by calling
    compression_obj.run_update_step(session=session).
  """
  if hparams.prune_option in [
      'weight', 'first_order_gradient', 'second_order_gradient']:
    return matrix_compression_obj.conditional_mask_update_op()
  elif hparams.get_operator_hparam('update_option') == 0:
    # 'update_option' == 0 means matrix compression, for which we can
    # return an update op here. 'update_option' == 1 means dictionary learning,
    # for which we cannot return an update op here, and need to explicitly call
    # run_update_step(), see graph_compression/compression_lib/compression_op.py
    # for more details.
    return matrix_compression_obj.all_update_op()
  else:
    raise NotImplementedError()
