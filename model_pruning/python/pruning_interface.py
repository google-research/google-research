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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Helper functions for applying Pruning / Compression to tensors.

Examples:

  # Get compression object.
  compression_obj = get_matrix_compression_object(hparams, global_step)

  # Creating a compressed tensor.
  compressed_tensor = apply_matrix_compression(compression_obj, matrix_tensor)

  # Create an update op.
  update_op = get_matrix_compression_update_op(scompression_obj, hparams)

  # Group update_op with train_op, and used the grouped op in place of train_op.
  train_op = tf.group(train_op, update_op)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from lingvo.core import py_utils
import tensorflow.compat.v1 as tf

from graph_compression.compression_lib import compression_wrapper_py2 as compression_wrapper
from model_pruning.python import pruning


UPDATE_OP_COLLECTION = 'update_op'


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
  if global_step is None:
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
    compressed_matrix = matrix_compression_obj.apply_compression(weight, scope)
    hparams = matrix_compression_obj.get_spec()
    if hparams.use_collection:
      tf.add_to_collection(UPDATE_OP_COLLECTION,
                           matrix_compression_obj.all_update_op())
    return compressed_matrix


def apply_customized_lstm_matrix_compression(matrix_compression_obj,
                                             weight_params_fn,
                                             weight_init_obj,
                                             layer_obj,
                                             weight_name,
                                             weight_shape,
                                             weight_dtype,
                                             scope_name='pruning_interface'):
  """Apply pruning or compression to a LSTM layer.

  This provides a unified interface to perform pruning or compression for a
  lingvo LSTM layer.

  Args:
    matrix_compression_obj: A Pruning or
      compression_lib.lingvo_compression_op.ApplyCompression object;
    weight_params_fn: functional handle to create model parameters;
    weight_init_obj: a weight initialization object;
    layer_obj: a LSTM cell object in the lingvo package, weight matrix of this
      layer is pruned or compressed;
    weight_name: name of the tensor that is compressed, str;
    weight_shape: shape of the weight matrix;
    weight_dtype: data type of the weight matrix;
    scope_name: TensorFlow scope for creating relavant variables.

  Returns:
    None.
  """
  if isinstance(matrix_compression_obj, pruning.Pruning):
    prune_option = matrix_compression_obj.matrix_compression_spec.prune_option

    with tf.variable_scope(scope_name):
      # Create mask and threshold variable and add them to pruning collection.
      mask_pc = weight_params_fn(weight_shape, weight_init_obj.Constant(1.0),
                                 weight_dtype)
      threshold_pc = weight_params_fn([], weight_init_obj.Constant(0.0),
                                      tf.float32)
      layer_obj.CreateVariable('mask', mask_pc, theta_fn=None, trainable=False)
      layer_obj.CreateVariable(
          'threshold', threshold_pc, theta_fn=None, trainable=False)
      if layer_obj.vars.mask not in tf.get_collection(pruning.MASK_COLLECTION):
        tf.add_to_collection(pruning.WEIGHT_COLLECTION,
                             getattr(layer_obj.vars, weight_name))
        tf.add_to_collection(pruning.MASK_COLLECTION, layer_obj.vars.mask)
        tf.add_to_collection(pruning.THRESHOLD_COLLECTION,
                             layer_obj.vars.threshold)
      if prune_option in ['first_order_gradient', 'second_order_gradient']:
        grad_pc = weight_params_fn(weight_shape, weight_init_obj.Constant(0.0),
                                   weight_dtype)
        layer_obj.CreateVariable(
            'gradient', grad_pc, theta_fn=None, trainable=False)
        layer_obj.CreateVariable(
            'old_weight', grad_pc, theta_fn=None, trainable=False)
        layer_obj.CreateVariable(
            'old_old_weight', grad_pc, theta_fn=None, trainable=False)
        tf.add_to_collection(pruning.WEIGHT_GRADIENT_COLLECTION,
                             layer_obj.vars.gradient)
        tf.add_to_collection(pruning.OLD_WEIGHT_COLLECTION,
                             layer_obj.vars.old_weight)
        tf.add_to_collection(pruning.OLD_OLD_WEIGHT_COLLECTION,
                             layer_obj.vars.old_old_weight)
  else:
    _ = matrix_compression_obj.customized_apply_compression(
        getattr(layer_obj.vars, weight_name), layer_obj, weight_params_fn,
        weight_init_obj, scope_name)
    hparams = matrix_compression_obj.get_spec()
    if hparams.use_collection:
      tf.add_to_collection(UPDATE_OP_COLLECTION,
                           matrix_compression_obj.all_update_op())


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


def get_matrix_compression_update_op(matrix_compression_obj):
  """Return pruning/compression update op.

  For pruning, this returns a contional_mask_update_op; for compression, this
  returns an ApplyCompression.all_update_op.

  Args:
    matrix_compression_obj: a Pruning or a compression_lib.ApplyCompression
      object;

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
  hparams = matrix_compression_obj.get_spec()
  if hparams.prune_option in [
      'weight', 'first_order_gradient', 'second_order_gradient']:
    return matrix_compression_obj.conditional_mask_update_op()
  elif hparams.update_option == 0 or hparams.update_option == 2:
    # 'update_option' == 0 means matrix compression, for which we can
    # return an update op here. 'update_option' == 1 means dictionary learning,
    # for which we cannot return an update op here, and need to explicitly call
    # run_update_step(), see graph_compression/compression_lib/compression_op.py
    # for more details.
    if hparams.use_collection:
      # If use_collection is True, then update_ops are retrieved from
      # UPDATE_OP_COLLECTION, to ensure the same behavior as pruning.
      update_ops = tf.get_collection(UPDATE_OP_COLLECTION)
      return tf.group(*update_ops)
    else:
      return matrix_compression_obj.all_update_op()
  else:
    raise NotImplementedError()


def run_update_step(matrix_compression_obj, session, step_number=None):
  """This the update step that needs to be called periodically."""

  hparams = matrix_compression_obj.get_spec()
  if (hparams.prune_option in [
      'weight', 'first_order_gradient', 'second_order_gradient'] or
      hparams.update_option == 0):
    update_op = get_matrix_compression_update_op(matrix_compression_obj)
    session.run(update_op)
  else:
    matrix_compression_obj.run_update_step(session, step_number)


def add_compression_summaries(matrix_compression_obj):
  """Add compression summaries.

  Args:
    matrix_compression_obj: a Pruning or a compression_lib.ApplyCompression
      object.

  Returns:
    None
  """
  if isinstance(matrix_compression_obj, pruning.Pruning):
    matrix_compression_obj.add_pruning_summaries()


class PruningOp(object):
  """A pruning op object.

  This class encapsulates the methods that are needed for pruning (and
  compression) so that both pruning and compression can be called in lingvo
  using the same API.
  """

  _pruning_hparams_dict = {}
  _global_step = None
  _pruning_obj = None
  _pruning_hparams = None

  @classmethod
  def Setup(cls, pruning_hparams_dict, global_step):  # pylint:disable=invalid-name
    """Set up the pruning op with pruning hyperparameters and global step.

    Args:
      pruning_hparams_dict: a dict containing pruning hyperparameters;
      global_step: global step in TensorFlow.
    """
    if cls._pruning_obj is not None:
      pass
    assert pruning_hparams_dict is not None
    assert isinstance(pruning_hparams_dict, dict)
    cls._pruning_hparams_dict = pruning_hparams_dict
    cls._global_step = global_step
    cls._pruning_hparams = pruning.get_pruning_hparams().override_from_dict(
        pruning_hparams_dict)
    cls._pruning_obj = get_matrix_compression_object(
        cls._pruning_hparams, global_step=global_step)
    add_compression_summaries(cls._pruning_obj)

  @classmethod
  def ApplyPruning(cls, pruning_hparams_dict, lstmobj, weight_name, wm_pc,  # pylint:disable=invalid-name
                   dtype, scope):
    if not cls._pruning_obj:
      cls.Setup(pruning_hparams_dict, global_step=py_utils.GetGlobalStep())
    return apply_customized_lstm_matrix_compression(cls._pruning_obj,
                                                    py_utils.WeightParams,
                                                    py_utils.WeightInit,
                                                    lstmobj, weight_name,
                                                    wm_pc.shape, dtype, scope)

  @classmethod
  def GetMixResult(cls, theta, concat, lstmobj):  # pylint:disable=invalid-name
    """Compute the mix result.

    Args:
      theta: a theta object in the LSTM cells;
      concat: Tensor, concat of previous output and current state vector;
      lstmobj: a LSTM cell object.

    Returns:
      result Tensor.

    Raises:
      NotImplementedError if prune_option is not 'weight',
      'first_order_gradient', or 'second_order_gradient'.
    """
    if cls._pruning_hparams.prune_option in [
        'weight', 'first_order_gradient', 'second_order_gradient'
    ]:
      return tf.matmul(
          concat,
          lstmobj.QWeight(tf.multiply(theta.wm, theta.mask, 'masked_weight')))
    elif cls._pruning_obj:
      return cls._pruning_obj.get_mix_operator(theta, concat)
    else:
      raise NotImplementedError()

  @classmethod
  def GetPruningUpdate(cls):  # pylint:disable=invalid-name
    # for pruning, it returns pruning_obj.conditional_mask_update_op()
    return get_matrix_compression_update_op(cls._pruning_obj)

  @classmethod
  def ApplyTensorflowUpdate(cls):  # pylint:disable=invalid-name
    if not cls._pruning_obj:
      return False
    hparams = cls._pruning_obj.get_spec()
    return (hparams.prune_option in [
        'weight', 'first_order_gradient', 'second_order_gradient'
    ] or hparams.update_option == 0 or hparams.update_option == 2)

  @classmethod
  def ApplyPythonUpdate(cls):  # pylint:disable=invalid-name
    if not cls._pruning_obj:
      return False
    hparams = cls._pruning_obj.get_spec()
    return hparams.update_option == 1 or hparams.update_option == 2

  @classmethod
  def ApplyTensorflowAndPythonUpdate(cls):  # pylint:disable=invalid-name
    """Returns True if both Tensorflow and Python updates need to run."""
    if not cls._pruning_obj:
      return False
    hparams = cls._pruning_obj.get_spec()
    return hparams.update_option == 2

  @classmethod
  def RunPythonUpdate(cls, session, global_step):  # pylint:disable=invalid-name
    run_update_step(cls._pruning_obj, session, global_step)
