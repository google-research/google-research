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
"""Helper functions to add support for magnitude-based model pruning.

  # Adds variables and ops to the graph to enable
  # elementwise masking of weights
  apply_mask(weights)

  # Returns a list containing the sparsity of each of the weight tensors
  get_weight_sparsity()

  # Returns a list of all the masked weight tensorflow variables
  get_masked_weights()

  # Returns a list of all the mask tensorflow variables
  get_masks()

  # Returns a list of all the thresholds
  get_thresholds()

  # Returns a list of all the weight tensors that have been masked
  get_weights()

  The Pruning class uses a tf.hparams object to set up the
  parameters for a model pruning. Here's a typical usage:

  # Parse pruning hyperparameters
  pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)

  # Create a pruning object using the pruning_hparams
  p = pruning.Pruning(pruning_hparams)

  # Add mask update ops to the graph
  mask_update_op = p.conditional_mask_update_op()

  # Add the summaries
  p.add_pruning_summaries()

  # Run the op
  session.run(mask_update_op)

  # An object of the pruning also accepts externally defined sparsity:
  sparsity = tf.Variable(0.5, name = "ConstantSparsity")
  p = pruning.Pruning(pruning_hparams, sparsity=sparsity)

  # Group pruning.
  # Use apply_mask_with_group() instead of apply_mask(), and
  # set group_pruning to be True in pruning_hparams.

  var1 = tf.Variable(tf.random.normal(shape=(16, 16)), name='var1')
  var2 = tf.Variable(tf.random.normal(shape=(32, 16)), name='var2')
  var3 = tf.Variable(tf.random.normal(shape=(48, 8)), name='var3')

  _ = apply_mask_with_group(var1, group_name='group1')
  _ = apply_mask_with_group(var2, group_name='group1')
  _ = apply_mask_with_group(var3, group_name='group2')
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow.compat.v1 as tf

from graph_compression.compression_lib import compression_op_utils as comp_op_utils
from model_pruning.python import hparam
from model_pruning.python import pruning_utils
from tensorflow.python.ops import variables  # pylint: disable=g-direct-tensorflow-import

MASK_COLLECTION = 'masks'
THRESHOLD_COLLECTION = 'thresholds'
MASKED_WEIGHT_COLLECTION = 'masked_weights'
WEIGHT_COLLECTION = 'kernel'
# The 'weights' part of the name is needed for the quantization library
# to recognize that the kernel should be quantized.
MASKED_WEIGHT_NAME = 'weights/masked_weight'
WEIGHT_GRADIENT_COLLECTION = 'gradient_weights'
OLD_WEIGHT_COLLECTION = 'old_weights'
OLD_OLD_WEIGHT_COLLECTION = 'old_old_weights'
# Group name for 'ungrouped' weights.
UNGROUPED_GROUP_NAME = 'ungrouped'


def attach_group_suffix(collection_name, group_name=None, separator='_'):
  if group_name is None:
    return collection_name
  return collection_name + separator + group_name


def add_to_pruning_collections(mask,
                               threshold,
                               weight,
                               masked_weight,
                               gradient=None,
                               old_weight=None,
                               old_old_weight=None,
                               group_name=None):
  """Add mask, threshold, weight, masked_weight tensors to collections.

  Args:
    mask: mask tensor, a tf.Tensor object.
    threshold: threshold tensor, a tf.Tensor object.
    weight: weight tensor, a tf.Tensor object.
    masked_weight: masked weight tensor, a tf.Tensor object.
    gradient: gradient tensor, a tf.Tensor object or None (default).
    old_weight: weight tensor at the last pruning step, a tf.Tensor object or
      None (default).
    old_old_weight: weight tensor 2 steps before, a tf.Tensor object or None
      (default).
    group_name: group name, a Python str object or None (default).
  """
  mask_collection = attach_group_suffix(MASK_COLLECTION, group_name)
  threshold_collection = attach_group_suffix(THRESHOLD_COLLECTION, group_name)
  weight_collection = attach_group_suffix(WEIGHT_COLLECTION, group_name)
  masked_weight_collection = attach_group_suffix(MASKED_WEIGHT_COLLECTION,
                                                 group_name)
  gradient_collection = attach_group_suffix(WEIGHT_GRADIENT_COLLECTION,
                                            group_name)
  old_weight_collection = attach_group_suffix(OLD_WEIGHT_COLLECTION, group_name)
  old_old_weight_collection = attach_group_suffix(OLD_OLD_WEIGHT_COLLECTION,
                                                  group_name)

  for var, collection in zip([
      mask, threshold, weight, masked_weight, gradient, old_weight,
      old_old_weight
  ], [
      mask_collection, threshold_collection, weight_collection,
      masked_weight_collection, gradient_collection, old_weight_collection,
      old_old_weight_collection
  ]):
    if var is not None:
      tf.add_to_collection(collection, var)


def apply_mask(x, scope='', prune_option='weight'):
  """Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope. Defaults to "".
    prune_option: pruning option. Defaults to 'weight'. option =
      'first_order_gradient' means using |weight| * |first order gradient| for
      pruning. option = 'second_order_gradient' means using |weight| * |second
      order gradient| for pruning.

  Returns:
    Tensor representing masked_weights
  """
  return apply_mask_with_group(x, scope, prune_option, group_name=None)


def apply_mask_with_group(x,
                          scope='',
                          prune_option='weight',
                          group_name=UNGROUPED_GROUP_NAME):
  """Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope. Defaults to "".
    prune_option: pruning option. Defaults to 'weight'. option =
      'first_order_gradient' means using |weight| * |first order gradient| for
      pruning. option = 'second_order_gradient' means using |weight| * |second
      order gradient| for pruning.
    group_name: group name for this weight tensor, str. Defaults to
      UNGROUPED_GROUP_NAME.

  Returns:
    Tensor representing masked_weights
  """

  mask = pruning_utils.weight_mask_variable(x, scope)
  threshold = pruning_utils.weight_threshold_variable(x, scope)
  # Add masked_weights in the weights namescope so as to make it easier
  # for the quantization library to add quant ops.
  masked_weights = tf.multiply(mask, x, MASKED_WEIGHT_NAME)

  gradient = None
  old_weight = None
  old_old_weight = None
  if prune_option in ('first_order_gradient', 'second_order_gradient'):
    # absolute value of gradients for gradient based pruning
    gradient = pruning_utils.weight_gradient_variable(x, scope)
    old_weight = pruning_utils.old_weight_variable(x, scope)
    old_old_weight = pruning_utils.old_old_weight_variable(x, scope)

  # Make sure the mask for a given variable are not added multiple times to the
  # collection. This is particularly important when applying mask to RNN's
  # weight variables
  if mask not in tf.get_collection_ref(MASK_COLLECTION):
    add_to_pruning_collections(mask, threshold, x, masked_weights, gradient,
                               old_weight, old_old_weight)
    if group_name is not None:
      add_to_pruning_collections(
          mask,
          threshold,
          x,
          masked_weights,
          gradient,
          old_weight,
          old_old_weight,
          group_name=group_name)
  return masked_weights


def apply_mask_and_return(x, scope='', prune_option='weight'):
  """Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope. Defaults to "".
    prune_option: pruning option. Defaults to 'weight'. option =
      'first_order_gradient' means using |weight| * |first order gradient| for
      pruning. option = 'second_order_gradient' means using |weight| * |second
      order gradient| for pruning.

  Returns:
    masked_weights: a TensorFlow tensor representing masked weights.
    mask: a TensorFlow tensor representing the pruning mask.
  """

  mask = pruning_utils.weight_mask_variable(x, scope)
  threshold = pruning_utils.weight_threshold_variable(x, scope)
  # Add masked_weights in the weights namescope so as to make it easier
  # for the quantization library to add quant ops.
  masked_weights = tf.multiply(mask, x, MASKED_WEIGHT_NAME)

  if prune_option in ('first_order_gradient', 'second_order_gradient'):
    # absolute value of gradients for gradient based pruning
    gradient = pruning_utils.weight_gradient_variable(x, scope)
    old_weight = pruning_utils.old_weight_variable(x, scope)
    old_old_weight = pruning_utils.old_old_weight_variable(x, scope)

  # Make sure the mask for a given variable are not added multiple times to the
  # collection. This is particularly important when applying mask to RNN's
  # weight variables
  if mask not in tf.get_collection_ref(MASK_COLLECTION):
    tf.add_to_collection(THRESHOLD_COLLECTION, threshold)
    tf.add_to_collection(MASK_COLLECTION, mask)
    tf.add_to_collection(MASKED_WEIGHT_COLLECTION, masked_weights)
    tf.add_to_collection(WEIGHT_COLLECTION, x)
    if prune_option in ('first_order_gradient', 'second_order_gradient'):
      tf.add_to_collection(WEIGHT_GRADIENT_COLLECTION, gradient)
      tf.add_to_collection(OLD_WEIGHT_COLLECTION, old_weight)
      tf.add_to_collection(OLD_OLD_WEIGHT_COLLECTION, old_old_weight)
  return [masked_weights, mask]


def get_masked_weights(group_name=None):
  if group_name:
    return tf.get_collection(
        attach_group_suffix(MASKED_WEIGHT_COLLECTION, group_name))
  return tf.get_collection(MASKED_WEIGHT_COLLECTION)


def get_masks(group_name=None):
  if group_name:
    return tf.get_collection(attach_group_suffix(MASK_COLLECTION, group_name))
  return tf.get_collection(MASK_COLLECTION)


def get_thresholds(group_name=None):
  if group_name:
    return tf.get_collection(
        attach_group_suffix(THRESHOLD_COLLECTION, group_name))
  return tf.get_collection(THRESHOLD_COLLECTION)


def get_weights(group_name=None):
  if group_name:
    return tf.get_collection(attach_group_suffix(WEIGHT_COLLECTION, group_name))
  return tf.get_collection(WEIGHT_COLLECTION)


def get_gradients(group_name=None):
  if group_name:
    return tf.get_collection(
        attach_group_suffix(WEIGHT_GRADIENT_COLLECTION, group_name))
  return tf.get_collection(WEIGHT_GRADIENT_COLLECTION)


def get_old_weights(group_name=None):
  if group_name:
    return tf.get_collection(
        attach_group_suffix(OLD_WEIGHT_COLLECTION, group_name))
  return tf.get_collection(OLD_WEIGHT_COLLECTION)


def get_old_old_weights(group_name=None):
  if group_name:
    return tf.get_collection(
        attach_group_suffix(OLD_OLD_WEIGHT_COLLECTION, group_name))
  return tf.get_collection(OLD_OLD_WEIGHT_COLLECTION)


def get_weight_sparsity():
  """Get sparsity of the weights.

  Args: None

  Returns:
    A list containing the sparsity of each of the weight tensors
  """
  masks = get_masks()
  return [tf.nn.zero_fraction(mask) for mask in masks]


def get_pruning_hparams():
  """Get a tf.HParams object with the default values for the hyperparameters.

    name: string
      name of the pruning specification. Used for adding summaries and ops under
      a common tensorflow name_scope
    begin_pruning_step: integer
      the global step at which to begin pruning
    end_pruning_step: integer
      the global step at which to terminate pruning. Defaults to -1 implying
      that pruning continues till the training stops
    weight_sparsity_map: list of strings
       comma separed list of {weight_variable_name:target sparsity} or
       {regex:target sparsity} pairs.
       For layers/weights not in this list, sparsity as specified by the
       target_sparsity hyperparameter is used.
       Eg. [conv1:0.9,conv2/kernel:0.8]
    block_dims_map: list of strings
       comma separated list of {weight variable name:block_height x block_width}
       or {regex:block_height x block_width} pairs. For layers/weights not in
       this list, block dims are specified by the block_height, block_width
       hyperparameters are used Eg. [dense1:4x4,dense2:1x16,dense3:1x1]
    threshold_decay: float
      the decay factor to use for exponential decay of the thresholds
    pruning_frequency: integer
      How often should the masks be updated? (in # of global_steps)
    nbins: integer
      number of bins to use for histogram computation
    block_height: integer
      number of rows in a block (defaults to 1), can be -1 in which
      case it is set to the size of the corresponding weight tensor.
    block_width: integer
      number of cols in a block (defaults to 1), can be -1 in which
      case it is set to the size of the corresponding weight tensor.
    block_pooling_function: string
      Whether to perform average (AVG) or max (MAX) pooling in the block
      (default: AVG)
    initial_sparsity: float
      initial sparsity value
    target_sparsity: float
      target sparsity value
    sparsity_function_begin_step: integer
      the global step at this which the gradual sparsity function begins to
      take effect
    sparsity_function_end_step: integer
      the global step used as the end point for the gradual sparsity function
    sparsity_function_exponent: float
      exponent = 1 is linearly varying sparsity between initial and final.
      exponent > 1 varies more slowly towards the end than the beginning
    use_tpu: False
      Indicates whether to use TPU
    gradient_decay_rate: float
      when prune_option is gradient based pruning, decay factor for gradient
      decay
    prune_option: string
      option = 'weight' means using |weight| for pruning.
      option = 'first_order_gradient' means using |weight| * |first order
      gradient| for pruning.
      option = 'second_order_gradient' means using |weight| * |second order
      gradient| for pruning.
        second order gradient is approximated by |weight + old_old_weight -
        2*old_weight|.
      option = 'compression' means using compression.
    alpha_decrement_value: only effective when prune_option is 'compression',
      see graph_compression/compression_lib/compression_op.py. The following
      arguments are all only effective when prune_option == 'compression', see
      graph_compression/compression_lib/compression_op.py for details.
    begin_compression_step: only effective when prune_option is 'compression',
                           see graph_compression/compression_op.py.
    end_compresson_step: only effective when prune_option is 'compression',
                           see graph_compression/compression_op.py.
    compression_frequency: only effective when prune_option is 'compression',
                           see graph_compression/compression_op.py.
    compression_option: only effective when prune_option is 'compression',
                        see graph_compression/compression_op.py.
    rank: only effective when prune_option is 'compression',
          see graph_compression/compression_op.py.
    update_option: only effective when prune_option is 'compression',
                   see graph_compression/compression_op.py.
    run_update_interval_check: only effective when prune_option is 'compression'
                               see graph_compression/compression_op.py.
    pruning_fraction: only effective when prune_option is 'compression',
                      see graph_compression/compression_op.py.
    use_collection: only effective when prune_option is 'compression',
                    update_ops are retrieved from UPDATE_OP_COLLECTION if True,
                    otherwise update_ops are obtained from
                    matrix_compression_obj.all_update_op() directly. Default is
                    True.
    compress_input: boolean flag indicating whether to compress input.
                    only used when prune_option is 'compression' and compression
                    option is 9.
    compress_output: boolean flag indicating whether to compress output.
                     only used when prune_option is 'compression' and
                     compression option is 9.
    input_compression_factor: ratio of the size of original input to compressed
                              input. currently only support positive integers.
    output_compression_factor: ratio of the size of original output to
                               compressed output. currently only support
                               positive integers.
    input_block_size: size of input blocks for input compression.
    output_block_size: size of output blocks for input compression.
    block_method: string.
                  option = "mask" implements block compression using a mask.
                  option = 'loop' stores the blocks as a rank 3 tensor and loops
                           through them.
    block_compression_factor: ratio of size of original weight matrix to
                              (nonzero entries in) compressed matrix for block
                              compression. Equivalently, number of blocks on
                              diagonal.
    compression_factor: Compression factor to use for MixedBlockCompressionOp.
    num_bases: Number of basis matrices to use for MixedBlockCompressionOp.
    group_pruning: perform group pruning if True. Default is False.
    group_sparsity_map: list of strings
      comma separated list of {group_name:target sparsity} or
      {regex:target sparsity} pairs.
      For groups not in this list, sparsity as specified by the
      target_sparsity hyperparameter is used.
      Eg. [conv1:0.9, conv2/kernel:0.8, .*dense:0.5]
    group_block_dims_map: comma separated list of
      {group_name:block_height x block_width} or
      {regex:block_height x block_width} pairs. For any groups not in this list,
      the block_height, block_width hyperparameters are used.
      Eg. [dense1:4x4,dense2:1x16,dense3:1x1].

    We use the following sparsity function:

    num_steps = (sparsity_function_end_step -
                 sparsity_function_begin_step)/pruning_frequency
    sparsity(step) = (initial_sparsity - target_sparsity)*
                     [1-step/(num_steps -1)]**exponent + target_sparsity

    intra_block_sparsity: default False, otherwise indicates pruning within a
      block. The block size is specified using the `block_width` parameters.
      `block_height` must be 1. As an example, this paper
      https://arxiv.org/abs/2104.08378 proposes 2-in-4 sparsity.

  Args: None

  Returns:
    tf.HParams object initialized to default values

  """
  return hparam.HParams(
      name='model_pruning',
      begin_pruning_step=0,
      end_pruning_step=-1,
      weight_sparsity_map=[''],
      block_dims_map=[''],
      group_sparsity_map=[''],
      group_block_dims_map=[''],
      threshold_decay=0.0,
      pruning_frequency=10,
      nbins=256,
      block_height=1,
      block_width=1,
      block_pooling_function='AVG',
      initial_sparsity=0.0,
      target_sparsity=0.5,
      sparsity_function_begin_step=0,
      sparsity_function_end_step=100,
      sparsity_function_exponent=3.0,
      use_tpu=False,
      gradient_decay_rate=0.99,
      prune_option='weight',
      alpha_decrement_value=0.01,
      begin_compression_step=0,
      end_compression_step=-1,
      compression_frequency=10,
      compression_option=comp_op_utils.CompressionOptions.NO_MATRIX_COMPRESSION,
      rank=7,
      block_size=1,
      update_option=comp_op_utils.UpdateOptions.NO_UPDATE,
      run_update_interval_check=1,
      pruning_fraction=0.4,
      use_collection=False,
      do_transpose=False,
      compress_input=True,
      input_compression_factor=1,
      input_block_size=1,
      compress_output=False,
      output_compression_factor=1,
      output_block_size=1,
      block_method='loop',
      block_compression_factor=1,
      compression_factor=1,
      num_bases=1,
      add_summary=True,
      group_pruning=False,
      intra_block_sparsity=False)


class Pruning(object):

  def __init__(self, spec=None, global_step=None, sparsity=None):
    """Set up the specification for model pruning.

    If a spec is provided, the sparsity is set up based on the sparsity_function
    in the spec. The effect of sparsity_function is overridden if the sparsity
    variable is passed to the constructor. This enables setting up arbitrary
    sparsity profiles externally and passing it to this pruning functions.

    Args:
      spec: Pruning spec, a tf.HParams object
      global_step: A tensorflow variable that is used while setting up the
        sparsity function
      sparsity: A tensorflow scalar variable storing the sparsity
    """

    # Pruning specification
    self._spec = spec if spec else get_pruning_hparams()
    if spec:
      self._spec = self._normalize_spec(self._spec)
    tf.logging.vlog(0, 'Pruning spec...')
    self.print_hparams()

    self.matrix_compression_spec = self._spec

    # Sanity check for pruning hparams
    self._validate_spec()

    # A tensorflow variable that tracks the sparsity function.
    # If not provided as input, the graph must already contain the global_step
    # variable before calling this constructor.
    self._global_step = self._setup_global_step(global_step)

    # Stores the tensorflow sparsity variable.
    # Built using self._setup_sparsity() or provided externally
    self._sparsity = (
        sparsity if sparsity is not None else self._setup_sparsity())

    # List of tensorflow assignments ops for new masks and thresholds
    self._assign_ops = []

    self._assign_gradient_ops = []

    self._assign_old_weight_ops = []

    self._assign_old_old_weight_ops = []

    # Tensorflow variable keeping track of the last global step when the masks
    # and gradients were updated
    self._last_update_step = self._setup_last_update_step()
    self._last_gradient_update_step = self._setup_last_gradient_update_step()

    # Block dimensions
    self._block_dims = [self._spec.block_height, self._spec.block_width]

    # Block pooling function
    self._block_pooling_function = self._spec.block_pooling_function

    # Mapping of layer/weight names and block dims
    self._block_dims_map = self._get_block_dims_map()

    # Mapping of weight names and target sparsity
    self._weight_sparsity_map = self._get_weight_sparsity_map()

    # Group pruning.
    self._group_pruning = self._spec.group_pruning
    self._group_sparsity_map = self._get_group_sparsity_map()
    self._group_sparsity_map_raw = self._get_group_sparsity_map_raw()
    self._group_block_dims_map = self._get_group_block_dims_map()

  def _normalize_spec(self, spec):
    """Normalize the HParams spec.

    Compare `spec` with default spec from `get_pruning_hparams()` and add
    missing fields. This ensures that old specs still works when we add new
    key:value pairs.

    Args:
      spec: Pruning spec, a tf.contrib.training.HParams object.

    Returns:
      Normalized pruning spec, a tf.contrib.training.HParam object.
    """
    spec_dict = spec.values()
    default_spec_dict = get_pruning_hparams().values()
    for k in default_spec_dict:
      if k not in spec_dict:
        spec_dict[k] = default_spec_dict[k]
        spec.add_hparam(name=k, value=default_spec_dict[k])
    return spec

  def _validate_spec(self):
    spec = self._spec
    if spec.begin_pruning_step < 0:
      raise ValueError('Illegal value for begin_pruning_step')

    if spec.begin_pruning_step >= spec.end_pruning_step:
      if spec.end_pruning_step != -1:
        raise ValueError(
            'Pruning must begin before it can end. begin_step=%d, end_step=%d.'
            'Set end_pruning_step to -1 if pruning is required till training'
            'stops' % (spec.begin_pruning_step, spec.end_pruning_step))

    if spec.sparsity_function_begin_step < 0:
      raise ValueError('Illegal value for sparsity_function_begin_step')

    if spec.sparsity_function_begin_step >= spec.sparsity_function_end_step:
      raise ValueError('Sparsity function requires begin_step < end_step')

    if not 0.0 <= spec.threshold_decay < 1.0:
      raise ValueError('threshold_decay must be in range [0,1)')

    if not 0.0 <= spec.initial_sparsity < 1.0:
      raise ValueError('initial_sparsity must be in range [0,1)')

    if not 0.0 <= spec.target_sparsity < 1.0:
      raise ValueError('target_sparsity must be in range [0,1)')

    if spec.prune_option not in ('weight', 'first_order_gradient',
                                 'second_order_gradient'):
      raise ValueError('prune option specified is not supported')

  def _setup_global_step(self, global_step):
    graph_global_step = global_step
    if graph_global_step is None:
      graph_global_step = tf.train.get_global_step()
      if not graph_global_step:
        raise ValueError(
            'Could not get the global step. Either pass it explicitly, or '
            'ensure that the library is called within a TF graph.')

    return tf.cast(graph_global_step, tf.int32)

  def _setup_sparsity(self):
    begin_step = self._spec.sparsity_function_begin_step
    end_step = self._spec.sparsity_function_end_step
    initial_sparsity = self._spec.initial_sparsity
    target_sparsity = self._spec.target_sparsity
    exponent = self._spec.sparsity_function_exponent

    with tf.name_scope(self._spec.name):
      p = tf.minimum(
          1.0,
          tf.maximum(
              0.0,
              tf.div(
                  tf.cast(self._global_step - begin_step, tf.float32),
                  end_step - begin_step)))
      sparsity = tf.add(
          tf.multiply(initial_sparsity - target_sparsity,
                      tf.pow(1 - p, exponent)),
          target_sparsity,
          name='sparsity')

    return sparsity

  def _setup_last_update_step(self):
    with tf.variable_scope(self._spec.name, use_resource=True) as scope:
      try:
        last_update_step = tf.get_variable(
            'last_mask_update_step', [],
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.int32)
      except ValueError:
        scope.reuse_variables()
        last_update_step = tf.get_variable(
            'last_mask_update_step', dtype=tf.int32)
    return last_update_step

  def _get_block_dims_map(self):
    """Returns the map of layer name: block dims."""
    block_dims_map = {}
    val_list = self._spec.block_dims_map
    filtered_val_list = [l for l in val_list if l]
    for val in filtered_val_list:
      weight_name, block_dims_str = val.split(':')
      block_dims_str = block_dims_str.split('x')
      if len(block_dims_str) != 2:
        raise ValueError('Expected 2 values for block dim for %s, got %s' %
                         (weight_name, block_dims_str))
      block_dims = [int(block_dims_str[0]), int(block_dims_str[1])]
      block_dims_map[re.compile(weight_name)] = block_dims

    return block_dims_map

  def _get_block_dims(self, weight_name):
    """Returns the block dims for the given layer/weight name."""
    block_dims_list = [
        block_dims for regexp, block_dims in self._block_dims_map.items()
        if regexp.search(weight_name)
    ]
    if not block_dims_list:
      return self._block_dims

    if len(block_dims_list) > 1:
      raise ValueError('Multiple matches in block_dims_map for weight %s' %
                       weight_name)

    return block_dims_list[0]

  def _setup_last_gradient_update_step(self):
    with tf.variable_scope(self._spec.name, use_resource=True) as scope:
      try:
        last_gradient_update_step = tf.get_variable(
            'last_gradient_update_step', [],
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.int32)
      except ValueError:
        scope.reuse_variables()
        last_gradient_update_step = tf.get_variable(
            'last_gradient_update_step', dtype=tf.int32)
    return last_gradient_update_step

  def _get_weight_sparsity_map(self):
    """Returns the map of weight_name:sparsity parsed from the hparams."""
    weight_sparsity_map = {}
    val_list = self._spec.weight_sparsity_map
    filtered_val_list = [l for l in val_list if l]
    for val in filtered_val_list:
      weight_name, sparsity = val.split(':')
      if float(sparsity) >= 1.0:
        raise ValueError('Weight sparsity can not exceed 1.0')
      weight_sparsity_map[re.compile(weight_name)] = float(sparsity)

    return weight_sparsity_map

  def _get_group_sparsity_map(self):
    """Returns the map of group_name:sparsity parsed from the hparams."""
    group_sparsity_map = {}
    val_list = self._spec.group_sparsity_map
    filtered_val_list = [l for l in val_list if l]
    for val in filtered_val_list:
      group_name, sparsity = val.split(':')
      if float(sparsity) >= 1.0:
        raise ValueError('Weight sparsity can not exceed 1.0')
      group_sparsity_map[re.compile(group_name)] = float(sparsity)
    if UNGROUPED_GROUP_NAME not in group_sparsity_map:
      group_sparsity_map[UNGROUPED_GROUP_NAME] = float(
          self._spec.target_sparsity)

    return group_sparsity_map

  def _get_group_sparsity_map_raw(self):
    """Returns the map of group_name:sparsity parsed from the hparams."""
    group_sparsity_map = {}
    val_list = self._spec.group_sparsity_map
    filtered_val_list = [l for l in val_list if l]
    for val in filtered_val_list:
      group_name, sparsity = val.split(':')
      if float(sparsity) >= 1.0:
        raise ValueError('Weight sparsity can not exceed 1.0')
      group_sparsity_map[group_name] = float(sparsity)
    if UNGROUPED_GROUP_NAME not in group_sparsity_map:
      group_sparsity_map[UNGROUPED_GROUP_NAME] = float(
          self._spec.target_sparsity)

    return group_sparsity_map

  def _get_group_block_dims_map(self):
    block_dims_map = {}
    val_list = self._spec.group_block_dims_map
    filtered_val_list = [l for l in val_list if l]
    for val in filtered_val_list:
      group_name, block_dims_str = val.split(':')
      block_dims_str = block_dims_str.split('x')
      if len(block_dims_str) != 2:
        raise ValueError('Expected 2 values for block dim for %s, got %s' %
                         (group_name, block_dims_str))
      block_dims = [int(block_dims_str[0]), int(block_dims_str[1])]
      block_dims_map[group_name] = block_dims

    return block_dims_map

  def _get_sparsity(self, weight_name):
    """Returns target sparsity for the given layer/weight name."""
    target_sparsity = [
        sparsity for regexp, sparsity in self._weight_sparsity_map.items()
        if regexp.search(weight_name)
    ]
    if not target_sparsity:
      return self._sparsity

    if len(target_sparsity) > 1:
      raise ValueError('Multiple matches in weight_sparsity_map for weight %s' %
                       weight_name)
    # TODO(suyoggupta): This will work when initial_sparsity = 0. Generalize
    # to handle other cases as well.
    return tf.multiply(self._sparsity,
                       tf.div(target_sparsity[0], self._spec.target_sparsity))

  def _get_group_sparsity(self, group_name):
    """Returns target sparsity for the given group name."""
    target_sparsity = [
        sparsity for regexp, sparsity in self._group_sparsity_map.items()
        if regexp.search(group_name)
    ]
    if not target_sparsity:
      return self._sparsity

    if len(target_sparsity) > 1:
      raise ValueError('Multiple matches in group_sparsity_map for group %s' %
                       group_name)
    # TODO(suyoggupta): This will work when initial_sparsity = 0. Generalize
    # to handle other cases as well.
    return tf.multiply(self._sparsity,
                       tf.div(target_sparsity[0], self._spec.target_sparsity))

  def _get_group_sparsity_raw(self, group_name):
    target_sparsity = self._group_sparsity_map_raw.get(group_name)

    if not target_sparsity:
      return self._sparsity

    return tf.multiply(self._sparsity,
                       tf.div(target_sparsity, self._spec.target_sparsity))

  def _get_group_block_dims(self, group_name):
    return self._group_block_dims_map.get(group_name, self._block_dims)

  def _update_mask(self, weights, threshold, gradients):  # pylint: disable=unused-argument
    """Updates the mask for a given weight tensor.

    This functions first computes the cdf of the weight tensor, and estimates
    the threshold value such that 'desired_sparsity' fraction of weights
    have magnitude less than the threshold.

    Args:
      weights: The weight tensor that needs to be masked.
      threshold: The current threshold value. The function will compute a new
        threshold and return the exponential moving average using the current
        value of threshold
      gradients: The gradient tensor that is used for salience calculation.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if sparsity is not defined
    """
    if self._sparsity is None:
      raise ValueError('Sparsity variable undefined')

    sparsity = self._get_sparsity(weights.op.name)
    with tf.name_scope(weights.op.name + '_pruning_ops'):
      tf.logging.info('Applying option %s pruning', self._spec.prune_option)
      if self._spec.prune_option == 'weight':
        abs_weights = tf.abs(weights)
      elif self._spec.prune_option in ('first_order_gradient',
                                       'second_order_gradient'):
        if gradients is None:
          raise ValueError('gradient tensor cannot be None.')
        # gradient variable stores absolute value already
        abs_weights = tf.multiply(tf.abs(weights), gradients)
      else:
        raise ValueError('undefined option')

      k = tf.cast(
          tf.round(tf.cast(tf.size(abs_weights), tf.float32) * (1 - sparsity)),
          tf.int32)

      # Generate a random shuffling of the weights s.t. the tie-breaker on
      # weight magnitude is random uniform.
      shuffling = tf.random_shuffle(
          tf.range(tf.size(abs_weights)))
      shuffling = tf.reshape(shuffling, [-1, 1])

      # Flatten the weights and scatter the values randomly.
      abs_weights = tf.reshape(abs_weights, [-1])
      abs_weights = tf.scatter_nd(
          shuffling,
          abs_weights,
          tf.shape(abs_weights))

      # Sort the entire array
      _, indices = tf.nn.top_k(abs_weights, k=tf.size(abs_weights))

      # `k` is how many non-zero weights we're going to have. Create a new
      # mask where the first `k` elements are set to one and all others are
      # set to zero.
      mask_staging = tf.range(tf.size(abs_weights))
      mask_staging = tf.cast(
          tf.less(mask_staging, k),
          tf.float32)

      # Scatter the mask back into the proper positions for the weight matrix.
      indices = tf.reshape(indices, [-1, 1])
      new_mask = tf.scatter_nd(
          indices,
          mask_staging,
          tf.shape(mask_staging))

      # Un-shuffle the newly created mask.
      new_mask = tf.reshape(
          tf.gather_nd(
              new_mask,
              shuffling),
          tf.shape(weights))
    return tf.constant(0, tf.float32), new_mask

  def _update_mask_sparsity_m_by_n(self, weights, block_size=4):
    """Updates the mask m-by-n block sparsity.

    Args:
      weights: The weight tensor that needs to be masked.
      block_size: Block size to enforce block sparsity pattern.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if sparsity is not defined
    """
    if self._sparsity is None:
      raise ValueError('Sparsity variable undefined')

    sparsity = self._get_sparsity(weights.op.name)

    with tf.name_scope(weights.op.name + '_pruning_ops'):
      tf.logging.info(
          'Applying block sparsity pruning for %s, block size (1, %d), shape %s',
          weights.op.name, block_size, weights.shape)

      # Rearrange weights tensor so that m by n sparsity structure applied in
      # last channel.
      # In case of Conv2D weights:
      #   TF data format is [height, width, channel_in, channel_out],
      #   TFLite data format is [channel_out, height, width, channel_in]
      #   Rearranged Conv2D weights format:
      #   [channel_out x height x width, channel_in]
      # In case of Dense weights:
      #   TF data format is [channel_in, channel_out],
      #   TFLite data format is [width, channel_in]
      #   Rearranged Dense weights format: [width, channel_in]
      # In case of Multi-head Attention weights:
      #   Outermost dimension is the reduction dimension, intra block sparsity
      #   is applied to the reduction dimension. Therefore transpose the
      #   weight's outermost to innermost and reshape it to 2D.
      if weights.shape.rank == 2:
        prepared_weights = tf.transpose(weights)
      elif weights.shape.rank == 3:
        prepared_weights = tf.transpose(
            tf.reshape(weights, [weights.shape[0], -1]))
      elif weights.shape.rank == 4:
        perm_weights = tf.transpose(weights, perm=[3, 0, 1, 2])
        prepared_weights = tf.reshape(
            perm_weights, [tf.reduce_prod(perm_weights.shape[:-1]), -1])
      else:
        raise ValueError(
            f'weight tensor with shape: {weights.shape} is not supported.')

      # Generate m-by-n sparsity mask.
      num_zeros = tf.cast(tf.math.floor(block_size * sparsity), dtype=tf.int32)
      block_size = tf.constant(block_size, tf.int32)
      num_non_zeros = block_size - num_zeros
      abs_weights = tf.abs(prepared_weights)

      # add zero-padding
      pad_after = block_size - tf.shape(abs_weights)[-1] % block_size
      abs_weights_pad = tf.pad(abs_weights, [[0, 0], [0, pad_after]],
                               'CONSTANT')

      num_blocks = tf.size(abs_weights_pad) // block_size
      reshaped_weights_into_blocks = tf.reshape(abs_weights_pad,
                                                [num_blocks, block_size])

      # Sort the weights based on magnitude.
      row_sorted_weights = tf.sort(
          reshaped_weights_into_blocks, axis=1, direction='DESCENDING')
      # Calculate cut-off value (thresholds) of the weights.
      # num_non_zeros is in [1, block_size], therefore pad row_sorted_weights
      # tensor with one more column of -1.0 to cover the case that sparsity is
      # 0.0. In case two weights have exactly the same value, they are either
      # both pruned or both not pruned. Therefore we use tf.greater() rather
      # than tf.greater_equal(). This guarantees that the number of non-zeros
      # after pruning is always less than floor(block_size * (1 - sparsity)).
      thresholds = tf.slice(
          tf.pad(row_sorted_weights, [[0, 0], [0, 1]], constant_values=-1.0),
          begin=[0, num_non_zeros],
          size=[row_sorted_weights.shape[0], 1])
      expanded_thresholds = tf.repeat(
          thresholds, repeats=row_sorted_weights.shape[1], axis=1)
      sparsity_mask_pad = tf.dtypes.cast(
          tf.math.greater(reshaped_weights_into_blocks, expanded_thresholds),
          weights.dtype)
      reshaped_sparsity_mask_pad = tf.reshape(sparsity_mask_pad,
                                              tf.shape(abs_weights_pad))

      # remove padding from mask
      sparsity_mask = tf.slice(reshaped_sparsity_mask_pad, [0, 0],
                               abs_weights.shape)

      # Reshape and permute sparsity mask, so that it match original weights
      # data format.
      tf.debugging.assert_equal(
          tf.size(sparsity_mask),
          tf.reduce_prod(weights.shape),
          message='number of elements mismatch between mask and weights.',
      )

      if sparsity_mask.shape.rank != 2:
        raise ValueError(
            f'rank of mask(rank:{sparsity_mask.shape.rank}) should be 2.')

      if weights.shape.rank == 2:
        prepared_mask = tf.transpose(sparsity_mask)
      elif weights.shape.rank == 3:
        prepared_mask = tf.reshape(tf.transpose(sparsity_mask), weights.shape)
      elif weights.shape.rank == 4:
        weights_shape = tf.shape(weights)
        reshaped_mask = tf.reshape(
            sparsity_mask,
            [
                weights_shape[-1], weights_shape[0], weights_shape[1],
                weights_shape[2]
            ],
        )
        prepared_mask = tf.transpose(reshaped_mask, perm=[1, 2, 3, 0])
      else:
        raise ValueError(
            f'weight tensor with shape: {weights.shape} is not supported.')

      tf.debugging.assert_equal(
          prepared_mask.shape,
          weights.shape,
          message='shape of prepared mask mismatch shape of weights.')

    # Need to return some numbers for threshold.
    return tf.constant(999.0, tf.float32), prepared_mask

  def _maybe_update_block_mask(self, weights, threshold, gradients=None):
    """Performs block-granular masking of the weights.

    If intra_block_sparsity is selected, then we return the relevant pruning
    mask, that nullify m out of n elements in the block.

    Block pruning occurs only if the block_height or block_width is > 1 and
    if the weight tensor, when squeezed, has ndims = 2. Otherwise, elementwise
    pruning occurs.
    Args:
      weights: The weight tensor that needs to be masked.
      threshold: The current threshold value. The function will compute a new
        threshold and return the exponential moving average using the current
        value of threshold
      gradients: The gradient tensor that used for salience calculation.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step. In case of intra_block_sparsity,
        the returned threshold is an arbitrary number.
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if block pooling function is not AVG or MAX
    """

    block_dims = self._get_block_dims(weights.op.name)

    # Intra block sparsity is only enabled when:
    # 1. `intra_block_sparsity` is set to True.
    # 2. weights is 2D, 3D or 4D.
    if (self._spec.intra_block_sparsity and
        weights.get_shape().ndims in [2, 3, 4]):
      return self._update_mask_sparsity_m_by_n(weights, block_dims[1])

    squeezed_weights = tf.squeeze(weights)
    if squeezed_weights.get_shape().ndims != 2 or block_dims == [1, 1]:
      return self._update_mask(weights, threshold, gradients)

    if (self._spec.prune_option in ('first_order_gradient',
                                    'second_order_gradient') and
        gradients is None):
      raise ValueError(
          'Gradient based pruning implementation for block sparsity is not supported.'
      )

    for i in range(2):
      if block_dims[i] == -1:
        block_dims[i] = squeezed_weights.get_shape()[i]

    if self._block_pooling_function not in ['AVG', 'MAX']:
      raise ValueError('Unknown pooling function for block sparsity: %s' %
                       self._block_pooling_function)

    with tf.name_scope(weights.op.name + '_pruning_ops'):
      abs_weights = tf.abs(squeezed_weights)
      if gradients is not None:
        abs_gradients = tf.abs(tf.squeeze(gradients))

      pool_window = block_dims
      pool_fn = pruning_utils.factorized_pool
      squeeze_axis = None
      if not self._spec.use_tpu:
        pool_fn = tf.nn.pool
        abs_weights = tf.reshape(
            abs_weights,
            [1, abs_weights.get_shape()[0],
             abs_weights.get_shape()[1], 1])
        if gradients is not None:
          # Reshape gradients to be a rank 4 tensor of shape [1, .., .., 1].
          abs_gradients = tf.reshape(
              abs_gradients,
              [1, gradients.get_shape()[0], gradients.get_shape()[1], 1])
        squeeze_axis = [0, 3]

      pooled_weights = pool_fn(
          abs_weights,
          window_shape=pool_window,
          pooling_type=self._block_pooling_function,
          strides=pool_window,
          padding='SAME',
          name=weights.op.name + '_pooled')

      if gradients is not None:
        pooled_gradients = pool_fn(
            abs_gradients,
            window_shape=pool_window,
            pooling_type=self._block_pooling_function,
            strides=pool_window,
            padding='SAME',
            name=gradients.op.name + '_pooled')
      else:
        pooled_gradients = None

      if pooled_weights.get_shape().ndims != 2:
        pooled_weights = tf.squeeze(pooled_weights, axis=squeeze_axis)

      if gradients is not None and pooled_gradients.get_shape().ndims != 2:
        pooled_gradients = tf.squeeze(pooled_gradients, axis=squeeze_axis)

      smoothed_threshold, new_mask = self._update_mask(pooled_weights,
                                                       threshold,
                                                       pooled_gradients)

      updated_mask = pruning_utils.expand_tensor(new_mask, block_dims)
      sliced_mask = tf.slice(
          updated_mask, [0, 0],
          [squeezed_weights.get_shape()[0],
           squeezed_weights.get_shape()[1]])

    return smoothed_threshold, tf.reshape(sliced_mask, tf.shape(weights))

  def _get_assign_old_weight_ops(self):
    if self._assign_old_weight_ops:
      raise ValueError(
          'Assign op list not empty. _get_old_weight_assign_ops() called twice?'
      )

    weights = get_weights()
    old_weights = get_old_weights()

    if len(weights) != len(old_weights):
      raise ValueError(
          'Number of weights %s and number of old_weights %s mismatch' %
          (len(weights), len(old_weights)))

    for index, weight in enumerate(weights):
      old_weight = old_weights[index]

      self._assign_old_weight_ops.append(
          pruning_utils.variable_assign(old_weight, weight))

  def _get_assign_old_old_weight_ops(self):
    if self._assign_old_old_weight_ops:
      raise ValueError(
          'Assign op list not empty. _get_old_old_weight_assign_ops() called twice?'
      )

    old_old_weights = get_old_old_weights()
    old_weights = get_old_weights()

    if len(old_old_weights) != len(old_weights):
      raise ValueError(
          'Number of old_old_weights %s and number of old_weights %s mismatch' %
          (len(old_old_weights), len(old_weights)))

    for index, old_old_weight in enumerate(old_old_weights):
      old_weight = old_weights[index]

      self._assign_old_old_weight_ops.append(
          pruning_utils.variable_assign(old_old_weight, old_weight))

  def _get_assign_gradient_ops(self):
    # Make sure the assignment ops have not already been added to the list
    if self._assign_gradient_ops:
      raise ValueError(
          'Assign op list not empty. _get_mask_assign_ops() called twice?')

    weights = get_weights()
    old_weights = get_old_weights()
    old_old_weights = get_old_old_weights()
    gradients = get_gradients()

    if len(weights) != len(old_weights):
      raise ValueError(
          'Number of weights %s and number of old_weights %s mismatch' %
          (len(weights), len(old_weights)))

    if len(weights) != len(gradients):
      raise ValueError(
          'Number of weights %s and number of gradients %s mismatch' %
          (len(weights), len(gradients)))

    for index, _ in enumerate(weights):
      weight = weights[index]
      old_weight = old_weights[index]
      old_old_weight = old_old_weights[index]
      gradient = gradients[index]

      if weight.shape.as_list() != old_weight.shape.as_list():
        raise ValueError('weight tensor has different shape from old_weight')

      if weight.shape.as_list() != gradient.shape.as_list():
        raise ValueError('weight tensor has different shape from gradient')

      if weight.shape.as_list() != old_old_weight.shape.as_list():
        raise ValueError('weight tensor has different shape from old_weight')

      is_partitioned = isinstance(weight, variables.PartitionedVariable)
      if is_partitioned:
        weight = weight.as_tensor()
        old_weight = old_weight.as_tensor()
        old_old_weight = old_old_weight.as_tensor()

      decay = self._spec.gradient_decay_rate
      if self._spec.prune_option == 'first_order_gradient':
        tf.logging.info('Applying first order gradient pruning')
        normalized_weight_delta = tf.nn.l2_normalize(
            tf.abs(weight - old_weight))
      elif self._spec.prune_option == 'second_order_gradient':
        tf.logging.info('Applying second order gradient pruning')
        normalized_weight_delta = tf.nn.l2_normalize(
            tf.abs(weight + old_old_weight - 2 * old_weight))
      else:
        raise ValueError('Unknown prune option. Should not execute this code.')
      new_gradient = decay * gradient + (1 - decay) * normalized_weight_delta

      self._assign_gradient_ops.append(
          pruning_utils.variable_assign(gradient, new_gradient))

  def _get_mask_assign_ops(self):
    # Make sure the assignment ops have not already been added to the list
    if self._assign_ops:
      raise ValueError(
          'Assign op list not empty. _get_mask_assign_ops() called twice?')

    masks = get_masks()
    weights = get_weights()
    thresholds = get_thresholds()
    gradients = get_gradients()

    if len(masks) != len(thresholds):
      raise ValueError(
          'Number of masks %s and number of thresholds %s mismatch' %
          (len(masks), len(thresholds)))

    for index, mask in enumerate(masks):
      threshold = thresholds[index]
      weight = weights[index]
      if self._spec.prune_option in ('first_order_gradient',
                                     'second_order_gradient'):
        gradient = gradients[index]
      else:
        gradient = None

      is_partitioned = isinstance(weight, variables.PartitionedVariable)
      if is_partitioned:
        weight = weight.as_tensor()

      new_threshold, new_mask = self._maybe_update_block_mask(
          weight, threshold, gradient)
      self._assign_ops.append(
          pruning_utils.variable_assign(threshold, new_threshold))

      self._assign_ops.append(
          pruning_utils.partitioned_variable_assign(mask, new_mask)
          if is_partitioned else pruning_utils.variable_assign(mask, new_mask))

  def use_gradient(self):
    return self._spec.prune_option in ('first_order_gradient',
                                       'second_order_gradient')

  def _update_group_masks(self, weights, gradients, group_name, sparsity):
    if self.use_gradient() and not gradients:
      raise ValueError('Gradient is not available.')
    # TODO(wanxin): check if the random uniform tie-breaker is necessary.
    # Create a single flat tensor containing all the tensors in the group.
    with tf.name_scope(group_name + '_pruning_ops'):
      tf.logging.info('Applying option %s pruning', self._spec.prune_option)

      flattened_weight = tf.concat(
          [tf.reshape(weight, [-1]) for weight in weights], axis=0)
      flattened_gradient = None
      if self.use_gradient():
        flattened_gradient = tf.concat(
            [tf.reshape(gradient, [-1]) for gradient in gradients], axis=0)

      if self._spec.prune_option == 'weight':
        importance_score = tf.abs(flattened_weight)
      elif self.use_gradient():
        importance_score = tf.multiply(
            tf.abs(flattened_weight), flattened_gradient)
      else:
        raise ValueError('Undefined option.')

      k = tf.cast(
          tf.round(
              tf.cast(tf.size(importance_score), tf.float32) * (1 - sparsity)),
          tf.int32)
      # Shuffle and find the threshold.
      shuffling = tf.random_shuffle(tf.range(tf.size(importance_score)))
      shuffling = tf.reshape(shuffling, [-1, 1])

      importance_score = tf.scatter_nd(shuffling, importance_score,
                                       tf.shape(importance_score))

      _, indices = tf.math.top_k(importance_score, k=tf.size(importance_score))

      mask_staging = tf.range(tf.size(importance_score))
      mask_staging = tf.cast(tf.less(mask_staging, k), tf.float32)

      indices = tf.reshape(indices, [-1, 1])
      new_mask = tf.scatter_nd(indices, mask_staging, tf.shape(mask_staging))

      new_mask = tf.reshape(
          tf.gather_nd(new_mask, shuffling), tf.shape(importance_score))

      group_masks = tf.split(new_mask, [tf.size(weight) for weight in weights])
      masks = [
          tf.reshape(mask, weight.shape)
          for mask, weight in zip(group_masks, weights)
      ]
    return tf.constant(0, tf.float32), masks

  def _maybe_update_block_group_mask(self, weights, gradients, group_name,
                                     sparsity, block_dims):
    """Update masks for all the weights in the group."""
    if block_dims == [1, 1]:
      return self._update_group_masks(weights, gradients, group_name, sparsity)
    if self.use_gradient() and not gradients:
      raise ValueError('Gradients are not available.')
    if self._block_pooling_function not in ['AVG', 'MAX']:
      raise ValueError('Unknown pooling function for block sparsity: %s' %
                       self._block_pooling_function)

    with tf.name_scope(group_name + '_pruning_ops'):
      # Create pooled_weights and pooled_gradients.
      pooled_weights = []
      pooled_gradients = []
      for index, weight in enumerate(weights):
        pool_window = block_dims
        pool_fn = pruning_utils.factorized_pool
        squeeze_axis = None

        abs_weight = tf.abs(tf.squeeze(weight))
        if self.use_gradient():
          abs_gradient = tf.abs(tf.squeeze(gradients[index]))
        if not self._spec.use_tpu:
          pool_fn = tf.nn.pool
          abs_weight = tf.reshape(
              abs_weight,
              [1, abs_weight.get_shape()[0],
               abs_weight.get_shape()[1], 1])
          if self.use_gradient():
            abs_gradient = tf.reshape(abs_gradient, [
                1,
                abs_gradient.get_shape()[0],
                abs_gradient.get_shape()[1], 1
            ])
          squeeze_axis = [0, 3]
        pooled_weight = pool_fn(
            abs_weight,
            window_shape=pool_window,
            pooling_type=self._block_pooling_function,
            strides=pool_window,
            padding='SAME',
            name=weight.op.name + '_pooled')
        if pooled_weight.get_shape().ndims != 2:
          pooled_weight = tf.squeeze(pooled_weight, axis=squeeze_axis)
        pooled_weights.append(pooled_weight)
        pooled_gradient = None
        if self.use_gradient():
          pooled_gradient = pool_fn(
              abs_gradient,
              window_shape=pool_window,
              pooling_type=self._block_pooling_function,
              strides=pool_window,
              padding='SAME',
              name=abs_gradient.op.name + '_pooled')
          if pooled_gradient.get_shape().ndims != 2:
            pooled_gradient = tf.squeeze(pooled_gradient, axis=squeeze_axis)
          pooled_gradients.append(pooled_gradient)
      smoothed_group_threshold, new_masks = self._update_group_masks(
          pooled_weights, pooled_gradients, group_name, sparsity)

      updated_masks = [
          pruning_utils.expand_tensor(mask, block_dims) for mask in new_masks
      ]
      sliced_masks = []
      for idx, mask in enumerate(updated_masks):
        squeezed_weight = tf.squeeze(weights[idx])
        sliced_masks.append(
            tf.reshape(
                tf.slice(mask, [0, 0], [
                    squeezed_weight.get_shape()[0],
                    squeezed_weight.get_shape()[1]
                ]), weights[idx].shape))
    return smoothed_group_threshold, sliced_masks

  def _get_group_mask_assign_ops(self):
    if not self._group_pruning:
      raise ValueError('Group pruning is not enabled.')

    for group_name in self._group_sparsity_map_raw:
      masks = get_masks(group_name)
      weights = get_weights(group_name)
      thresholds = get_thresholds(group_name)
      gradients = get_gradients(group_name)

      if len(masks) != len(thresholds):
        raise ValueError(
            'Number of masks %s and number of thresholds %s mismatch' %
            (len(masks), len(thresholds)))

      if not masks:
        continue

      group_weights = []
      for index, weight in enumerate(weights):
        is_partitioned = isinstance(weight, variables.PartitionedVariable)
        if is_partitioned:
          weight = weight.as_tensor()
        group_weights.append(weight)

      if group_name != UNGROUPED_GROUP_NAME:
        block_dims = self._get_group_block_dims(group_name)
        sparsity = self._get_group_sparsity_raw(group_name)
        new_threshold, new_masks = self._maybe_update_block_group_mask(
            group_weights, gradients, group_name, sparsity, block_dims)
        for idx, weight in enumerate(weights):
          is_partitioned = isinstance(weight, variables.PartitionedVariable)
          self._assign_ops.append(
              pruning_utils.variable_assign(thresholds[idx], new_threshold))
          self._assign_ops.append(
              pruning_utils.partitioned_variable_assign(
                  masks[idx], new_masks[idx]) if is_partitioned else
              pruning_utils.variable_assign(masks[idx], new_masks[idx]))
      else:
        for idx, weight in enumerate(weights):
          is_partitioned = isinstance(weight, variables.PartitionedVariable)
          if is_partitioned:
            weight = weight.as_tensor()
          threshold = thresholds[idx]
          mask = masks[idx]
          gradient = None
          if self.use_gradient():
            gradient = gradients[idx]
          new_threshold, new_mask = self._maybe_update_block_mask(
              weight, threshold, gradient)
          self._assign_ops.append(
              pruning_utils.variable_assign(threshold, new_threshold))

          self._assign_ops.append(
              pruning_utils.partitioned_variable_assign(mask, new_mask) if
              is_partitioned else pruning_utils.variable_assign(mask, new_mask))

  def old_weight_update_op(self):
    with tf.name_scope(self._spec.name):
      if self._spec.prune_option not in ('first_order_gradient',
                                         'second_order_gradient'):
        return tf.no_op('gradient_update_no_op')
      if not self._assign_old_weight_ops:
        self._get_assign_old_weight_ops()
      with tf.control_dependencies(self._assign_old_weight_ops):
        tf.logging.info('Updating old weights.')
        return tf.no_op('old_weight_update')

  def old_old_weight_update_op(self):
    with tf.name_scope(self._spec.name):
      if self._spec.prune_option != 'second_order_gradient':
        return tf.no_op('gradient_update_no_op')
      if not self._assign_old_old_weight_ops:
        self._get_assign_old_old_weight_ops()
      with tf.control_dependencies(self._assign_old_old_weight_ops):
        tf.logging.info('Updating old old weights.')
        return tf.no_op('old_old_weight_update')

  def gradient_update_op(self):
    with tf.name_scope(self._spec.name):
      if self._spec.prune_option not in ('first_order_gradient',
                                         'second_order_gradient'):
        return tf.no_op('gradient_update_no_op')
      if not self._assign_gradient_ops:
        self._get_assign_gradient_ops()
      with tf.control_dependencies([
          tf.assign(
              self._last_gradient_update_step,
              self._global_step,
              name='last_gradient_update_step_assign')
      ]):
        with tf.control_dependencies(self._assign_gradient_ops):
          tf.logging.info('Updating gradients.')
          return tf.no_op('gradient_update')

  def conditional_gradient_update_op(self):

    def maybe_update_gradients():
      with tf.name_scope(self._spec.name):
        is_step_within_pruning_range = tf.logical_and(
            tf.greater_equal(self._global_step, self._spec.begin_pruning_step),
            # If end_pruning_step is negative, keep pruning forever!
            tf.logical_or(
                tf.less_equal(self._global_step, self._spec.end_pruning_step),
                tf.less(self._spec.end_pruning_step, 0)))
        return is_step_within_pruning_range

    def gradient_update_op():
      return self.gradient_update_op()

    def no_update_op():
      return tf.no_op()

    return tf.cond(maybe_update_gradients(), gradient_update_op, no_update_op)

  def mask_update_op(self):
    with tf.name_scope(self._spec.name):
      if not self._assign_ops:
        if self._group_pruning:
          self._get_group_mask_assign_ops()
        else:
          self._get_mask_assign_ops()

      grad_update_ops = self.gradient_update_op()
      old_weight_update_ops = self.old_weight_update_op()
      old_old_weight_update_ops = self.old_old_weight_update_op()

      with tf.control_dependencies([
          tf.assign(
              self._last_update_step,
              self._global_step,
              name='last_mask_update_step_assign')
      ]):
        with tf.control_dependencies([grad_update_ops]):
          with tf.control_dependencies([old_old_weight_update_ops]):
            with tf.control_dependencies([old_weight_update_ops]):
              with tf.control_dependencies(self._assign_ops):
                tf.logging.info('Updating masks.')
                return tf.no_op('mask_update')

  def conditional_mask_update_op(self):

    def maybe_update_masks():
      with tf.name_scope(self._spec.name):
        is_step_within_pruning_range = tf.logical_and(
            tf.greater_equal(self._global_step, self._spec.begin_pruning_step),
            # If end_pruning_step is negative, keep pruning forever!
            tf.logical_or(
                tf.less_equal(self._global_step, self._spec.end_pruning_step),
                tf.less(self._spec.end_pruning_step, 0)))
        is_pruning_step = tf.less_equal(
            tf.add(self._last_update_step, self._spec.pruning_frequency),
            self._global_step)
        return tf.logical_and(is_step_within_pruning_range, is_pruning_step)

    def mask_update_op():
      return self.mask_update_op()

    def no_update_op():
      return tf.no_op()

    return tf.cond(maybe_update_masks(), mask_update_op, no_update_op)

  def add_pruning_summaries(self):
    """Adds summaries of weight sparsities and thresholds."""
    with tf.name_scope(self._spec.name + '_summaries'):
      tf.summary.scalar('sparsity', self._sparsity)
      tf.summary.scalar('last_mask_update_step', self._last_update_step)
      tf.summary.scalar('last_gradient_update_step',
                        self._last_gradient_update_step)
      masks = get_masks()
      thresholds = get_thresholds()
      gradients = get_gradients()

      for mask, threshold, gradient in zip(masks, thresholds, gradients):
        tf.summary.scalar(mask.op.name + '/sparsity', tf.nn.zero_fraction(mask))
        tf.summary.scalar(threshold.op.name + '/threshold', threshold)
        tf.summary.scalar(gradient.op.name + '/gradient', tf.norm(gradient))
        tf.summary.scalar(gradient.op.name + '/gradient-sparsity',
                          tf.nn.zero_fraction(gradient))
        tf.summary.histogram(gradient.op.name + '/abs.gradient', gradient)

  def apply_mask(self, x, scope=''):
    return apply_mask(x, scope, self._spec.prune_option)

  def print_hparams(self):
    tf.logging.vlog(0, self._spec.to_json())

  def get_spec(self):
    """Get the spec / hparams used to create the ApplyCompression object."""
    return self._spec
