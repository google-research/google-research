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

r"""Functions to add magnitude-based model pruning or noise based pruning.

Pruning broadly refers to a set of approaches that reduce the number of
parameters in a model. Inducing sparsity by pruning should ideally lead to
model compression, acceleration at inference time + provide theoretical
insights as to why we need such large networks to begin with.

"""
from absl import flags
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.pruning_tools import core_layers as core
from pruning_identified_exemplars.pruning_tools import pruning_utils
from tensorflow.python.ops import sort_ops as sort  # pylint: disable=g-direct-tensorflow-import

_MASK_COLLECTION = core.MASK_COLLECTION
_THRESHOLD_COLLECTION = core.THRESHOLD_COLLECTION
_MASKED_WEIGHT_COLLECTION = core.MASKED_WEIGHT_COLLECTION
_WEIGHT_COLLECTION = core.WEIGHT_COLLECTION
_MASKED_WEIGHT_NAME = core.MASKED_WEIGHT_NAME

FLAGS = flags.FLAGS


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
       comma separed list of weight variable name:target sparsity pairs.
       For layers/weights not in this list, sparsity as specified by the
       target_sparsity hyperparameter is used.
       Eg. [conv1:0.9,conv2/kernel:0.8]
    threshold_decay: float
      the decay factor to use for exponential decay of the thresholds
    pruning_frequency: integer
      How often should the masks be updated? (in # of global_steps)
    nbins: integer
      number of bins to use for histogram computation
    block_height: integer
      number of rows in a block (defaults to 1)
    block_width: integer
      number of cols in a block (defaults to 1)
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
      option = 3 means using effective resistance for pruning.
      option > 3 reserved for future use

    We use the following sparsity function:

    num_steps = (sparsity_function_end_step -
                 sparsity_function_begin_step)/pruning_frequency
    sparsity(step) = (initial_sparsity - target_sparsity)*
                     [1-step/(num_steps -1)]**exponent + target_sparsity

  Args: None

  Returns:
    tf.HParams object initialized to default values

  """
  return tf.HParams(
      name='model_pruning',
      begin_pruning_step=0,
      end_pruning_step=-1,
      weight_sparsity_map=[''],
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
      prune_option='weight')


def apply_mask(x, scope=''):
  """Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope."".

  Returns:
    Tensor representing masked_weights
  """

  mask = pruning_utils.weight_mask_variable(x, scope)
  threshold = pruning_utils.weight_threshold_variable(x, scope)
  # Add masked_weights in the weights namescope so as to make it easier
  # for the quantization library to add quant ops.
  masked_weights = tf.multiply(mask, x, _MASKED_WEIGHT_NAME)

  # Make sure the mask for a given variable are not added multiple times to the
  # collection. This is particularly important when applying mask to an RNN.
  # weight variables
  if mask not in tf.get_collection_ref(_MASK_COLLECTION):
    tf.add_to_collection(_THRESHOLD_COLLECTION, threshold)
    tf.add_to_collection(_MASK_COLLECTION, mask)
    tf.add_to_collection(_MASKED_WEIGHT_COLLECTION, masked_weights)
    tf.add_to_collection(_WEIGHT_COLLECTION, x)
  return masked_weights


def get_masked_weights():
  return tf.get_collection(_MASKED_WEIGHT_COLLECTION)


def get_masks():
  return tf.get_collection(_MASK_COLLECTION)


def get_thresholds():
  return tf.get_collection(_THRESHOLD_COLLECTION)


def get_weights():
  return tf.get_collection(_WEIGHT_COLLECTION)


def get_weight_sparsity():
  """Get sparsity of the weights.

  Args: None

  Returns:
    A list containing the sparsity of each of the weight tensors
  """
  masks = get_masks()
  return [tf.nn.zero_fraction(mask) for mask in masks]


def rescale_input(x):
  """Rescales image input to be in range [0,1]."""

  current_min = tf.reduce_min(x)
  current_max = tf.reduce_max(x)

  # we add an epsilon value to prevent division by zero
  epsilon = 1e-5
  rescaled_x = tf.div(
      tf.subtract(x, current_min),
      tf.maximum(tf.subtract(current_max, current_min), epsilon))
  return rescaled_x


class Pruning(object):
  """class to prune model at each pruning step."""

  def __init__(self,
               spec=None,
               global_step=None,
               sparsity=None,
               pruning_method=None,
               end_sparsity=None,
               labels=None,
               num_classes=None,
               use_tpu=False):
    """Set up the specification for model pruning.

    If a spec is provided, the sparsity is set up based on the sparsity_function
    in the spec. The effect of sparsity_function is overridden if the sparsity
    variable is passed to the constructor. This enables setting up arbitrary
    sparsity profiles externally and passing it to this pruning functions.

    Args:
      spec: Pruning spec as defined in pruning.proto
      global_step: A tensorflow variable that is used while setting up the
        sparsity function
      sparsity: A tensorflow scalar variable storing the sparsity
      pruning_method: The pruning methodology used to identify which weights to
        remove.
      end_sparsity: Desired sparsity as a fraction of total weights by the end
        of training.
      labels: The true labels associated with each input.
      num_classes: The number of unique labels for all input images in the
        dataset.
      use_tpu: Whether the code is being run on a tpu or not.
    """

    # Pruning specification
    self._spec = spec if spec else get_pruning_hparams()

    # Sanity check for pruning hparams
    self._validate_spec()

    self._target_sparsity = end_sparsity

    # A tensorflow variable that tracks the sparsity function.
    # If not provided as input, the graph must already contain the global_step
    # variable before calling this constructor.
    self._global_step = self._setup_global_step(global_step)

    # Stores the tensorflow sparsity variable.
    # Built using self._setup_sparsity() or provided externally
    self._sparsity = sparsity if sparsity else self._setup_sparsity()

    # List of tensorflow assignments ops for new masks and thresholds
    self._assign_ops = []

    self._use_tpu = use_tpu

    # Tensorflow variable keeping track of the last global step when the masks
    # were updated
    self._last_update_step = self._setup_last_update_step()

    # Block dimensions
    self._block_dim = [self._spec.block_height, self._spec.block_width]

    # Block pooling function
    self._block_pooling_function = self._spec.block_pooling_function

    self._pruning_method = pruning_method

    self._labels = labels

    self._num_classes = num_classes

    # Mapping of weight names and target sparsity
    self._weight_sparsity_map = self._get_weight_sparsity_map()

  def _setup_global_step(self, global_step):
    """Fetches the global step."""
    graph_global_step = global_step
    if graph_global_step is None:
      graph_global_step = tf.training_util.get_global_step()

    return tf.cast(graph_global_step, tf.int32)

  def _setup_sparsity(self):
    """Defines the current level of sparsity required."""
    begin_step = self._spec.sparsity_function_begin_step
    end_step = self._spec.sparsity_function_end_step
    initial_sparsity = self._spec.initial_sparsity
    exponent = self._spec.sparsity_function_exponent

    if begin_step >= end_step:
      raise ValueError(
          'Pruning must begin before it can end. begin_step=%d, end_step=%d' %
          (begin_step, end_step))

    with tf.name_scope(self._spec.name):
      # is the fraction of the total training completed
      p = tf.minimum(
          1.0,
          tf.maximum(
              0.0,
              tf.div(
                  tf.cast(self._global_step - begin_step, tf.float32),
                  end_step - begin_step)))

      # slowly introduce sparsity
      sparsity = tf.cast(
          tf.add(
              tf.multiply((initial_sparsity - self._target_sparsity),
                          tf.pow(1 - p, exponent)),
              self._target_sparsity,
              name='sparsity'), tf.float32)
    return sparsity

  def _setup_last_update_step(self):
    """Defines the last update step for pruning."""
    with tf.variable_scope(
        self._spec.name, use_resource=self._use_tpu) as scope:
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

  def _get_weight_sparsity_map(self):
    """Return the map of weight_name:sparsity parsed from the hparams."""
    weight_sparsity_map = {}
    val_list = self._spec.weight_sparsity_map
    filtered_val_list = [l for l in val_list if l]
    for val in filtered_val_list:
      weight_name, sparsity = val.split(':')
      if float(sparsity) >= 1.0:
        raise ValueError('Weight sparsity can not exceed 1.0')
      weight_sparsity_map[weight_name] = float(sparsity)

    return weight_sparsity_map

  def _validate_spec(self):
    spec = self._spec
    if spec.begin_pruning_step < 0:
      raise ValueError('Illegal value for begin_pruning_step')

  def _get_sparsity(self, weight_name):
    """Return target sparsity for the given layer/weight name."""
    target_sparsity = [
        sparsity for name, sparsity in self._weight_sparsity_map.items()
        if weight_name.find(name) != -1
    ]
    if not target_sparsity:
      return self._sparsity

    if len(target_sparsity) > 1:
      raise ValueError(
          'Multiple matches in weight_sparsity_map for weight %s' % weight_name)
    return tf.mul(self._sparsity,
                  tf.div(target_sparsity[0], self._spec.target_sparsity))

  def _exists_in_do_not_prune_list(self, tensor_name):
    """Checks whether the weight matrix is on the no prune list."""
    do_not_prune_list = self._spec.do_not_prune
    if not do_not_prune_list[0]:
      return False
    for layer_name in do_not_prune_list:
      if tensor_name.find(layer_name) != -1:
        return True

    return False

  def _update_mask(self, weights, threshold):
    """Updates the mask for a given weight tensor for magnitude based pruning.

    This functions first computes the cdf of the weight tensor, and estimates
    the threshold value such that 'desired_sparsity' fraction of weights
    have magnitude less than the threshold.

    Args:
      weights: The weight tensor that needs to be masked.
      threshold: The current threshold value. The function will compute a new
        threshold and return the exponential moving average using the current
        value of threshold

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A tensor of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if sparsity is not defined
    """

    if self._sparsity is None:
      raise ValueError('Sparsity variable undefined')

    sparsity = self._get_sparsity(weights.op.name)
    with tf.name_scope(weights.op.name + '_pruning_ops'):
      abs_weights = tf.abs(weights)
      k = tf.cast(
          tf.round(tf.cast(tf.size(abs_weights), tf.float32) * (1 - sparsity)),
          tf.int32)
      # Sort the entire array.
      values, _ = tf.math.top_k(
          tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))
      # Grab the (k-1) th value.
      current_threshold = tf.gather(values, k - 1)
      smoothed_threshold = tf.add_n([
          tf.multiply(current_threshold, 1 - self._spec.threshold_decay),
          tf.multiply(threshold, self._spec.threshold_decay)
      ])

      new_mask = tf.cast(
          tf.greater_equal(abs_weights, smoothed_threshold), tf.float32)

    return smoothed_threshold, new_mask

  def _update_random_mask(self, weights, mask):
    """Randomly identifies subset of weights to be set to zero in the network.

       If pruning method is specified as 'random_cumulative', at each pruning
       step a random subset of weights is set to zero taking into account which
       weights are still non-zero.

       If pruning method is specified to be 'random_independent', the random
       weights selected at each pruning step are entirely independent
       of previous pruning steps.

    Args:
      weights: The weight tensor that needs to be masked.
      mask: The mask from the previous pruning update.

    Returns:
      new_mask: A tensor of the same size and shape as weights containing
        0 or 1.
    Raises:
      ValueError: Raises ValueError if sparsity is not defined
    """

    if self._sparsity is None:
      raise ValueError('Sparsity variable undefined')

    sparsity = self._get_sparsity(weights.op.name)
    with tf.name_scope(weights.op.name + '_pruning_ops'):

      if self._pruning_method == 'random_cumulative':
        # compute the total number of weights in the layer.
        total_weights = tf.size(weights)
        mask = tf.reshape(mask, [total_weights])

        # adding random vector because if there are ties sort simply
        # selects based upon index position (starts from beginning of vector).
        random_noise = tf.random_uniform(
            shape=mask.shape, minval=0.0001, maxval=0.0003)
        mask = tf.cast(tf.add(random_noise, mask), tf.float32)

        # rank the binary mask by magnitude. Weights already on are selected
        # plus a random subset of all other weights.
        sorted_mask = sort(mask, direction='DESCENDING')

        # multiply desired sparsity fraction by the number of weights.
        num_weights = tf.reshape(
            tf.cast(tf.cast(total_weights, tf.float32) * sparsity, tf.int32),
            [1])
        percentile = tf.gather_nd(sorted_mask, num_weights)

        one_mask = tf.ones([total_weights])
        zero_mask = tf.zeros([total_weights])

        feature_ranking = tf.where(
            tf.greater_equal(percentile, mask), one_mask, zero_mask)
        new_mask = tf.reshape(feature_ranking, weights.get_shape())

      else:
        drop_out = tf.nn.dropout(
            tf.ones_like(weights), keep_prob=(1. - self._sparsity))
        new_mask = tf.cast(drop_out, tf.float32)

    return self._sparsity, new_mask

  def _update_block_mask(self, weights, threshold, mask):
    """Performs block-granular masking of the weights.

    Block pruning occurs only if the block_height or block_width is > 1 and
    if the weight tensor, when squeezed, has ndims = 2. Otherwise, elementwise
    pruning occurs.
    Args:
      weights: The weight tensor that needs to be masked.
      threshold: The current threshold value. The function will compute a new
        threshold and return the exponential moving average using the current
        value of threshold
      mask: The mask from the previous pruning update.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if block pooling function is not AVG or MAX
    """
    squeezed_weights = tf.squeeze(weights)
    if squeezed_weights.get_shape().ndims != 2 or self._block_dim == [1, 1]:
      if self._pruning_method == 'threshold':
        return self._update_mask(weights, threshold)
      # random_cumulative removes at random taking into account previous
      # random modification. random_indepent simply removes at random.
      elif self._pruning_method in ['random_independent', 'random_cumulative']:
        return self._update_random_mask(weights, mask)
      else:
        raise ValueError('Unknown pruning method: %s' % self._pruning_method)

    if self._block_pooling_function not in ['AVG', 'MAX']:
      raise ValueError('Unknown pooling function for block sparsity: %s' %
                       self._block_pooling_function)

    with tf.name_scope(weights.op.name + '_pruning_ops'):
      abs_weights = tf.abs(squeezed_weights)

      pool_window = [self._block_dim[0], self._block_dim[1]]
      pool_fn = pruning_utils.factorized_pool

      if not self._use_tpu:
        pool_fn = tf.pool
        abs_weights = tf.reshape(
            abs_weights,
            [1, abs_weights.get_shape()[0],
             abs_weights.get_shape()[1], 1])

      pooled_weights = pool_fn(
          abs_weights,
          window_shape=pool_window,
          pooling_type=self._block_pooling_function,
          strides=pool_window,
          padding='SAME',
          name=weights.op.name + '_pooled')

      if pooled_weights.get_shape().ndims != 2:
        pooled_weights = tf.squeeze(pooled_weights)

      if self._pruning_method == 'threshold':
        smoothed_threshold, new_mask = self._update_mask(
            pooled_weights, threshold)
      elif self._pruning_method in ['random_independent', 'random_cumulative']:
        smoothed_threshold, new_mask = self._update_random_mask(
            pooled_weights, mask)
      else:
        raise ValueError('Unknown pruning method: %s' % self._pruning_method)

      ## this is the process that updates the mask.
      updated_mask = pruning_utils.kronecker_product(new_mask,
                                                     tf.ones(self._block_dim))
      sliced_mask = tf.slice(
          updated_mask, [0, 0],
          [squeezed_weights.get_shape()[0],
           squeezed_weights.get_shape()[1]])

    return smoothed_threshold, tf.reshape(sliced_mask, tf.shape(weights))

  def _get_mask_assign_ops(self):
    """retrieve all masks, and iteratively update each based upon method."""
    # Make sure the assignment ops have not already been added to the list
    if self._assign_ops:
      raise ValueError(
          'Assign op list not empty. _get_mask_assign_ops() called twice?')

    masks = get_masks()
    weights = get_weights()
    thresholds = get_thresholds()

    if len(masks) != len(thresholds):
      raise ValueError(
          'Number of masks %s and number of thresholds %s mismatch' %
          (len(masks), len(thresholds)))

    for index, mask in enumerate(masks):
      threshold = thresholds[index]
      weight = weights[index]
      is_partitioned = isinstance(weight, tf.PartitionedVariable)
      if is_partitioned:
        weight = weight.as_tensor()

      if self._spec.do_not_prune:
        if self._exists_in_do_not_prune_list(mask.name):
          continue

      new_threshold, new_mask = self._update_block_mask(weight, threshold, mask)
      self._assign_ops.append(
          pruning_utils.variable_assign(threshold, new_threshold))

      self._assign_ops.append(
          pruning_utils.partitioned_variable_assign(mask, new_mask)
          if is_partitioned else pruning_utils.variable_assign(mask, new_mask))

  def mask_update_op(self):
    """update mask and all dependencies."""
    with tf.name_scope(self._spec.name):
      if not self._assign_ops:
        self._get_mask_assign_ops()
      with tf.control_dependencies([
          tf.assign(
              self._last_update_step,
              self._global_step,
              name='last_mask_update_step_assign')
      ]):
        with tf.control_dependencies(self._assign_ops):
          tf.logging.info('Updating masks.')
          return tf.no_op('mask_update')

  def conditional_mask_update_op(self):
    """update mask if all pruning criteria are met."""

    def update_masks():
      """check whether all pruning conditions are met before pruning."""
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

    return tf.cond(update_masks(), mask_update_op, no_update_op)

  def add_pruning_summaries(self):
    """Adds summaries for this pruning spec."""
    with tf.name_scope(self._spec.name + '_summaries'):
      tf.summary.scalar('sparsity', self._sparsity)
      tf.summary.scalar('last_mask_update_step', self._last_update_step)
      masks = get_masks()
      thresholds = get_thresholds()
      for mask, threshold in zip(masks, thresholds):
        if not self._exists_in_do_not_prune_list(mask.name):
          tf.summary.scalar(mask.op.name + '/sparsity',
                            tf.nn.zero_fraction(mask))
          tf.summary.scalar(threshold.op.name + '/threshold', threshold)

  def print_hparams(self):
    tf.logging.info(self._spec.to_json())
