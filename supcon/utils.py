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

# Lint as: python3
"""Utility functions to build losses, optimizers, EMA etc."""

import re

import tensorflow.compat.v1 as tf

from supcon import enums

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
# pylint: enable=g-direct-tensorflow-import

# LARS Optimizer: Scaling of learning rate to compute trust ratio
ETA_DEFAULT = 0.001

# Default global step used by create_train_op function
_USE_GLOBAL_STEP = 0


class LARSOptimizer(tf.train.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

  Implementation based on: //third_party/py/simclr/lars_optimizer.py
  """

  def __init__(self,
               learning_rate,
               momentum=0.9,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eta=ETA_DEFAULT,
               name='LARSOptimizer'):
    r"""Constructs a LARSOptimizer.

    Notation based on: https://arxiv.org/pdf/1708.03888.pdf

    Args:
      learning_rate: A `float` for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momeuntum update in
          classic momentum, but after momentum for popular momentum.
      eta: A `float` for scaling of learning rate when computing trust ratio.
        This is used in the apply gradient function for every weight variable
        $w$ and it's corresponding gradient $g$.
        The trust ratio is computed as $\tau = \eta * ||w|| / ||g||$ if both
        ||w|| and ||g|| are > 0. The update that is applied is
        $w_new = w_old - \tau * g$.
      name: The name for the scope.
    """
    super(LARSOptimizer, self).__init__(use_locking=False, name=name)

    self.learning_rate = learning_rate
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.use_nesterov = use_nesterov
    self.classic_momentum = classic_momentum
    self.eta = eta
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if global_step is None:
      global_step = tf.train.get_or_create_global_step()
    new_global_step = global_step + 1

    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      if tf.executing_eagerly() or tf.executing_eagerly_outside_functions():
        param_name = param.name
      else:
        param_name = param.op.name

      v = tf.get_variable(
          name=f'{param_name}/{self.get_name()}/Momentum',
          shape=param.shape.as_list(),
          dtype=param.dtype,
          trainable=False,
          initializer=tf.zeros_initializer())

      grad = tf.cast(grad, param.dtype)

      if self._use_weight_decay(param_name):
        grad += self.weight_decay * param

      if callable(self.learning_rate):
        base_learning_rate = self.learning_rate()
      else:
        base_learning_rate = self.learning_rate

      if self.classic_momentum:
        trust_ratio = 1.0
        if self._do_layer_adaptation(param_name):
          w_norm = tf.norm(param, ord=2)
          g_norm = tf.norm(grad, ord=2)
          trust_ratio = tf.where(
              tf.greater(w_norm, 0), tf.where(
                  tf.greater(g_norm, 0), (self.eta * w_norm / g_norm),
                  1.0),
              1.0)
          trust_ratio = tf.cast(trust_ratio, param.dtype)

        base_learning_rate = tf.cast(base_learning_rate, param.dtype)
        scaled_lr = base_learning_rate * trust_ratio

        next_v = tf.multiply(
            tf.cast(self.momentum, v.dtype), v) + scaled_lr * grad
        if self.use_nesterov:
          update = tf.multiply(
              tf.cast(self.momentum, v.dtype), next_v) + scaled_lr * grad
        else:
          update = next_v
        next_param = param - update
      else:
        next_v = tf.multiply(
            tf.cast(self.momentum, v.dtype), v) + grad
        if self.use_nesterov:
          update = tf.multiply(
              tf.cast(self.momentum, v.dtype), next_v) + grad
        else:
          update = next_v

        trust_ratio = 1.0
        if self._do_layer_adaptation(param_name):
          w_norm = tf.norm(param, ord=2)
          v_norm = tf.norm(update, ord=2)
          trust_ratio = tf.where(
              tf.greater(w_norm, 0), tf.where(
                  tf.greater(v_norm, 0), (self.eta * w_norm / v_norm),
                  1.0),
              1.0)
          trust_ratio = tf.cast(trust_ratio, param.dtype)
        scaled_lr = trust_ratio * base_learning_rate
        next_param = param - scaled_lr * update

      assignments.extend(
          [param.assign(next_param),
           v.assign(next_v),
           global_step.assign(new_global_step)])
    return tf.group(*assignments, name=name)

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True


def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    transform_grads_fn=None,
                    gate_gradients=tf.train.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    check_numerics=True):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.train.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`, a
      warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.compat.v1.trainable_variables().
    transform_grads_fn: A function which takes a single argument, a list of
      gradient to variable pairs (tuples), performs any requested gradient
      updates, such as gradient clipping or multipliers, and returns the updated
      list.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    check_numerics: Whether or not we apply check_numerics.

  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  if global_step is _USE_GLOBAL_STEP:  # pylint: disable=g-int-id-comparison
    # global_step can be None when passed into the optimizer in case we do not
    # want apply_gradients to factor that in. This is different from default
    # behaviour where we use the standard global step.
    global_step = tf.train.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    tf.logging.warning('update_ops in create_train_op does not contain all the '
                       'update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with tf.control_dependencies(update_ops):
      barrier = tf.no_op(name='update_barrier')
    with tf.control_dependencies([barrier]):
      total_loss = tf.identity(total_loss)

  if variables_to_train is None:
    # Default to tf.compat.v1.trainable_variables()
    variables_to_train = tf.trainable_variables()
  else:
    # Make sure that variables_to_train are in
    # tf.compat.v1.trainable_variables()
    for v in variables_to_train:
      assert v.trainable or v in tf.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  if transform_grads_fn:
    grads = transform_grads_fn(grads)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with tf.name_scope('train_op'):
    # Make sure total_loss is valid.
    if check_numerics:
      total_loss = tf.check_numerics(total_loss,
                                     'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    with tf.control_dependencies([grad_updates]):
      train_op = tf.identity(total_loss)

  # Add the operation used for training to the 'train_op' collection
  train_ops = tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
    train_ops.append(train_op)

  return train_op


def maybe_add_warmup_to_lr(
    warmup_target, non_warmup_learning_rate, warmup_step_counter,
    warmup_epochs, steps_per_epoch):
  """Add warmup to the start of learning rate if warmup_steps is more than 0.

  Args:
    warmup_target: Learning rate at the end of the warmup period, before
      the decay kicks-in.
    non_warmup_learning_rate: The learning rate Tensor use after `warmup_steps`
      which also inherently depends on `global_step`. This could be a Tensor or
      python scalar.
    warmup_step_counter: The step Tensor to used to compute the learning rate.
      warmup_step_counter should take value 0 at the step warmup starts.
      Tensor has dtype tf.int32 or tf.int64.
    warmup_epochs: Tensor for number of epochs for which warmup runs. Dtype must
      be one of tf.int32, tf.int64 or tf.float32.
    steps_per_epoch: Tensor which defines the number of steps that are run for
      every epoch, with dtype

  Returns:
    Tensor which can be used as learning rate for the training process.
  """

  if warmup_epochs is not None:
    warmup_steps = tf.cast(warmup_epochs * steps_per_epoch,
                           warmup_step_counter.dtype)
    warmup_learning_rate = (
        warmup_target * tf.cast(warmup_step_counter, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    warmup_learning_rate = tf.cond(
        warmup_steps > 0,
        lambda: warmup_learning_rate,
        lambda: non_warmup_learning_rate)
    learning_rate = tf.cond(
        warmup_step_counter < warmup_steps,
        lambda: warmup_learning_rate,
        lambda: non_warmup_learning_rate)

  return learning_rate


def exponential_decay(initial_lr,
                      global_step,
                      total_epochs,
                      steps_per_epoch,
                      decay_rate=0.97,
                      epochs_per_decay=2.4):
  """Exponential decay learning rate decay.

  Args:
    initial_lr: Learning rate at the start of training if warmup is not applied.
    global_step: The global step Tensor to use for learning rate computation.
    total_epochs: Not used.
    steps_per_epoch: Number of training steps per epoch of training.
    decay_rate: The factor by which learning rate decays after every
      `epochs_per_decay` steps.
    epochs_per_decay: Scaling factor used to control the rate of decay.

  Returns:
    Tensor which can be used as learning rate for the training process.
  """
  del total_epochs
  epochs_per_decay = tf.convert_to_tensor(epochs_per_decay)
  decay_steps = tf.cast(
      steps_per_epoch, epochs_per_decay.dtype) * epochs_per_decay
  learning_rate = tf.train.exponential_decay(
      initial_lr, global_step, decay_steps, decay_rate, staircase=True)
  return learning_rate


def cosine_decay(initial_lr,
                 global_step,
                 total_epochs,
                 steps_per_epoch):
  r"""Cosine shaped learning rate decay.

  The learning rate multiplier varies as (1. + cos(x)) / 2. where x varies from
  [0, 2\pi] between step 0 and total_steps.

  Args:
    initial_lr: Learning rate at the start of training if warmup is not applied.
    global_step: The global step Tensor to use for learning rate computation.
    total_epochs: Total number of epochs over which the decay happens after
      which the learning rate is fixed at 0.
    steps_per_epoch: Number of training steps per epoch of training.

  Returns:
    Tensor which can be used as learning rate for the training
      process.
  """
  total_steps = tf.cast(
      steps_per_epoch * total_epochs,
      global_step.dtype)

  learning_rate = tf.train.cosine_decay(
      initial_lr, global_step, total_steps)

  return learning_rate


def piecewise_linear_decay(initial_lr,
                           global_step,
                           total_epochs,
                           steps_per_epoch,
                           boundary_epochs=(30, 60, 80, 90),
                           decay_rate=0.1):
  """Piece-wise linear learning rate schedule.

  Args:
    initial_lr: Learning rate at the start of training (without accounting for
      warmup).
    global_step: The global step to use for learning rate computation.
    total_epochs: Not used.
    steps_per_epoch: Number of training steps per epoch of training.
    boundary_epochs: Iterable of python ints containing epochs at which learning
      rate changes.
    decay_rate: At each `boundary_epoch`, `initial_lr` is decayed by an
      additional factor of `decay_rate`.

  Returns:
    Tensor which can be used as learning rate for the training process.
  """
  del total_epochs
  assert steps_per_epoch is not None
  boundaries = [tf.cast(steps_per_epoch * epoch, global_step.dtype)
                for epoch in boundary_epochs]
  rates = [initial_lr * decay_rate**n for n in range(len(boundary_epochs) + 1)]
  learning_rate = tf.compat.v1.train.piecewise_constant(global_step,
                                                        boundaries, rates)
  return learning_rate


def build_learning_rate_schedule(
    learning_rate,
    decay_type,
    warmup_start_epoch,
    max_learning_rate_epoch,
    decay_end_epoch,
    global_step,
    steps_per_epoch,
    **decay_type_specific_kwargs):
  """Build learning rate from base learning rate and other details.

  We note that warmup_start_epoch <= max_learning_rate_epoch < decay_end_epoch
  since the warmup happens at the start of learning rate schedule.

  Args:
    learning_rate: Learning rate for the model.
    decay_type: Name of the decay that should be applied to the learning rate.
    warmup_start_epoch: Epoch at which learning rate warmup starts.
    max_learning_rate_epoch: Epoch at which learning rate warmup ends and the
      decay kicks in.
    decay_end_epoch: Epoch at which learning rate decays ends, at which point
      learning rate becomes 0.
    global_step: The global step to use for learning rate computation.
    steps_per_epoch: Integer which defines the number of steps that are run for
      every epoch.
    **decay_type_specific_kwargs: Specific key-word arguments which are unique
      to a said `decay_type`.


  Returns:
    Scalar tensor which stores the learning rate at a given global step.
  """
  if decay_end_epoch == max_learning_rate_epoch:
    # This stage of training is 0 epochs long, so just return learning_rate and
    # avoid potential divide by 0 problems.
    if warmup_start_epoch < max_learning_rate_epoch:
      raise ValueError(
          'Cannot have warmup for a 0-step learning rate schedule.')

    return learning_rate

  assert warmup_start_epoch <= max_learning_rate_epoch
  assert max_learning_rate_epoch < decay_end_epoch

  max_learning_rate_epoch_tensor = tf.convert_to_tensor(max_learning_rate_epoch)
  warmup_start_epoch_tensor = tf.convert_to_tensor(
      warmup_start_epoch, max_learning_rate_epoch_tensor.dtype)
  decay_end_epoch_tensor = tf.convert_to_tensor(
      decay_end_epoch, max_learning_rate_epoch_tensor.dtype)
  steps_per_epoch_tensor = tf.cast(steps_per_epoch,
                                   max_learning_rate_epoch_tensor.dtype)

  # Learning rate decay kicks in starting max_learning_rate_epoch
  # Before max_learning_rate_epoch either there is a warmup or the learning rate
  # is set to the constant value of `initial_lr`.
  learning_rate_step = global_step - tf.cast(
      max_learning_rate_epoch_tensor * steps_per_epoch_tensor,
      global_step.dtype)

  def _no_decay_fn(initial_lr, *args, **kwargs):
    del args, kwargs
    return initial_lr

  decay_type_fn_map = {
      enums.DecayType.EXPONENTIAL: exponential_decay,
      enums.DecayType.COSINE: cosine_decay,
      enums.DecayType.PIECEWISE_LINEAR: piecewise_linear_decay,
      enums.DecayType.NO_DECAY: _no_decay_fn,
  }
  if decay_type not in decay_type_fn_map:
    raise ValueError(f'Unknown decay type {decay_type}')

  decayed_learning_rate = decay_type_fn_map[decay_type](
      initial_lr=learning_rate,
      global_step=learning_rate_step,
      total_epochs=decay_end_epoch_tensor - max_learning_rate_epoch_tensor,
      steps_per_epoch=steps_per_epoch,
      **decay_type_specific_kwargs)

  # The learning rate is set to 0 once global_step is more than total_steps.
  total_steps = tf.cast(
      steps_per_epoch_tensor * (
          decay_end_epoch_tensor - max_learning_rate_epoch_tensor),
      global_step.dtype)
  decayed_learning_rate = tf.cond(
      learning_rate_step <= total_steps,
      lambda: decayed_learning_rate,
      lambda: 0.0)

  warmup_step_counter = global_step - tf.cast(
      warmup_start_epoch_tensor * steps_per_epoch_tensor, global_step.dtype)
  return maybe_add_warmup_to_lr(
      learning_rate, decayed_learning_rate, warmup_step_counter,
      max_learning_rate_epoch - warmup_start_epoch_tensor,
      steps_per_epoch_tensor)


def build_optimizer(learning_rate,
                    optimizer_type=enums.Optimizer.RMSPROP,
                    decay=0.9,
                    epsilon=1.0,
                    momentum=0.9,
                    lars_weight_decay=1e-6,
                    lars_exclude_from_weight_decay=('batch_normalization',),
                    is_tpu=False,
                    name=''):
  """Build optimizer."""
  if optimizer_type == enums.Optimizer.MOMENTUM:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        use_nesterov=False,
        name=f'MomentumOptimizer_{name}')
  elif optimizer_type == enums.Optimizer.NESTEROV:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        use_nesterov=True,
        name=f'NesterovMomentumOptimizer_{name}')
  elif optimizer_type == enums.Optimizer.RMSPROP:
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay,
        momentum,
        epsilon,
        name=f'RMSPropOptimizer_{name}')
  elif optimizer_type == enums.Optimizer.LARS:
    optimizer = LARSOptimizer(
        learning_rate,
        momentum=momentum,
        weight_decay=lars_weight_decay,
        exclude_from_weight_decay=lars_exclude_from_weight_decay,
        name=f'LARSOptimizer_{name}')
  elif optimizer_type == enums.Optimizer.ADAM:
    optimizer = tf.train.AdamOptimizer(learning_rate)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer_type}')

  if is_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  return optimizer


def stacked_multiview_image_channels_to_batch(images,
                                              data_format='channels_last'):
  """Split 2 views from the channel dim and concatenate back on the batch dim.

  Args:
    images: A 4-D batched image tensor, with 2 images stacked in the channel
      dimension.
    data_format: Either 'channels_first' or 'channels_last' to indicate whether
      `images` is formatted [N, 2C, H, W] or [N, H, W, 2C].

  Returns:
    Images reformated so that the extra views are now stacked in the batch
    dimension. If the input was [N, 2C, H, W] the output is [2N, C, H, W]. If
    the input was [N, H, W, 2C] the output is [2N, H, W, C].
  """
  with tf.name_scope('channels_to_batch'):
    if data_format == 'channels_first':
      images_a = images[:, :3, :, :]
      images_b = images[:, -3:, :, :]
    else:
      images_a = images[:, :, :, :3]
      images_b = images[:, :, :, -3:]
    return tf.concat([images_a, images_b], axis=0)


def stacked_multiview_embeddings_to_channel(embeddings):
  """Stack multiviewed embeddings in the channel dimension instead of batch.

  Args:
    embeddings: A 2D tensor of shape [2N, D].

  Returns:
    The embeddings reformatted to [N, 2D].
  """
  with tf.name_scope('batch_to_channels'):
    return tf.concat(tf.split(embeddings, 2, 0), 1)


def local_tpu_replica_id():
  """Returns the index of the current TPU replica."""
  num_tpu_replicas = tpu_function.get_tpu_context().number_of_shards
  if num_tpu_replicas is not None:
    # Need tf.control_dependencies(None) in order to make sure this is run
    # on CPU (not TPU)
    with tf.control_dependencies(None):
      return tpu_ops.tpu_replicated_input(
          list(range(num_tpu_replicas)), name='local_replica_id')
  else:
    # The non-TPU case.
    return 0


def cross_replica_concat(tensor):
  """A cross-replica concatenation of a single Tensor across TPU cores.

  Input tensor is assumed to have batch dimension as the first dimension. The
  concatenation is done along the batch dimension.

  Args:
    tensor: Input Tensor which should be concatenated across TPU cores.

  Returns:
    The concatenated Tensor with batch dimension multiplied by the number of
      TPU cores.
  """
  num_tpu_replicas = tpu_function.get_tpu_context().number_of_shards

  if num_tpu_replicas is not None:
    # Scattered tensor has shape [num_replicas, local_batch_size, ...]
    scattered_tensor = tf.scatter_nd(
        indices=[[local_tpu_replica_id()]],
        updates=[tensor],
        shape=[num_tpu_replicas] + tensor.shape.as_list())
    reduced_tensor = tf.tpu.cross_replica_sum(scattered_tensor)
    # Returned tensor has shape [num_replicas * local_batch_size, ...]
    return tf.reshape(reduced_tensor,
                      [-1] + scattered_tensor.shape.as_list()[2:])
  else:
    # This is a no op if not running on TPU
    return tensor


def estimator_mode_to_model_mode(estimator_mode):
  return {
      tf.estimator.ModeKeys.TRAIN: enums.ModelMode.TRAIN,
      tf.estimator.ModeKeys.EVAL: enums.ModelMode.EVAL,
      tf.estimator.ModeKeys.PREDICT: enums.ModelMode.INFERENCE,
  }[estimator_mode]
