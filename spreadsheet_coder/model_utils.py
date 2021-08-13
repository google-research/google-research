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

"""Model config utils."""

import collections
import re
from six.moves import zip
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.rnn as contrib_rnn
from tensorflow.python.ops import math_ops


def configure_tpu(flags):
  """Configures the TPU from the command line flags."""
  session_config = tf.ConfigProto(
      allow_soft_placement=True)
  # Uncomment the following line if you hope to monitor GPU RAM growth
  # session_config.gpu_options.allow_growth = True

  if flags.use_tpu:
    strategy = None
    tf.logging.info("Use TPU without distribute strategy.")
  elif flags.num_core_per_host == 1:
    strategy = None
    tf.logging.info("Single device mode.")
  else:
    strategy = tf.distribute.MirroredStrategy(
        num_gpus=flags.num_core_per_host)
    tf.logging.info("Use MirroredStrategy with %d devices.",
                    strategy.num_replicas_in_sync)

  per_host_input = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      master=flags.master,
      model_dir=flags.output_dir,
      save_checkpoints_steps=flags.save_checkpoints_steps,
      session_config=session_config,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=flags.iterations_per_loop,
          num_shards=flags.num_hosts * flags.num_core_per_host,
          per_host_input_for_training=per_host_input),
      keep_checkpoint_max=flags.max_save,
      save_checkpoints_secs=None,
      train_distribute=strategy
  )
  return run_config


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               include_in_weight_decay=("r_s_bias", "r_r_bias", "r_w_bias"),
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.include_in_weight_decay = include_in_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    for r in self.include_in_weight_decay:
      if re.search(r, param_name) is not None:
        return True

    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          tf.logging.info("Adam WD excludes {}".format(param_name))
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


def get_train_op(flags, total_loss, ema=None, tvars=None):
  """Generates the training operation."""
  global_step = tf.train.get_or_create_global_step()

  # increase the learning rate linearly
  if flags.warmup_steps > 0:
    warmup_lr = (tf.cast(global_step, tf.float32)
                 / tf.cast(flags.warmup_steps, tf.float32)
                 * flags.learning_rate)
  else:
    warmup_lr = 0.0

  # decay the learning rate
  if flags.decay_method == "poly":
    decay_lr = tf.train.polynomial_decay(
        flags.learning_rate,
        global_step=global_step - flags.warmup_steps,
        decay_steps=flags.train_steps - flags.warmup_steps,
        end_learning_rate=flags.learning_rate * flags.min_lr_ratio)
  elif flags.decay_method == "cos":
    decay_lr = tf.train.cosine_decay(
        flags.learning_rate,
        global_step=global_step - flags.warmup_steps,
        decay_steps=flags.train_steps - flags.warmup_steps,
        alpha=flags.min_lr_ratio)
  else:
    raise ValueError(flags.decay_method)

  learning_rate = tf.where(global_step < flags.warmup_steps,
                           warmup_lr, decay_lr)

  if flags.weight_decay == 0:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        epsilon=flags.adam_epsilon)
  elif flags.weight_decay > 0 and flags.num_core_per_host == 1:
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        epsilon=flags.adam_epsilon,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        weight_decay_rate=flags.weight_decay)
  else:
    raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                     "training so far.")

  if flags.use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  if tvars is None:
    grads_and_vars = optimizer.compute_gradients(total_loss)
  else:
    grads_and_vars = optimizer.compute_gradients(total_loss,
                                                 var_list=tvars)
  gradients, variables = zip(*grads_and_vars)
  clipped, gnorm = tf.clip_by_global_norm(gradients, flags.clip)

  train_op = optimizer.apply_gradients(
      zip(clipped, variables), global_step=global_step)

  # Manually increment `global_step` for AdamWeightDecayOptimizer
  if isinstance(optimizer, AdamWeightDecayOptimizer):
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  if ema is not None:
    # Update the variables with the EMA after the train op.
    with tf.control_dependencies([train_op]):
      train_op = ema.apply(tf.trainable_variables())
  return train_op, learning_rate, gnorm


def construct_scalar_host_call(monitor_dict, model_dir,
                               prefix="", reduce_fn=None):
  """Construct host calls to monitor training progress on TPUs."""

  metric_names = list(monitor_dict.keys())

  def host_call_fn(global_step, *args):
    """actual host call function."""
    step = global_step[0]
    with tf2.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf2.summary.record_if(True):
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          with tf2.summary.record_if(lambda: math_ops.equal(step % 100, 0)):
            tf.summary.scalar(prefix + name, scalar, step=step)

        return tf.summary.all_v2_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(monitor_dict[key], [1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def build_lstm(num_units):
  return contrib_rnn.LSTMCell(
      num_units=num_units,
      initializer=initializers.xavier_initializer(),
      state_is_tuple=True,
      activation=tf.tanh)


def print_tensors(**tensors):
  """Host call function to print Tensors from the TPU during training."""
  tf.logging.info("Print these tensors: %s", tensors)
  first_n = 25
  print_op = tf.no_op()
  for name in sorted(tensors):
    with tf.control_dependencies([print_op]):
      tensor = tensors[name]
      print_op = tf.Print(
          tensor, [tensor],
          message=name + "=",
          first_n=first_n,
          summarize=1024 * 2 * 8)
  with tf.control_dependencies([print_op]):
    return tf.Print(0., [0.], message="------", first_n=first_n)


def get_assignment_map_from_checkpoint(vars_to_restore, init_checkpoint,
                                       bert_prefix=""):
  """Compute the union of the current variables and checkpoint variables.

  A fork from BERT's modeling.py.
  the original implementation returned a map of string values, this triggers
  issues that can mitigated by switching values of the map to the variables
  themselves.

  Args:
    vars_to_restore: a list of variables to be initialized from the checkpoint.
    init_checkpoint: a path to the model checkpoint.
    bert_prefix: The string prefix of BERT variables.

  Returns:
    a tuple of two maps:
      - assignment map: lookup name in checkpoint to variable to initialize.
      - initialized_variable_names: a map of names of variables to be
        initialized from the checkpoint.
  """

  name_to_variable = collections.OrderedDict()
  for var in vars_to_restore:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)  # change /scope/var1:0 to /scope/var1
    if name.startswith(bert_prefix + "bert/"):
      name = name[len(bert_prefix):]
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  initialized_variable_names = {}
  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name_to_variable[name]
    init_var_name = name_to_variable[name].name
    initialized_variable_names[init_var_name] = 1
    initialized_variable_names[init_var_name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
