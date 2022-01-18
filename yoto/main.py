# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Train and eval yoto models.

The script works in one of two modes (passed as --schedule). It can either
train, which instantiates the models and calls TPUEstimator.train, or run in
eval mode, which waits for checkpoints to be produced, saves them to tfhub
modules, computes accuracies and logs them.
"""
import enum
import functools
import os.path

from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf.external_configurables
import tensorflow.compat.v1 as tf

# We need to import them so that gin can discover them.
from yoto import architectures  # pylint: disable=unused-import
from yoto import optimizers
from yoto import problems  # pylint: disable=unused-import
from yoto.optimizers import distributions
from yoto.utils import data
from yoto.utils import preprocessing
from tensorflow.python.training import checkpoint_utils  # pylint: disable=g-direct-tensorflow-import


FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Where to store files.")
flags.DEFINE_string(
    "schedule", "train", "Schedule to run. Options: 'train' and 'eval'.")
flags.DEFINE_multi_string(
    "gin_config", [],
    "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Newline separated list of Gin parameter bindings.")
flags.DEFINE_bool("use_tpu", False, "Whether running on TPU or not.")
flags.DEFINE_integer("seed", 0, "The random seed.")
flags.DEFINE_integer("validation_percent", 20,
                     "Percent of training data to be used for validation.")
flags.DEFINE_string(
    "master", None, "Name of the TensorFlow master to use. Defaults to GPU.")


@gin.constants_from_enum
class Task(enum.Enum):
  VARIATIONAL_AUTOENCODER = 0


@gin.configurable("TrainingParams")
class TrainingParams(object):
  """Parameters for network training.

  Includes learning rate with schedule and weight decay.
  """

  def __init__(self, initial_lr=gin.REQUIRED, lr_decay_factor=gin.REQUIRED,
               lr_decay_steps_str=gin.REQUIRED, weight_decay=gin.REQUIRED):
    self.initial_lr = initial_lr
    self.lr_decay_factor = lr_decay_factor
    self.lr_decay_steps_str = lr_decay_steps_str
    self.weight_decay = weight_decay


def iterate_checkpoints_until_file_exists(checkpoints_dir,
                                          path_to_file,
                                          timeout_in_mins=60):
  """Yields checkpoints as long as the file does not exist."""
  remaining_mins = timeout_in_mins
  last_checkpoint = None
  while remaining_mins > 0:
    checkpoint = checkpoint_utils.wait_for_new_checkpoint(
        checkpoints_dir, last_checkpoint=last_checkpoint, timeout=60)  # 1 min.
    if checkpoint:
      last_checkpoint = checkpoint
      remaining_mins = timeout_in_mins  # Reset the remaining time.
      yield checkpoint
    elif tf.gfile.Exists(path_to_file):
      logging.info("Found %s, exiting", path_to_file)
      return
    else:
      remaining_mins -= 1


def get_decay_op(weight_decay, learning_rate, opt_step, vars_to_decay=None):
  """Generates the weight decay op for the given variables."""
  with tf.control_dependencies([opt_step]):
    if vars_to_decay is None:
      vars_to_decay = tf.trainable_variables()
    decay_ops = []
    for v in vars_to_decay:
      decayed_val = v * (1. - learning_rate * weight_decay)
      decay_ops.append(v.assign(decayed_val))
    decay_op = tf.group(decay_ops)
  return decay_op


def get_learning_rate(training_params):
  """Produces piece-wise learning rate tensor that decays exponentially.

  Args:
    training_params: TrainingParams instance.
      training_params.initial_lr: initial learning rate.
      training_params.lr_decay_steps_str: a list of step numbers, for which
        learning rate decay should be performed.
      training_params.lr_decay_factor: learning rate decay factor.

  Returns:
    lr: Learning rate tensor that decays exponentially according to given
      parameters.
  """

  initial_lr = training_params.initial_lr
  lr_decay_factor = training_params.lr_decay_factor
  lr_decay_steps_str = training_params.lr_decay_steps_str
  if lr_decay_steps_str:
    global_step = tf.train.get_or_create_global_step()
    lr_decay_steps = [int(s) for s in lr_decay_steps_str.split(",")]

    lr = tf.train.piecewise_constant(
        global_step,
        lr_decay_steps,
        [initial_lr * (lr_decay_factor ** i)
         for i in range(len(lr_decay_steps) + 1)]
    )
  else:
    lr = initial_lr
  return lr


def get_optimizer(optimizer_class, learning_rate, use_tpu):
  optimizer = optimizer_class(learning_rate=learning_rate)
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  return optimizer


def construct_model_fn(problem, optimizer_class, base_optimizer_class,
                       eval_weights=None, eval_num_samples=10,
                       training_params_class=None,
                       training_params_conditioning_class=None,
                       base_optimizer_conditioning_class=None):
  """Constructs a model_fn for the given problem and optimizer.

  Args:
    problem: An instance of the Problem class, defining the learning problem.
    optimizer_class: MultiLossOptimizer class (gin-injected), used to generate
      an instance used to optimize the problem. This optimizer handles
      problems with parametrized loss functions.
    base_optimizer_class: A tf.Optimizer class (gin-injected), used to create
      an optimizer instance which is actually used to minimize the objective.
    eval_weights: a specification of eval_weights, either as a random
      distribution or as a list of weight dictionaries (see
      distributions.get_samples_as_dicts for details)
    eval_num_samples: Int. If eval_weights are given as a distribution, this
      defines how many vectors to sample from it for evaluation.
    training_params_class: TrainingParams class (gin_injected). Stores training
      parameters (learning rate parameters as in get_learning_rate(...) and
      weight_decay).
    training_params_conditioning_class: TrainingParams class (gin_injected).
      Same as training_params_class, but, if provided, to be used for the
      conditioning part of the network.
    base_optimizer_conditioning_class: A tf.Optimizer class (gin-injected).
      If proivided, used to create an optimizer instance that minimizes the
      objective for the conditioning variables.

  Returns:
    model_fn: A function that creates a model, to be used by TPU Estimator.
  """
  def model_fn(features, mode, params):
    """Returns a TPU estimator spec for the task at hand."""
    problem.initialize_model()
    optimizer = optimizer_class(problem, batch_size=params["batch_size"])
    training_params = training_params_class()
    learning_rate_normal = get_learning_rate(training_params)
    separate_conditioning_optimizer = (
        training_params_conditioning_class and base_optimizer_conditioning_class
        and isinstance(optimizer,
                       optimizers.MultiLossOptimizerWithConditioning))
    if not separate_conditioning_optimizer and (
        training_params_conditioning_class
        or base_optimizer_conditioning_class):
      raise ValueError("training_params_conditioning_class and "
                       "base_optimizer_conditioning_class should be provided "
                       "together and only when the optimizer is "
                       "MultiLossOptimizerWithConditioning.")

    tf.logging.info("separate_conditioning_optimizer: %s",
                    separate_conditioning_optimizer)

    if separate_conditioning_optimizer:
      training_params_conditioning = training_params_conditioning_class()
      learning_rate_conditioning = get_learning_rate(
          training_params_conditioning)

    if mode == tf.estimator.ModeKeys.TRAIN:

      base_optimizer = get_optimizer(base_optimizer_class, learning_rate_normal,
                                     params["use_tpu"])
      if separate_conditioning_optimizer:
        base_optimizer_conditioning = get_optimizer(
            base_optimizer_conditioning_class, learning_rate_conditioning,
            params["use_tpu"])
        loss, opt_step = optimizer.compute_train_loss_and_update_op(
            features, base_optimizer, base_optimizer_conditioning)
        all_vars_str = "\n".join([str(v) for v in optimizer.all_vars])
        normal_vars_str = "\n".join([str(v) for v in optimizer.normal_vars])
        conditioning_vars_str = "\n".join([str(v) for
                                           v in optimizer.conditioning_vars])
        tf.logging.info("\n\nall_vars\n %s", all_vars_str)
        tf.logging.info("\n\nnormal_vars\n %s", normal_vars_str)
        tf.logging.info("\n\nconditioning_vars\n %s", conditioning_vars_str)
      else:
        loss, opt_step = optimizer.compute_train_loss_and_update_op(
            features, base_optimizer)

      # weight decay op
      decay_op = get_decay_op(training_params.weight_decay,
                              learning_rate_normal, opt_step,
                              vars_to_decay=optimizer.normal_vars)
      if separate_conditioning_optimizer:
        decay_op_conditioning = get_decay_op(
            training_params_conditioning.weight_decay,
            learning_rate_conditioning,
            opt_step, vars_to_decay=optimizer.conditioning_vars)
        decay_op = tf.group([decay_op, decay_op_conditioning])
      # batch norm update ops
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = tf.group([opt_step, decay_op] + update_ops)
      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def unstack_metrics(**metrics):
        """Unstack separate metrics from one big aggregate tensor.

        This is needed because otherwise evaluation on TPU with many metrics
        gets horribly slow. Concatenating all metrics into one tensor makes
        things much better.

        Args:
          **metrics: Dict[ Str: tf.Tensor ]. Dictionary with one element, for
            which the key the concatenation of all metric names separated by "!"
            and the value are all metric values stacked along axis 1.

        Returns:
          metrics_dict: Dict[ Str: tf.Tensor ]. Dictionary mapping metrics names
            to tensors with their per-sample values.
        """
        if len(metrics) != 1:
          raise ValueError("Stacked metrics dict should have one element, got "
                           "{}".format(len(metrics)))
        names_stacked = list(metrics.keys())[0]
        values_stacked = metrics[names_stacked]
        names = names_stacked.split("!")
        values = tf.unstack(values_stacked, axis=1)
        return {name: tf.metrics.mean(value) for name, value in
                zip(names, values)}

      loss = optimizer.compute_eval_loss(features)

      if isinstance(optimizer, optimizers.MultiLossOptimizerWithConditioning):
        sampled_weights = distributions.get_samples_as_dicts(
            eval_weights, num_samples=eval_num_samples,
            names=problem.losses_keys, seed=17)
        all_metrics = {}
        for idx, weights in enumerate(sampled_weights):
          with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            losses_id, metrics_id = \
                optimizer.compute_eval_losses_and_metrics_for_weights(features,
                                                                      weights)
          all_metrics.update({"{}/{}".format(key, idx): value
                              for key, value in losses_id.items()})
          all_metrics.update({"{}/{}".format(key, idx): value
                              for key, value in metrics_id.items()})
          full_loss = 0.
          for loss_name in losses_id.keys():
            full_loss += weights[loss_name] * losses_id[loss_name]
          all_metrics.update({"full_loss/{}".format(idx): full_loss})
      else:
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
          losses, metrics = problem.losses_and_metrics(features, training=False)
        all_metrics = losses
        all_metrics.update(metrics)
      metrics_shape_out = all_metrics[list(all_metrics.keys())[0]].get_shape()
      # Need this broadcasting because on TPU all output tensors should have
      # the same shape
      all_metrics.update(
          {"learning_rate_normal": tf.broadcast_to(
              learning_rate_normal, metrics_shape_out)})
      if separate_conditioning_optimizer:
        all_metrics.update(
            {"learning_rate_conditioning": tf.broadcast_to(
                learning_rate_conditioning, metrics_shape_out)})
      # Stacking all metrics for efficiency (otherwise eval is horribly slow)
      sorted_keys = sorted(all_metrics.keys())
      sorted_values = [all_metrics[key] for key in sorted_keys]
      metrics_stacked = {"!".join(sorted_keys): tf.stack(sorted_values, axis=1)}
      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(unstack_metrics, metrics_stacked))
    else:
      raise ValueError("Unknown mode: {}".format(mode))

  return model_fn


@gin.configurable("experiment")
def run(model_dir,
        schedule,
        problem_class=gin.REQUIRED,
        optimizer_class=gin.REQUIRED,
        dataset_name=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        eval_batch_size=64,
        train_steps=gin.REQUIRED,
        eval_steps=gin.REQUIRED,
        base_optimizer_class=gin.REQUIRED,
        base_optimizer_conditioning_class=None,
        iterations_per_loop=gin.REQUIRED,
        eval_weights=None,
        training_params_class=gin.REQUIRED,
        training_params_conditioning_class=None,
        preprocess="",
        preprocess_eval="",
        save_checkpoints_steps=None,
        keep_checkpoint_max=0,
        eval_on_test=False):
  """Main training function. Most of the parameters come from Gin."""
  assert schedule in ("train", "eval")

  if save_checkpoints_steps:
    kwargs = {"save_checkpoints_steps": save_checkpoints_steps}
  else:
    kwargs = {"save_checkpoints_secs": 60*10}  # Every 10 minutes.

  run_config = tf.estimator.tpu.RunConfig(
      keep_checkpoint_max=keep_checkpoint_max,
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop),
      **kwargs)
  # We use one estimator (potentially on TPU) for training and evaluation.
  problem = problem_class()
  model_fn = construct_model_fn(
      problem, optimizer_class, base_optimizer_class,
      eval_weights=eval_weights,
      base_optimizer_conditioning_class=base_optimizer_conditioning_class,
      training_params_class=training_params_class,
      training_params_conditioning_class=training_params_conditioning_class)
  tpu_estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      model_dir=model_dir,
      train_batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      config=run_config)


  def input_fn_train(params):
    preprocess_fn = preprocessing.get_preprocess_fn(preprocess)
    return data.get_dataset(dataset_name, data.DatasetSplit.TRAIN,
                            FLAGS.validation_percent, params["batch_size"],
                            preprocess_fn)

  def input_fn_eval(params, split):
    preprocess_fn = preprocessing.get_preprocess_fn(preprocess_eval)
    return data.get_dataset(dataset_name, split, FLAGS.validation_percent,
                            params["batch_size"], preprocess_fn).repeat()

  path_to_finished_file = os.path.join(model_dir, "FINISHED")
  if schedule == "train":
    gin_hook = gin.tf.GinConfigSaverHook(model_dir, summarize_config=True)
    tpu_estimator.train(input_fn=input_fn_train,
                        hooks=[gin_hook],
                        max_steps=train_steps)
    with tf.gfile.GFile(path_to_finished_file, "w") as finished_file:
      finished_file.write("1")
  else:
    for checkpoint in iterate_checkpoints_until_file_exists(
        model_dir, path_to_finished_file):
      if eval_on_test:
        train_split = data.DatasetSplit.TRAIN_FULL
        test_split = data.DatasetSplit.TEST
        test_summary_name = "test"
      else:
        train_split = data.DatasetSplit.TRAIN
        test_split = data.DatasetSplit.VALID
        test_summary_name = "valid"

      eval_train = tpu_estimator.evaluate(
          input_fn=functools.partial(input_fn_eval, split=train_split),
          checkpoint_path=checkpoint,
          steps=eval_steps,
          name="train")
      eval_test = tpu_estimator.evaluate(
          input_fn=functools.partial(input_fn_eval, split=test_split),
          checkpoint_path=checkpoint,
          steps=eval_steps,
          name="test")

      current_step = eval_train["global_step"]


      hub_modules_dir = os.path.join(model_dir, "hub_modules")
      if not tf.gfile.Exists(hub_modules_dir):
        tf.gfile.MkDir(hub_modules_dir)
      else:
        if not tf.gfile.IsDirectory(hub_modules_dir):
          raise ValueError("{0} exists and is not a directory".format(
              hub_modules_dir))

      hub_module_path = os.path.join(hub_modules_dir,
                                     "step-{:0>9}".format(current_step))
      if not tf.gfile.Exists(hub_module_path):
        problem.module_spec.export(hub_module_path,
                                   checkpoint_path=checkpoint)
      else:
        logging.info("Not saving the hub module, since the path"
                     " %s already exists", hub_module_path)


def main(argv):
  if len(argv) > 1:
    raise ValueError("Too many command-line arguments.")
  logging.info("Gin config: %s\nGin bindings: %s",
               FLAGS.gin_config, FLAGS.gin_bindings)
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  run(model_dir=FLAGS.model_dir, schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  flags.mark_flag_as_required("model_dir")
  app.run(main)
