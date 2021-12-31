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

"""Helper utils for training on TPU using the Estimator framework.

The functions in this build estimator objects and invoke training loops, on TPU
and on GPU/CPU.

Since exporting summaries is a bit involved on TPU, we use a `summarizer`
training loop instead. This loop is intended to run on CPU, load the checkpoints
written by the TPU trainer, perform a single training step and write the
summaries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import flags
from absl import logging
import six
import tensorflow.compat.v1 as tf

from depth_and_motion_learning import maybe_summary
from depth_and_motion_learning.parameter_container import ParameterContainer
from tensorflow.contrib import estimator as contrib_estimator

FLAGS = flags.FLAGS


flags.DEFINE_string('master', '', 'TensorFlow session address.')

flags.DEFINE_string('model_dir', '', 'Directory where the model is saved.')

flags.DEFINE_string('param_overrides', '', 'Parameters for the trainer and the '
                    'model')

# Defaults for various parameters common to training configurations.

TRAINER_PARAMS = {
    # Learning rate
    'learning_rate': 2e-4,

    # If not None, gradients will be clipped to this value.
    'clip_gradients': 10.0,

    # Number of iterations in the TPU internal on-device loop.
    'iterations_per_loop': 20,

    # If not None, the training will be initialized form this checkpoint.
    'init_ckpt': None,

    # A string, specifies the format of a checkpoint form which to initialize.
    # The model code is expected to convert this string into a
    # vars_to_restore_fn (see below),
    'init_ckpt_type': None,

    # Master address
    'master': None,

    # Directory where checkpoints will be saved.
    'model_dir': None,

    # Maximum number of training steps.
    'max_steps': int(1e6),

    # Number of hours between each checkpoint to be saved.
    # The default value of 10,000 hours effectively disables the feature.
    'keep_checkpoint_every_n_hours': 10000,
}


class InitFromCheckpointHook(tf.estimator.SessionRunHook):
  """A hook for initializing training from a checkpoint.

  Although the Estimator framework supports initialization from a checkpoint via
  https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings,
  the only way to build mapping between the variables and the checkpoint names
  is via providing a regex. This class provides the same functionality, but the
  mapping can be built by a callback, which provides more flexibility and
  readability.
  """

  def __init__(self, model_dir, ckpt_to_init_from, vars_to_restore_fn=None):
    """Creates an instance.

    Args:
      model_dir: A string, path where checkpoints are saved during training.
        Used for checking whether a checkpoint already exists there, in which
        case we want to continue training from there rather than initialize from
        another checkpoint.
      ckpt_to_init_from: A string, path to a checkpoint to initialize from.
      vars_to_restore_fn: A callable that receives no arguments. When called,
        expected to provide a dictionary that maps the checkpoint name of each
        variable to the respective variable object. This dictionary will be used
        as `var_list` in a Saver object used for initializing from
        `ckpt_to_init_from`. If None, the default saver will be used.
    """
    self._ckpt = None if tf.train.latest_checkpoint(
        model_dir) else ckpt_to_init_from
    self._vars_to_restore_fn = vars_to_restore_fn

  def begin(self):
    if not self._ckpt:
      return
    logging.info('%s will be used for initialization.', self._ckpt)
    # Build a saver object for initializing from a checkpoint, or use the
    # default one if no vars_to_restore_fn was given.
    self._reset_step = None
    if tf.train.get_global_step() is not None:
      self._reset_step = tf.train.get_global_step().assign(0)
    if not self._vars_to_restore_fn:
      logging.info('All variables will be initialized form the checkpoint.')
      self._saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
      return

    vars_to_restore = self._vars_to_restore_fn()
    restored_vars_string = (
        'The following variables are to be initialized from the checkpoint:\n')
    for ckpt_name in sorted(vars_to_restore):
      restored_vars_string += '%s --> %s\n' % (
          ckpt_name, vars_to_restore[ckpt_name].op.name)

    logging.info(restored_vars_string)
    self._saver = tf.train.Saver(vars_to_restore)

  def after_create_session(self, session, coord):
    del coord  # unused
    if not self._ckpt:
      return
    self._saver.restore(session, self._ckpt)
    self._saver.restore(session, self._ckpt)
    if self._reset_step is not None:
      session.run(self._reset_step)


def _build_estimator_spec(losses, trainer_params, mode, use_tpu=False):
  """Builds an EstimatorSpec/TPUEstimatorSpec based on trainer_params.

  Args:
    losses: A dictionary of {string: tf.Tensor} containing the various losses.
      The keys will be used as display names for the summaries, the values will
      be summed up to obtain the total loss, which is to be minimized.
    trainer_params: A ParameterContainer object with parameters relevant to the
      training.
    mode: One of tf.estimator.ModeKeys: TRAIN, PREDICT or EVAL.
    use_tpu: A boolean, if True, a TPU-compatible version of EstimatorSpec will
      be built.

  Returns:
    A EstimatorSpec or a TPUEstimatorSpec object.
  """
  if mode == tf.estimator.ModeKeys.TRAIN:
    total_loss = 0.0
    for loss_name, loss in six.iteritems(losses):
      if not use_tpu:
        tf.summary.scalar('Loss/%s' % loss_name, loss)
      total_loss += loss

    learning_rate = trainer_params.learning_rate
    maybe_summary.scalar('Learning Rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)
    optimizer = contrib_estimator.clip_gradients_by_norm(
        optimizer, trainer_params.clip_gradients)

    if use_tpu:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.minimize(
        total_loss, global_step=tf.train.get_global_step())
  else:
    total_loss = None
    train_op = None

  if use_tpu:
    estimator_spec = tf.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN, loss=total_loss, train_op=train_op)
  else:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN, loss=total_loss, train_op=train_op)

  return estimator_spec


def run_local_training(losses_fn,
                       input_fn,
                       trainer_params_overrides,
                       model_params,
                       vars_to_restore_fn=None):
  """Run a simple single-mechine traing loop.

  Args:
    losses_fn: A callable that receives two arguments, `features` and `params`,
      both are dictionaries, and returns a dictionary whose values are the
      losses. Their sum is the total loss to be minimized.
    input_fn: A callable that complies with tf.Estimtor's definition of
      input_fn.
    trainer_params_overrides: A dictionary or a ParameterContainer with
      overrides for the default values in TRAINER_PARAMS above.
    model_params: A ParameterContainer that will be passed to the model (i. e.
      to losses_fn and input_fn).
    vars_to_restore_fn: A callable that receives no arguments. When called,
      expected to provide a dictionary that maps the checkpoint name of each
      variable to the respective variable object. This dictionary will be used
      as `var_list` in a Saver object used for initializing from the checkpoint
      at trainer_params.init_ckpt. If None, the default saver will be used.
  """
  trainer_params = ParameterContainer.from_defaults_and_overrides(
      TRAINER_PARAMS, trainer_params_overrides, is_strict=True)

  run_config_params = {
      'model_dir':
          trainer_params.model_dir,
      'save_summary_steps':
          50,
      'keep_checkpoint_every_n_hours':
          trainer_params.keep_checkpoint_every_n_hours,
      'log_step_count_steps':
          50,
  }
  logging.info(
      'Estimators run config parameters:\n%s',
      json.dumps(run_config_params, indent=2, sort_keys=True, default=str))
  run_config = tf.estimator.RunConfig(**run_config_params)

  def estimator_spec_fn(features, labels, mode, params):
    del labels  # unused
    return _build_estimator_spec(
        losses_fn(features, mode, params),
        trainer_params=trainer_params,
        mode=mode,
        use_tpu=False)

  init_hook = InitFromCheckpointHook(trainer_params.model_dir,
                                     trainer_params.init_ckpt,
                                     vars_to_restore_fn)

  estimator = tf.estimator.Estimator(
      model_fn=estimator_spec_fn,
      config=run_config,
      params=model_params.as_dict())

  estimator.train(
      input_fn=input_fn, max_steps=trainer_params.max_steps, hooks=[init_hook])


def train(input_fn, loss_fn, get_vars_to_restore_fn=None):
  """Run training.

  Args:
    input_fn: A tf.Estimator compliant input_fn.
    loss_fn: a callable with the signature loss_fn(features, mode, params),
      where `features` is a dictionary mapping strings to tf.Tensors, `mode` is
      a tf.estimator.ModeKeys (TRAIN, EVAL, PREDICT), and `params` is a
      dictionary mapping strings to hyperparameters (could be nested). It
      returns a dictionary mapping strings to scalar tf.Tensor-s representing
      losses. Their sum is the total training loss.
    get_vars_to_restore_fn: A callable that receives a string argument
      (intdicating the type of initialization) and returns a vars_to_restore_fn.
      The latter is a callable that receives no arguments and returns a
      dictionary that can be passed to a tf.train.Saver object's constructor as
      a `var_list` to indicate which variables to load from what names in the
      checnpoint.
  """
  params = ParameterContainer({
      'model': {
          'batch_size': 16,
          'input': {}
      },
  }, {'trainer': {
      'master': FLAGS.master,
      'model_dir': FLAGS.model_dir
  }})

  params.override(FLAGS.param_overrides)

  init_ckpt_type = params.trainer.get('init_ckpt_type')

  if init_ckpt_type and not get_vars_to_restore_fn:
    raise ValueError(
        'An init_ckpt_type was specified (%s), but no get_vars_to_restore_fn '
        'was provided.' % init_ckpt_type)

  vars_to_restore_fn = (
      get_vars_to_restore_fn(init_ckpt_type) if init_ckpt_type else None)

  logging.info(
      'Starting training with the following parameters:\n%s',
      json.dumps(params.as_dict(), indent=2, sort_keys=True, default=str))

  run_local_training(loss_fn, input_fn, params.trainer, params.model,
                     vars_to_restore_fn)
