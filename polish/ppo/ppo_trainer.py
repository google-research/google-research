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

"""Wrapper class around TF estimator to perform training."""
import gin
import tensorflow.compat.v1 as tf
from polish.utils import tf_utils


@gin.configurable
class PpoTrainer(object):
  """Wrapper class for PPO TF estimator training.

  This class mainly receives any compatible `input_fn` and `model_fn` functions
  for a TF estimator and launches estimator training. `input_fn` is a function
  that feeds dictionary of arrays into the model. `model_fn` is a function that
  defines the network architecture and training operation.
  """

  def __init__(self,
               input_fn,
               model_fn,
               num_iterations=156160,
               iterations_per_loop=320,
               checkpoint_dir=gin.REQUIRED,
               keep_checkpoint_max=20,
               use_tpu=False):
    """Creates a PPO training class.

    Args:
      input_fn: The function to feed in input data during training.
      model_fn: The model to train on.
      num_iterations: The number of iterations to run the training for.
      iterations_per_loop: Number of steps to run on TPU before outfeeding
        metrics to the CPU. If the number of iterations in the loop would exceed
        the number of train steps, the loop will exit before reaching
        --iterations_per_loop. The larger this value is, the higher the
        utilization on the TPU.
      checkpoint_dir: The directory to save checkpoints to.
      keep_checkpoint_max: The maximum number of checkpoints to keep.
      use_tpu: If True, use TPU for model training.
    """
    self._input_fn = input_fn
    self._model_fn = model_fn
    self._num_iterations = num_iterations
    self._iterations_per_loop = iterations_per_loop
    self._checkpoint_dir = checkpoint_dir
    self._keep_checkpoint_max = keep_checkpoint_max
    self._use_tpu = use_tpu

  def get_estimator(self):
    """Obtain estimator for the working directory.

    Returns:
      an (TPU/non-TPU) estimator.
    """
    if self._use_tpu:
      return tf_utils.get_tpu_estimator(self._checkpoint_dir, self._model_fn)

    run_config = tf.estimator.RunConfig(
        save_summary_steps=self._iterations_per_loop,
        save_checkpoints_steps=self._iterations_per_loop,
        keep_checkpoint_max=self._keep_checkpoint_max)
    return tf.estimator.Estimator(
        self._model_fn, model_dir=self._checkpoint_dir, config=run_config)

  def train(self):
    """A wrapper to launch training on the estimator."""
    estimator = self.get_estimator()
    hooks = [self._input_fn]
    estimator.train(
        input_fn=self._input_fn, hooks=hooks, max_steps=self._num_iterations)
