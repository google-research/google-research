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

"""Utils for TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib import tpu as contrib_tpu

gfile = tf.gfile


USE_MOVING_AVERAGE = 'USE_MOVING_AVERAGE'


def get_lr(curr_step, params):
  """Compute learning rate at step depends on `params`."""
  lr = tf.constant(params.learning_rate, dtype=tf.float32)
  if 'num_warmup_steps' in params and params.num_warmup_steps > 0:
    num_warmup_steps = tf.cast(params.num_warmup_steps, dtype=tf.float32)
    step = tf.cast(curr_step, dtype=tf.float32)
    warmup_lr = params.learning_rate * step / num_warmup_steps
    lr = tf.cond(tf.less(step, num_warmup_steps), lambda: warmup_lr, lambda: lr)
  return lr


def strip_var_name(var_name):
  """Strips variable name of sub-strings blocking variable name matching."""
  # Strip trailing number, e.g. convert
  # 'lstm/W_0:0' to 'lstm/W_0'.
  var_name = re.sub(r':\d+$', '', var_name)
  # Strip partitioning info, e.g. convert
  # 'W_0/part_3/Adagrad' to 'W_0/Adagrad'.
  var_name = re.sub(r'/part_\d+', '', var_name)
  return var_name


def create_estimator(params, model_dir, model_fn):
  """Create a `TPUEstimator`."""

  tpu_config = contrib_tpu.TPUConfig(
      iterations_per_loop=params.save_every,
      num_cores_per_replica=2,
      per_host_input_for_training=contrib_tpu.InputPipelineConfig.PER_HOST_V2,  # pylint: disable=line-too-long
      input_partition_dims=[{
          'x': [1, 2],
          'y': [1, 2]
      }, None],
      tpu_job_name=params.tpu_job_name,
  )

  session_config = tf.ConfigProto(
      operation_timeout_in_ms=int(6e9),
      allow_soft_placement=True,
      isolate_session_state=True)

  run_config = contrib_tpu.RunConfig(
      tpu_config=tpu_config,
      master=params.master,
      session_config=session_config,
      log_step_count_steps=None,
      keep_checkpoint_max=5,
      save_checkpoints_steps=params.save_every)

  estimator = contrib_tpu.TPUEstimator(
      model_fn=model_fn,
      model_dir=model_dir,
      train_batch_size=params.train_batch_size,
      eval_batch_size=params.eval_batch_size,
      config=run_config,
      params=params,
      use_tpu=params.use_tpu,
      eval_on_tpu=True)

  return estimator


def build_host_call_fn(params, names_and_tensors):
  """Wrapper to build `host_call` for `TPUEstimator`.

  Args:
    params: a `tf.contrib.train.HParams` object.
    names_and_tensors: list of elemens such as
      `("loss", loss)`. These are the tensors' names and values.

  Returns:
    A pair of `(host_call_fn, tensors)` for `TPUEstimatorSpec`.
  """

  names, tensors = zip(*names_and_tensors)

  def host_call_fn(global_step, *tensors):
    """Training host call."""
    global_step = global_step[0]
    with contrib_summary.create_file_writer(params.output_dir).as_default():
      with contrib_summary.record_summaries_every_n_global_steps(
          n=params.log_every, global_step=global_step):
        for i, tensor in enumerate(tensors):
          if 'images' not in names[i]:
            contrib_summary.scalar(names[i], tensor[0], step=global_step)
        return contrib_summary.all_summary_ops()

  global_step = tf.reshape(tf.train.get_or_create_global_step(), [1])
  tensors = [tf.expand_dims(tf.cast(t, dtype=tf.float32), axis=0)
             for t in tensors]

  return (host_call_fn, [global_step] + tensors)
