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

"""APIs for building host call function for TF estimators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
import tensorflow.compat.v1 as tf
from tensorflow.contrib import summary as contrib_summary


@gin.configurable
def build_host_call_fn_every_n_global_steps(
    params,
    names_and_tensors,
    n,
    summary_dir=None):
  """Wrapper to build `host_call` for `TPUEstimator`.

  This function records the summaries if global_step % n == 0

  Args:
    params: A `tf.contrib.train.HParams` object.
    names_and_tensors: List of elemens such as `("loss", loss)`. These are the
      tensors' names and values.
    n: Defines the frequency of recording the summaries.
          Performance-wise on TPU, it is better to set n equal to
          the number of iterations per loop.
       In TPU, each training loop (each call to estimator.train)
         consists of multiple iterations. There is a communication overhead
         between host and TPU per training loop to send/receive data.
         As such, it is better not to interrupt the TPU loop for saving
         the summaries. You may also need to save the summaries
         after multiple training loops.
    summary_dir: Summary directory used to store TF summaries.

  Returns:
    A pair of `(host_call_fn, tensors)` for `TPUEstimatorSpec`.
  """
  del params
  assert summary_dir, 'Please specify a directory for summaries.'

  names, tensors = zip(*names_and_tensors)

  def host_call_fn(global_step, *tensors):
    """Training host call."""
    global_step = global_step[0]
    with contrib_summary.create_file_writer(summary_dir +
                                            '/metrics').as_default():
      with contrib_summary.record_summaries_every_n_global_steps(
          n=n, global_step=global_step):
        for i, tensor in enumerate(tensors):
          contrib_summary.scalar(names[i], tensor[0], step=global_step)
        return contrib_summary.all_summary_ops()

  global_step = tf.reshape(tf.train.get_or_create_global_step(), [1])
  tensors = [
      tf.expand_dims(tf.cast(t, dtype=tf.float32), axis=0) for t in tensors
  ]
  return (host_call_fn, [global_step] + tensors)


@gin.configurable
def build_host_call_fn(
    params,
    names_and_tensors,
    summary_dir=None):
  """Wrapper to build `host_call` for `TPUEstimator`.

  Adopted from: experimental/users/hyhieu/patch_based_unsup/utils.py

  Args:
    params: A `tf.contrib.train.HParams` object.
    names_and_tensors: List of elemens such as `("loss", loss)`. These are the
      tensors' names and values.
    summary_dir: Summary directory used to store TF summaries.

  Returns:
    A pair of `(host_call_fn, tensors)` for `TPUEstimatorSpec`.
  """
  del params
  assert summary_dir, 'Please specify a directory for summaries.'

  names, tensors = zip(*names_and_tensors)

  def host_call_fn(global_step, *tensors):
    """Training host call."""
    global_step = global_step[0]
    with contrib_summary.create_file_writer(summary_dir +
                                            '/metrics').as_default():
      with contrib_summary.always_record_summaries():
        for i, tensor in enumerate(tensors):
          contrib_summary.scalar(names[i], tensor[0], step=global_step)
        return contrib_summary.all_summary_ops()

  global_step = tf.reshape(tf.train.get_or_create_global_step(), [1])
  tensors = [
      tf.expand_dims(tf.cast(t, dtype=tf.float32), axis=0) for t in tensors
  ]
  return (host_call_fn, [global_step] + tensors)
