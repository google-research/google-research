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

# Lint as: python2, python3
# pylint: disable=invalid-name,g-bad-import-order,missing-docstring,logging-not-lazy
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import csv
import functools
import gzip
import os
import time

from absl import flags
from absl import logging
import gin
import numpy as np
import pandas as pd
import simplejson
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import typing
from typing import Any, Dict, List, Optional, Tuple
import yaml

tfd = tfp.distributions
tfb = tfp.bijectors


LOGGING_OUTPUTS = "logging"


@gin.configurable("process_gradients")
def ProcessGradients(grads_and_vars,
                     global_gradient_clip = 0.0,
                     sanitize_gradients=False,
                     normalize_gradients=False):
  tf.logging.info("Prcessing gradients")
  grads, vars_ = list(zip(*grads_and_vars))
  if sanitize_gradients:
    new_grads = []
    for g in grads:
      if g is not None:
        g = tf.where(tf.is_finite(g), g, tf.zeros_like(g))
      new_grads.append(g)
    grads = new_grads
  if normalize_gradients:
    new_grads = []
    for g in grads:
      if g is not None:
        g *= tf.rsqrt(tf.maximum(1e-12, tf.reduce_sum(tf.square(g))))
      new_grads.append(g)
    grads = new_grads
  if global_gradient_clip > 0:
    grads, grad_norm = tf.clip_by_global_norm(grads, global_gradient_clip)
    grads_and_vars = list(zip(grads, vars_))
  else:
    grad_norm = tf.global_norm(grads)
  tf.summary.scalar("global_grad_norm", grad_norm)
  return grads_and_vars


def LogAndSummarizeMetrics(metrics,
                           use_streaming_mean = True):
  """Logs and summarizes metrics.

  Metrics are added to the LOGGING_OUTPUTS collection.

  Args:
    metrics: A dictionary of scalar metrics.
    use_streaming_mean: If true, the metrics will be averaged using a running
      mean.

  Returns:
    If use_streaming_mean is true, then this will be the op that you need to
    regularly call to update the running mean. Otherwise, this is a no-op.
  """

  prefix = tf.get_default_graph().get_name_scope()
  if prefix:
    prefix += "/"
  logging_collection = tf.get_collection_ref(LOGGING_OUTPUTS)

  update_ops = [tf.no_op()]
  for name, value in metrics.items():
    if use_streaming_mean:
      value, update_op = tf.metrics.mean(value)
      update_ops.append(update_op)
    logging_collection.append((prefix + name, value))
    tf.summary.scalar(name, value)

  return tf.group(*update_ops)


def GetLoggingOutputs():
  return dict(tf.get_collection(LOGGING_OUTPUTS))


def MakeTFTemplate(function):
  """A decorator that converts a function into a TensorFlow template.

  This is done by calling tf.make_template with the wrapped function. In
  addition to setting the name, this decorator sets the create_scope_now_
  parameter to True, which is a useful default. See tf.make_template
  documentation for more information.

  The wrapper takes the template name as the first argument, with the remaining
  kwargs passed to tf.make_template (which passes them to the wrapped function).
  The wrapper returns the return value of tf.make_template.

  When used with Gin, the template will remember the scope used when
  constructing the template, and then re-apply it when instantiating the
  template. E.g. if the current scope was 'scope' when template was constructed,
  when you call the resulting instance later, the wrapped function will be
  called withing 'scope' config scope.

  Usage:
    @MakeTFTemplate
    def ModelComponent(x, variable_name):
      return x + tf.get_variable(variable_name, initializer=1.0)

    component = ModelComponent('template', variable_name='variable_name')
    output1 = component(tf.constant(1.0))
    output2 = component(tf.constant(2.0))

  Args:
    function: Function to decorate, passed to tf.make_template.

  Returns:
    A decorator applied to the function.
  """

  # No *args because we fully specify them.
  @functools.wraps(function)
  def Wrapper(template_name, **kwargs):
    # Do a bit of hackery with scopes so that the current config scope used when
    # constructing the template is also used when the template is instantiated.
    saved_scopes = gin.current_scope()

    def GinWrapper(*args, **kwargs):
      with gin.config_scope(saved_scopes):
        return function(*args, **kwargs)

    return tf.make_template(
        template_name,
        GinWrapper,
        create_scope_now_=True,
        unique_name_=template_name,
        **kwargs)

  return Wrapper


AISOutputs = collections.namedtuple("AISOutputs", "log_p, p_accept, z_fin")


@gin.configurable("ais")
def AIS(proposal_log_prob_fn,
        target_log_prob_fn,
        z_init,
        num_steps=10000,
        step_size=0.1,
        num_leapfrog_steps=10,
        bijector=None):
  tf.logging.info("About to call tfp ais")

  def MakeKernelFn(tlp_fn):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=tlp_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps)
    if bijector is not None:
      kernel = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=kernel, bijector=bijector)
    return kernel

  z_fin, ais_weights, ais_outputs = tfp.mcmc.sample_annealed_importance_chain(
      num_steps=num_steps,
      proposal_log_prob_fn=proposal_log_prob_fn,
      target_log_prob_fn=target_log_prob_fn,
      current_state=z_init,
      make_kernel_fn=MakeKernelFn)
  tf.logging.info("Tfp ais done")
  log_accept_ratio = ais_outputs.inner_results.log_accept_ratio
  log_accept_ratio = tf.Print(
      log_accept_ratio, [log_accept_ratio], "Log accept\n", summarize=999)
  p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))

  return AISOutputs(log_p=ais_weights, p_accept=p_accept, z_fin=z_fin)


class YAMLDictParser(flags.ArgumentParser):
  syntactic_help = """Expects YAML one-line dictionaries without braces, e.g.
  'key1: val1, key2: val2'."""

  def parse(self, argument):
    return yaml.load("{" + argument + "}")

  def flag_type(self):
    return "Dict[str, Any]"


def BindHParams(hparams):
  """Binds all Gin parameters from a dictionary.

  Args:
    hparams: HParams to bind.
  """
  for k, v in hparams.items():
    gin.bind_parameter(k, v)


def LogAndSaveHParams():
  """Logs and saves the operative parameters to the graph."""
  hparams_str = gin.operative_config_str()
  logging.info("Config:\n%s", hparams_str)
  tf.get_collection_ref("operative_hparams").append(hparams_str)


class Dataset(object):

  @property
  def train_size(self):
    return 0

  @property
  def name(self):
    return ""

  @property
  def test_size(self):
    return 0

  def TrainBatch(self):
    return None

  def TestBatch(self):
    return None

  def AISIterator(self):
    return None


def Binarize(image):
  d = tfd.Bernoulli(probs=image)
  return tf.to_float(d.sample())


def GetBatch(data,
             batch_size,
             shuffle_buffer,
             epochs,
             data_size,
             binarize = True):
  if binarize:
    data = data.map(lambda x, l: (Binarize(x), l), num_parallel_calls=16)
  data = tf.data.Dataset.zip((tf.data.Dataset.range(data_size), data))
  data = data.map(lambda i, x_l: (i, x_l[0], x_l[1]))
  if shuffle_buffer:
    data = data.shuffle(shuffle_buffer)
  data = data.repeat(epochs)
  data = data.batch(batch_size).prefetch(1)
  return data.make_one_shot_iterator().get_next()


class MNISTDataset(Dataset):

  def __init__(self, data_dir, test_is_valid = False):
    self._data_dir = data_dir
    self._test_is_valid = test_is_valid

  @property
  def name(self):
    return "mnist"

  @property
  def train_size(self):
    return 60000

  @property
  def test_size(self):
    return 10000

  def _dataset(self, images_file, labels_file):

    def DecodeImage(image):
      image = tf.decode_raw(image, tf.uint8)
      image = tf.cast(image, tf.float32)
      image = tf.reshape(image, [28, 28, 1])
      return image / 255.0

    def DecodeLabel(label):
      label = tf.decode_raw(label, tf.uint8)
      label = tf.reshape(label, [])
      return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(
            DecodeImage, num_parallel_calls=16)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(
            DecodeLabel, num_parallel_calls=16)
    return tf.data.Dataset.zip((images, labels))

  def TrainBatch(self, batch_size, epochs):
    data = self._dataset(
        os.path.join(self._data_dir, "train-images-idx3-ubyte"),
        os.path.join(self._data_dir, "train-labels-idx1-ubyte"))
    data_idx, images, _ = GetBatch(
        data,
        batch_size=batch_size,
        shuffle_buffer=self.train_size,
        data_size=self.train_size,
        epochs=epochs,
        binarize=True)
    return data_idx, images

  def TestBatch(self, batch_size, binarize=True):
    if self._test_is_valid:
      data = self._dataset(
          os.path.join(self._data_dir, "train-images-idx3-ubyte"),
          os.path.join(self._data_dir, "train-labels-idx1-ubyte"))
      data = data.take(self.test_size)
    else:
      data = self._dataset(
          os.path.join(self._data_dir, "t10k-images-idx3-ubyte"),
          os.path.join(self._data_dir, "t10k-labels-idx1-ubyte"))
    data_idx, images, _ = GetBatch(
        data,
        batch_size=batch_size,
        shuffle_buffer=None,
        data_size=self.test_size,
        epochs=-1,
        binarize=binarize)
    return data_idx, images

  def AISIterator(self, batch_size, shard_idx, num_workers):
    if self._test_is_valid:
      data = self._dataset(
          os.path.join(self._data_dir, "train-images-idx3-ubyte"),
          os.path.join(self._data_dir, "train-labels-idx1-ubyte"))
      data = data.take(self.test_size)
    else:
      data = self._dataset(
          os.path.join(self._data_dir, "t10k-images-idx3-ubyte"),
          os.path.join(self._data_dir, "t10k-labels-idx1-ubyte"))

    # Can this be done more efficiently? Need some sort of dynamic seed PRNG.
    rs = np.random.RandomState(0)
    masks = tf.to_float(rs.rand(self.test_size, 28, 28, 1))
    mask_data = tf.data.Dataset.from_tensor_slices(masks)
    data = tf.data.Dataset.zip((mask_data, data))
    data = data.shard(num_workers, shard_idx)

    def StaticBinarize(mask, image_and_label):
      image, label = image_and_label
      return tf.to_float(tf.to_float(image) > mask), label

    data = data.map(StaticBinarize)
    data = data.batch(batch_size)
    data = data.prefetch(1)
    return data.make_initializable_iterator()


class CIFAR10Dataset(Dataset):

  def __init__(self,
               data_dir,
               test_is_valid = False,
               parallel_interleave = False):
    self._data_dir = data_dir
    self._test_is_valid = test_is_valid
    self._parallel_interleave = parallel_interleave

  @property
  def name(self):
    return "cifar10"

  @property
  def train_size(self):
    return 60000

  @property
  def test_size(self):
    return 10000

  def _dataset(self, data_files,
               parallel_interleave = False):
    if parallel_interleave:
      data_files = tf.data.Dataset.from_tensor_slices(data_files)
      data = data_files.interleave(
          lambda f: tf.data.FixedLengthRecordDataset(f, 1 + 32 * 32 * 3),
          cycle_length=6)
    else:
      data = tf.data.FixedLengthRecordDataset(data_files, 1 + 32 * 32 * 3)

    def DecodeLabelAndImage(r):
      r = tf.decode_raw(r, tf.uint8)
      return tf.to_float(
          tf.transpose(tf.reshape(r[1:], [3, 32, 32]),
                       [1, 2, 0])) / 255.0, tf.to_int32(r[0])

    return data.map(DecodeLabelAndImage)

  def TrainBatch(self, batch_size, epochs):
    data = self._dataset([
        os.path.join(self._data_dir, "data_batch_%d.bin" % i)
        for i in range(1, 6)
    ],
                         parallel_interleave=True)
    data_idx, images, _ = GetBatch(
        data,
        batch_size=batch_size,
        shuffle_buffer=self.train_size,
        data_size=self.train_size,
        epochs=epochs,
        binarize=False)
    return data_idx, images

  def TestBatch(self, batch_size):
    if self._test_is_valid:
      data = self._dataset([
          os.path.join(self._data_dir, "data_batch_%d.bin" % i)
          for i in range(1, 6)
      ],
                           parallel_interleave=True)
      data = data.take(self.test_size)
    else:
      data = self._dataset([os.path.join(self._data_dir, "test_batch.bin")])
    data_idx, images, _ = GetBatch(
        data,
        batch_size=batch_size,
        shuffle_buffer=None,
        data_size=self.test_size,
        epochs=-1,
        binarize=False)
    return data_idx, images

  def AISIterator(self, batch_size, shard_idx, num_workers):
    if self._test_is_valid:
      data = self._dataset([
          os.path.join(self._data_dir, "data_batch_%d.bin" % i)
          for i in range(1, 6)
      ],
                           parallel_interleave=True)
      data = data.take(self.test_size)
    else:
      data = self._dataset([os.path.join(self._data_dir, "test_batch.bin")])

    data = data.shard(num_workers, shard_idx)
    data = data.batch(batch_size)
    data = data.prefetch(1)
    return data.make_initializable_iterator()


class FakeMNISTDataset(Dataset):

  @property
  def name(self):
    return "mnist"

  @property
  def train_size(self):
    return 250

  @property
  def test_size(self):
    return 250

  def _dataset(self, num):
    rs = np.random.RandomState(num)
    images = tf.data.Dataset.from_tensor_slices(rs.rand(num, 28, 28, 1))
    labels = tf.data.Dataset.from_tensor_slices(rs.randint(1, size=(num)))
    return tf.data.Dataset.zip((images, labels))

  def TrainBatch(self, batch_size, epochs):
    data = self._dataset(num=self.train_size)
    data_idx, images, _ = GetBatch(
        data,
        batch_size=batch_size,
        shuffle_buffer=self.train_size,
        data_size=self.train_size,
        epochs=epochs,
        binarize=True)
    return data_idx, images

  def TestBatch(self, batch_size, binarize=True):
    data = self._dataset(num=self.test_size)
    data_idx, images, _ = GetBatch(
        data,
        batch_size=batch_size,
        shuffle_buffer=None,
        data_size=self.test_size,
        epochs=-1,
        binarize=binarize)
    return data_idx, images

  def AISIterator(self, batch_size, shard_idx, num_workers):
    data = self._dataset(num=self.test_size)

    # Can this be done more efficiently? Need some sort of dynamic seed PRNG.
    rs = np.random.RandomState(0)
    masks = tf.to_float(rs.rand(self.test_size, 28, 28, 1))
    mask_data = tf.data.Dataset.from_tensor_slices(masks)
    data = tf.data.Dataset.zip((mask_data, data))
    data = data.shard(num_workers, shard_idx)

    def StaticBinarize(mask, image_and_label):
      image, label = image_and_label
      return tf.to_float(tf.to_float(image) > mask), label

    data = data.map(StaticBinarize)
    data = data.batch(batch_size)
    data = data.prefetch(1)
    return data.make_initializable_iterator()


def SanitizedAutoCorrelation(x, axis, *args, **kwargs):
  res = tfp.stats.auto_correlation(x, axis, *args, **kwargs)
  res = tf.where(tf.is_nan(res), tf.ones_like(res), res)
  res = tf.where(tf.is_inf(res), tf.ones_like(res), res)
  return res


def EffectiveSampleSize(states,
                        filter_beyond_lag=300,
                        filter_threshold=0.05,
                        center=True,
                        normalize=True):
  """ESS computation for one single Tensor argument."""

  def _axis_size(x, axis=None):
    """Get number of elements of `x` in `axis`, as type `x.dtype`."""
    if axis is None:
      return tf.cast(tf.size(x), x.dtype)
    return tf.cast(tf.reduce_prod(tf.gather(tf.shape(x), axis)), x.dtype)

  with tf.name_scope(
      "effective_sample_size_single_state",
      values=[states, filter_beyond_lag, filter_threshold]):

    states = tf.convert_to_tensor(states, name="states")
    dt = states.dtype

    # filter_beyond_lag == None ==> auto_corr is the full sequence.
    auto_corr = SanitizedAutoCorrelation(
        states,
        axis=0,
        center=center,
        normalize=normalize,
        max_lags=filter_beyond_lag)
    auto_corr = tf.reduce_mean(auto_corr, 1)
    if filter_threshold is not None:
      filter_threshold = tf.convert_to_tensor(
          filter_threshold, dtype=dt, name="filter_threshold")
      # Get a binary mask to zero out values of auto_corr below the threshold.
      #   mask[i, ...] = 1 if auto_corr[j, ...] > threshold for all j <= i,
      #   mask[i, ...] = 0, otherwise.
      # So, along dimension zero, the mask will look like [1, 1, ..., 0, 0,...]
      # Building step by step,
      #   Assume auto_corr = [1, 0.5, 0.0, 0.3], and filter_threshold = 0.2.
      # Step 1:  mask = [False, False, True, False]
      mask = tf.abs(auto_corr) < filter_threshold
      # Step 2:  mask = [0, 0, 1, 1]
      mask = tf.cast(mask, dtype=dt)
      # Step 3:  mask = [0, 0, 1, 2]
      mask = tf.cumsum(mask, axis=0)
      # Step 4:  mask = [1, 1, 0, 0]
      mask = tf.maximum(1. - mask, 0.)
      auto_corr *= mask

    # With R[k] := auto_corr[k, ...],
    # ESS = N / {1 + 2 * Sum_{k=1}^N (N - k) / N * R[k]}
    #     = N / {-1 + 2 * Sum_{k=0}^N (N - k) / N * R[k]} (since R[0] = 1)
    #     approx N / {-1 + 2 * Sum_{k=0}^M (N - k) / N * R[k]}
    # where M is the filter_beyond_lag truncation point chosen above.

    # Get the factor (N - k) / N, and give it shape [M, 1,...,1], having total
    # ndims the same as auto_corr
    n = _axis_size(states, axis=0)
    k = tf.range(0., _axis_size(auto_corr, axis=0))
    nk_factor = (n - k) / n
    if auto_corr.shape.ndims is not None:
      new_shape = [-1] + [1] * (auto_corr.shape.ndims - 1)
    else:
      new_shape = tf.concat(
          ([-1], tf.ones([tf.rank(auto_corr) - 1], dtype=tf.int32)), axis=0)
    nk_factor = tf.reshape(nk_factor, new_shape)

    #return tf.reduce_mean(n / (-1 + 2 * tf.reduce_sum(nk_factor * auto_corr, axis=0)), 0)
    return n / (1.0 + 2 * tf.reduce_sum(
        nk_factor[1:, Ellipsis] * auto_corr[1:, Ellipsis], axis=0))
    #return tf.reduce_mean(n / (-auto_corr[0] + 2 * tf.reduce_sum(nk_factor * auto_corr, axis=0)), 0)


@gin.configurable("l2hmc_initializer")
def L2HMCInitializer(factor = 1.0):
  return tf.variance_scaling_initializer(2.0 * factor)


@gin.configurable("dense_shift_log_scale")
@MakeTFTemplate
def DenseShiftLogScale(x,
                       output_units,
                       h=None,
                       hidden_layers=[],
                       activation=tf.nn.relu,
                       log_scale_clip=None,
                       train=False,
                       dropout_rate=0.0,
                       sigmoid_scale=False,
                       log_scale_factor=1.0,
                       log_scale_reg=0.0,
                       *args,
                       **kwargs):
  for units in hidden_layers:
    x = tf.layers.dense(
        inputs=x, units=units, activation=activation, *args, **kwargs)
    if h is not None:
      x += tf.layers.dense(h, units, use_bias=False, *args, **kwargs)
    if dropout_rate > 0:
      x = tf.layers.dropout(x, dropout_rate, training=train)
  if log_scale_factor == 1.0 and log_scale_reg == 0.0:
    x = tf.layers.dense(inputs=x, units=2 * output_units, *args, **kwargs)
    if h is not None:
      x += tf.layers.dense(h, 2 * output_units, use_bias=False, *args, **kwargs)

    shift, log_scale = tf.split(x, 2, axis=-1)
  else:
    shift = tf.layers.dense(h, output_units, *args, **kwargs)
    if log_scale_reg > 0.0:
      regularizer = lambda w: log_scale_reg * 2.0 * tf.nn.l2_loss(w)
    else:
      regularizer = None
    log_scale = tf.layers.dense(
        h,
        output_units,
        use_bias=False,
        kernel_regularizer=regularizer,
        *args,
        **kwargs)
    log_scale = log_scale * log_scale_factor + tf.get_variable(
        "log_scale_bias", [1, output_units], initializer=tf.zeros_initializer())
    if h is not None:
      shift += tf.layers.dense(h, output_units, use_bias=False, *args, **kwargs)
      log_scale += tf.layers.dense(
          h, output_units, use_bias=False, *args, **kwargs)

  if sigmoid_scale:
    log_scale = tf.log_sigmoid(log_scale)

  if log_scale_clip:
    log_scale = log_scale_clip * tf.nn.tanh(log_scale / log_scale_clip)

  return shift, log_scale


@gin.configurable("dense_ar")
@MakeTFTemplate
def DenseAR(x,
            h=None,
            hidden_layers=[],
            activation=tf.nn.relu,
            log_scale_clip=None,
            log_scale_clip_pre=None,
            train=False,
            dropout_rate=0.0,
            sigmoid_scale=False,
            log_scale_factor=1.0,
            log_scale_reg=0.0,
            shift_only=False,
            *args,
            **kwargs):
  input_depth = x.shape.with_rank_at_least(1)[-1].value
  if input_depth is None:
    raise NotImplementedError(
        "Rightmost dimension must be known prior to graph execution.")
  input_shape = (
      np.int32(x.shape.as_list())
      if x.shape.is_fully_defined() else tf.shape(x))
  for i, units in enumerate(hidden_layers):
    x = tfb.masked_dense(
        inputs=x,
        units=units,
        num_blocks=input_depth,
        exclusive=True if i == 0 else False,
        activation=activation,
        *args,
        **kwargs)
    if h is not None:
      x += tf.layers.dense(h, units, use_bias=False, *args, **kwargs)
    if dropout_rate > 0:
      x = tf.layers.dropout(x, dropout_rate, training=train)

  if shift_only:
    shift = tfb.masked_dense(
        inputs=x,
        units=input_depth,
        num_blocks=input_depth,
        activation=None,
        *args,
        **kwargs)
    return shift, None
  else:
    if log_scale_factor == 1.0 and log_scale_reg == 0.0 and not log_scale_clip_pre:
      x = tfb.masked_dense(
          inputs=x,
          units=2 * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,
          **kwargs)
      if h is not None:
        x += tf.layers.dense(h, 2 * input_depth, use_bias=False, *args, **kwargs)
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
    else:
      shift = tfb.masked_dense(
          inputs=x,
          units=input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,
          **kwargs)
      if log_scale_reg > 0.0:
        regularizer = lambda w: log_scale_reg * 2.0 * tf.nn.l2_loss(w)
      else:
        regularizer = None
      log_scale = tfb.masked_dense(
          inputs=x,
          units=input_depth,
          num_blocks=input_depth,
          activation=None,
          use_bias=False,
          kernel_regularizer=regularizer,
          *args,
          **kwargs)
      log_scale *= log_scale_factor
      if log_scale_clip_pre:
        log_scale = log_scale_clip_pre * tf.nn.tanh(log_scale / log_scale_clip_pre)
      log_scale += tf.get_variable(
          "log_scale_bias", [1, input_depth], initializer=tf.zeros_initializer())
      if h is not None:
        shift += tf.layers.dense(h, input_depth, use_bias=False, *args, **kwargs)
        log_scale += tf.layers.dense(
            h, input_depth, use_bias=False, *args, **kwargs)

    if sigmoid_scale:
      log_scale = tf.log_sigmoid(log_scale)

    if log_scale_clip:
      log_scale = log_scale_clip * tf.nn.tanh(log_scale / log_scale_clip)

    return shift, log_scale


def Expectation(dist, num_steps, batch_size, f=tf.identity):

  def Mean():
    return tf.reduce_mean(f(dist.sample(batch_size)), 0)

  def Cond(step, _):
    return step < num_steps

  def Body(step, running_mean):
    new_mean = Mean()

    running_mean = (
        running_mean * tf.to_float(step) + new_mean) / tf.to_float(step + 1)

    return step + 1, running_mean

  return tf.while_loop(Cond, Body, [0, Mean()])[1]


def Covariance(x):
  x = tf.to_float(x)
  mean = tf.reduce_mean(x, 0)
  x -= mean
  return tf.matmul(x, x, transpose_a=True) / tf.to_float(tf.shape(x)[0] - 1)


def GetPercentile(v, ps=(5, 50, 95)):
  return [tfp.stats.percentile(v, p) for p in ps]


class NumpyEncoder(simplejson.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return simplejson.JSONEncoder.default(self, obj)


def SaveJSON(obj, path):
  with tf.gfile.Open(path, "w") as f:
    simplejson.dump(obj, f, cls=NumpyEncoder)


def SaveNumpy(arr, path):
  with tf.gfile.Open(path, "w") as f:
    np.save(f, arr, allow_pickle=False)


@gin.configurable("covertype")
def LoadCovertype(path):
  with tf.gfile.Open(path, "rb") as f:
    # pytype: disable=wrong-arg-types
    data = np.genfromtxt(
        gzip.GzipFile(fileobj=f), delimiter=",")
    # pytype: enable=wrong-arg-types

  x = data[:, :-1]
  y = data[:, -1].astype(np.int32)
  y -= 1

  return x, y

@gin.configurable("cloud")
def LoadCloud(path):
  with tf.gfile.Open(path) as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]
    cols = list(zip(*rows))[1:]

    numeric_cols = []
    for col in cols:
      try:
        x = np.zeros([len(col), 1])
        for i, v in enumerate(col):
          x[i, 0] = float(v)

        x_min = np.min(x, 0, keepdims=True)
        x_max = np.max(x, 0, keepdims=True)

        x /= (x_max - x_min)
        x = 2.0 * x - 1.0

      except ValueError:
        keys = list(sorted(set(col)))
        vocab = {k: v for v, k in enumerate(keys)}
        x = np.zeros([len(col), len(keys)])
        for i, v in enumerate(col):
          one_hot = np.zeros(len(keys))
          one_hot[vocab[v]] = 1.
          x[i] = one_hot
      numeric_cols.append(x)

    data = np.concatenate(numeric_cols, -1)
    return data[:, :-1], data[:, -1]

@gin.configurable("german")
def LoadGerman(path, numeric=True):
  if numeric:
    with tf.gfile.Open(path, "rb") as f:
      data = np.genfromtxt(f)

    x = tf.convert_to_tensor(data[:, :-1])
    y = tf.convert_to_tensor(data[:, -1] - 1)
  else:
    with tf.gfile.Open(path) as f:
      data = pd.read_csv(f, delim_whitespace=True, header=None)

    def categorical_to_int(x):
      d = {u: i for i, u in enumerate(np.unique(x))}
      return np.array([d[i] for i in x])

    categoricals = []
    numericals = []
    for column in data.columns[:-1]:
      column = data[column]
      if column.dtype == 'O':
        categoricals.append(categorical_to_int(column))
      else:
        numericals.append((column - column.mean()) / column.std())
    numericals = np.array(numericals).T
    status = np.array(data[20] == 1, dtype=np.int32)

    x_numeric = tf.constant(numericals.astype(np.float32))
    x_categorical = [tf.one_hot(c, c.max() + 1) for c in categoricals]
    x = tf.concat([x_numeric] + x_categorical, 1)
    y = tf.constant(status)

  return x, y

def StitchImages(images):
  # images is [batch, x, y, c]
  batch, width, _, channels = tf.unstack(tf.shape(images))
  num_per_side = tf.to_int32(tf.ceil(tf.sqrt(tf.to_float(batch))))
  new_width = num_per_side * width
  paddings = tf.concat([tf.zeros([4, 1], dtype=tf.int32), tf.stack([num_per_side * num_per_side - batch, 0, 0, 0])[Ellipsis, tf.newaxis]], -1)
  images = tf.pad(images, paddings)

  images = tf.transpose(images, [1, 0, 2, 3])
  images = tf.reshape(images, [width, num_per_side, new_width, channels])
  images = tf.transpose(images, [1, 0, 2, 3])
  images = tf.reshape(images, [1, new_width, new_width, channels])

  return images

class TensorRef(object):

  def __init__(self, value):
    self._value = value

  def to_tensor(self, dtype=None, name=None, as_ref=False):
    return tf.convert_to_tensor(self._value, dtype=dtype, name=name)

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value


tf.register_tensor_conversion_function(
    TensorRef, conversion_func=TensorRef.to_tensor)


class AdaptStepResults(collections.namedtuple("AdaptStepResults", "inner_results, step_size, num_leapfrog_steps, cur_step")):
  __slots__ = ()


class AdaptStep(tfp.mcmc.TransitionKernel):

  def __init__(self, inner_kernel, step_size, num_leapfrog_steps,
               trajectory_length, target_accept_rate, gain, num_adapt_steps):
    self.inner_kernel = inner_kernel
    self.step_size = step_size
    self.num_leapfrog_steps = num_leapfrog_steps
    self.trajectory_length = trajectory_length
    self.target_accept_rate = target_accept_rate
    self.gain = gain
    self.num_adapt_steps = num_adapt_steps

  def compute_num_leapfrog_steps(self, step_size):
    return tf.cast(tf.ceil(self.trajectory_length / step_size), tf.int64)

  def one_step(self, current_state, current_results):
    step_size = current_results.step_size
    cur_step = current_results.cur_step

    self.step_size.value = step_size
    self.num_leapfrog_steps.value = current_results.num_leapfrog_steps

    new_state, new_results = self.inner_kernel.one_step(
        current_state, current_results.inner_results)

    def adapt():
      accept_prob = tf.exp(tf.minimum(0., new_results.inner_results.log_accept_ratio))
      return tf.minimum(
          step_size *
          tf.exp(self.gain *
                 (tf.reduce_mean(accept_prob) - self.target_accept_rate)),
          self.trajectory_length)

    step_size = tf.cond(cur_step < self.num_adapt_steps,
                        adapt, lambda: tf.identity(step_size))

    return new_state, AdaptStepResults(
        inner_results=new_results,
        step_size=step_size,
        num_leapfrog_steps=self.compute_num_leapfrog_steps(step_size),
        cur_step=cur_step + 1)

  def bootstrap_results(self, current_state):
    new_results = self.inner_kernel.bootstrap_results(current_state)

    return AdaptStepResults(
        inner_results=new_results,
        step_size=tf.convert_to_tensor(self.step_size),
        num_leapfrog_steps=tf.convert_to_tensor(
            self.num_leapfrog_steps, dtype=tf.int64),
        cur_step=tf.cast(0, tf.int64),
    )

  def is_calibrated(self):
    return True


class LogProbDist(tf.distributions.Distribution):

  def __init__(self,
               num_dims,
               log_prob_fn,
               name="LogProbDist",
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True):
    parameters = dict(locals())

    self._log_prob_fn = log_prob_fn
    self._num_dims = num_dims

    super(LogProbDist, self).__init__(
        dtype=tf.float32,
        reparameterization_type=tf.distributions.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[],
        name=name,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats)

  def _batch_shape_tensor(self):
    return tf.constant([])

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape_tensor(self):
    return tf.constant([self._num_dims])

  def _event_shape(self):
    return tf.TensorShape([self._num_dims])

  def _sample_n(self, n, seed=None):
    return tf.zeros([n, self._num_dims])

  def _log_prob(self, value):
    value = tf.convert_to_tensor(value)
    return self._log_prob_fn(value)


def CreateTrainOp(total_loss, optimizer, global_step, variables_to_train,
                  transform_grads_fn):
  grads_and_vars = optimizer.compute_gradients(total_loss, variables_to_train)
  if transform_grads_fn:
    grads_and_vars = transform_grads_fn(grads_and_vars)
  with tf.name_scope("summarize_grads"):
    for grad, var in grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
        tf.summary.histogram(var.op.name + "_gradient", grad_values)
        tf.summary.scalar(var.op.name + "_gradient_norm",
                          tf.global_norm([grad_values]))
      else:
        logging.info("Var %s has no gradient", var.op.name)
  grad_updates = optimizer.apply_gradients(
      grads_and_vars, global_step=global_step)
  with tf.name_scope("train_op"):
    with tf.control_dependencies([grad_updates]):
      total_loss = tf.check_numerics(total_loss, "LossTensor is inf or nan")

  return total_loss


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1):
  """Borrowed from tf.contrib.training."""
  logging.info("Waiting for new checkpoint at %s", checkpoint_dir)
  while True:
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      time.sleep(seconds_to_sleep)
    else:
      logging.info("Found new checkpoint at %s", checkpoint_path)
      return checkpoint_path


def checkpoints_iterator(
    checkpoint_dir,
    min_interval_secs=0,
):
  """Borrowed from tf.contrib.training."""
  checkpoint_path = None
  while True:
    new_checkpoint_path = wait_for_new_checkpoint(checkpoint_dir,
                                                  checkpoint_path)
    start = time.time()
    checkpoint_path = new_checkpoint_path
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


def evaluate_repeatedly(
    checkpoint_dir,
    eval_dir,
    stop_after_n_evals,
    eval_ops,
    master="",
    scaffold=None,
    eval_interval_secs=60,
    max_number_of_evaluations=None,
):
  """Borrowed from tf.contrib.training."""

  summary_op = tf.summary.merge_all()
  global_step = tf.train.get_or_create_global_step()
  summary_writer = tf.summary.FileWriterCache.get(eval_dir)

  num_evaluations = 0
  for checkpoint_path in checkpoints_iterator(
      checkpoint_dir, min_interval_secs=eval_interval_secs):

    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_filename_with_path=checkpoint_path,
        master=master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      logging.info("Starting evaluation at " +
                   time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()))
      for _ in range(stop_after_n_evals):
        if sess.should_stop():
          break
        sess.run(eval_ops)
      global_step_, summary_str, log_outputs = sess.run(
          [global_step, summary_op,
           GetLoggingOutputs()])
      logging.info(", ".join(
          "{} = {}".format(k, v) for k, v in sorted(list(log_outputs.items()))))
      summary_writer.add_summary(summary_str, global_step_)
      summary_writer.flush()

      logging.info("Finished evaluation at " +
                   time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()))
    num_evaluations += 1

    if (max_number_of_evaluations is not None and
        num_evaluations >= max_number_of_evaluations):
      return
