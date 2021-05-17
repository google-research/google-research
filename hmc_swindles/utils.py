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

"""Various utilities."""
# pylint: disable=invalid-name,missing-docstring
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import contextlib
import functools
import json

from typing import Any, Callable, Dict, Text, Union
from absl import flags
from absl import logging
import gin
import numpy as np
import simplejson
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import yaml

# pylint: disable=g-import-not-at-top
USE_LOCAL_FUN_MCMC = True
if USE_LOCAL_FUN_MCMC:
  from discussion import fun_mcmc  # pylint: disable=reimported

tfd = tfp.distributions
tfb = tfp.bijectors
tfkl = tf.keras.layers

LOGGING_OUTPUTS = "logging"


class PiecewiseConstantDecay(tf.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses a piecewise constant decay schedule."""

  def __init__(self, boundaries, values, name="PiecewiseConstantDecay"):
    super(PiecewiseConstantDecay, self).__init__()

    if len(boundaries) != len(values) - 1:
      raise ValueError(
          "The length of boundaries should be 1 less than the length of values")

    self.boundaries = boundaries
    self.values = values
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name):
      boundaries = tf.nest.map_structure(tf.convert_to_tensor, self.boundaries)
      values = tf.nest.map_structure(tf.convert_to_tensor, self.values)
      x_recomp = tf.convert_to_tensor(step)
      lr = values[0]
      for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
        lr = tf.where((x_recomp > low) & (x_recomp <= high), v, lr)
      lr = tf.where(x_recomp > boundaries[-1], values[-1], lr)

      return lr

  def get_config(self):
    return {
        "boundaries": self.boundaries,
        "values": self.values,
        "name": self.name
    }


def Compile(fn):

  @tf.function(autograph=False)
  def Wrapper(*args, **kwargs):
    return tf.xla.experimental.compile(lambda: fn(*args, **kwargs))

  return Wrapper


class Template(tf.Module):
  """A class to automatically track and reuse variables in a callable."""

  def __init__(self, name, fn, **kwargs):
    self._fn = fn
    self._built = False
    self._fn_kwargs = kwargs
    self._var_cache = []
    super(Template, self).__init__(name)

  def __call__(self, *args, **kwargs):
    new_kwargs = dict(self._fn_kwargs).copy()
    new_kwargs.update(kwargs)
    kwargs = new_kwargs

    def _Creator(next_creator, **kwargs):
      if self._built:
        ret = self._var_cache[_Creator.counter]
      else:
        ret = next_creator(**kwargs)
        self._var_cache.append(ret)
      _Creator.counter += 1
      return ret

    _Creator.counter = 0

    with tf.variable_creator_scope(_Creator):
      ret = self._fn(*args, **kwargs)
    self._built = True
    return ret


def DeferBijector(make_bijector_fn):
  # pylint: disable=protected-access
  return tfb.Inline(
      forward_fn=lambda x: make_bijector_fn()._forward(),
      inverse_fn=lambda y: make_bijector_fn()._inverse(y),
      forward_log_det_jacobian_fn=(
          lambda x: make_bijector_fn()._forward_log_det_jacobian(x)),
      inverse_log_det_jacobian_fn=(
          lambda x: make_bijector_fn()._inverse_log_det_jacobian(x)),
      forward_min_event_ndims=make_bijector_fn().forward_min_event_ndims)


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
  called within 'scope' config scope.

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

    return Template(template_name, GinWrapper, **kwargs)

  return Wrapper


class NotDict(object):
  """You don't want to know."""

  def __init__(self, d):
    self._d = d

  def items(self):
    return self._d.items()

  def __getitem__(self, k):
    return self._d[k]


class YAMLDictParser(flags.ArgumentParser):
  syntactic_help = """Expects YAML one-line dictionaries without braces, e.g.
  'key1: val1, key2: val2'."""

  def parse(self, argument):
    return NotDict(yaml.safe_load("{" + argument + "}"))

  def flag_type(self):
    return "Dict[Text, Any]"


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


def SanitizedAutoCorrelation(x, axis, **kwargs):
  res = tfp.stats.auto_correlation(x, axis, **kwargs)
  res = tf.where(tf.math.is_nan(res), tf.ones_like(res), res)
  res = tf.where(tf.math.is_inf(res), tf.ones_like(res), res)
  return res


def SanitizedAutoCorrelationMean(x, axis, reduce_axis, max_lags=None, **kwargs):
  shape_arr = np.array(list(x.shape))
  axes = list(sorted(set(range(len(shape_arr))) - set([reduce_axis])))
  mean_shape = shape_arr[axes]
  if max_lags is not None:
    mean_shape[axis] = max_lags + 1
  mean_state = fun_mcmc.running_mean_init(mean_shape, x.dtype)
  new_order = list(range(len(shape_arr)))
  new_order[0] = new_order[reduce_axis]
  new_order[reduce_axis] = 0
  x = tf.transpose(x, new_order)
  x_arr = tf.TensorArray(x.dtype, x.shape[0]).unstack(x)
  mean_state, _ = fun_mcmc.trace(
      state=mean_state,
      fn=lambda state: fun_mcmc.running_mean_step(  # pylint: disable=g-long-lambda
          state,
          SanitizedAutoCorrelation(
              x_arr.read(state.num_points), axis, max_lags=max_lags, **kwargs)),
      num_steps=x.shape[0],
      trace_fn=lambda *_: ())
  return mean_state.mean


def EffectiveSampleSize(states,
                        filter_beyond_lag=300,
                        filter_threshold=0.05,
                        use_geyer=False,
                        center=True,
                        normalize=True):
  """ESS computation for one single Tensor argument."""

  def _axis_size(x, axis=None):
    """Get number of elements of `x` in `axis`, as type `x.dtype`."""
    if axis is None:
      return tf.cast(tf.size(x), x.dtype)
    return tf.cast(tf.reduce_prod(tf.gather(tf.shape(x), axis)), x.dtype)

  with tf.name_scope("effective_sample_size_single_state"):

    states = tf.convert_to_tensor(states, name="states")
    dt = states.dtype

    # filter_beyond_lag == None ==> auto_corr is the full sequence.
    auto_corr = SanitizedAutoCorrelationMean(
        states,
        axis=0,
        reduce_axis=1,
        center=center,
        normalize=normalize,
        max_lags=filter_beyond_lag)
    orig_auto_corr = auto_corr
    if use_geyer:

      def _sum_pairs(x):
        if x.shape[0] % 2 != 0:
          x = tf.concat([x, tf.zeros(tf.concat([[1], tf.shape(x)[1:]], 0))], 0)
        return tf.reduce_sum(tf.reshape(x, [tf.shape(x)[0] // 2, 2, -1]), 1)

      def _make_pairs(x):
        return tf.reshape(
            tf.tile(x[:, tf.newaxis, :], [1, 2, 1]), [-1, x.shape[-1]])

      auto_corr_pairs = _make_pairs(_sum_pairs(auto_corr))[:auto_corr.shape[0]]
      mask = auto_corr_pairs < 0.
      mask = tf.cast(mask, dt)
      mask = tf.cumsum(mask, axis=0)
      mask = tf.maximum(1. - mask, 0.)
      auto_corr *= mask
    elif filter_threshold is not None:
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

    # return tf.reduce_mean(n / (
    #   -1 + 2 * tf.reduce_sum(nk_factor * auto_corr, axis=0)), 0)
    # return n / (1.0 + 2 *
    #             tf.reduce_sum(nk_factor[1:, ...] * auto_corr[1:, ...],
    #             axis=0))
    # return tf.reduce_mean(n / (-auto_corr[0] + 2 *
    #   tf.reduce_sum(nk_factor * auto_corr, axis=0)), 0)
    # print(auto_corr[0])
    return n / (
        orig_auto_corr[0] +
        2 * tf.reduce_sum(nk_factor[1:, Ellipsis] * auto_corr[1:, Ellipsis], axis=0))


@gin.configurable("l2hmc_initializer")
def L2HMCInitializer(factor = 1.0):
  return tf.keras.initializers.VarianceScaling(2.0 * factor)


@gin.configurable("dense_shift_log_scale")
@MakeTFTemplate
def DenseShiftLogScale(x,
                       output_units,
                       h=None,
                       hidden_layers=(),
                       activation=tf.nn.relu,
                       log_scale_clip=None,
                       train=False,
                       dropout_rate=0.0,
                       sigmoid_scale=False,
                       log_scale_factor=1.0,
                       log_scale_reg=0.0,
                       **kwargs):
  for units in hidden_layers:
    x = tfkl.Dense(units=units, activation=activation, **kwargs)(x)
    if h is not None:
      x += tfkl.Dense(units, use_bias=False, **kwargs)(h)
    if dropout_rate > 0:
      x = tfkl.Dropout(dropout_rate)(x, training=train)
  if log_scale_factor == 1.0 and log_scale_reg == 0.0:
    x = tfkl.Dense(units=2 * output_units, **kwargs)(x)
    if h is not None:
      x += tfkl.Dense(2 * output_units, use_bias=False, **kwargs)(h)

    shift, log_scale = tf.split(x, 2, axis=-1)
  else:
    shift = tfkl.Dense(output_units, **kwargs)(h)
    if log_scale_reg > 0.0:
      regularizer = lambda w: log_scale_reg * 2.0 * tf.nn.l2_loss(w)
    else:
      regularizer = None
    log_scale = tfkl.Dense(
        output_units, use_bias=False, kernel_regularizer=regularizer, **kwargs)(
            h)
    log_scale = log_scale * log_scale_factor + tf.Variable(
        tf.zeros([1, output_units]), name="log_scale_bias")
    if h is not None:
      shift += tfkl.Dense(output_units, use_bias=False, **kwargs)(h)
      log_scale += tfkl.Dense(output_units, use_bias=False, **kwargs)(h)

  if sigmoid_scale:
    log_scale = tf.math.log_sigmoid(log_scale)

  if log_scale_clip:
    log_scale = log_scale_clip * tf.nn.tanh(log_scale / log_scale_clip)

  return shift, log_scale


def MaskedDense(inputs,
                num_blocks,
                units,
                kernel_initializer,
                exclusive=False,
                **kwargs):
  input_depth = inputs.shape[-1]
  if input_depth is None:
    raise ValueError("input depth is None")

  MASK_INCLUSIVE = "inclusive"
  MASK_EXCLUSIVE = "exclusive"

  def _gen_slices(num_blocks, n_in, n_out, mask_type=MASK_EXCLUSIVE):
    """Generate the slices for building an autoregressive mask."""
    # TODO(b/67594795): Better support of dynamic shape.
    slices = []
    col = 0
    d_in = n_in // num_blocks
    d_out = n_out // num_blocks
    row = d_out if mask_type == MASK_EXCLUSIVE else 0
    for _ in range(num_blocks):
      row_slice = slice(row, None)
      col_slice = slice(col, col + d_in)
      slices.append([row_slice, col_slice])
      col += d_in
      row += d_out
    return slices

  def _gen_mask(num_blocks,
                n_in,
                n_out,
                mask_type=MASK_EXCLUSIVE,
                dtype=tf.float32):
    """Generate the mask for building an autoregressive dense layer."""
    # TODO(b/67594795): Better support of dynamic shape.
    mask = np.zeros([n_out, n_in], dtype=dtype.as_numpy_dtype)
    slices = _gen_slices(num_blocks, n_in, n_out, mask_type=mask_type)
    for [row_slice, col_slice] in slices:
      mask[row_slice, col_slice] = 1
    return mask

  mask = _gen_mask(num_blocks, input_depth, units,
                   MASK_EXCLUSIVE if exclusive else MASK_INCLUSIVE).T

  def masked_initializer(shape, dtype=None, partition_info=None):
    # If no `partition_info` is given, then don't pass it to `initializer`, as
    # `initializer` may be a `tf.compat.v2.initializers.Initializer` (which
    # don't accept a `partition_info` argument).
    if partition_info is None:
      x = kernel_initializer(shape, dtype)
    else:
      x = kernel_initializer(shape, dtype, partition_info)
    return tf.cast(mask, x.dtype) * x

  return tfkl.Dense(
      units=units,
      kernel_initializer=masked_initializer,
      kernel_constraint=lambda x: mask * x,
      **kwargs)(
          inputs)


@gin.configurable("dense_ar")
@MakeTFTemplate
def DenseAR(x,
            h=None,
            hidden_layers=(),
            activation=tf.nn.relu,
            log_scale_clip=None,
            log_scale_clip_pre=None,
            train=False,
            dropout_rate=0.0,
            sigmoid_scale=False,
            log_scale_factor=1.0,
            log_scale_reg=0.0,
            shift_only=False,
            **kwargs):
  input_depth = int(x.shape.with_rank_at_least(1)[-1])
  if input_depth is None:
    raise NotImplementedError(
        "Rightmost dimension must be known prior to graph execution.")
  input_shape = (
      np.int32(x.shape.as_list())
      if x.shape.is_fully_defined() else tf.shape(x))
  for i, units in enumerate(hidden_layers):
    x = MaskedDense(
        inputs=x,
        units=units,
        num_blocks=input_depth,
        exclusive=True if i == 0 else False,
        activation=activation,
        **kwargs)
    if h is not None:
      x += tfkl.Dense(units, use_bias=False, **kwargs)(h)
    if dropout_rate > 0:
      x = tfkl.Dropout(dropout_rate)(x, training=train)

  if shift_only:
    shift = MaskedDense(
        inputs=x,
        units=input_depth,
        num_blocks=input_depth,
        activation=None,
        **kwargs)
    return shift, None
  else:
    if log_scale_factor == 1.0 and log_scale_reg == 0.0 and not log_scale_clip_pre:
      x = MaskedDense(
          inputs=x,
          units=2 * input_depth,
          num_blocks=input_depth,
          activation=None,
          **kwargs)
      if h is not None:
        x += tfkl.Dense(2 * input_depth, use_bias=False, **kwargs)(h)
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
    else:
      shift = MaskedDense(
          inputs=x,
          units=input_depth,
          num_blocks=input_depth,
          activation=None,
          **kwargs)
      if log_scale_reg > 0.0:
        regularizer = lambda w: log_scale_reg * 2.0 * tf.nn.l2_loss(w)
      else:
        regularizer = None
      log_scale = MaskedDense(
          inputs=x,
          units=input_depth,
          num_blocks=input_depth,
          activation=None,
          use_bias=False,
          kernel_regularizer=regularizer,
          **kwargs)
      log_scale *= log_scale_factor
      if log_scale_clip_pre:
        log_scale = log_scale_clip_pre * tf.nn.tanh(
            log_scale / log_scale_clip_pre)
      log_scale += tf.get_variable(
          "log_scale_bias", [1, input_depth],
          initializer=tf.zeros_initializer())
      if h is not None:
        shift += tfkl.Dense(input_depth, use_bias=False, **kwargs)(h)
        log_scale += tfkl.Dense(input_depth, use_bias=False, **kwargs)(h)

    if sigmoid_scale:
      log_scale = tf.log_sigmoid(log_scale)

    if log_scale_clip:
      log_scale = log_scale_clip * tf.nn.tanh(log_scale / log_scale_clip)

    return shift, log_scale


class NumpyEncoder(simplejson.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, tf.Tensor):
      obj = obj.numpy()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return simplejson.JSONEncoder.default(self, obj)


def SaveJSON(obj, path):
  with tf.io.gfile.GFile(path, "w") as f:
    simplejson.dump(obj, f, cls=NumpyEncoder)


def SaveNumpy(arr, path):
  with tf.io.gfile.GFile(path, "w") as f:
    np.save(f, arr, allow_pickle=False)


class LogProbDist(tfd.Distribution):

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
        dtype=dtype,
        reparameterization_type=tfd.NOT_REPARAMETERIZED,
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


class Probit(tfd.Distribution):

  def __init__(self,
               logits,
               name="Probit",
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True):
    parameters = dict(locals())
    self._logits = logits
    super(Probit, self).__init__(
        dtype=dtype,
        reparameterization_type=tfd.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[],
        name=name,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats)

  def _batch_shape_tensor(self):
    return tf.shape(input=self._logits)

  def _batch_shape(self):
    return self._logits.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    raise NotImplementedError()

  def _log_prob(self, value):
    logits = tf.convert_to_tensor(self._logits, self.dtype)
    value = tf.cast(value, self.dtype)

    def phi(x):
      return 0.5 * (1 + tf.math.erf(x * tf.math.sqrt(0.5)))

    def phic(x):
      return 0.5 * tf.math.erfc(x * tf.math.sqrt(0.5))

    pi = np.array(np.pi).astype(self.dtype.as_numpy_dtype)

    def logphi(x):
      is_small = x < -5
      no_small_x = tf.where(is_small, tf.ones_like(x), x)
      x = tf.where(is_small, x, tf.ones_like(x))
      return tf.where(is_small, -0.5 * x * x - tf.math.log(-x * pi),
                      tf.math.log(phi(no_small_x)))

    def logphic(x):
      is_large = x > 5
      no_large_x = tf.where(is_large, tf.ones_like(x), x)
      x = tf.where(is_large, x, tf.ones_like(x))
      return tf.where(is_large, -0.5 * x * x - tf.math.log(x * pi),
                      tf.math.log(phic(no_large_x)))

    ret = value * logphi(logits) + (1 - value) * logphic(logits)
    return ret


_USE_XLA = False


@contextlib.contextmanager
def use_xla(enable = True):
  """Context manager to dis/enable XLA for `compile`."""
  global _USE_XLA
  try:
    old_setting = _USE_XLA
    _USE_XLA = enable
    yield
  finally:
    _USE_XLA = old_setting


def compile(fn):  # pylint: disable=redefined-builtin
  """Decorator to optionally compile a function."""

  @tf.function(autograph=False)
  def _wrapper(*args, **kwargs):
    use_xla_val = kwargs.pop("_use_xla")

    if use_xla_val:
      logging.info("%s using XLA", fn.__name__)
    else:
      logging.info("%s not using XLA", fn.__name__)

    return tf.function(
        lambda: fn(*args, **kwargs),
        autograph=False,
        experimental_compile=use_xla_val)()

  ret = lambda *args, **kwargs: _wrapper(*args, _use_xla=_USE_XLA, **kwargs)  # pylint: disable=unnecessary-lambda
  ret = functools.wraps(fn)(ret)
  return ret


def encode_tree(tree):
  """Encodes a tree into a decodable tree of basic Python types."""
  if isinstance(tree, tf.Tensor):
    tree = np.array(tree)
  if isinstance(tree, np.generic):
    tree = np.array(tree)
  if isinstance(tree, np.ndarray):
    return dict(
        __type__="ndarray",
        dtype=np.dtype(tree.dtype).name,
        data=tree.tolist(),
    )
  if isinstance(tree, tuple) and hasattr(tree, "_asdict"):
    ret = {k: encode_tree(v) for k, v in tree._asdict().items()}
    ret["__type__"] = type(tree).__name__
    return ret
  if isinstance(tree,
                collections.Sequence) and not isinstance(tree, (str, bytes)):
    return [encode_tree(v) for v in tree]
  if isinstance(tree, collections.Mapping):
    return {k: encode_tree(v) for k, v in tree.items()}
  return tree


def decode_tree(tree):
  """Decodes a tree encoded by `encode_tree`.

  Uses `register_namedtuple` to decode namedtuples.

  Args:
    tree: Input tree.

  Returns:
    encoded_tree: Encoded tree.
  """
  if isinstance(tree, collections.Mapping):
    if tree.get("__type__") == "ndarray":
      return np.array(tree["data"]).astype(np.dtype(tree["dtype"]))
    if tree.get("__type__") in _NAMEDTUPLE_REGISTRY:
      tree = tree.copy()
      name = tree.pop("__type__")
      tree = {k: decode_tree(v) for k, v in tree.items()}
      return _NAMEDTUPLE_REGISTRY[name](**tree)
    return {k: decode_tree(v) for k, v in tree.items()}
  if isinstance(tree,
                collections.Sequence) and not isinstance(tree, (str, bytes)):
    return [decode_tree(v) for v in tree]
  return tree


_NAMEDTUPLE_REGISTRY = {}


def register_namedtuple(namedtuple):
  """Registers a namedtuple for `decode_tree`."""
  _NAMEDTUPLE_REGISTRY[namedtuple.__name__] = namedtuple
  return namedtuple


def save_json(tree, path):
  """Saves a `tree` to json.

  Uses `encode_tree` to set the format.

  Args:
    tree: Input tree.
    path: Where to save the JSON.
  """
  encoded = encode_tree(tree)
  s = json.dumps(encoded, indent=2)
  with tf.io.gfile.GFile(path, "w") as f:
    f.write(s)


def load_json(path):
  """Loads a `tree` from json.

  Uses `decode_tree` to parse the format.

  Args:
    path: From where to load the JSON.

  Returns:
    tree: The decoded tree.
  """
  with tf.io.gfile.GFile(path, "r") as f:
    decoded = json.load(f)
    return decode_tree(decoded)


class VectorTargetDensity(object):
  """Converts a variously shaped TargetDensity into a vector variate one."""

  def __init__(self, target_density):
    self._target_density = target_density

    def _make_reshaped_bijector(b, s):
      return tfb.Reshape(
          event_shape_in=s, event_shape_out=[s.num_elements()])(b)(
              tfb.Reshape(event_shape_out=b.inverse_event_shape(s)))

    reshaped_bijector = tf.nest.map_structure(
        _make_reshaped_bijector, self._target_density.constraining_bijectors,
        self._target_density.event_shape)

    self._bijector = tfb.Blockwise(
        bijectors=tf.nest.flatten(reshaped_bijector),
        block_sizes=tf.nest.flatten(
            tf.nest.map_structure(lambda s: s.num_elements(),
                                  self._target_density.event_shape)))

  def _flatten_and_concat_event(self, x):
    """Flattens and concatenates a structure-valued event `x`."""

    def _reshape_part(part, event_shape):
      part = tf.cast(part, self.dtype)
      rank = event_shape.rank
      if rank == 1:
        return part
      new_shape = tf.concat([
          tf.shape(part)[:tf.size(tf.shape(part)) - tf.size(event_shape)], [-1]
      ],
                            axis=-1)
      return tf.reshape(part, tf.cast(new_shape, tf.int32))

    x = tf.nest.map_structure(_reshape_part, x,
                              self._target_density.event_shape)
    return tf.concat(tf.nest.flatten(x), axis=-1)

  def _split_and_reshape_event(self, x):
    """Splits and reshapes of a vector-valued event `x`."""
    splits = [
        tf.maximum(1, tf.reduce_prod(s))
        for s in tf.nest.flatten(self._target_density.event_shape)
    ]
    x = tf.nest.pack_sequence_as(self._target_density.event_shape,
                                 tf.split(x, splits, axis=-1))

    def _reshape_part(part, dtype, event_shape):
      part = tf.cast(part, dtype)
      rank = event_shape.rank
      if rank == 1:
        return part
      new_shape = tf.concat([tf.shape(part)[:-1], event_shape], axis=-1)
      return tf.reshape(part, tf.cast(new_shape, tf.int32))

    x = tf.nest.map_structure(_reshape_part, x, self._target_density.dtype,
                              self._target_density.event_shape)
    return x

  @property
  def event_shape(self):
    event_sizes = tf.nest.map_structure(lambda s: s.num_elements(),
                                        self._target_density.event_shape)
    return tf.TensorShape([sum(tf.nest.flatten(event_sizes))])

  @property
  def dtype(self):
    return tf.nest.flatten(self._target_density.dtype)[0]

  def __call__(self, x):
    return self._target_density(self._split_and_reshape_event(x))

  @property
  def constraining_bijectors(self):
    return self._bijector

  @property
  def expectations(self):
    """See `TargetDensity.expectations`."""
    ret = collections.OrderedDict()

    def body(k, exp):
      new_exp = exp._replace(fn=lambda x: exp(self._split_and_reshape_event(x)))
      ret[k] = new_exp

    for k, exp in self._target_density.expectations.items():
      body(k, exp)
    return ret

  @property
  def distribution(self):
    return tfd.Blockwise(self._target_density.distribution)
