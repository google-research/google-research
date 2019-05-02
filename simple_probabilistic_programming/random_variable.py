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

"""Random variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops

__all__ = [
    "RandomVariable",
]


class RandomVariable(object):
  """Class for random variables.

  `RandomVariable` encapsulates properties of a random variable, namely, its
  distribution, sample shape, and (optionally overridden) value. Its `value`
  property is a `tf.Tensor`, which embeds the `RandomVariable` object into the
  TensorFlow graph. `RandomVariable` also features operator overloading and
  registration to TensorFlow sessions, enabling idiomatic usage as if one were
  operating on `tf.Tensor`s.

  The random variable's shape is given by

  `sample_shape + distribution.batch_shape + distribution.event_shape`,

  where `sample_shape` is an optional argument describing the shape of
  independent, identical draws from the distribution (default is `()`, meaning
  a single draw); `distribution.batch_shape` describes the shape of
  independent-but-not-identical draws (determined by the shape of the
  distribution's parameters); and `distribution.event_shape` describes the
  shape of dependent dimensions (e.g., `Normal` has scalar `event_shape`;
  `Dirichlet` has vector `event_shape`).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  from tensorflow_probability import edward2 as ed
  tfd = tfp.distributions

  z1 = tf.constant([[1.0, -0.8], [0.3, -1.0]])
  z2 = tf.constant([[0.9, 0.2], [2.0, -0.1]])
  x = ed.RandomVariable(tfd.Bernoulli(logits=tf.matmul(z1, z2)))

  loc = ed.RandomVariable(tfd.Normal(0., 1.))
  x = ed.RandomVariable(tfd.Normal(loc, 1.), sample_shape=50)
  assert x.shape.as_list() == [50]
  assert x.sample_shape.as_list() == [50]
  assert x.distribution.batch_shape.as_list() == []
  assert x.distribution.event_shape.as_list() == []
  ```
  """

  def __init__(self,
               distribution,
               sample_shape=(),
               value=None):
    """Create a new random variable.

    Args:
      distribution: tfd.Distribution governing the distribution of the random
        variable, such as sampling and log-probabilities.
      sample_shape: tf.TensorShape of samples to draw from the random variable.
        Default is `()` corresponding to a single sample.
      value: Fixed tf.Tensor to associate with random variable. Must have shape
        `sample_shape + distribution.batch_shape + distribution.event_shape`.
        Default is to sample from random variable according to `sample_shape`.

    Raises:
      ValueError: `value` has incompatible shape with
        `sample_shape + distribution.batch_shape + distribution.event_shape`.
    """
    self._distribution = distribution
    self._sample_shape = sample_shape
    if value is not None:
      value = tf.cast(value, self.distribution.dtype)
      value_shape = value.shape
      expected_value_shape = self.sample_shape.concatenate(
          self.distribution.batch_shape).concatenate(
              self.distribution.event_shape)
      if not value_shape.is_compatible_with(expected_value_shape):
        raise ValueError(
            "Incompatible shape for initialization argument 'value'. "
            "Expected %s, got %s." % (expected_value_shape, value_shape))
    self._value = value

  @property
  def distribution(self):
    """Distribution of random variable."""
    return self._distribution

  @property
  def dtype(self):
    """`Dtype` of elements in this random variable."""
    return self.value.dtype

  @property
  def sample_shape(self):
    """Sample shape of random variable as a `TensorShape`."""
    if isinstance(self._sample_shape, tf.Tensor):
      return tf.TensorShape(tf.contrib.util.constant_value(self._sample_shape))
    return tf.TensorShape(self._sample_shape)

  def sample_shape_tensor(self, name="sample_shape_tensor"):
    """Sample shape of random variable as a 1-D `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      sample_shape: `Tensor`.
    """
    with tf.compat.v1.name_scope(name):
      if isinstance(self._sample_shape, tf.Tensor):
        return self._sample_shape
      return tf.convert_to_tensor(self.sample_shape.as_list(), dtype=tf.int32)

  @property
  def shape(self):
    """Shape of random variable."""
    return self.value.shape

  @property
  def value(self):
    """Get tensor that the random variable corresponds to."""
    if self._value is None:
      try:
        self._value = self.distribution.sample(self.sample_shape_tensor())
      except NotImplementedError:
        raise NotImplementedError(
            "sample is not implemented for {0}. You must either pass in the "
            "value argument or implement sample for {0}."
            .format(self.distribution.__class__.__name__))
    return self._value

  def __str__(self):
    if not isinstance(self.value, ops.EagerTensor):
      name = self.distribution.name
    else:
      name = _numpy_text(self.value)
    return "RandomVariable(\"%s\"%s%s%s)" % (
        name,
        ", shape=%s" % self.shape if self.shape.ndims is not None else "",
        ", dtype=%s" % self.dtype.name if self.dtype else "",
        ", device=%s" % self.value.device if self.value.device else "")

  def __repr__(self):
    string = "ed.RandomVariable '%s' shape=%s dtype=%s" % (
        self.distribution.name, self.shape, self.dtype.name)
    if hasattr(self.value, "numpy"):
      string += " numpy=%s" % _numpy_text(self.value, is_repr=True)
    return "<%s>" % string

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return id(self) == id(other)

  def __ne__(self, other):
    return not self == other

  def eval(self, session=None, feed_dict=None):
    """In a session, computes and returns the value of this random variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.

    Args:
      session: tf.BaseSession.
        The `tf.Session` to use to evaluate this random variable. If
        none, the default session is used.
      feed_dict: dict.
        A dictionary that maps `tf.Tensor` objects to feed values. See
        `tf.Session.run()` for a description of the valid feed values.

    Returns:
      Value of the random variable.

    #### Examples

    ```python
    x = Normal(0.0, 1.0)
    with tf.Session() as sess:
      # Usage passing the session explicitly.
      print(x.eval(sess))
      # Usage with the default session.  The 'with' block
      # above makes 'sess' the default session.
      print(x.eval())
    ```
    """
    return self.value.eval(session=session, feed_dict=feed_dict)

  def numpy(self):
    """Value as NumPy array, only available for TF Eager."""
    if not isinstance(self.value, ops.EagerTensor):
      raise NotImplementedError("value argument must be a EagerTensor.")

    return self.value.numpy()

  def get_shape(self):
    """Get shape of random variable."""
    return self.shape

  # This enables the RandomVariable's overloaded "right" binary operators to
  # run when the left operand is an ndarray, because it accords the
  # RandomVariable class higher priority than an ndarray, or a numpy matrix.
  __array_priority__ = 100


def _numpy_text(tensor, is_repr=False):
  """Human-readable representation of a tensor's numpy value."""
  if tensor.dtype.is_numpy_compatible:
    text = repr(tensor.numpy()) if is_repr else str(tensor.numpy())
  else:
    text = "<unprintable>"
  if "\n" in text:
    text = "\n" + text
  return text


def _overload_operator(cls, op):
  """Defer an operator overload to `tf.Tensor`.

  We pull the operator out of tf.Tensor dynamically to avoid ordering issues.

  Args:
    cls: Class to overload operator.
    op: Python string representing the operator name.
  """
  @functools.wraps(getattr(tf.Tensor, op))
  def _run_op(a, *args):
    return getattr(tf.Tensor, op)(a.value, *args)

  setattr(cls, op, _run_op)


def _session_run_conversion_fetch_function(tensor):
  return ([tensor.value], lambda val: val[0])


def _session_run_conversion_feed_function(feed, feed_val):
  return [(feed.value, feed_val)]


def _session_run_conversion_feed_function_for_partial_run(feed):
  return [feed.value]


def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
  del name, as_ref  # unused
  if dtype and not dtype.is_compatible_with(v.dtype):
    raise ValueError(
        "Incompatible type conversion requested to type '%s' for variable "
        "of type '%s'" % (dtype.name, v.dtype.name))
  return v.value


for operator in tf.Tensor.OVERLOADABLE_OPERATORS.union({"__iter__",
                                                        "__bool__",
                                                        "__nonzero__"}):
  _overload_operator(RandomVariable, operator)

tf_session.register_session_run_conversion_functions(  # enable sess.run, eval
    RandomVariable,
    _session_run_conversion_fetch_function,
    _session_run_conversion_feed_function,
    _session_run_conversion_feed_function_for_partial_run)

tf.register_tensor_conversion_function(  # enable tf.convert_to_tensor
    RandomVariable, _tensor_conversion_function)
