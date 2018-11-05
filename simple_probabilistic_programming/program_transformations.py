# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Transformations of Edward2 programs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import six
import tensorflow as tf

from simple_probabilistic_programming.trace import trace

__all__ = [
    "make_log_joint_fn",
]


def make_log_joint_fn(model):
  """Takes Edward probabilistic program and returns its log joint function.

  Args:
    model: Python callable which executes the generative process of a
      computable probability distribution using `ed.RandomVariable`s.

  Returns:
    A log-joint probability function. Its inputs are `model`'s original inputs
    and random variables which appear during the program execution. Its output
    is a scalar tf.Tensor.

  #### Examples

  Below we define Bayesian logistic regression as an Edward program,
  representing the model's generative process. We apply `make_log_joint_fn` in
  order to represent the model in terms of its joint probability function.

  ```python
  import simple_probabilistic_programming as ed

  def logistic_regression(features):
    coeffs = ed.Normal(loc=0., scale=1.,
                       sample_shape=features.shape[1], name="coeffs")
    outcomes = ed.Bernoulli(logits=tf.tensordot(features, coeffs, [[1], [0]]),
                            name="outcomes")
    return outcomes

  log_joint = ed.make_log_joint_fn(logistic_regression)

  features = tf.random_normal([3, 2])
  coeffs_value = tf.random_normal([2])
  outcomes_value = tf.round(tf.random_uniform([3]))
  output = log_joint(features, coeffs=coeffs_value, outcomes=outcomes_value)
  ```

  """
  def log_joint_fn(*args, **kwargs):
    """Log-probability of inputs according to a joint probability distribution.

    Args:
      *args: Positional arguments. They are the model's original inputs and can
        alternatively be specified as part of `kwargs`.
      **kwargs: Keyword arguments, where for each key-value pair `k` and `v`,
        `v` is passed as a `value` to the random variable(s) whose keyword
        argument `name` during construction is equal to `k`.

    Returns:
      Scalar tf.Tensor, which represents the model's log-probability summed
      over all Edward random variables and their dimensions.

    Raises:
      TypeError: If a random variable in the model has no specified value in
        `**kwargs`.
    """
    log_probs = []

    def tracer(rv_constructor, *rv_args, **rv_kwargs):
      """Overrides a random variable's `value` and accumulates its log-prob."""
      # Set value to keyword argument indexed by `name` (an input tensor).
      rv_name = rv_kwargs.get("name")
      if rv_name is None:
        raise KeyError("Random variable constructor {} has no name "
                       "in its arguments.".format(rv_constructor.__name__))
      value = kwargs.get(rv_name)
      if value is None:
        raise LookupError("Keyword argument specifying value for {} is "
                          "missing.".format(rv_name))
      rv_kwargs["value"] = value

      rv = rv_constructor(*rv_args, **rv_kwargs)
      log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
      log_probs.append(log_prob)
      return rv

    model_kwargs = _get_function_inputs(model, kwargs)
    with trace(tracer):
      model(*args, **model_kwargs)
    log_prob = sum(log_probs)
    return log_prob
  return log_joint_fn


def _get_function_inputs(f, src_kwargs):
  """Filters inputs to be compatible with function `f`'s signature.

  Args:
    f: Function according to whose input signature we filter arguments.
    src_kwargs: Keyword arguments to filter according to `f`.

  Returns:
    kwargs: Dict of key-value pairs in `src_kwargs` which exist in `f`'s
      signature.
  """
  if hasattr(f, "_func"):  # functions returned by tf.make_template
    f = f._func  # pylint: disable=protected-access

  try:  # getargspec was deprecated in Python 3.6
    argspec = inspect.getfullargspec(f)
  except AttributeError:
    argspec = inspect.getargspec(f)

  fkwargs = {k: v for k, v in six.iteritems(src_kwargs) if k in argspec.args}
  return fkwargs
