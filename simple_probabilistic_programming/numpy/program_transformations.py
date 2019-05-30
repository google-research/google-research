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

"""Transformations of Edward2 programs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys
import numpy as np
import six

from simple_probabilistic_programming.trace import trace


def make_log_joint_fn(model):
  """Takes Edward2 probabilistic program and returns its log joint function.

  Args:
    model: Python callable which executes the generative process of a
      computable probability distribution using Edward2 random variables.

  Returns:
    A log-joint probability function. Its inputs are `model`'s original inputs
    and random variables which appear during the program execution. Its output
    is a scalar `np.ndarray`.

  #### Examples

  Below we define Bayesian logistic regression as an Edward2 program, which
  represents the model's generative process. We apply `make_log_joint_fn` in
  order to alternatively represent the model in terms of its joint probability
  function.

  ```python
  import edward2.numpy as ed

  def model(X):
    beta = ed.norm.rvs(loc=0., scale=0.1, size=X.shape[1])
    loc = np.einsum('ij,j->i', X, beta)
    y = ed.norm.rvs(loc=loc, scale=1.)
    return y

  log_joint = ed.make_log_joint_fn(model)

  X = np.random.normal(size=[3, 2])
  beta = np.random.normal(size=[2])
  y = np.random.normal(size=[3])
  out = log_joint(X, beta, y)
  ```

  One can use kwargs in `log_joint` if `rvs` are given `name` kwargs.

  ```python
  def model(X):
    beta = ed.norm.rvs(loc=0., scale=0.1, size=X.shape[1], name="beta")
    loc = np.einsum('ij,j->i', X, beta)
    y = ed.norm.rvs(loc=loc, scale=1., name="y")
    return y

  log_joint = ed.make_log_joint_fn(model)
  out = log_joint(X, y=y, beta=beta)
  ```

  #### Notes

  For implementation, we make several requirements:

  1. A random variable's `rvs` method has the same kwargs as scipy.stats'
    `logpmf`/`logpdf` up to `size` and `random_state`.
  2. User must use explicit kwargs (no positional arguments) when specifying
     `size` and `random_state` in the `rvs` method.
     TODO(trandustin): Relax this requirement.
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
      Scalar `np.ndarray`, which represents the model's log-probability summed
      over all Edward2 random variables and their dimensions.

    Raises:
      TypeError: If a random variable in the model has no specified value in
        `**kwargs`.
    """
    log_probs = []
    args_counter = []

    def tracer(rv_call, *rv_args, **rv_kwargs):
      """Overrides a random variable's `value` and accumulates its log-prob."""
      if len(args) - len(args_counter) > 0:
        value = args[len(args_counter)]
        args_counter.append(0)
      else:
        # Set value to keyword argument indexed by `name` (an input tensor).
        rv_name = rv_kwargs.get("name")
        if rv_name is None:
          raise KeyError("Random variable call {} has no name in its arguments."
                         .format(rv_call.im_class.__name__))
        value = kwargs.get(rv_name)
        if value is None:
          raise LookupError("Keyword argument specifying value for {} is "
                            "missing.".format(rv_name))
      if sys.version_info < (3,):
        cls = rv_call.im_class
      else:
        cls = rv_call.__self__.__class__
      log_prob_fn = getattr(cls, "logpdf", getattr(cls, "logpmf", None))
      rv_kwargs.pop("size", None)
      rv_kwargs.pop("random_state", None)
      rv_kwargs.pop("name", None)
      log_prob = np.sum(log_prob_fn(cls(), value, *rv_args, **rv_kwargs))
      log_probs.append(log_prob)
      return value

    args, model_args, model_kwargs = _get_function_inputs(
        model, *args, **kwargs)
    with trace(tracer):
      model(*model_args, **model_kwargs)
    log_prob = sum(log_probs)
    return log_prob
  return log_joint_fn


def _get_function_inputs(f, *args, **kwargs):
  """Filters inputs to be compatible with function `f`'s signature.

  Args:
    f: Function according to whose input signature we filter arguments.
    *args: Keyword arguments to filter according to `f`.
    **kwargs: Keyword arguments to filter according to `f`.

  Returns:
    New original args, args of f, kwargs of f.
  """
  if hasattr(f, "_func"):  # functions returned by tf.make_template
    argspec = inspect.getargspec(f._func)  # pylint: disable=protected-access
  else:
    argspec = inspect.getargspec(f)

  fkwargs = {}
  for k, v in six.iteritems(kwargs):
    if k in argspec.args:
      fkwargs[k] = v
      kwargs.pop(k)
  num_args = len(argspec.args) - len(fkwargs)
  fargs = args[:num_args]
  new_args = args[num_args:]
  return new_args, fargs, fkwargs
