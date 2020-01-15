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

# Lint as: python2, python3
"""Transformations of Edward2 programs."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import inspect
import six
import tensorflow.compat.v1 as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.experimental.edward2.generated_random_variables import Normal
from tensorflow_probability.python.experimental.edward2.interceptor import interceptable
from tensorflow_probability.python.experimental.edward2.interceptor import interception

__all__ = [
    'make_log_joint_fn', 'make_variational_model', 'make_value_setter', 'ncp',
    'get_trace'
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
  from tensorflow_probability import edward2 as ed

  def logistic_regression(features):
    coeffs = ed.Normal(loc=0., scale=1.,
                       sample_shape=features.shape[1], name='coeffs')
    outcomes = ed.Bernoulli(logits=tf.tensordot(features, coeffs, [[1], [0]]),
                            name='outcomes')
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

    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
      """Overrides a random variable's `value` and accumulates its log-prob."""
      # Set value to keyword argument indexed by `name` (an input tensor).
      rv_name = rv_kwargs.get('name')
      if rv_name is None:
        raise KeyError('Random variable constructor {} has no name '
                       'in its arguments.'.format(rv_constructor.__name__))
      value = kwargs.get(rv_name)
      if value is None:
        raise LookupError('Keyword argument specifying value for {} is '
                          'missing.'.format(rv_name))
      rv_kwargs['value'] = value

      rv = rv_constructor(*rv_args, **rv_kwargs)
      log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
      log_probs.append(log_prob)
      return rv

    model_kwargs = _get_function_inputs(model, **kwargs)
    with interception(interceptor):
      try:
        model(*args, **model_kwargs)
      except TypeError as err:
        raise Exception(
            'Wrong number of arguments in log_joint function definition. {}'
            .format(err))
    log_prob = sum(log_probs)
    return log_prob
  return log_joint_fn


def make_value_setter(*positional_args, **model_kwargs):
  """Creates a value-setting interceptor.

  Args:
    *positional_args: Optional positional `Tensor` values. If provided, these
      will be used to initialize variables in the order they occur in the
      program's execution trace.
    **model_kwargs: dict of str to Tensor. Keys are the names of random variable
      in the model to which this interceptor is being applied. Values are
      Tensors to set their value to.

  Returns:
    set_values: Function which sets the value of intercepted ops.
  """

  consumable_args = [x for x in positional_args]
  if len(consumable_args) and len(model_kwargs):
    raise ValueError(
        'make_value_setter does not support simultaneous '
        'use of positional and keyword args.'
    )

  def set_values(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    name = kwargs.get('name')
    if name in model_kwargs:
      kwargs['value'] = model_kwargs[name]
    elif consumable_args:
      kwargs['value'] = consumable_args.pop(0)
    return interceptable(f)(*args, **kwargs)
  return set_values


def get_trace(model, *args, **kwargs):

  trace_result = {}

  def trace(rv_constructor, *rv_args, **rv_kwargs):
    rv = interceptable(rv_constructor)(*rv_args, **rv_kwargs)
    name = rv_kwargs['name']
    trace_result[name] = rv.value
    return rv

  with interception(trace):
    model(*args, **kwargs)

  return trace_result


def make_variational_model(model, *args, **kwargs):

  variational_parameters = collections.OrderedDict()

  def get_or_init(name, shape=None):

    loc_name = model.__name__ + '_' + name + '_loc'
    scale_name = model.__name__ + '_' + name + '_scale'

    if loc_name in variational_parameters.keys() and \
       scale_name in variational_parameters.keys():
      return (variational_parameters[loc_name],
              variational_parameters[scale_name])
    else:
      # shape must not be None
      variational_parameters[loc_name] = \
          tf.get_variable(name=loc_name,
                          initializer=1e-10*tf.ones(shape, dtype=tf.float32))

      variational_parameters[scale_name] = tf.nn.softplus(
          tf.get_variable(name=scale_name,
                          initializer=-10*tf.ones(shape, dtype=tf.float32)))
      return (variational_parameters[loc_name],
              variational_parameters[scale_name])

  def mean_field(rv_constructor, *rv_args, **rv_kwargs):

    name = rv_kwargs['name']
    if name not in kwargs.keys():
      rv = rv_constructor(*rv_args, **rv_kwargs)
      loc, scale = get_or_init('q_' + name, rv.shape)

      # NB: name must be the same as original variable,
      # in order to be able to do black-box VI (setting
      # parameters to variational values obtained via trace).
      return Normal(loc=loc, scale=scale, name=name)

    else:
      rv_kwargs['value'] = kwargs[name]
      return rv_constructor(*rv_args, **rv_kwargs)

  def variational_model(*args):
    with interception(mean_field):
      return model(*args)

  _ = variational_model(*args)

  return variational_model, variational_parameters


# FIXME: Assumes the name of the data starts with y... Need to fix so that
# it works with user-specified data.
def ncp(rv_constructor, *rv_args, **rv_kwargs):
  if (rv_constructor.__name__ == 'Normal'
      and not rv_kwargs['name'].startswith('y')):
    loc = rv_kwargs['loc']
    scale = rv_kwargs['scale']
    name = rv_kwargs['name']

    shape = rv_constructor(*rv_args, **rv_kwargs).shape

    kwargs_std = {}
    kwargs_std['loc'] = tf.zeros(shape)
    kwargs_std['scale'] = tf.ones(shape)
    kwargs_std['name'] = name + '_std'

    b = tfb.AffineScalar(scale=scale, shift=loc)
    if 'value' in rv_kwargs:
      kwargs_std['value'] = b.inverse(rv_kwargs['value'])

    rv_std = interceptable(rv_constructor)(*rv_args, **kwargs_std)

    return b.forward(rv_std)

  else:
    return interceptable(rv_constructor)(*rv_args, **rv_kwargs)


def make_learnable_parametrisation(init_val_loc=None,
                                   init_val_scale=None,
                                   learnable_parameters=None,
                                   tau=1.):

  allow_new_variables = False
  if learnable_parameters is None:
    learnable_parameters = collections.OrderedDict()
    allow_new_variables = True

  def get_or_init(name, shape):
    loc_name = name + '_a'
    scale_name = name + '_b'

    if loc_name in learnable_parameters.keys() and \
        scale_name in learnable_parameters.keys():
      return learnable_parameters[loc_name], learnable_parameters[scale_name]
    else:
      if not allow_new_variables:
        raise Exception(
            'trying to create a variable for {}, but '
            'parameterization was already passed in ({})'
            .format(name, learnable_parameters))
      learnable_parameters[loc_name] = tf.sigmoid(  # tf.nn.relu(
          tau * tf.get_variable(
              name=loc_name + '_unconstrained',
              initializer=tf.ones(shape) * init_val_loc))

      learnable_parameters[scale_name] = tf.sigmoid(  # tf.nn.relu(
          tau * tf.get_variable(
              name=scale_name + '_unconstrained',
              initializer=tf.ones(shape) * init_val_scale))

      return learnable_parameters[loc_name], learnable_parameters[scale_name]

  bijectors = collections.OrderedDict()
  def recenter(rv_constructor, *rv_args, **rv_kwargs):
    if (rv_constructor.__name__ == 'Normal'
        and not rv_kwargs['name'].startswith('y')):

      # NB: assume everything is kwargs for now.
      x_loc = rv_kwargs['loc']
      x_scale = rv_kwargs['scale']

      name = rv_kwargs['name']
      shape = rv_constructor(*rv_args, **rv_kwargs).shape

      a, b = get_or_init(name, shape)  # w

      kwargs_std = {}
      kwargs_std['loc'] = tf.multiply(x_loc, a)
      kwargs_std['scale'] = tf.pow(x_scale, b)
      kwargs_std['name'] = name + '_param'

      scale = tf.pow(x_scale, 1. - b)
      b = tfb.AffineScalar(
          scale=scale, shift=x_loc + tf.multiply(scale, -kwargs_std['loc']))
      if 'value' in rv_kwargs:
        kwargs_std['value'] = b.inverse(rv_kwargs['value'])

      rv_std = interceptable(rv_constructor)(*rv_args, **kwargs_std)
      bijectors[name] = b
      return b.forward(rv_std)

    else:
      return interceptable(rv_constructor)(*rv_args, **rv_kwargs)

  return learnable_parameters, recenter, bijectors


def _get_function_inputs(f, **kwargs):
  """Filters inputs to be compatible with function `f`'s signature.

  Args:
    f: Function according to whose input signature we filter arguments.
    **kwargs: Keyword arguments to filter according to `f`.

  Returns:
    Dict of key-value pairs in `kwargs` which exist in `f`'s signature.
  """
  if hasattr(f, '_func'):  # functions returned by tf.make_template
    f = f._func  # pylint: disable=protected-access

  try:  # getargspec was deprecated in Python 3.6
    argspec = inspect.getfullargspec(f)
  except AttributeError:
    argspec = inspect.getargspec(f)  # pylint: disable=deprecated-method

  fkwargs = {k: v for k, v in six.iteritems(kwargs) if k in argspec.args}
  return fkwargs
