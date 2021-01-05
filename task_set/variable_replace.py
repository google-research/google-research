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

"""Custom getter that allows one to use custom values instead of tf.Variables.

This converts functions with implicit weights -- f(x),
to explicit -- f(x, params). See `VariableReplaceGetter` for more info.
"""

import collections
import contextlib
import tensorflow.compat.v1 as tf

# Two modes for the custom getter to be in.
_UseVariables = collections.namedtuple("UseVariables", [])
_UseValues = collections.namedtuple("UseValues", ["values"])


class VariableReplaceGetter(object):
  """Getter that swaps out internal tf.Variable with tf.Tensor values.

  By default tensorflow hides away access to variables. A function that would
  normally be a function of both data and variables: f(variables, data)
  is presented as a function of just data: f(data) with variables hidden
  away in variable scopes. This custom getter can be used to create functions
  that use tensorflow's neural network construction libraries while exposing
  the underlying variables in a way that users can swap other values in.


  This can be used from things like evolutionary strategies, to unrolled
  optimization.

  ```
  context = VariableReplaceGetter()
  mod = snt.Linear(123, custom_getter=context)

  with context.use_variables():
    y1 = mod(x1) # use variables

  values = context.get_variable_dict()
  # modify the current set of weights
  new_values = {k: v+1 for k,v in values.items()}

  with context.use_value_dict(new_values):
    y2 = mod(x1)
  ```
  """

  def __init__(self, verbose=False):
    """Initializer.

    Args:
      verbose: bool If true, log when inside these contexts.
    """
    self._verbose = verbose

    # Store initializers for each variable created.
    self._variable_initializer_dict = collections.OrderedDict()

    # Store the variables created for each variable created.
    self._variable_dict = collections.OrderedDict()

    # store the current state of the custom getter.
    # Is either an instance of:
    #  * _UseVariables which causes the custom getter to return tf.Variable
    #    created upon first call of tf.get_variables
    #  * _UseValues: which uses the set tf.Tensors instead of variables
    self._context_state = None

  def __call__(self, getter, name, *args, **kwargs):
    """The custom getter.

    Do not call directly, instead pass to a variable_scope.

    Args:
      getter: callable the default getter
      name: str name of variable to get
      *args: args forwarded to `getter`
      **kwargs: kwargs forwarded to `getter`

    Returns:
      tf.Tensor or tf.Variable
    """
    # Only do variable replacement on trainable variables.
    # If not trainable, skip this custom getter.
    if not kwargs["trainable"]:
      if self._verbose:
        tf.logging.info("Skipping non-trainable %s" % name)

      # If in the _UseValues context ensure that the name of the non-trainable
      # variable has not been given a value.
      if isinstance(self._context_state, _UseValues):
        values = self._context_state.values
        if name in values:
          raise ValueError(
              "The name [%s] was found in the value_dict but it is a"
              " non-trainable variable. Either remove it from the"
              "value_dict or make it trainable!")

      return getter(name, *args, **kwargs)

    if self._context_state is None:
      raise ValueError("Only call in a `use_variables`, `use_value_dict`,"
                       " context!")

    # Store the variable created by the default getter
    if self._verbose:
      tf.logging.info("Getting %s with normal getter" % name)
    orig_var = getter(name, *args, **kwargs)
    self._variable_dict[name] = orig_var

    # Store the default initializer
    if name not in self._variable_initializer_dict:
      if kwargs["initializer"] is not None:
        shape = tf.TensorShape(kwargs["shape"]).as_list()
        # pylint: disable=g-long-lambda
        init_fn = lambda: kwargs["initializer"](
            shape=shape, dtype=kwargs["dtype"])
      else:
        # If there is no initializer set, just use the initial value.
        init_fn = lambda: orig_var.initial_value

      self._variable_initializer_dict[name] = init_fn

    # This custom getter is in 1 of two modes -- _UseVariables or _UseValues.
    # The mode is determined by _context_state.
    if isinstance(self._context_state, _UseVariables):
      return orig_var

    elif isinstance(self._context_state, _UseValues):
      if self._verbose:
        tf.logging.info("Getting %s from values" % name)
      values = self._context_state.values
      if name not in values:
        message = ("Name: %s not specified in the values. \nValid names:\n %s" %
                   (name, "\n".join("    %s" % k for k in values.keys())))
        raise ValueError(message)
      if self._verbose:
        tf.logging.info("Tensor returned %s" % values[name])
      return values[name]
    else:
      raise ValueError("Bad type of self._context_state. Got [%s]" %
                       type(self._context_state))

  @contextlib.contextmanager
  def use_variables(self):
    """Context for using the tf.Variables which are stored in the tf Graph."""
    if self._context_state is not None:
      raise NotImplementedError("Nested contexts not allowed at this point.")
    self._context_state = _UseVariables()
    yield
    self._context_state = None

  @contextlib.contextmanager
  def use_value_dict(self, values):
    """Context for using the values, instead of the local variables.

    Args:
      values: dict maps name to value to use when a tf.get_variable is called

    Yields:
      None
    """
    if self._context_state is not None:
      raise NotImplementedError("Nested contexts not allowed at this point." "")
    self._context_state = _UseValues(values=values)
    yield
    self._context_state = None

  def get_initialized_value_dict(self):
    """Return a dictionary of names to tf.Tensor with the initial values.

    Returns:
      collections.OrderedDict with initialized tf.Tensor values.
    """
    d = [(name, init())
         for name, init in self._variable_initializer_dict.items()]
    return collections.OrderedDict(d)

  def get_variable_dict(self):
    """Return a dictionary of names to tf.Variable with the variables created.

    Returns:
      collections.OrderedDict with tf.Variable.
    """
    return self._variable_dict
