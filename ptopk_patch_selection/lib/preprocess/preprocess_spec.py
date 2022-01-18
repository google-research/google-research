# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""Library parsing a preprocessing spec.

A preprocessing spec is a list of preprocessing ops separated by '|' that can be
applied sequentially as a preprocessing function. The preprocessing ops are
provided as input.

By convention the preprocessing function operates on dictionaries of features.
Each op can change the dictionary by modifying, adding or removing dictionary
entries. Dictionary entries should be tensors, keys are strings.
The first argument of the op must be named `features` to which the feature
dictionary will be passed. Additional positional and keyword only arguments can
be defined. The processing spec can define values that will passed to those. The
op must return the feature dictionary.

For convenience ops can also operate an tensors of the feature dictionary
directly. In this case they must accept the names of the tensors (one or
multiple of _FEATURE_TENSORS} and return values for all input tensors.

Example spec: 'fn1|fn2(3)|fn3(keyword=5)'
This will construct the following preprocessing function:
def preprocess_fn(features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  features = fn1(features)
  features = fn2(features, 3)
  features = fn3(features, keyword=5)
  return features

WARNING: Do not use decorators when defining ops.
"""

import ast
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from absl import logging
import tensorflow as tf

# Any type of TF tensor.
Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]

# Dictionary with features.
Features = Dict[str, Tensor]

TPU_SUPPORTED_TYPES = frozenset(
    [tf.float32, tf.int32, tf.complex64, tf.int64, tf.bool, tf.bfloat16])


_FEATURE_TENSORS = ("features", "image", "label", "video", "segmentation_mask",
                    "instance_masks", "instance_labels", "rng")
OpFn = Callable[Ellipsis, Union[Features, tf.Tensor, Sequence[tf.Tensor]]]


def remove_non_tpu_features(features):
  """Removes all features which types are not supported on TPU."""
  for name in list(features):
    dtype = features[name].dtype
    if dtype not in TPU_SUPPORTED_TYPES:
      del features[name]
      msg = f"Removing {name!r} because dtype {dtype} is not supported on TPU."
      logging.warning(msg)
    elif isinstance(features[name], tf.SparseTensor):
      del features[name]
      msg = f"Removing features {name!r}. Sparse tensors not supported on TPU."
      logging.warning(msg)
  return features


class PreprocessOp:
  """Represents a processing operating.

  A process op consists of a method arguments passed to the method. The method
  can modify/add/remove features. For convenience the method can also directly
  operate on the tensors directly (_FEATURE_TENSORS).
  """

  def __init__(self, fn, kwargs = None):
    self._fn = fn
    self._kwargs = kwargs or {}
    self._argspec = inspect.getfullargspec(inspect.unwrap(fn))
    if not self._argspec.args or self._argspec.args[0] not in _FEATURE_TENSORS:
      raise ValueError(
          f"Method {fn} with argspec {self._argspec} cannot be used as "
          f"preprocessing operation. First argument must be one of "
          f"{_FEATURE_TENSORS} but was {self._argspec.args}.")

  def __call__(self, features):
    """Applies the process op to the given features."""
    try:
      return self._apply(features)
    except:
      msg = f"Failed to apply {self!r} to features {features}."
      logging.error(msg)
      raise

  def _apply(self, features):
    """Applies the preprocess op to given features."""
    features = features.copy()

    # Simple case: Function accepts a feature dictionary.
    if self._argspec.args[0] == "features":
      # These function should always return a feature dictionary, but PyType
      # doesn't know this.
      return self._fn(features, **self._kwargs)  # pytype: disable=bad-return-type

    # Handle special case with tensors passed directly.
    tensor_names = []
    for argname in self._argspec.args:
      if argname not in _FEATURE_TENSORS:
        break
      if argname not in features:
        raise ValueError(
            f"Tensor {argname} requested by {self._fn} but not available "
            f"features ({features}).")
      tensor_names.append(argname)
    if not tensor_names:
      raise ValueError(
          f"{self._fn} must either accept a dictionary with features as first "
          f"argument called 'features' or any number of tensors (with names in "
          f"{_FEATURE_TENSORS}) as positional arguments.")
    returned_tensors = self._fn(**{n: features[n] for n in tensor_names},
                                **self._kwargs)
    if len(tensor_names) == 1:
      returned_tensors = [returned_tensors]
    if len(returned_tensors) != len(tensor_names):
      raise ValueError(
          f"Number of returned tensors ({returned_tensors}) does not match "
          f"number of input tensors ({tensor_names}).")
    for i, name in enumerate(tensor_names):
      features[name] = returned_tensors[i]
    return features

  def __str__(self):
    """Returns a valid preprocess spec for this operations."""
    name = self._fn.__name__
    args = ", ".join([f"{k}={v}" for k, v in self._kwargs.items()])
    return f"{name}({args})"

  def __repr__(self):
    """Returns a representation string."""
    return (f"PreprocessOp(fn={self._fn}, kwargs={self._kwargs}, "
            f"argspec={self._argspec})")

  def __eq__(self, other):
    """Returns True if other is the same op with the same arguments."""
    if not isinstance(other, PreprocessOp):
      return False
    # We do not check if kwargs simply match default  argument values.
    # pylint: disable=protected-access
    return self._fn == other._fn and self._kwargs == other._kwargs
    # pylint: enable=protected-access


class PreprocessFn(object):
  """Chain of preprocessing ops combined to a single preprocessing function."""

  def __init__(self,
               ops,
               *,
               only_tpu_features = True):
    self._ops = ops
    self._only_tpu_features = only_tpu_features

  def __call__(self, features):
    logging.info("Features before preprocessing: %s", features)
    features = features.copy()
    for op in self._ops:
      features = op(features)
      logging.info("Features after op %s: %s", op, features)
    if self._only_tpu_features:
      features = remove_non_tpu_features(features)
    logging.info("Features after preprocessing and cleaning: %s", features)
    return features

  def __str__(self):
    """Returns a valid preprocess spec for this preprocess function."""
    return "|".join([str(op) for op in self._ops])


def _get_op_fn(expr, available_ops):
  """Gets the process op fn from the given expression."""
  if isinstance(expr, ast.Call):
    fn_name = expr.func.id
  elif isinstance(expr, ast.Name):
    fn_name = expr.id
  else:
    raise ValueError(
        f"Could not parse function name from expression: {expr!r}.")
  name_to_op = {op.__name__: op for op in available_ops}
  if fn_name in name_to_op:
    return name_to_op[fn_name]
  raise ValueError(
      f"'{fn_name}' is not available (available ops: {list(name_to_op)}).")


def parse_single_preprocess_op(spec,
                               available_ops):
  """Parsing the spec for a single preprocess op.

  The op can just be the method name or the method name followed by any
  arguments (both positional and keyword) to the method.
  See the test cases for some valid examples.

  Args:
    spec: String specifying a single processing operations.
    available_ops: Available preprocessing ops.

  Returns:
    The ProcessOp corresponding to the spec.
  """
  try:
    expr = ast.parse(spec, mode="eval").body  # pytype: disable=attribute-error
  except SyntaxError:
    raise ValueError(f"{spec!r} is not a valid preprocess op spec.")
  fn = _get_op_fn(expr, available_ops)

  # Simple case without arguments.
  if isinstance(expr, ast.Name):
    return PreprocessOp(fn)

  assert isinstance(expr, ast.Call)
  args = [ast.literal_eval(arg) for arg in expr.args]
  kwargs = {kv.arg: ast.literal_eval(kv.value) for kv in expr.keywords}
  if not args:
    return PreprocessOp(fn, kwargs)

  # Translate positional arguments into keyword arguments.
  argspec = inspect.getfullargspec(inspect.unwrap(fn))
  available_arg_names = [n for n in argspec.args if n not in _FEATURE_TENSORS]
  for i, arg in enumerate(args):
    name = available_arg_names[i]
    if name in kwargs:
      raise ValueError(
          f"Argument {name} to op {fn} given both as positional argument "
          f"(value: {arg}) and keyword argument (value: {kwargs[name]}).")
    kwargs[name] = arg

  return PreprocessOp(fn, kwargs)


def parse(spec,
          available_ops,
          *,
          only_tpu_features = True):
  """Parses a preprocess spec; a '|' separated list of preprocess ops."""
  if not spec.strip():
    ops = []
  else:
    ops = [
        parse_single_preprocess_op(s, available_ops) for s in spec.split("|")
    ]
  return PreprocessFn(ops, only_tpu_features=only_tpu_features)
