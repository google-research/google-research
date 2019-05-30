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

"""Tracing operations.

This file collects common tracing operations, i.e., traces that each control the
execution of programs in a specific manner. For example, 'condition' traces a
program and fixes the value of random variables; and 'tape' traces the program
and records the executed random variables onto an ordered dictionary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

from simple_probabilistic_programming.trace import trace
from simple_probabilistic_programming.trace import traceable

__all__ = [
    "condition",
    "tape",
]


@contextlib.contextmanager
def condition(**model_kwargs):
  """Context manager for setting the values of random variables.

  Args:
    **model_kwargs: dict of str to Tensor. Keys are the names of random variable
      in the model. Values are Tensors to set their corresponding value to.

  Yields:
    None.

  #### Examples

  `condition` is typically used for binding observations to random variables
  in the model, or equivalently binding posterior samples to random variables
  in the model. This lets one compute likelihoods or prior densities.

  ```python
  import simple_probabilistic_programming as ed  # pylint: disable=line-too-long

  def probabilistic_matrix_factorization():
    users = ed.Normal(0., 1., sample_shape=[5000, 128], name="users")
    items = ed.Normal(0., 1., sample_shape=[7500, 128], name="items")
    ratings = ed.Normal(loc=tf.matmul(users, items, transpose_b=True),
                        scale=0.1,
                        name="ratings")
    return ratings

  users = tf.zeros([5000, 128])
  items = tf.zeros([7500, 128])
  with ed.condition(users=users, items=items):
    ratings = probabilistic_matrix_factorization()

  # Compute the likelihood given latent user preferences and item attributes set
  # to zero matrices, p(data | users=0, items=0).
  ratings.distribution.log_prob(data)
  ```
  """
  def _condition(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    name = kwargs.get("name")
    if name in model_kwargs:
      kwargs["value"] = model_kwargs[name]
    return traceable(f)(*args, **kwargs)

  with trace(_condition):
    yield


@contextlib.contextmanager
def tape():
  """Context manager for recording traceable executions onto a tape.

  Similar to `tf.GradientTape`, operations are recorded if they are executed
  within this context manager. In addition, the operation must be registered
  (decorated) as `ed.traceable`.

  Yields:
    tape: OrderedDict where operations are recorded in sequence. Keys are
      the `name` keyword argument to the operation (typically, a random
      variable's `name`) and values are the corresponding output of the
      operation. If the operation has no name, it is not recorded.

  #### Examples

  ```python
  import simple_probabilistic_programming as ed  # pylint: disable=line-too-long

  def probabilistic_matrix_factorization():
    users = ed.Normal(0., 1., sample_shape=[5000, 128], name="users")
    items = ed.Normal(0., 1., sample_shape=[7500, 128], name="items")
    ratings = ed.Normal(loc=tf.matmul(users, items, transpose_b=True),
                        scale=0.1,
                        name="ratings")
    return ratings

  with ed.tape() as model_tape:
    ratings = probabilistic_matrix_factorization()

  assert model_tape["users"].shape == (5000, 128)
  assert model_tape["items"].shape == (7500, 128)
  assert model_tape["ratings"] == ratings
  ```

  """
  tape_data = collections.OrderedDict({})

  def record(f, *args, **kwargs):
    """Records execution to a tape."""
    name = kwargs.get("name")
    output = traceable(f)(*args, **kwargs)
    if name:
      tape_data[name] = output
    return output

  with trace(record):
    yield tape_data
