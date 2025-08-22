# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Common trainer utils."""

import jax
import jax.numpy as jnp


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceContext."""

  def __init__(self, name, init_step_num):
    self.name = name
    self.step_num = init_step_num

  def __enter__(self):
    self.context = jax.profiler.StepTraceAnnotation(
        self.name, step_num=self.step_num
    )
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    self.context.__exit__(None, None, None)
    self.__enter__()


def compute_grad_norm(grad):
  return global_norm(grad)


def global_norm(pytree):
  return jnp.sqrt(
      jnp.sum(
          jnp.asarray([jnp.sum(jnp.square(x)) for x in jax.tree.leaves(pytree)])
      )
  )
