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

"""Methods for the propose stage of EV3."""

import optax

from ev3 import base


def get_propose_tx(
    propose_state,
    propose_init_fn,
    propose_update_fn,
):
  """Generates an Optax transformation that proposes updates to the model(s).

  Note on tx: In the JAX ecosystem, it is customary to refer to the gradient
  transformation as tx.

  Args:
    propose_state: Parameters for the propose stage.
    propose_init_fn: A function that initializes the state object of the propose
      step.
    propose_update_fn: A function that proposes model updates.

  Returns:
    An Optax transformation object consisting of a pair of functions that
    encode how to initialize the proposal state and how to generate proposed
    updates.
  """

  def init_fn(model):
    return propose_init_fn(propose_state, model)

  def update_fn(updates, state, model, **extra_args):
    del updates, extra_args
    return propose_update_fn(state, model)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)
