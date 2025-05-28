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

"""Methods for EV3's optimize step."""

from typing import Callable, Tuple

import jax.numpy as jnp
import pandas as pd

from ev3.model_history import eval_util
from ev3.model_history import struct


PARAMS = eval_util.PARAMS
STABLE_PARAMS = eval_util.STABLE_PARAMS


def select_best_update(
    model_updates,
    eval_results_lcb,
    **kwargs,
):
  """Selects the best update among the proposed model updates.

  Args:
    model_updates: The collection of updates to be chosen from.
    eval_results_lcb: Lower confidence bounds on the performance of each update
      according to the metric that we would like to optimize.
    **kwargs: Other arguments that were returned by the evaluation method and
      which this method does not need.

  Returns:
    The model update with the highest lower confidence bound, which is
    considered to be the safest bet.

  Raises:
    NotImplementedError: The evaluation results in eval_results_lcb contain
      results for multiple metrics.
  """
  del kwargs
  if len(eval_results_lcb.index.unique(level='metric index')) > 1:
    raise NotImplementedError(
        'This function can only deal with a single metric.'
    )
  # Note that the following only uses the first metric (i.e. the one with
  # index 0).
  lcb = eval_results_lcb[0]

  # Choose the best update based on the maximum of the lower confidence bound
  # of the metric.
  best_param_key = lcb.idxmax()
  if best_param_key in range(len(model_updates)):
    best_model_update = model_updates[best_param_key]
  else:
    raise ValueError(
        f'best_param_key should be in {list(range(len(model_updates)))}, but it'
        f' is {best_param_key}.'
    )

  if best_model_update.logs is None:
    best_model_update = best_model_update.replace(logs={})
  return best_model_update.replace(
      logs=best_model_update.logs | {'selected_loss_idx': best_param_key}
  )


def modify_state(
    state, **kwargs
):
  del kwargs
  return state


def optimize_init(
    state,
    model,
    get_batch_fn = eval_util.get_batch,
):
  """Initializes the optimize state.

  This methods makes sure that the metric functions included in the 'state'
  return arrays when evaluated on a batch, rather than a single values. This is
  needed in order to compute confidence intervals for the evluation results.

  Args:
    state: An object containing information relevant to the optimize step of
      EV3, including metrics and a data iterator.
    model: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.

  Returns:
    The initialized optimize state.

  Raise:
    AttributeError: One or more of the metric functions included in the 'state'
      do not return an array of values when evaluated on a batch of data.
  """
  batch = get_batch_fn(state)
  for m_ind, metric_fn in enumerate(state.metric_fn_list):
    m_arr = metric_fn(model.params, model.graph, batch)
    if not jnp.shape(m_arr):
      raise AttributeError(
          f'The output of the metric indexed {m_ind} is not an array.'
      )
    if max(jnp.shape(m_arr)) <= 1:
      raise AttributeError(
          'The output of each metric needs to be a non-trivial vector, but '
          f'the metric indexed {m_ind} generated a single value.'
      )
  return state


def optimize_update(
    updates,
    state,
    models,
    get_batch_fn = eval_util.get_batch,
    evaluate_updates_fn = eval_util.evaluate_updates,
    select_best_updates_fn = select_best_update,
    modify_state_fn = modify_state,
):
  """The default update function for the optimize step of EV3.

  Args:
    updates: A connection of proposed updates to model parameters that we can
      choose from.
    state: An object containing information relevant to the optimize step of
      EV3, including metrics and a data iterator.
    models: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    evaluate_updates_fn: A function that evaluates the proposed updates.
    select_best_updates_fn: A function that choose the best option among the
      proposed updates based on the evaluation results returned by
      evaluate_updates_fn.
    modify_state_fn: A function that allows for 'state' to be updated in case
      that is needed. Note that the default function is a no-op that does not
      modify the state.

  Returns:
    A tuple (updates, state), where 'updates' contains the most promising
    updates and 'state' is the updated optimize state.
  """
  # Evaluate the updates.
  eval_results_kwargs = evaluate_updates_fn(
      updates, state, models, add_model_params=False, get_batch_fn=get_batch_fn
  )

  # Select the best updates.
  best_updates = select_best_updates_fn(updates, **eval_results_kwargs)

  # Update the stable params.
  new_state = modify_state_fn(state, **eval_results_kwargs)

  return best_updates, new_state
