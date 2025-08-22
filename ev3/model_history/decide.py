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

"""Methods for EV3's decide step."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import pandas as pd

from ev3 import base
from ev3.model_history import eval_util
from ev3.model_history import struct


PARAMS = eval_util.PARAMS
STABLE_PARAMS = eval_util.STABLE_PARAMS


def select_best_update(
    model_updates,
    eval_results_ucb,
    eval_results_lcb,
    all_params,
    **kwargs,
):
  """Selects the best overall update including the current model.

  Args:
    model_updates: The collection of updates to be chosen from.
    eval_results_ucb: Upper confidence bounds on the performance of each update
      according to the metric that we would like to optimize.
    eval_results_lcb: Lower confidence bounds on the performance of each update
      according to the metric that we would like to optimize.
    all_params: The updated parameters whose evaluations are included in
      eval_results_lcb.
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
  ucb = eval_results_ucb[0]
  lcb = eval_results_lcb[0]

  # Choose the best update based on the maximum of the lower confidence bound
  # of the metric.
  best_param_key = lcb.idxmax()
  # Check if the best params are from a model update.
  if best_param_key in range(len(model_updates)):
    best_model_update = model_updates[best_param_key]
  else:
    best_model_update = struct.ModelUpdates(
        params_list=(all_params[best_param_key],)
    )  # pytype: disable=wrong-keyword-args  # dataclass_transform

  # If the best update is significantly better than the stable params, update
  # the stable params.
  significantly_better = False
  if lcb[best_param_key] > ucb[STABLE_PARAMS]:
    significantly_better = True
    best_model_update = best_model_update.replace(
        stable_params=all_params[best_param_key]
    )

  if best_model_update.logs is None:
    best_model_update = best_model_update.replace(
        logs={
            'selected_loss_idx': -1,
        },
    )
  # Add eval results to model update logs.
  return best_model_update.replace(
      logs=best_model_update.logs
      | {
          'ucb': ucb,
          'lcb': lcb,
          'acc': (lcb + ucb) / 2,
          'best_param_key': best_param_key,
          'significantly_better': significantly_better,
      }
  )


def modify_state(state, **kwargs):
  del kwargs
  return state


def update_model_graph(
    best_updates,
    state,
    models,
    get_batch_fn,
    history_tracking_length = 5,
):
  """Updates the model graph for MLP.

  Args:
    best_updates: A ModelUpdates object to store the new graph.
    state: The DecideState object.
    models: A Model object to be updated.
    get_batch_fn: To get a batch of data to initialize the new graph.
    history_tracking_length: How many events to consider whether to update the
      graph.

  Returns:
    A ModelUpdates that contains the new graph.
  """
  if models.history is None:
    return best_updates
  if len(models.history) < history_tracking_length:
    return best_updates

  significance = [
      logs['significantly_better']
      for logs in models.history[-history_tracking_length:]
  ]
  if any(significance):
    return best_updates
  else:
    batch = get_batch_fn(state)
    new_key, expansion_key = jax.random.split(models.rand_key, 2)
    new_model_graph, new_params, new_stable_params = models.graph.expand(
        models.params, models.stable_params, expansion_key, batch
    )
    if best_updates.logs is None:
      logs = {}
    else:
      logs = best_updates.logs
    logs['updated_graph'] = new_model_graph

    return best_updates.replace(
        params_list=[new_params],
        graph=new_model_graph,
        stable_params=new_stable_params,
        rand_key=new_key,
        logs=logs,
    )


def decide_init(
    state,
    model,
    get_batch_fn = eval_util.get_batch,
):
  """Initializes the decide state.

  This methods makes sure that the metric functions included in the 'state'
  return arrays when evaluated on a batch, rather than a single values. This is
  needed in order to compute confidence intervals for the evluation results.

  Args:
    state: An object containing information relevant to the decide step of EV3,
      including metrics and a data iterator.
    model: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.

  Returns:
    The initialized decide state.

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


def decide_update(
    updates,
    state,
    models,
    get_batch_fn = eval_util.get_batch,
    evaluate_updates_fn = eval_util.evaluate_updates,
    select_best_updates_fn = select_best_update,
    modify_state_fn = modify_state,
    update_model_graph_fn = update_model_graph,
):
  """The default update function for the decide step of EV3.

  Args:
    updates: A connection of proposed updates to model parameters that we can
      choose from.
    state: An object containing information relevant to the decide step of EV3,
      including metrics and a data iterator.
    models: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    evaluate_updates_fn: A function that evaluates the proposed updates.
    select_best_updates_fn: A function that choose the best option among the
      proposed updates based on the evaluation results returned by
      evaluate_updates_fn.
    modify_state_fn: A function that allows for 'state' to be updated in case
      that is needed.
    update_model_graph_fn: A function that modifies the model graph.

  Returns:
    A tuple (updates, state), where 'updates' contained the most promising
    updates and 'state' is the updated optimize state.
  """
  # Evaluate the updates as well as the current params.
  eval_results_kwargs = evaluate_updates_fn(
      updates, state, models, add_model_params=True, get_batch_fn=get_batch_fn
  )

  # Decide on the best updates.
  best_updates = select_best_updates_fn(updates, **eval_results_kwargs)

  # Update the stable params.
  new_state = modify_state_fn(state, **eval_results_kwargs)

  # Update the model graph.
  best_updates = update_model_graph_fn(
      best_updates, new_state, models, get_batch_fn
  )

  return best_updates, new_state


def trival_decide_update(
    updates,
    state,
    models,
    get_batch_fn = eval_util.get_batch,
    evaluate_updates_fn = eval_util.evaluate_updates,
):
  """A simple update function to just evaluate and log metrics.

  Args:
    updates: A connection of proposed updates to model parameters that we can
      choose from.
    state: An object containing information relevant to the decide step of EV3,
      including metrics and a data iterator.
    models: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    evaluate_updates_fn: A function that evaluates the proposed updates.

  Returns:
    A tuple (updates, state), where 'updates' contained the most promising
    updates and 'state' is the updated optimize state.
  """
  # Evaluate the updates.
  eval_results = evaluate_updates_fn(
      updates, state, models, add_model_params=False, get_batch_fn=get_batch_fn
  )

  # Decide on the best updates.
  best_model_update = updates[0]

  if best_model_update.logs is None:
    best_model_update = best_model_update.replace(logs={})
  # Add eval results to model update logs.
  best_model_update = best_model_update.replace(
      logs=best_model_update.logs
      | {
          'ucb': eval_results['eval_results_ucb'][0],
          'lcb': eval_results['eval_results_lcb'][0],
          'acc': (
              eval_results['eval_results_lcb'][0]
              + eval_results['eval_results_ucb'][0]
          ) / 2,
          'best_param_key': 0,
          'significantly_better': False,
      }
  )
  return best_model_update, state


def trival_decide_update_with_expansion(
    updates,
    state,
    models,
    get_batch_fn = eval_util.get_batch,
    evaluate_updates_fn = eval_util.evaluate_updates,
    update_model_graph_fn = update_model_graph,
):
  """A simple update function to just evaluate and log metrics.

  Args:
    updates: A connection of proposed updates to model parameters that we can
      choose from.
    state: An object containing information relevant to the decide step of EV3,
      including metrics and a data iterator.
    models: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    evaluate_updates_fn: A function that evaluates the proposed updates.
    update_model_graph_fn: A function that modifies the model graph.

  Returns:
    A tuple (updates, state), where 'updates' contains the most promising
    updates and 'state' is the updated optimize state.
  """
  # Evaluate the updates.
  eval_results = evaluate_updates_fn(
      updates, state, models, add_model_params=False, get_batch_fn=get_batch_fn
  )

  # Decide on the best updates.
  best_model_update = updates[0]

  # Update the model graph.
  best_model_update = update_model_graph_fn(
      best_model_update, state, models, get_batch_fn
  )

  if best_model_update.logs is None:
    best_model_update = best_model_update.replace(logs={})
  # Add eval results to model update logs.
  best_model_update = best_model_update.replace(
      logs=best_model_update.logs
      | {
          'ucb': eval_results['eval_results_ucb'][0],
          'lcb': eval_results['eval_results_lcb'][0],
          'acc': (
              eval_results['eval_results_lcb'][0]
              + eval_results['eval_results_ucb'][0]
          ) / 2,
          'best_param_key': 0,
          'significantly_better': False,
      }
  )
  return best_model_update, state
