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

"""Testing EV3 end-to-end."""

from ev3 import decide
from ev3 import optimize
from ev3 import propose
from ev3.model_history import decide as model_history_decide
from ev3.model_history import optimize as model_history_optimize
from ev3.model_history import propose as model_history_propose
from ev3.model_history import struct as model_history_struct
from ev3.model_history import test_util as model_history_test_util
import jax.numpy as jnp
import optax


def main():
  print('\nRunning EV3 end-to-end on a sample training example:')

  num_features = 2
  num_labels = 2
  batch_size = 32
  rand_seed = 42

  (
      propose_data_iter,
      optimize_data_iter,
      decide_data_iter,
      loss_fn,
      loss_states,
      vec_metric_fn,
      mean_metric_fn,
      model,
  ) = model_history_test_util.get_ev3_state_inputs(
      num_features,
      num_labels,
      batch_size,
      rand_seed,
      model_name='2lp',
  )

  tx_list = [
      optax.sgd(learning_rate=0.1),
      optax.sgd(learning_rate=0.01, momentum=0.95),
  ]
  p_state = model_history_struct.ProposeState(
      data_iter=propose_data_iter,
      loss_fn_list=tuple([loss_fn] * len(loss_states)),
      loss_states=tuple(loss_states),
      trajectory_length=100,
      tx_list=tuple(tx_list),
      traj_mul_factor=1,
  )
  p_tx = propose.get_propose_tx(
      p_state,
      propose_init_fn=model_history_propose.propose_init,
      propose_update_fn=model_history_propose.propose_update,
  )

  o_state = model_history_struct.OptimizeState(
      data_iter=optimize_data_iter,
      metric_fn_list=tuple([vec_metric_fn]),
      ucb_alpha=0.5,
  )
  o_tx = optimize.get_optimize_tx(
      o_state,
      optimize_init_fn=model_history_optimize.optimize_init,
      optimize_update_fn=model_history_optimize.optimize_update,
  )

  d_state = model_history_struct.DecideState(
      data_iter=decide_data_iter,
      metric_fn_list=tuple([vec_metric_fn]),
      ucb_alpha=4.0,
  )
  d_tx = decide.get_decide_tx(
      d_state,
      decide_init_fn=model_history_decide.decide_init,
      decide_update_fn=model_history_decide.decide_update,
  )

  tx = optax.chain(p_tx, o_tx, d_tx)
  state = tx.init(model)
  model_update, _ = tx.update(None, state, model)
  new_model = model + model_update

  # Make sure that the best update was a significant improvement.
  assert model_update.stable_params is not None

  # Sample a new batch for evaluation.
  test_batch = next(decide_data_iter)

  # Make sure that all losses were improved by the best update.
  old_losses = jnp.array(
      [
          loss_fn(model.params, model.graph, loss_state, test_batch)
          for loss_state in loss_states
      ]
  )
  new_losses = jnp.array(
      [
          loss_fn(new_model.params, new_model.graph, loss_state, test_batch)
          for loss_state in loss_states
      ]
  )
  assert (new_losses < old_losses).all()

  # Make sure that the metric was improved by the best update.
  old_metric = mean_metric_fn(model.params, model.graph, test_batch)
  new_metric = mean_metric_fn(new_model.params, new_model.graph, test_batch)
  assert new_metric > old_metric

  print(
      'Successfully trained a 2-layer feedforward network on a problem with'
      f' {num_features} features and {num_labels} labels using batches of size'
      f' {batch_size}.'
  )
  print(
      f'The starting accuracy was {round(float(old_metric), 3)}'
      f' and the final accuracy was {round(float(new_metric), 3)}.\n'
  )


if __name__ == '__main__':
  main()
