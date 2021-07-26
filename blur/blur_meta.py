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

"""Functions for setting up meta-learning graph."""

from typing import List, Optional, Sequence, Callable, Tuple

import tensorflow.compat.v1 as tf

from blur import blur
from blur import blur_env
from blur import genome_util
from blur import synapse_util


def init_first_state(
    genome,
    data,
    hidden_layers,
    synapse_initializer,
    create_synapses_fn = synapse_util.create_synapses,
    network_spec = blur.NetworkSpec(),
    env = blur_env.tf_env
):
  """Initialize the very first state of the graph."""
  if create_synapses_fn is None:
    create_synapses_fn = synapse_util.create_synapses

  if callable(genome):
    num_neuron_states = genome(0).synapse.transform.pre.shape[-1]
  else:
    num_neuron_states = genome.synapse.transform.pre.shape[-1]
  output_shape = data.element_spec[1].shape
  num_outputs = output_shape[-1]
  num_inputs = data.element_spec[0].shape[-1]
  batch_dims = output_shape[:-1]

  layer_sizes = (num_inputs, *hidden_layers, num_outputs)
  layers = [tf.zeros((*batch_dims, h, num_neuron_states)) for h in layer_sizes]
  synapses = create_synapses_fn(layers, synapse_initializer)
  if network_spec.symmetric_in_out_synapses:
    for i in range(len(synapses)):
      synapses[i] = synapse_util.sync_in_and_out_synapse(
          synapses[i], layers[i].shape[-2], env)
  if network_spec.symmetric_states_synapses:
    for i in range(len(synapses)):
      synapses[i] = synapse_util.sync_states_synapse(synapses[i], env)
  num_updatable_units = len(hidden_layers) + 1
  ground_truth = tf.zeros((*batch_dims, num_outputs))

  return blur.NetworkState(
      layers=layers, synapses=synapses,
      ground_truth=ground_truth,
      updatable=[False] + [True] * num_updatable_units)


def build_graph(
    genome,
    unroll_steps=5,
    *,
    data,
    hidden_layers,
    synapse_initializer,
    create_synapses_fn = synapse_util.create_synapses,
    input_fn = None,
    out_intermediate_states = None,
    network_spec = None,
    ):
  """Builds a tensorflow graph with given number of unroll steps."""
  if network_spec is None:
    network_spec = blur.default_network_spec(env=blur_env.tf_env)
  state = init_first_state(genome,
                           data,
                           hidden_layers,
                           synapse_initializer,
                           create_synapses_fn,
                           network_spec)
  if input_fn is None:
    input_fn = tf.data.make_one_shot_iterator(data).get_next
  return unroll_graph(
      genome=genome, initial_state=state,
      unroll_steps=unroll_steps,
      input_fn=input_fn,
      network_spec=network_spec,
      out_intermediate_states=out_intermediate_states)


def unroll_graph(
    genome, initial_state, unroll_steps, input_fn,
    network_spec=None,
    out_intermediate_states = None
):
  """Unrolls initial step given number of unroll steps."""

  states = [initial_state]
  state = initial_state
  for i in range(unroll_steps):
    has_more_steps = i < unroll_steps - 1
    state = blur.network_step(
        state,
        genome(i) if callable(genome) else genome,
        input_fn=input_fn,
        backward_pass_neurons=has_more_steps,
        backward_pass_synapses=has_more_steps,
        network_spec=network_spec,
        debug_hidden_states=out_intermediate_states,
        step=i,
        env=blur_env.tf_env,
    )
    states.append(state)
  return states


def l2_score_per_species(states, weights=None):
  """Returns l2_score (-l2_loss) per species.

  Result is averaged over replicas/batch/channels but not species.
  Args:
    states: Sequence[blur.NetworkState]
    weights: Should be broadcastable into last layer of states[-1].

  Returns:
    tf.Tensor
  """
  return l2_score(states, weights=weights, average_replicas=True)


def l2_score(states, average_replicas=False, average_batch=True,
             weights=None):
  """Returns l2_score (-l2_loss) per species.

  Args:
    states: Sequence[blur.NetworkState]
    average_replicas: whether to average across replicas dimension
    average_batch: whether to average across batch dimension.
    weights: weights to scale the score. Should be broadcastable onto
             last layer.
  Returns:
    tf.Tensor
  """
  # Summing over channels and batch and replicas.
  prediction = states[-1].layers[-1][Ellipsis, 0]
  ground_truth = states[-1].ground_truth
  # species x replicas x batch_size x channel x state
  # Can add support here for optional species/replicas.
  diff = (prediction - ground_truth)**2
  if weights is not None:
    diff = diff * weights
  axis = [-1]
  if average_batch:
    axis.append(-2)

  if average_replicas:
    assert len(states[-1].layers[-1].shape) >= 4
    axis.append(-3)

  return -tf.reduce_mean(diff, axis=axis)
