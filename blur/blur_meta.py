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

import dataclasses as dc
from typing import List, Optional, Sequence, Callable, Tuple

import tensorflow.compat.v1 as tf

from blur import blur
from blur import blur_env
from blur import genome_util
from blur import synapse_util


def freeze_network_state(state, sess):
  layers, synapses = sess.run([state.layers, state.synapses])
  return dc.replace(state, layers=layers, synapses=synapses)


def init_first_state(
    genome,
    data,
    hidden_layers,
    synapse_initializer,
    create_synapses_fn = synapse_util.create_synapses,
    network_spec = blur.NetworkSpec(),
    env = blur_env.tf_env):
  """Initialize the very first state of the graph."""
  if create_synapses_fn is None:
    create_synapses_fn = synapse_util.create_synapses

  if callable(genome):
    num_neuron_states = genome(0).synapse.transform.pre.shape[-1]
  else:
    num_neuron_states = genome.synapse.transform.pre.shape[-1]
  output_shape = data.element_spec['support'][1].shape
  num_outputs = output_shape[-1]
  num_inputs = data.element_spec['support'][0].shape[-1]
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
      layers=layers,
      synapses=synapses,
      ground_truth=ground_truth,
      updatable=[False] + [True] * num_updatable_units)


def episode_data_fn_split(input_fn, fixed_batches=False):
  """Split episodic input data function into support tna query data functions.

  Args:
    input_fn: input function that returns joint episodic data.
    fixed_batches: whether to return same fixed batch during learning. If True
      same batch would be returned for each function invocation.
  Returns:
    A tuple of two functions: one for support and query data.
  """
  if fixed_batches:
    data_dict = input_fn()
    data_support_fn = lambda: data_dict['support']
    if 'query' in data_dict:
      data_query_fn = lambda: data_dict['query']
    else:
      query_data_dict = input_fn()
      data_query_fn = lambda: query_data_dict['support']
  else:
    def map_data_fn(data_fn):
      data_dict = data_fn()
      return data_dict['support']
    data_support_fn = lambda: map_data_fn(input_fn)
    data_query_fn = lambda: map_data_fn(input_fn)
  return data_support_fn, data_query_fn


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
  state = init_first_state(genome, data, hidden_layers, synapse_initializer,
                           create_synapses_fn, network_spec)
  if input_fn is None:
    input_fn = tf.data.make_one_shot_iterator(data).get_next

  data_support_fn, data_query_fn = episode_data_fn_split(
      input_fn, fixed_batches=network_spec.fixed_batches)
  training_states = unroll_graph(
      genome=genome,
      initial_state=state,
      unroll_steps=unroll_steps,
      input_fn=data_support_fn,
      network_spec=network_spec,
      out_intermediate_states=out_intermediate_states)

  final_state = training_states[-1]
  validation_state = blur.network_step(
      final_state,
      genome=genome,
      input_fn=data_query_fn,
      backward_pass_neurons=False,
      backward_pass_synapses=False,
      network_spec=network_spec,
      env=blur_env.tf_env)
  return training_states, validation_state


def unroll_graph(
    genome,
    initial_state,
    unroll_steps,
    input_fn,
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


def get_accuracy(final_layer,
                 ground_truth,
                 onehot=False,
                 inverted=False):
  """Computes accuracy for the tensor."""
  batch_size = final_layer.shape[-3]
  if onehot:
    if inverted:
      pred = tf.argmin(final_layer[Ellipsis, 0], axis=-1)
    else:
      pred = tf.argmax(final_layer[Ellipsis, 0], axis=-1)
    gt = tf.argmax(ground_truth, axis=-1)
  else:
    assert final_layer.shape[-2] == 1
    pred = final_layer[Ellipsis, 0, 0] > 0
    if inverted:
      pred = tf.logical_not(pred)
    gt = ground_truth[Ellipsis, 0] > 0

  agreed = tf.cast(tf.equal(pred, gt), tf.float32)
  # all this acrobatics is because tf1 uses Dimension for shape, while tf2
  # uses tuples.
  batch_size = tf.cast(tf.convert_to_tensor(batch_size), tf.float32)
  accuracy = tf.reduce_sum(agreed, axis=-1) / batch_size
  return accuracy


def l2_score_per_species(validation_state, weights=None):
  """Returns l2_score (-l2_loss) per species.

  Result is averaged over replicas/batch/channels but not species.
  Args:
    validation_state: state with synapses from training run on validation data.
    weights: Should be broadcastable into last layer of states[-1].

  Returns:
    tf.Tensor
  """
  return l2_score(validation_state, weights=weights, average_replicas=True)


def l2_score(validation_state,
             average_replicas=False,
             average_batch=True,
             weights=None):
  """Returns l2_score (-l2_loss) per species.

  Args:
    validation_state: state with synapses from training run on validation data.
    average_replicas: whether to average across replicas dimension.
    average_batch: whether to average across batch dimension.
    weights: weights to scale the score. Should be broadcastable onto last
      layer.

  Returns:
    tf.Tensor
  """
  # Summing over channels and batch and replicas.
  prediction = validation_state.layers[-1][Ellipsis, 0]
  ground_truth = validation_state.ground_truth
  # species x replicas x batch_size x channel x state
  # Can add support here for optional species/replicas.
  diff = (prediction - ground_truth)**2
  if weights is not None:
    diff = diff * weights
  axis = [-1]
  if average_batch:
    axis.append(-2)

  if average_replicas:
    assert len(validation_state.layers[-1].shape) >= 4
    axis.append(-3)

  return -tf.reduce_mean(diff, axis=axis)


def get_backprop_network_spec():
  state = blur.NetworkSpec()
  state.backward_update = 'multiplicative_second_state'
  state.symmetric_in_out_synapses = True
  state.symmetric_states_synapses = True
  return state


def compute_weight_by_frequency(labels):
  """Computes weights to keep positive/negative labels in balance.

   For instance if labels contains 1 positive out of 9 negative, the positive
   one will have weight 0.9, while negative will have weight 0.1

  Args:
    labels: +1/-1 tensor of shape [..., num_channels]

  Returns:
    tensor of the same shape as labels.
  """
  p = tf.greater(labels, 0)
  pf = tf.to_float(p)
  positives = tf.reduce_sum(pf, axis=-1, keepdims=True) + tf.zeros_like(pf)
  negatives = tf.reduce_sum(1 - pf, axis=-1, keepdims=True) + tf.zeros_like(pf)
  total = positives + negatives
  weights = tf.where(p, negatives / total, positives / total)
  return weights
