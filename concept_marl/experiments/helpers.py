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

"""Helpers for meltingpot environment."""

from typing import Any, Dict, Sequence

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_jax_utils
from acme.multiagent import types as ma_types
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from concept_marl import concept_ppo
from concept_marl.utils import factories


def encode_meltingpot_obs(obs,
                          conv_filters = (16, 32),
                          conv_kernels = (8, 4),
                          strides = (8, 1),
                          scalar_fc = 5,
                          mlp_layer_sizes = (128, 128),
                          ):
  """Conducts preprocessing and encoding of 'meltingpot' dict observations.

  This is similar to what is done in:
  https://github.com/google-research/google-research/blob/master/social_rl/multiagent_tfagents/multigrid_networks.py

  Args:
    obs: meltingpot observation dict, which can include observation inputs such
      as 'RGB', 'POSITION', 'ORIENTATION' and a global observation 'WORLD.RGB'
      that each agent carries.
    conv_filters: Number of convolution filters.
    conv_kernels: Size of the convolution kernel.
    strides: Size of the stride for CNN layers.
    scalar_fc: Number of neurons in the fully connected layer processing the
      scalar input.
    mlp_layer_sizes: Size of mlp layers to use after initial encoding.

  Returns:
    out: output observation.
  """

  def _cast_and_scale(x, scale_by=10.0):
    if isinstance(x, jnp.ndarray):
      x = x.astype(jnp.float32)
    return x / scale_by

  def _cast(x):
    if isinstance(x, jnp.ndarray):
      x = x.astype(jnp.float32)
    return x

  def _cast_and_one_hot(x, num_classes=4):
    if isinstance(x, jnp.ndarray):
      x = jax.nn.one_hot(x, num_classes).astype(jnp.float32)
      # account for frame-stacking (if any)
      x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    return x

  outputs = []
  # custom CNN backbone for RGB observations
  if 'RGB' in obs.keys():
    layers = [_cast]

    # add CNN cnn block for each input filter/kernel/stride
    for conv_filter, kernel, stride in zip(conv_filters, conv_kernels, strides):
      layers.append(
          hk.Conv2D(
              output_channels=conv_filter, kernel_shape=kernel, stride=stride))
      layers.append(jax.nn.relu)

    # flatten and reshape
    layers.append(hk.Flatten())
    layers.append(hk.nets.MLP(mlp_layer_sizes))

    # apply encoding stack and return
    image_enc = hk.Sequential(layers)
    outputs.append(image_enc(obs['RGB']))

  # encode position observations with single fc layer
  if 'POSITION' in obs.keys():
    position_enc = hk.Sequential([_cast_and_scale, hk.Linear(scalar_fc)])
    out = position_enc(obs['POSITION'])
    outputs.append(out)

  # one-hot encode orientations observations and pass thru single fc layer
  if 'ORIENTATION' in obs.keys():
    orientation_enc = hk.Sequential(
        [_cast_and_one_hot, hk.Linear(scalar_fc)])
    out = orientation_enc(obs['ORIENTATION'])
    outputs.append(out)

  out = jnp.concatenate(outputs, axis=-1)
  return out


def concept_bottleneck(
    inputs,
    concepts,
    is_training,
    hidden_layer_sizes = (128, 128),
    categorical_values = 4
):
  """"Implements concept bottleneck layer.

  Args:
    inputs: Pre-processed inputs.
    concepts: Concept observations.
    is_training: Flag whether or not network is in training mode (used for
      masking).
    hidden_layer_sizes: Size of the hidden layers in bottleneck.
    categorical_values: number of values for categorical concepts.

  Returns:
    out: concept estimates.
  """

  # extract relevant info from concept obs
  num_scalar_concepts = concepts['scalar_concepts']['concept_values'].shape[1]
  num_categorical_concepts = int(
      concepts['cat_concepts']['concept_values'].shape[1] / categorical_values)
  raw_outputs, outputs = [], []  # also want to hold onto raw

  # pre-process inputs
  concept_backbone = hk.nets.MLP(hidden_layer_sizes)
  processed_inputs = concept_backbone(inputs)

  # scalar concepts head
  scalar_concept_network = hk.Linear(num_scalar_concepts)
  scalar_out = scalar_concept_network(processed_inputs)
  raw_outputs.append(scalar_out)
  outputs.append(scalar_out)

  # categorical concept heads (pass logits for precision)
  cat_concept_networks = [
      networks_lib.CategoricalHead(num_values=categorical_values)
      for num_classes in range(num_categorical_concepts)
  ]
  for network in cat_concept_networks:
    cat_out = network(processed_inputs).logits
    raw_outputs.append(cat_out)
    outputs.append(cat_out)

  # concat all concept predictions
  out = jnp.concatenate(outputs, axis=-1)
  raw_out = jnp.concatenate(raw_outputs, axis=-1)

  if not is_training and 'interventions' in concepts.keys():
    concept_interventions = concepts['interventions']

    # apply interventions
    if 'override_masks' in concept_interventions.keys():
      # replace specific concept estimates with override values
      masked_out = out * concept_interventions[
          'intervention_masks'] + concept_interventions['override_masks'] * (
              1 - concept_interventions['intervention_masks'])
    else:
      # mask out completely instead of applying noise
      masked_out = out * concept_interventions['intervention_masks']
    return masked_out, raw_out  # pytype: disable=bad-return-type  # jax-ndarray
  else:
    return out, raw_out  # pytype: disable=bad-return-type  # jax-ndarray


def make_meltingpot_concept_ppo_networks(
    environment_spec,
    hidden_layer_sizes = (128, 128),
):
  """Returns ConceptPPO networks used by the agent in the multigrid environments."""

  # Check that meltingpot is defined with discrete actions, 0-indexed
  assert np.issubdtype(environment_spec.actions.dtype, np.integer), (
      'Expected meltingpot environment to have discrete actions with int dtype'
      f' but environment_spec.actions.dtype == {environment_spec.actions.dtype}'
  )
  assert environment_spec.actions.minimum == 0, (
      'Expected meltingpot environment to have 0-indexed action indices, but'
      f' environment_spec.actions.minimum == {environment_spec.actions.minimum}'
  )

  # actions from environment
  num_actions = environment_spec.actions.maximum + 1

  def forward_fn(inputs, is_training=False):
    raw_inputs = inputs['raw_observations']
    concept_obs = inputs['concept_observations']

    # policy network
    policy_network = hk.Sequential([
        hk.nets.MLP(hidden_layer_sizes, activation=jnp.tanh),
        networks_lib.CategoricalValueHead(num_values=num_actions)
    ])

    # encode observations
    processed_inputs = encode_meltingpot_obs(raw_inputs)

    # pass encoded observations through bottleneck layer
    concepts, unmodified_concepts = concept_bottleneck(processed_inputs,
                                                       concept_obs, is_training)
    return policy_network(concepts), unmodified_concepts

  # transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  # dummy observation for network initialization.
  dummy_obs = acme_jax_utils.zeros_like(environment_spec.observations)
  dummy_obs = acme_jax_utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = concept_ppo.networks.ConceptPolicyNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
  return concept_ppo.make_concept_ppo_networks(network)


def init_default_meltingpot_network(
    agent_type, agent_spec):
  """Returns concept ppo networks for meltingpot environment."""
  if agent_type == factories.DefaultSupportedAgent.CONCEPT_PPO:
    return make_meltingpot_concept_ppo_networks(agent_spec)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')
