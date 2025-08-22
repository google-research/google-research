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

"""Concept PPO network definitions."""

import dataclasses
from typing import Any, Callable, Optional, Protocol, Sequence

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np

EntropyFn = Callable[[Any], jnp.ndarray]


class ApplyFn(Protocol):

  def __call__(self,
               params,
               observation,
               *args,
               is_training,
               **kwargs):
    Ellipsis


@dataclasses.dataclass
class ConceptPolicyNetwork:
  """Holds a pair of pure functions defining a policy network for ConceptPPO.

  This is an extension of feed-forward network that takes params,
  obs, is_training as input.

  Attributes:
    init: A pure function. Initializes and returns the networks parameters.
    apply: A pure function. Computes and returns the outputs of a forward pass.
  """
  init: Callable[[types.PRNGKey], networks_lib.Params]
  apply: ApplyFn


@dataclasses.dataclass
class ConceptPPONetworks:
  """Network and pure functions for the Concept PPO agent.

  If 'network' returns tfd.Distribution, you can use make_concept_ppo_networks()
  to create this object properly.
  If one is building this object manually, one has a freedom to make 'network'
  object return anything that is later being passed as input to
  log_prob/entropy/sample functions to perform the corresponding computations.
  """
  network: ConceptPolicyNetwork
  log_prob: networks_lib.LogProbFn
  entropy: EntropyFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def make_inference_fn(
    concept_ppo_networks,
    evaluation = False):
  """Returns a function to be used for inference by a ConceptPPO actor."""

  def inference(params, key,
                observations):
    is_training = not evaluation
    (distribution, _), concepts = concept_ppo_networks.network.apply(
        params, observations, is_training=is_training)
    if evaluation and concept_ppo_networks.sample_eval:
      actions = concept_ppo_networks.sample_eval(distribution, key)
    else:
      actions = concept_ppo_networks.sample(distribution, key)
    if evaluation:
      return actions, {'concepts': concepts}
    log_prob = concept_ppo_networks.log_prob(distribution, actions)
    return actions, {'log_prob': log_prob}

  return inference


def make_networks(
    spec, hidden_layer_sizes = (256, 256)
):
  if isinstance(spec.actions, specs.DiscreteArray):
    return make_discrete_networks(spec, hidden_layer_sizes)
  else:
    return make_continuous_networks(
        spec,
        policy_layer_sizes=hidden_layer_sizes,
        value_layer_sizes=hidden_layer_sizes)


def make_concept_ppo_networks(
    network):
  """Constructs a ConceptPPONetworks instance from the given FeedForwardNetwork.

  Args:
    network: a transformed Haiku network that takes in observations and returns
      the action distribution and value.

  Returns:
    A ConceptPPONetworks instance with pure functions wrapping the input
    network.
  """
  return ConceptPPONetworks(
      network=network,
      log_prob=lambda distribution, action: distribution.log_prob(action),
      entropy=lambda distribution: distribution.entropy(),
      sample=lambda distribution, key: distribution.sample(seed=key),
      sample_eval=lambda distribution, key: distribution.mode())


def make_discrete_networks(
    environment_spec,
    hidden_layer_sizes = (512,),
    use_conv = True,
):
  """Creates networks used by the agent for discrete action environments.

  Args:
    environment_spec: Environment spec used to define number of actions.
    hidden_layer_sizes: Network definition.
    use_conv: Whether to use a conv or MLP feature extractor.
  Returns:
    ConceptPPONetworks
  """

  num_actions = environment_spec.actions.num_values

  def forward_fn(inputs):
    layers = []
    if use_conv:
      layers.extend([networks_lib.AtariTorso()])
    layers.extend([
        hk.nets.MLP(hidden_layer_sizes, activate_final=True),
        networks_lib.CategoricalValueHead(num_values=num_actions)
    ])
    policy_value_network = hk.Sequential(layers)
    return policy_value_network(inputs)

  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))
  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = ConceptPolicyNetwork(lambda rng: forward_fn.init(rng, dummy_obs),
                                 forward_fn.apply)

  # Create ConceptPPONetworks to add functionality required by the agent.
  return make_concept_ppo_networks(network)


def make_continuous_networks(
    environment_spec,
    policy_layer_sizes = (64, 64),
    value_layer_sizes = (64, 64),
):
  """Creates ConceptPPONetworks to be used for continuous action environments."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

  def forward_fn(inputs):
    policy_network = hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(policy_layer_sizes, activate_final=True),
        # We don't respect bounded action specs here and instead
        # rely on CanonicalSpecWrapper to clip actions accordingly.
        networks_lib.MultivariateNormalDiagHead(num_dimensions)
    ])
    value_network = hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(value_layer_sizes, activate_final=True),
        hk.Linear(1),
        lambda x: jnp.squeeze(x, axis=-1)
    ])

    action_distribution = policy_network(inputs)
    value = value_network(inputs)
    return (action_distribution, value)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = ConceptPolicyNetwork(lambda rng: forward_fn.init(rng, dummy_obs),
                                 forward_fn.apply)
  # Create ConceptPPONetworks to add functionality required by the agent.
  return make_concept_ppo_networks(network)
