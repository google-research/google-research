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

"""Specfies the behaviour of a single state."""
import dataclasses as dc
import functools as ft
from typing import Any, Callable, List, Optional, Text, Tuple, Union

import jax.numpy as jp
import numpy as np
import tensorflow.compat.v1 as tf
import typing_extensions

from blur import blur_env
from blur import genome_util
from blur import synapse_util


@dc.dataclass
class OjasSpec:
  use_ojas_rule: bool = False
  use_multiplier: bool = False
  extra_multiplier: float = 1.0
  # Replaces standard rule of sort W_ijA_j^2, with W_{ij}\sum W^2_kj
  use_synapse_weights: bool = False

Protocol = typing_extensions.Protocol
Tensor = Union[tf.Tensor, np.ndarray, jp.array]
ActivationFn = Callable[[Tensor], Tensor]

# Incorporates synaptic update for fully connected layers
# Here "s" represents species (With distinct genome),
# "r", replicas (different synapse patterns, sharing the same genome)
# "b" batch (same synapse pattern, different neuronal activation)
# "i" and "j" fully connected input and "output"
# "k", "o", "l", -neuronal state
# Terms are
# 1. the pre-synaptic input
# 2. Genome-based transformation of pre-synaptic states
# 3. Synapse state
FC_UPDATE_PATTERN = 'srbik,skl,srijl->srbjl'

# 1. Pre-synaptic state
# 2. Genome-based transformation of pre-synaptic state
# 3. Genome-based transformation of post-synaptic state
# 4. Post-synaptic state.
FC_SYNAPSE_UPDATE = 'srbik,sko,sol,srbjl->srijo'

# 1. Synapses
# 2. Negated squared post-synaptic activations
FC_OJAS_SYNAPSE_UPDATE = 'srijo,srbjo->srijo'


@dc.dataclass
class NetworkState:
  layers: List[Tensor] = dc.field(default_factory=list)
  synapses: List[Tensor] = dc.field(default_factory=list)
  ground_truth: Optional[Tensor] = None
  updatable: List[bool] = dc.field(default_factory=list)


class SynapseUpdateFn(Protocol):
  """Type declaring function signature for synapse update function."""

  def __call__(self, inp, out, synapse,
               *,
               synaptic_genome,
               update_type,
               global_spec, env): Ellipsis


class StateUpdateFn(Protocol):
  """Type declaring state update function called at the start of every step."""

  def __call__(self,
               state,
               genome,
               network_spec,
               env): Ellipsis


class NeuronUpdateFn(Protocol):
  """Type declaring function signature for neuron update function."""

  def __call__(self, inp, out, synapse,
               inp_act,
               out_act,
               *,
               neuron_genome,
               update_type,
               global_spec, env):
    """Computes a new value for neurons, given inp, out and synapse."""


class FeedbackFn(Protocol):
  """Type declaring function signature for feedback_fn."""

  def __call__(self,
               final_layer,
               ground_truth,
               env = blur_env.tf_env):
    """Computes a feedback function from the last layer and the ground truth."""


def concat_groundtruth_in(final_layer, ground_truth,
                          env = blur_env.tf_env):
  return env.concat(
      [final_layer[Ellipsis, 0:1], ground_truth[Ellipsis, None],
       env.zeros_like(final_layer[Ellipsis, 2:])], axis=-1)


@dc.dataclass
class SynapseNormalizationSpec:
  normalize_synapses: bool = False
  rescale: bool = False
  normalize_in_activations: bool = False


SynapseTransformFn = Callable[[Tensor], Tensor]


@dc.dataclass
class NetworkSpec:
  """Global specification of the network properties."""
  synapse_activation_fn: Optional[ActivationFn] = None
  forward_activation_fn: Optional[ActivationFn] = None
  backward_activation_fn: Optional[ActivationFn] = None
  last_layer_activation_fn: Optional[ActivationFn] = None
  zero_out_neurons: bool = True
  per_layer_genome: bool = False
  ojas_spec: Optional[OjasSpec] = None
  synapse_normalization_spec: Optional[SynapseNormalizationSpec] = None
  forward_synapse_update: bool = False
  supervised_every: Optional[int] = None
  synapse_transform_fn: Optional[SynapseTransformFn] = None
  fixed_batches: bool = False
  # Synapse saturation mechanism and parameters
  synapse_saturation_eps: Optional[float] = None
  synapse_saturation_type: Optional[str] = None

  neuron_update_fn: Optional[NeuronUpdateFn] = None
  synapse_update_fn: Optional[SynapseUpdateFn] = None
  get_genome_fn: Optional[Callable[[Any, int], Any]] = None
  state_update_fn: Optional[StateUpdateFn] = None
  feedback_fn: Optional[FeedbackFn] = None

  # Can be either `multiplicative_second_state`, 'multiplicative' or `additive`.
  # - `multiplicative_second_state` means the first state is passed through with
  # no changes, while the second state receives multipicative update.
  # - `multiplicative` means all the states are updated multiplicatively.
  # - `additive` means all the states are updated additively.
  backward_update: str = 'additive'
  # Whether to apply the same synapse matrix for the forward and backward pass.
  symmetric_in_out_synapses: bool = False
  symmetric_states_synapses: bool = False

  use_forward_activations_for_synapse_update: bool = False

  def __post_init__(self):
    if self.neuron_update_fn is None:
      self.neuron_update_fn = dense_neuron_update
    if self.synapse_update_fn is None:
      self.synapse_update_fn = ft.partial(
          dense_synapse_update,
          saturation_eps=self.synapse_saturation_eps,
          saturation_type=self.synapse_saturation_type)
    if self.get_genome_fn is None:
      self.get_genome_fn = ft.partial(genome_util.get_genome,
                                      per_layer_genome=self.per_layer_genome)
    if self.state_update_fn is None:
      self.state_update_fn = default_state_update
    if self.feedback_fn is None:
      self.feedback_fn = concat_groundtruth_in


def default_network_spec(env):
  return NetworkSpec(forward_activation_fn=env.relu_tanh,
                     backward_activation_fn=env.relu_tanh,
                     last_layer_activation_fn=env.tanh)


def backprop_network_spec(env):
  del env  # Unused.
  return NetworkSpec(
      backward_update='multiplicative_second_state',
      symmetric_in_out_synapses=True,
      symmetric_states_synapses=True)


def network_step(state,
                 genome,
                 input_fn,
                 backward_pass_neurons=True,
                 backward_pass_synapses=True,
                 *,
                 network_spec = None,
                 debug_hidden_states = None,
                 step = None,
                 env = blur_env.tf_env):
  """Given the current state of the network produces new state."""
  if network_spec is None:
    network_spec = default_network_spec(env)
  if network_spec.synapse_transform_fn is None:
    network_spec.synapse_transform_fn = default_synapse_transform_fn(
        genome, network_spec, env)
  if debug_hidden_states is None:
    debug_hidden_states = []
  with env.name_scope('step'):
    input_batch, gt_batch = input_fn()
    new_layers = list(state.layers)

    if network_spec.zero_out_neurons:
      for i, layer in enumerate(new_layers):
        new_layers[i] = env.zeros_like(layer)

    state = dc.replace(state, layers=new_layers)
    debug_hidden_states.append(state)
    # Concatenate input signal to the last axis
    state.layers[0] = env.concat(
        [input_batch[Ellipsis, None],
         env.zeros_like(state.layers[0][Ellipsis, 1:])],
        axis=-1)
    debug_hidden_states.append(state)
    state = network_spec.state_update_fn(
        state, genome, network_spec=network_spec, env=env)
    debug_hidden_states.append(state)
    with env.name_scope('forward'):
      state = update_network_neurons(
          state, genome.neuron, backward=False,
          network_spec=network_spec, env=env)
      debug_hidden_states.append(state)

      if network_spec.forward_synapse_update:
        with env.name_scope('synapse_update'):
          assert genome.forward_synapse is not None
          state = update_network_synapses(
              state, genome.forward_synapse, backward=False,
              network_spec=network_spec, env=env)
          debug_hidden_states.append(state)

    forward_state = None
    if network_spec.use_forward_activations_for_synapse_update:
      forward_state = state

    state.ground_truth = gt_batch

    backward_pass = (step is None or
                     network_spec.supervised_every is None or
                     step % network_spec.supervised_every == 0)
    with env.name_scope('backward'):
      if backward_pass_neurons and backward_pass:
        state.layers[-1] = network_spec.feedback_fn(
            final_layer=state.layers[-1], ground_truth=gt_batch, env=env)
        debug_hidden_states.append(state)
        state = update_network_neurons(state, genome.neuron,
                                       backward=True,
                                       network_spec=network_spec, env=env)
        debug_hidden_states.append(state)
      if backward_pass_synapses and backward_pass:
        with env.name_scope('synapse_update'):
          state = update_network_synapses(
              state, genome.synapse, backward=True,
              network_spec=network_spec, env=env,
              forward_state=forward_state)
          debug_hidden_states.append(state)
    return state


def update_network_neurons(
    state,
    neuron_genome,
    network_spec,
    backward=False,
    *, env):
  """Updates neuronal states.

  The updates are applied sequentially.
  if backward is False, the updates  are applied in increasing order (e.g.
  layer[1] is updated from layer[0] etc) if backward is True updates are
  applied in reversed order.

  E.g. for forward pass: we apply  updates like so
    for i in num_layers:
      Δl_i, Δl_{i+1} = synaptic_update(l_i, l_i+1).
      l_{i+1} += Δl_{i+1}

  where the next update is computed after previous have been applied.
  E.g. for backward mdoe: we compute updates like so
    for i in reverse(num_layers-1):
      Δl_i, Δl_{i+1} = synaptic_update(l_i, l_i+1).
      l_{i} += Δl_{i}

  Importantly: for efficiency  we don't (at the moment) apply
  l_{i} on forward pass and  l_{i+1} on backward pass.

  Args:
    state: NetworkState,
    neuron_genome: genome_util.NeuronGenome,
    network_spec: network spec
    backward: the direction of the pass.
    env: blur_env.Env environment.
  Returns:
    new NetworkState
  """
  network_spec = network_spec or NetworkSpec()
  enumerated_synapses = list(enumerate(state.synapses))
  if backward:
    base_act_fn = network_spec.backward_activation_fn
    update = synapse_util.UpdateType.BACKWARD
    enumerated_synapses = reversed(enumerated_synapses)
  else:
    base_act_fn = network_spec.forward_activation_fn
    update = synapse_util.UpdateType.FORWARD

  neuron_activations = [base_act_fn] * (len(state.layers) - 1) + [
      network_spec.last_layer_activation_fn]
  layers = list(state.layers)
  synapses = list(state.synapses)
  for i, synapse in enumerated_synapses:
    with env.name_scope(f'{update.name.lower()}_{i}_{i+1}'):
      new_layer = network_spec.neuron_update_fn(
          layers[i], layers[i+1], synapse,
          neuron_genome=network_spec.get_genome_fn(neuron_genome, i),
          inp_act=neuron_activations[i],
          out_act=neuron_activations[i+1],
          update_type=update,
          global_spec=network_spec,
          env=env)

      if state.updatable[i]:
        layers[i] = new_layer[0]
      if state.updatable[i+1]:
        layers[i+1] = new_layer[1]

  return NetworkState(
      layers=layers, synapses=synapses,
      ground_truth=state.ground_truth,
      updatable=state.updatable)


def update_network_synapses(state,
                            synaptic_genome,
                            network_spec = None,
                            backward=False,
                            forward_state=None, *, env):
  """Updates synapses."""
  network_spec = network_spec or NetworkSpec()
  layers = list(state.layers)
  synapses = list(state.synapses)
  update_type = (synapse_util.UpdateType.BACKWARD if backward else
                 synapse_util.UpdateType.FORWARD)
  enumerated_synapses = list(enumerate(state.synapses))
  if backward:
    enumerated_synapses = reversed(enumerated_synapses)

  if not network_spec.use_forward_activations_for_synapse_update:
    assert forward_state is None
    forward_state = state
  else:
    assert forward_state

  for i, synapse in enumerated_synapses:
    pre = env.identity(forward_state.layers[i], name='hebbian_pre')
    post = env.identity(layers[1+i], name='hebbian_post')
    synapses[i] = network_spec.synapse_update_fn(
        pre, post, synapse,
        synaptic_genome=network_spec.get_genome_fn(synaptic_genome, i),
        update_type=update_type, global_spec=network_spec, env=env)

  return NetworkState(
      layers=layers, synapses=synapses,
      ground_truth=state.ground_truth,
      updatable=state.updatable)


def default_synapse_transform_fn(
    genome,
    network_spec,
    env):
  """Creates additional parameters for the neuron update."""
  normalization_spec = network_spec.synapse_normalization_spec
  if (normalization_spec is None or
      not normalization_spec.normalize_in_activations):
    return None
  rescale_to = genome.synapse.rescale_to if normalization_spec.rescale else None
  return ft.partial(
      synapse_util.normalize_synapses, rescale_to=rescale_to, env=env)


def get_ojas_update(pre, post, synapse,
                    genome,
                    ojas_spec,
                    env):
  """Additional update term designed to match a version of the Oja's rule.

  Currently implemented expression is a version of:
    Δw_{ij} ~ - w_{ij} y_j^2

  where y are post-synaptic activations (additional state transform is not
  shown).

  For reference see, for example:
    "A meta-learning approach to (re)discover plasticity rules that carve a
    desired function into a neural network"
    https://www.biorxiv.org/content/10.1101/2020.10.24.353409v1.full.pdf

  Args:
   pre: [population] x batch_size x in_channels x num_states
   post: [population] x batch_size x out_channels x num_states
   synapse: [population] x in_channels x out_channels x num_states
   genome: Synaptic genome
   ojas_spec: Specification of the Oja's term
   env: Environment
  Returns:
   Update. [in_channels + 1 + out_channels, in_channels + 1 + out_channels, k]
  """
  if ojas_spec.use_synapse_weights:
    act_squared = -synapse ** 2 * ojas_spec.extra_multiplier
  else:
    activations = env.concat([env.concat_row(pre, 0), post], axis=-2)
    act_squared = env.einsum(
        'srbjl,sol->srbjo', activations, genome.transform.post)
    act_squared = -act_squared * act_squared * ojas_spec.extra_multiplier

  if ojas_spec.use_multiplier:
    act_squared *= env.abs(genome.transform.ojas_multiplier)
  # Negative `update` can turn this term into an amplifying force.
  # We solve this by properly adjusting the sign.
  act_squared *= env.sign(genome.update)
  return env.einsum(FC_OJAS_SYNAPSE_UPDATE, synapse, act_squared)


def get_synaptic_update(pre,
                        post,
                        synapse,
                        input_transform_gn,
                        *,
                        update_type=synapse_util.UpdateType.BOTH,
                        synapse_transform_fn=None,
                        env):
  """Performs synaptic update.

  [Δpre, Δpost] = synapse * input_transform @ [pre, post]

  This transformation allows both asymmetric updates with information only
  flows
  from pre to post, however it also allows reverse flow if input_transform
  is non-zero its upper left quadrant (similar to e.g. akin electrical
  synapse).

  This function can perform updates on a bunch of replicas sharing the
  same genome. As well as on population of species each containing bunch
  of agents with different genomes.

  The population dimensions (species/replica) are present in all tensors
  or not present at all.

  Args:
    pre: [#species x #replicas] x batch_size x input_shape x
      num_neuron_states,
    post: [#species x #replicas] x batch_size x output_shape x
      num_neuron_states
    synapse: [#species x #replicas] x (input_shape + output_shape + 1)^ 2 x
      num_neuron_states. the synapse weight connecting input_shape with
      output_shape
    input_transform_gn: [#species] x (2 num_neuron_states ) ^2: describes
      genome's input transformation.
    update_type: which update to compute (update to the input, update to the
      output or both).
    synapse_transform_fn: function used to transform synapses while
      calculating activations
    env: compute environment

  Returns:
    update signals
  """
  k = pre.shape[-1]
  # Verifies that we use the same embedding size.
  if pre.shape[-1] != post.shape[-1]:
    raise ValueError(f'pre: {pre.shape}[-1] != post: {post.shape}[-1]')

  # Note: we cheat here:
  # Update for "post" here only uses "pre" (ignoring current value of "post")
  # And vice versa for "pre".
  # That is we ignore the following elements of input_transform_gn
  # and synapse matrices
  #  [ x U ]
  #  [ V x ] (U and V are the only submatrices that are being used)
  # Add support for immediate feedback if and when desired.
  output_update = None
  if update_type in (synapse_util.UpdateType.FORWARD,
                     synapse_util.UpdateType.BOTH):
    output_update = single_synaptic_update(
        synapse,
        input_layer=pre,
        in_channels=pre.shape[-2],
        update_type=synapse_util.UpdateType.FORWARD,
        transform_gn=input_transform_gn[Ellipsis, :k, k:],
        synapse_transform_fn=synapse_transform_fn,
        env=env)
    output_update = env.identity(output_update, 'output_update')
  input_update = None
  if update_type in (synapse_util.UpdateType.BACKWARD,
                     synapse_util.UpdateType.BOTH):
    # Transforming "post"
    # Generating update for the "input" part of the synapse.
    input_update = single_synaptic_update(
        synapse,
        input_layer=post,
        in_channels=pre.shape[-2],
        update_type=synapse_util.UpdateType.BACKWARD,
        transform_gn=input_transform_gn[Ellipsis, k:, :k],
        synapse_transform_fn=synapse_transform_fn,
        env=env)
    input_update = env.identity(input_update, 'input_update')
  return input_update, output_update


def single_synaptic_update(
    synapse,
    input_layer,
    in_channels,
    update_type,
    transform_gn,
    synapse_transform_fn=None,
    *,
    env):
  """Computes one-way (Forward or Backward) synaptic update."""
  include_bias = False
  if update_type == synapse_util.UpdateType.FORWARD:
    # Input channels are +1 to include "1" channel to simulate bias.
    input_layer = env.concat_row(input_layer)
    include_bias = True
  subsynapse = synapse_util.synapse_submatrix(
      synapse,
      in_channels=in_channels,
      update_type=update_type,
      include_bias=include_bias)
  if synapse_transform_fn is not None:
    subsynapse = synapse_transform_fn(subsynapse)
  update = env.einsum(
      FC_UPDATE_PATTERN, input_layer, transform_gn, subsynapse)
  return update


def compute_neuron_state(old_neuron_state, synaptic_update,
                         g,
                         activation_fn, update_type, *,
                         env):
  """Computes new neuron state from the synaptic update."""
  if activation_fn is None:
    activation_fn = lambda x: x
    # TODO(sandler): Add inverse activation of old_state if needed.
  if synaptic_update is None:
    return old_neuron_state
  if update_type == 'multiplicative_second_state':
    # TODO(mxv): possibly rename this update_type, since technically it should
    # be called skip_first_state or something like this.
    second_state_update = old_neuron_state[Ellipsis, 1:] * synaptic_update[Ellipsis, 1:]
    new_neuron_state = env.concat(
        [old_neuron_state[Ellipsis, 0:1], second_state_update], axis=-1
        ) * env.right_pad_shape(g.keep, to=old_neuron_state)
  elif update_type == 'multiplicative':
    new_neuron_state = (
        old_neuron_state * synaptic_update *
        env.right_pad_shape(g.keep, to=old_neuron_state))
  elif update_type == 'additive':
    new_neuron_state = (
        old_neuron_state * env.right_pad_shape(g.keep, to=old_neuron_state) +
        synaptic_update * env.right_pad_shape(g.update, to=synaptic_update))
  elif update_type == 'none':
    new_neuron_state = synaptic_update
  return activation_fn(new_neuron_state)


def dense_neuron_update(
    inp, out, synapse,
    *,
    inp_act, out_act,
    neuron_genome,
    update_type = synapse_util.UpdateType.FORWARD,
    global_spec,  # pylint: disable=unused-argument
    env):
  """Default neuron update function."""
  input_update, output_update = get_synaptic_update(
      inp, out, synapse,
      input_transform_gn=neuron_genome.transform,
      update_type=update_type,
      synapse_transform_fn=global_spec.synapse_transform_fn,
      env=env)

  inp = compute_neuron_state(
      inp, input_update, neuron_genome, activation_fn=inp_act,
      update_type=global_spec.backward_update, env=env)
  out = compute_neuron_state(
      out, output_update, neuron_genome, activation_fn=out_act,
      update_type='additive', env=env)
  return inp, out


def get_hebbian_update(pre, post, transform,
                       global_spec,
                       env):
  """Performs hebbian update of the synapse weight matrix.

  Δ w = [pre, [1], post] @ transform.pre * transform.post @  [pre, [0],
  post]

  Args:
   pre: [population] x batch_size x in_channels x num_states
   post: [population] x batch_size x out_channels x num_states
   transform: genome_util.HebbianTransform
   global_spec: Specification of the network.
   env: Environment

  Returns:
   Update.  [in_channels+1 + in_channels, out_channels + 1 + out_channels, k]
  """
  inp = env.concat([env.concat_row(pre, 1), post], axis=-2)
  out = env.concat([env.concat_row(pre, 1), post], axis=-2)
  # inp: [b x (in+out) x k],
  # transforms: [k x k]
  hebbian_update = env.einsum(FC_SYNAPSE_UPDATE, inp, transform.pre,
                              transform.post, out)
  if global_spec.symmetric_in_out_synapses:
    hebbian_update = synapse_util.sync_in_and_out_synapse(
        hebbian_update, pre.shape[-2], env)
  if global_spec.symmetric_states_synapses:
    hebbian_update = synapse_util.sync_states_synapse(hebbian_update, env)
  return hebbian_update


def compute_synapse_state(old_synapse,
                          hebbian_update,
                          g,
                          activation_fn=None,
                          saturation_type = None,
                          saturation_eps = None,
                          *,
                          env):
  """Computes new synapse state given Hebbian update."""
  if activation_fn is None:
    activation_fn = lambda x: x
  if saturation_type == 'exp_update':
    assert saturation_eps is not None
    # This is necessary to avoid type issues during eval
    delta_keep = np.float32(-1.0)
    delta_keep += g.keep
    dw = (old_synapse * env.right_pad_shape(delta_keep, to=old_synapse) +
          hebbian_update * env.right_pad_shape(g.update, to=hebbian_update))
    dw *= env.exp(
        -env.abs(old_synapse) * (saturation_eps + env.abs(g.saturation)))
    return activation_fn(old_synapse + dw)
  elif (saturation_type in ['disabled', 'normalization'] or
        saturation_type is None):
    return activation_fn(
        old_synapse * env.right_pad_shape(g.keep, to=old_synapse) +
        hebbian_update * env.right_pad_shape(g.update, to=hebbian_update))
  else:
    raise AssertionError


# Note there might be multiple implementatios of "dense", we should rename
# this to as we have a better name.
def dense_synapse_update(
    inp, out, synapse,
    saturation_type = None,
    saturation_eps = None,
    *,
    synaptic_genome,
    update_type = synapse_util.UpdateType.FORWARD,
    global_spec,
    env):
  """Default synapse update function."""
  del update_type
  hebbian_update = get_hebbian_update(inp, out,
                                      synaptic_genome.transform,
                                      global_spec, env=env)
  if global_spec.ojas_spec:
    hebbian_update += get_ojas_update(inp, out, synapse,
                                      synaptic_genome,
                                      ojas_spec=global_spec.ojas_spec, env=env)
  hebbian_update = env.identity(hebbian_update, name='hebbian_update')
  return compute_synapse_state(
      synapse,
      hebbian_update,
      synaptic_genome,
      saturation_eps=saturation_eps,
      saturation_type=saturation_type,
      activation_fn=global_spec.synapse_activation_fn, env=env)


def default_state_update(state,
                         genome,
                         network_spec,
                         env):
  """Transforms system state (currently can normalize synapses)."""
  synapse_norm_spec = network_spec.synapse_normalization_spec
  if synapse_norm_spec is None:
    return state
  rescale_to = genome.synapse.rescale_to if synapse_norm_spec.rescale else None
  normalize = ft.partial(
      synapse_util.normalize_synapses, rescale_to=rescale_to, env=env)
  if synapse_norm_spec.normalize_synapses:
    for i in range(len(state.synapses)):
      synapse = state.synapses[i]
      in_channels = state.layers[i].shape[-2]
      forward = synapse[Ellipsis, :(in_channels + 1), (in_channels + 1):, :]
      # We include an extra dimension for axis=-2 here because it does not
      # influence normalization, but the tensors would have proper shapes:
      backward = synapse[Ellipsis, (in_channels + 1):, :(in_channels + 1), :]
      state.synapses[i] = synapse_util.combine_in_out_synapses(
          normalize(forward), normalize(backward), env)

  return state
