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

"""Deep Evolution solver.

Uses a neural net to predict the position and
mutation function to apply to a string. Neural net takes the string and predicts
1) [Batch x length] logits over positions in the string
2) [Batch x length x n_mutations] logits over mutation function for every
position in the string.
First, we sample the position from the position logits, take the logits
corresponding to the chosen position and sample the index of the mutation
function to apply to this position in the string. Currently, we apply
one mutation at a time. Finally, update the network parameters using REINFORCE
gradient estimator, where the advantage is the difference between parent and
child rewards. The log-likelihood is the sum of position and mutation
log-likelihoods.
By default, no selection is performed (we continue mutating the same batch,
use_selection_of_best = False). If use_selection_of_best=True, we choose best
samples from the previous batch and sample them with replacement to create
a new batch.
"""
import functools

from absl import logging
import gin
import jax
from jax.experimental import stax
from jax.experimental.optimizers import adam
import jax.numpy as jnp
import jax.random as jrand
from jax.scipy.special import logsumexp
import numpy as np

from amortized_bo import base_solver
from amortized_bo import data
from amortized_bo import utils


def logsoftmax(x, axis=-1):
  """Apply log softmax to an array of logits, log-normalizing along an axis."""
  return x - logsumexp(x, axis, keepdims=True)


def softmax(x, axis=-1):
  return jnp.exp(logsoftmax(x, axis))


def one_hot(x, k):
  """Create a one-hot encoding of x of size k."""
  return jnp.eye(k)[x]


def gumbel_max_sampler(logits, temperature, rng):
  """Sample fom categorical distribution using Gumbel-Max trick.

    Gumbel-Max trick:
    https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    https://arxiv.org/abs/1411.0030

  Args:
    logits: Unnormalized logits for categorical distribution.
      [batch x n_mutations_to_sample x n_mutation_types]
    temperature: temperature parameter for Gumbel-Max. The lower the
      temperature, the closer the sample is to one-hot-encoding.
    rng: Jax random number generator

  Returns:
    class_assignments: Sampled class assignments [batch]
    log_likelihoods: Log-likelihoods of the sampled mutations [batch]
  """

  # Normalize the logits
  logits = logsoftmax(logits)

  gumbel_noise = jrand.gumbel(rng, logits.shape)
  softmax_logits = (logits + gumbel_noise) / temperature
  soft_assignments = softmax(softmax_logits, -1)
  class_assignments = jnp.argmax(soft_assignments, -1)
  assert len(class_assignments.shape) == 2
  # Output shape: [batch x num_mutations]

  return class_assignments


##########################################
# Mutation-related helper functions
def _mutate_position(structure, pos_mask, fn):
  """Apply mutation fn to position specified by pos_mask."""
  structure = np.array(structure).copy()
  pos_mask = np.array(pos_mask).astype(int)
  structure[pos_mask == 1] = fn(structure[pos_mask == 1])
  return structure


def set_pos(x, pos_mask, val):
  return _mutate_position(x, pos_mask, fn=lambda x: val)


def apply_mutations(samples, mutation_types, pos_masks, mutations,
                    use_assignment_mutations=False):
  """Apply the mutations specified by mutation types to the batch of strings.

  Args:
    samples: Batch of strings [batch x str_length]
    mutation_types: IDs of mutation types to be applied to each string
      [Batch x num_mutations]
    pos_masks: One-hot encoding [Batch x num_mutations x str_length]
      of the positions to be mutate in each string.
      "num_mutations" positions will be mutated per string.
    mutations: A list of possible mutation functions.
      Functions should follow the format: fn(x, domain, pos_mask),
    use_assignment_mutations: bool. Whether mutations are defined as
      "Set position X to character C". If use_assignment_mutations=True,
      then vectorize procedure of applying mutations to the string.
      The index of mutation type should be equal to the index of the character.
      Gives considerable speed-up to this function.

  Returns:
    perturbed_samples: Strings perturbed according to the mutation list.
  """
  batch_size = samples.shape[0]
  assert len(mutation_types) == batch_size
  assert len(pos_masks) == batch_size

  str_length = samples.shape[1]
  assert pos_masks.shape[-1] == str_length

  # Check that number of mutations is consistent in mutation_types and positions
  assert mutation_types.shape[1] == pos_masks.shape[1]

  num_mutations = mutation_types.shape[1]

  # List of batched samples with 0,1,2,... mutations
  # First element of the list contains original samples
  # Last element has samples with all mutations applied to the string
  perturbed_samples_with_i_mutations = [samples]
  for i in range(num_mutations):

    perturbed_samples = []
    samples_to_perturb = perturbed_samples_with_i_mutations[-1]

    if use_assignment_mutations:
      perturbed_samples = samples_to_perturb.copy()
      mask = pos_masks[:, i].astype(int)
      # Assumes mutations are defined as "Set position to the character C"
      perturbed_samples[np.array(mask) == 1] = mutation_types[:, i]
    else:
      for j in range(batch_size):
        sample = samples_to_perturb[j].copy()

        pos = pos_masks[j, i]
        mut_id = mutation_types[j, i]

        mutation = mutations[int(mut_id)]
        perturbed_samples.append(mutation(sample, pos))
      perturbed_samples = np.stack(perturbed_samples)

    assert perturbed_samples.shape == samples.shape
    perturbed_samples_with_i_mutations.append(perturbed_samples)

  states = jnp.stack(perturbed_samples_with_i_mutations, 0)
  assert states.shape == (num_mutations + 1,) + samples.shape
  return states


##########################################
# pylint: disable=invalid-name
def OneHot(depth):
  """Layer for transforming inputs to one-hot encoding."""

  def init_fun(rng, input_shape):
    del rng
    return input_shape + (depth,), ()

  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    # Perform one-hot encoding
    return jnp.eye(depth)[inputs.astype(int)]

  return init_fun, apply_fun


def ExpandDims(axis=1):
  """Layer for expanding dimensions."""
  def init_fun(rng, input_shape):
    del rng
    input_shape = tuple(input_shape)
    if axis < 0:
      dims = len(input_shape)
      new_axis = dims + 1 - axis
    else:
      new_axis = axis
    return (input_shape[:new_axis] + (1,) + input_shape[new_axis:]), ()
  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    return jnp.expand_dims(inputs, axis)
  return init_fun, apply_fun


def AssertNonZeroShape():
  """Layer for checking that no dimension has zero length."""

  def init_fun(rng, input_shape):
    del rng
    return input_shape, ()

  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    assert 0 not in inputs.shape
    return inputs

  return init_fun, apply_fun

# pylint: enable=invalid-name


def squeeze_layer(axis=1):
  """Layer for squeezing dimension along the axis."""
  def init_fun(rng, input_shape):
    del rng
    if axis < 0:
      raise ValueError("squeeze_layer: negative axis is not supported")
    return (input_shape[:axis] + input_shape[(axis + 1):]), ()

  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    return inputs.squeeze(axis)

  return init_fun, apply_fun


def reduce_layer(reduce_fn=jnp.mean, axis=1):
  """Apply reduction function to the array along axis."""
  def init_fun(rng, input_shape):
    del rng
    assert axis >= 0
    assert len(input_shape) == 3
    return input_shape[:axis - 1] + input_shape[axis + 1:], ()

  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    return reduce_fn(inputs, axis=axis)

  return init_fun, apply_fun


def _create_positional_encoding(  # pylint: disable=invalid-name
    input_shape, max_len=10000):
  """Helper: create positional encoding parameters."""
  d_feature = input_shape[-1]
  pe = np.zeros((max_len, d_feature), dtype=np.float32)
  position = np.arange(0, max_len)[:, np.newaxis]
  div_term = np.exp(np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
  pe[:, 0::2] = np.sin(position * div_term)
  pe[:, 1::2] = np.cos(position * div_term)
  pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
  return jnp.array(pe)  # These are trainable parameters, initialized as above.


def positional_encoding():
  """Concatenate positional encoding to the last dimension."""
  def init_fun(rng, input_shape):
    del rng
    input_shape_for_enc = input_shape
    params = _create_positional_encoding(input_shape_for_enc)
    last_dim = input_shape[-1] + params.shape[-1]
    return input_shape[:-1] + (last_dim,), (params,)

  def apply_fun(params, inputs, **kwargs):
    del kwargs
    assert inputs.ndim == 4
    params = params[0]
    symbol_size = inputs.shape[-2]
    enc = params[None, :, :symbol_size, :]
    enc = jnp.repeat(enc, inputs.shape[0], 0)
    return jnp.concatenate((inputs, enc), -1)

  return init_fun, apply_fun


def cnn(conv_depth=300,
        kernel_size=5,
        n_conv_layers=2,
        across_batch=False,
        add_pos_encoding=False):
  """Build convolutional neural net."""
  # Input shape: [batch x length x depth]
  if across_batch:
    extra_dim = 0
  else:
    extra_dim = 1
  layers = [ExpandDims(axis=extra_dim)]
  if add_pos_encoding:
    layers.append(positional_encoding())

  for _ in range(n_conv_layers):
    layers.append(
        stax.Conv(conv_depth, (1, kernel_size), padding="same", strides=(1, 1)))
    layers.append(stax.Relu)
  layers.append(AssertNonZeroShape())
  layers.append(squeeze_layer(axis=extra_dim))
  return stax.serial(*layers)


def build_model_stax(output_size,
                     n_dense_units=300,
                     conv_depth=300,
                     n_conv_layers=2,
                     n_dense_layers=0,
                     kernel_size=5,
                     across_batch=False,
                     add_pos_encoding=False,
                     mean_over_pos=False,
                     mode="train"):
  """Build a model with convolutional layers followed by dense layers."""
  del mode
  layers = [
      cnn(conv_depth=conv_depth,
          n_conv_layers=n_conv_layers,
          kernel_size=kernel_size,
          across_batch=across_batch,
          add_pos_encoding=add_pos_encoding)
  ]
  for _ in range(n_dense_layers):
    layers.append(stax.Dense(n_dense_units))
    layers.append(stax.Relu)

  layers.append(stax.Dense(output_size))

  if mean_over_pos:
    layers.append(reduce_layer(jnp.mean, axis=1))
  init_random_params, predict = stax.serial(*layers)
  return init_random_params, predict


def sample_log_probs_top_k(log_probs, rng, temperature=1., k=1):
  """Sample categorical distribution of log probs using gumbel max trick."""
  noise = jax.random.gumbel(rng, shape=log_probs.shape)
  perturbed = (log_probs + noise) / temperature
  samples = jnp.argsort(perturbed)[Ellipsis, -k:]
  return samples


@jax.jit
def gather_positions(idx_to_gather, logits):
  """Collect logits corresponding to the positions in the string.

  Used for collecting logits for:
    1) positions in the string (depth = 1)
    2) mutation types (depth = n_mut_types)

  Args:
    idx_to_gather: [batch_size x num_mutations] Indices of the positions
      in the string to gather logits for.
    logits:  [batch_size x str_length x depth] Logits to index.

  Returns:
    Logits corresponding to the specified positions in the string:
    [batch_size, num_mutations, depth]
  """
  assert idx_to_gather.shape[0] == logits.shape[0]
  assert idx_to_gather.ndim == 2
  assert logits.ndim == 3

  batch_size, num_mutations = idx_to_gather.shape
  batch_size, str_length, depth = logits.shape

  oh = one_hot(idx_to_gather, str_length)
  assert oh.shape == (batch_size, num_mutations, str_length)

  oh = oh[Ellipsis, None]
  logits = logits[:, None, :, :]
  assert oh.shape == (batch_size, num_mutations, str_length, 1)
  assert logits.shape == (batch_size, 1, str_length, depth)

  # Perform element-wise multiplication (with broadcasting),
  # then sum over str_length dimension
  result = jnp.sum(oh * logits, axis=-2)
  assert result.shape == (batch_size, num_mutations, depth)
  return result


class JaxMutationPredictor(object):
  """Implements training and predicting from a Jax model.

  Attributes:
    output_size: Tuple containing the sizes of components to predict
    loss_fn: Loss function.
      Format of the loss fn: fn(params, batch, mutations, problem, predictor)
    loss_grad_fn: Gradient of the loss function
    temperature: temperature parameter for Gumbel-Max sampler.
    learning_rate: Learning rate for optimizer.
    batch_size: Batch size of input
    model_fn: Function which builds the model forward pass. Must have arguments
      `vocab_size`, `max_len`, and `mode` and return Jax float arrays.
    params: weights of the neural net
    make_state: function to make optimizer state given the network parameters
    rng: Jax random number generator
  """

  def __init__(self,
               vocab_size,
               output_size,
               loss_fn,
               rng,
               temperature=1,
               learning_rate=0.001,
               conv_depth=300,
               n_conv_layers=2,
               n_dense_units=300,
               n_dense_layers=0,
               kernel_size=5,
               across_batch=False,
               add_pos_encoding=False,
               mean_over_pos=False,
               model_fn=build_model_stax):
    self.output_size = output_size
    self.temperature = temperature

    # Setup randomness.
    self.rng = rng

    model_settings = {
        "output_size": output_size,
        "n_dense_units": n_dense_units,
        "n_dense_layers": n_dense_layers,
        "conv_depth": conv_depth,
        "n_conv_layers": n_conv_layers,
        "across_batch": across_batch,
        "kernel_size": kernel_size,
        "add_pos_encoding": add_pos_encoding,
        "mean_over_pos": mean_over_pos,
        "mode": "train"
    }

    self._model_init, model_train = model_fn(**model_settings)
    self._model_train = jax.jit(model_train)

    model_settings["mode"] = "eval"
    _, model_predict = model_fn(**model_settings)
    self._model_predict = jax.jit(model_predict)

    self.rng, subrng = jrand.split(self.rng)
    _, init_params = self._model_init(subrng, (-1, -1, vocab_size))
    self.params = init_params

    # Setup parameters for model and optimizer
    self.make_state, self._opt_update_state, self._get_params = adam(
        learning_rate)

    self.loss_fn = functools.partial(loss_fn, run_model_fn=self.run_model)
    self.loss_grad_fn = jax.grad(self.loss_fn)

    # Track steps of optimization so far.
    self._step_idx = 0

  def update_step(self, rewards, inputs, actions):
    """Performs a single update step on a batch of samples.

    Args:
      rewards: Batch [batch] of rewards for perturbed samples.
      inputs: Batch [batch x length] of original samples
      actions: actions applied on the samples

    Raises:
      ValueError: if any inputs are the wrong shape.
    """

    grad_update = self.loss_grad_fn(
        self.params,
        rewards=rewards,
        inputs=inputs,
        actions=actions,
    )

    old_params = self.params
    state = self.make_state(old_params)
    state = self._opt_update_state(self._step_idx, grad_update, state)

    self.params = self._get_params(state)
    del old_params, state

    self._step_idx += 1

  def __call__(self, x, mode="eval"):
    """Calls predict function of model.

    Args:
      x: Batch of input samples.
      mode: Mode for running the network: "train" or "eval"

    Returns:
      A list of tuples (class weights, log likelihood) for each of
        output components predicted by the model.
    """
    return self.run_model(x, self.params, mode="eval")

  def run_model(self, x, params, mode="eval"):
    """Run the Jax model.

    This function is used in __call__ to run the model in "eval" mode
    and in the loss function to run the model in "train" mode.

    Args:
      x: Batch of input samples.
      params: Network parameters
      mode: Mode for running the network: "train" or "eval"

    Returns:
      Jax neural network output.
    """
    if mode == "train":
      model_fn = self._model_train
    else:
      model_fn = self._model_predict
    self.rng, subrng = jax.random.split(self.rng)
    return model_fn(params, inputs=x, rng=subrng)


#########################################
# Loss function
def reinforce_loss(rewards, log_likelihood):
  """Loss function for Jax model.

  Args:
    rewards: List of rewards [batch] for the perturbed samples.
    log_likelihood: Log-likelihood of perturbations

  Returns:
    Scalar loss.
  """
  rewards = jax.lax.stop_gradient(rewards)

  # In general case, we assume that the loss is not differentiable
  # Use REINFORCE
  reinforce_estim = rewards * log_likelihood
  # Take mean over the number of applied mutations, then across the batch
  return -jnp.mean(jnp.mean(reinforce_estim, 1), 0)


def compute_entropy(log_probs):
  """Compute entropy of a set of log_probs."""
  return -jnp.mean(jnp.mean(stax.softmax(log_probs) * log_probs, axis=-1))


def compute_advantage(params, critic_fn, rewards, inputs):
  """Compute the advantage: difference between rewards and predicted value.

  Args:
    params: parameters for the critic neural net
    critic_fn: function to run critic neural net
    rewards: rewards for the perturbed samples
    inputs: original samples, used as input to the Jax model

  Returns:
    advantage: [batch_size x num_mutations]
  """
  assert inputs.ndim == 4

  num_mutations, batch_size, str_length, vocab_size = inputs.shape

  inputs_reshaped = inputs.reshape(
      (num_mutations * batch_size, str_length, vocab_size))

  predicted_value = critic_fn(inputs_reshaped, params, mode="train")
  assert predicted_value.shape == (num_mutations * batch_size, 1)
  predicted_value = predicted_value.reshape((num_mutations, batch_size))

  assert rewards.shape == (batch_size,)
  rewards = jnp.repeat(rewards[None, :], num_mutations, 0)
  assert rewards.shape == (num_mutations, batch_size)

  advantage = rewards - predicted_value
  advantage = jnp.transpose(advantage)
  assert advantage.shape == (batch_size, num_mutations)
  return advantage


def value_loss_fn(params, run_model_fn, rewards, inputs, actions=None):
  """Compute the loss for the value function.

  Args:
    params: parameters for the Jax model
    run_model_fn: Jax model to run
    rewards: rewards for the perturbed samples
    inputs: original samples, used as input to the Jax model
    actions: not used

  Returns:
    A scalar loss.
  """
  del actions
  advantage = compute_advantage(params, run_model_fn, rewards, inputs)
  advantage = advantage**2

  return jnp.sqrt(jnp.mean(advantage))


def split_mutation_predictor_output(output):
  return stax.logsoftmax(output[:, :, -1]), stax.logsoftmax(output[:, :, :-1])


def run_model_and_compute_reinforce_loss(params,
                                         run_model_fn,
                                         rewards,
                                         inputs,
                                         actions,
                                         n_mutations,
                                         entropy_weight=0.1):
  """Run Jax model and compute REINFORCE loss.

  Jax can compute the gradients of the model only if the model is called inside
  the loss function. Here we call the Jax model, re-compute the log-likelihoods,
  take log-likelihoods of the mutations and positions sampled before in
  _propose function of the solver, and compute the loss.

  Args:
    params: parameters for the Jax model
    run_model_fn: Jax model to run
    rewards: rewards for the perturbed samples
    inputs: original samples, used as input to the Jax model
    actions: Tuple (mut_types [Batch], positions [Batch]) of mutation types
      and positions sampled during the _propose() step of evolution solver.
    n_mutations: Number of mutations. Used for one-hot encoding of mutations
    entropy_weight: Weight on the entropy term added to the loss.

  Returns:
    A scalar loss.

  """
  mut_types, positions = actions
  mut_types_one_hot = one_hot(mut_types, n_mutations)

  batch_size, str_length, _ = inputs.shape
  assert mut_types.shape[0] == inputs.shape[0]
  batch_size, num_mutations = mut_types.shape
  assert mut_types.shape == positions.shape
  assert mut_types.shape == rewards.shape

  output = run_model_fn(inputs, params, mode="train")
  pos_log_probs, all_mut_log_probs = split_mutation_predictor_output(output)
  assert pos_log_probs.shape == (batch_size, str_length)
  pos_log_probs = jnp.expand_dims(pos_log_probs, -1)

  pos_log_likelihoods = gather_positions(positions, pos_log_probs)
  assert pos_log_likelihoods.shape == (batch_size, num_mutations, 1)

  # Sum over number of positions
  pos_log_likelihoods = jnp.sum(pos_log_likelihoods, -1)

  # all_mut_log_probs shape: [batch_size, str_length, n_mut_types]
  assert all_mut_log_probs.shape[:2] == (batch_size, str_length)

  # Get mutation logits corresponding to the chosen positions
  mutation_logprobs = gather_positions(positions, all_mut_log_probs)

  # Get log probs corresponding to the selected mutations
  mut_log_likelihoods_oh = mutation_logprobs * mut_types_one_hot

  # Sum over mutation types
  mut_log_likelihoods = jnp.sum(mut_log_likelihoods_oh, -1)
  assert mut_log_likelihoods.shape == (batch_size, num_mutations)

  joint_log_likelihood = mut_log_likelihoods + pos_log_likelihoods
  assert joint_log_likelihood.shape == (batch_size, num_mutations)

  loss = reinforce_loss(rewards, joint_log_likelihood)
  loss -= entropy_weight * compute_entropy(mutation_logprobs)
  return loss


############################################
# MutationPredictorSolver
def initialize_uniformly(domain, batch_size, random_state):
  return domain.sample_uniformly(batch_size, seed=random_state)


@gin.configurable
class MutationPredictorSolver(base_solver.BaseSolver):
  """Choose the mutation operator conditioned on the sample.

  Sample from categorical distribution over available mutation operators
  using Gumbel-Max trick
  """

  def __init__(self,
               domain,
               model_fn=build_model_stax,
               random_state=0,
               **kwargs):
    """Constructs solver.

    Args:
      domain: discrete domain
      model_fn: Function which builds the forward pass of predictor model.
      random_state: Random state to initialize jax & np RNGs.
      **kwargs: kwargs passed to config.
    """
    super(MutationPredictorSolver, self).__init__(
        domain=domain, random_state=random_state, **kwargs)
    self.rng = jrand.PRNGKey(random_state)
    self.rng, rng = jax.random.split(self.rng)

    if self.domain.length < self.cfg.num_mutations:
      logging.warning("Number of mutations to perform per string exceeds string"
                      " length. The number of mutation is set to be equal to "
                      "the string length.")
      self.cfg.num_mutations = self.domain.length

    # Right now the mutations are defined as "Set position X to character C".
    # It allows to vectorize applying mutations to the string and speeds up
    # the solver.
    # If using other types of mutations, set self.use_assignment_mut=False.
    self.mutations = []
    for val in range(self.domain.vocab_size):
      self.mutations.append(functools.partial(set_pos, val=val))
    self.use_assignment_mut = True

    mut_loss_fn = functools.partial(run_model_and_compute_reinforce_loss,
                                    n_mutations=len(self.mutations))

    # Predictor that takes the input string
    # Outputs the weights over the 1) mutations types 2) position in string
    if self.cfg.pretrained_model is None:
      self._mut_predictor = self.cfg.predictor(
          vocab_size=self.domain.vocab_size,
          output_size=len(self.mutations) + 1,
          loss_fn=mut_loss_fn,
          rng=rng,
          model_fn=build_model_stax,
          conv_depth=self.cfg.actor_conv_depth,
          n_conv_layers=self.cfg.actor_n_conv_layers,
          n_dense_units=self.cfg.actor_n_dense_units,
          n_dense_layers=self.cfg.actor_n_dense_layers,
          across_batch=self.cfg.actor_across_batch,
          add_pos_encoding=self.cfg.actor_add_pos_encoding,
          kernel_size=self.cfg.actor_kernel_size,
          learning_rate=self.cfg.actor_learning_rate,
      )

      if self.cfg.use_actor_critic:
        self._value_predictor = self.cfg.predictor(
            vocab_size=self.domain.vocab_size,
            output_size=1,
            rng=rng,
            loss_fn=value_loss_fn,
            model_fn=build_model_stax,
            mean_over_pos=True,
            conv_depth=self.cfg.critic_conv_depth,
            n_conv_layers=self.cfg.critic_n_conv_layers,
            n_dense_units=self.cfg.critic_n_dense_units,
            n_dense_layers=self.cfg.critic_n_dense_layers,
            across_batch=self.cfg.critic_across_batch,
            add_pos_encoding=self.cfg.critic_add_pos_encoding,
            kernel_size=self.cfg.critic_kernel_size,
            learning_rate=self.cfg.critic_learning_rate,
        )

      else:
        self._value_predictor = None
    else:
      self._mut_predictor, self._value_predictor = self.cfg.pretrained_model

    self._data_for_grad_update = []
    self._initialized = False

  def _config(self):
    cfg = super(MutationPredictorSolver, self)._config()

    cfg.update(
        dict(
            predictor=JaxMutationPredictor,
            temperature=1.,
            initialize_dataset_fn=initialize_uniformly,
            elite_set_size=10,
            use_random_network=False,
            exploit_with_best=True,
            use_selection_of_best=False,
            pretrained_model=None,

            # Indicator to BO to pass in previous weights.
            # As implemented in cl/318101597.
            warmstart=True,

            use_actor_critic=False,
            num_mutations=5,

            # Hyperparameters for actor
            actor_learning_rate=0.001,
            actor_conv_depth=300,
            actor_n_conv_layers=1,
            actor_n_dense_units=100,
            actor_n_dense_layers=0,
            actor_kernel_size=5,
            actor_across_batch=False,
            actor_add_pos_encoding=True,

            # Hyperparameters for critic
            critic_learning_rate=0.001,
            critic_conv_depth=300,
            critic_n_conv_layers=1,
            critic_n_dense_units=300,
            critic_n_dense_layers=0,
            critic_kernel_size=5,
            critic_across_batch=False,
            critic_add_pos_encoding=True,
        ))
    return cfg

  def _get_unique(self, samples):
    unique_population = data.Population()
    unique_structures = set()
    for sample in samples:
      hashed_structure = utils.hash_structure(sample.structure)
      if hashed_structure in unique_structures:
        continue
      unique_structures.add(hashed_structure)
      unique_population.add_samples([sample])
    return unique_population

  def _get_best_samples_from_last_batch(self,
                                        population,
                                        n=1,
                                        discard_duplicates=True):
    best_samples = population.get_last_batch().best_n(
        n, discard_duplicates=discard_duplicates)
    return best_samples.structures, best_samples.rewards

  def _select(self, population):
    if self.cfg.use_selection_of_best:
      # Choose best samples from the previous batch
      structures, rewards = self._get_best_samples_from_last_batch(
          population, self.cfg.elite_set_size)

      # Choose the samples to perturb with replacement
      idx = np.random.choice(len(structures), self.batch_size, replace=True)
      selected_structures = np.stack([structures[i] for i in idx])
      selected_rewards = np.stack([rewards[i] for i in idx])
      return selected_structures, selected_rewards
    else:
      # Just return the samples from the previous batch -- no selection
      last_batch = population.get_last_batch()
      structures = np.array([x.structure for x in last_batch])
      rewards = np.array([x.reward for x in last_batch])

      if len(last_batch) > self.batch_size:
        # Subsample the data
        idx = np.random.choice(len(last_batch), self.batch_size, replace=False)
        structures = np.stack([structures[i] for i in idx])
        rewards = np.stack([rewards[i] for i in idx])
      return structures, rewards

  def propose(self, num_samples, population=None, pending_samples=None):
    # Initialize population randomly.
    if self._initialized and population:
      if num_samples != self.batch_size:
        raise ValueError("Must maintain constant batch size between runs.")
      counter = population.max_batch_index
      if counter > 0:
        if not self.cfg.use_random_network:
          self._update_params(population)
    else:
      self.batch_size = num_samples
      self._initialized = True
      return self.cfg.initialize_dataset_fn(
          self.domain, num_samples, random_state=self._random_state)

    # Choose best samples so far -- [elite_set_size]
    samples_to_perturb, parent_rewards = self._select(population)

    perturbed, actions, mut_predictor_input = self._perturb(samples_to_perturb)

    if not self.cfg.use_random_network:
      self._data_for_grad_update.append({
          "batch_index": population.current_batch_index + 1,
          "mut_predictor_input": mut_predictor_input,
          "actions": actions,
          "parent_rewards": parent_rewards,
      })

    return np.asarray(perturbed)

  def _perturb(self, parents, mode="train"):
    length = parents.shape[1]
    assert length == self.domain.length

    parents_one_hot = one_hot(parents, self.domain.vocab_size)

    output = self._mut_predictor(parents_one_hot)
    pos_log_probs, all_mut_log_probs = split_mutation_predictor_output(output)

    self.rng, subrng = jax.random.split(self.rng)
    positions = sample_log_probs_top_k(
        pos_log_probs,
        subrng,
        k=self.cfg.num_mutations,
        temperature=self.cfg.temperature)

    pos_masks = one_hot(positions, length)

    mutation_logprobs = gather_positions(positions, all_mut_log_probs)
    assert mutation_logprobs.shape == (output.shape[0], self.cfg.num_mutations,
                                       output.shape[-1] - 1)

    self.rng, subrng = jax.random.split(self.rng)
    mutation_types = gumbel_max_sampler(
        mutation_logprobs, self.cfg.temperature, subrng)

    states = apply_mutations(parents, mutation_types, pos_masks, self.mutations,
                             use_assignment_mutations=self.use_assignment_mut)
    # states shape: [num_mutations+1, batch, str_length]
    # states[0] are original samples with no mutations
    # states[-1] are strings with all mutations applied to them
    states_oh = one_hot(states, self.domain.vocab_size)
    # states_oh shape: [n_mutations+1, batch, str_length, vocab_size]
    perturbed = states[-1]

    return perturbed, (mutation_types, positions), states_oh

  def _update_params(self, population):
    if not self._data_for_grad_update:
      return

    dat = self._data_for_grad_update.pop()
    assert dat["batch_index"] == population.current_batch_index

    child_rewards = jnp.array(population.get_last_batch().rewards)
    parent_rewards = dat["parent_rewards"]

    all_states = dat["mut_predictor_input"]
    # all_states shape: [num_mutations, batch_size, str_length, vocab_size]

    # TODO(rubanova): rescale the rewards
    terminal_rewards = child_rewards

    if self.cfg.use_actor_critic:
      # Update the value function
      # Compute the difference between predicted value of intermediate states
      # and the final reward.
      self._value_predictor.update_step(
          rewards=terminal_rewards,
          inputs=all_states[:-1],
          actions=None,
      )
      advantage = compute_advantage(self._value_predictor.params,
                                    self._value_predictor.run_model,
                                    terminal_rewards, all_states[:-1])
    else:
      advantage = child_rewards - parent_rewards
      advantage = jnp.repeat(advantage[:, None], self.cfg.num_mutations, 1)

    advantage = jax.lax.stop_gradient(advantage)

    # Perform policy update.
    # Compute policy on the original samples, like in _perturb function.
    self._mut_predictor.update_step(
        rewards=advantage,
        inputs=all_states[0],
        actions=dat["actions"])

    del all_states, advantage

  @property
  def trained_model(self):
    return (self._mut_predictor, self._value_predictor)
