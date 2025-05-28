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

"""Common functions for setting up inputs for adversarial prefix learning."""

import collections
import functools
import string
from typing import Any, List, Optional, Tuple, TypedDict

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from paxml import trainer_lib
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
import seqio
from tensorflow_probability.substrates import jax as tfp

# Define aliases for brevity
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE


def calc_max_onehot(x):
  return jax.nn.one_hot(jnp.argmax(x, -1), x.shape[-1], dtype=x.dtype)


class GumbelSoftmaxParams(TypedDict):
  temp: float
  hard: bool


class WrappedModel:
  """Wrapper for a Pax model that uses identity-based equality comparisons.

  This makes it possible to pass the model into jax functions such as `jit`,
  even if it is not hashable.  Note: jax will retrace jit-compiled functions
  whenever they are called with a new instance of the wrapped model.
  """

  def __init__(self, model):
    self.model = model

  def __eq__(self, other):
    return isinstance(other, WrappedModel) and self.model is other.model

  def __hash__(self):
    return id(self.model)


@functools.partial(jax.vmap, in_axes=(0, None, 0))
def _gumbel_softmax_part(
    logits, temp, rng
):
  # Temp must be >0 or we will get NaNs.
  # Can't use if statement to validate input or this won't jit well.
  converted_logits = jnp.array(logits, dtype=jnp.float32)
  dist = tfp.distributions.RelaxedOneHotCategorical(
      temp, logits=converted_logits
  )
  return jnp.array(dist.sample(seed=rng), dtype=logits.dtype)


def _gumbel_softmax_batch_keys(inputs, temp, hard,
                               all_rngs):
  """Helper function for gumbel_softmax()."""

  def flatten(x):
    return jnp.reshape(x, (-1, x.shape[-1]))

  flat_inputs = flatten(inputs)
  all_rngs = flatten(all_rngs)

  y = _gumbel_softmax_part(flat_inputs, temp, all_rngs)

  def _hard_fn():
    y_hard = calc_max_onehot(y)
    return jax.lax.stop_gradient(y_hard - y) + y

  result = jax.lax.cond(hard, _hard_fn, lambda: y)

  return jnp.reshape(result, inputs.shape)


def gumbel_softmax(
    inputs, temp, hard, rng
):
  """Draws from the gumbel softmax distribution over the given inputs.

  Args:
    inputs: Array
    temp: temperature of the gumbel softmax
    hard: If true, sample one-hot vector. Else return logits
    rng: A single random key for the operation

  Returns:
    Samples from a gumbel softmax distribution for each set of logits in inputs
  """
  all_rngs = jax.random.split(rng, np.prod(inputs.shape[:-1]))
  return _gumbel_softmax_batch_keys(inputs, temp, hard, all_rngs)


def smooth_logits(
    tokens, smooth_factor, logits_dim, dtype
):
  onehot = jax.nn.one_hot(tokens, logits_dim, dtype=dtype)
  smoothed_onehot = onehot * (1 - 2 * smooth_factor) + smooth_factor
  return jnp.log(smoothed_onehot / jnp.sum(smoothed_onehot, -1, keepdims=True))


def replicate_batch(x, batch_size):
  return jnp.array([x] * batch_size)


def replicate_batch_tree(tree, batch_size):
  return jax.tree.map(
      functools.partial(replicate_batch, batch_size=batch_size), tree)


def contains_only(vocab_string, chars):
  """Checks if the string only contains only the listed chars."""
  return all(c in chars for c in vocab_string)


def keep_alphanumeric_punct(
    index,
    vocabulary,
    exclude_no_space,
):
  """Returns True if the string contains ascii chars or punct, but not both."""
  if index == 1005:
    # Keep space.
    return True

  vocab_string = vocabulary.decode([index])

  if not vocab_string:
    return False

  alphanum_chars = string.ascii_letters + string.digits

  if contains_only(vocab_string, alphanum_chars):
    if exclude_no_space:
      return (len(vocab_string) == 1) or (
          # Add a non functional token to the beginning
          # to detect if there's a space.
          ' '
          in vocabulary.decode([1011, index])
      )
    else:
      return True

  return contains_only(vocab_string, string.punctuation)


def keep_vocab(
    index,
    vocabulary,
    exclude_tokens,
    exclude_no_space,
):
  return keep_alphanumeric_punct(index, vocabulary, exclude_no_space) and (
      vocabulary.decode([index]) not in exclude_tokens
  )


def get_vocab_mask(
    vocabulary,
    exclude_tokens,
    exclude_no_space,
):
  """Masks out tokens where keep_alphanumeric_punct returns false."""
  mask = jnp.array([
      keep_vocab(i, vocabulary, exclude_tokens, exclude_no_space)
      for i in range(vocabulary.vocab_size)
  ])

  return mask


def make_inputs(
    prefix,
    input_len,
    decode_len,
    input_for_classify,
    vocabulary,
    vocab_mask,
    dtype,
):
  """Returns the tokens, logits and parameters for the input to the model.

  Args:
    prefix: The input text before the adversarial input.
    input_len: The length of the adversarial input.
    decode_len: The length of the decode output.
    input_for_classify: The classifier uses this input instead of the
      adversarial input. Set to None to pass the adversarial input to the
      classifier.
    vocabulary: The vocabulary
    vocab_mask: Boolean mask of whether to keep or discard each vocab index.
    dtype: The type of the input to generate.

  Returns:
    tokens: A dict of tokens. These are assembled in loss_fn to get the full
    input to the model.
      prefix: Tokens which go before the adversarial input.
      sep: The tokens between the adversarial input and the model output.
      decode: The placeholder tokens for the model output.
      input_for_classify: The classifier input to use instead of the
      adversarial input.
      classify: The tokens which tell the model to classify.
    logits: The adversarial logits to optimize.
    vocab_mask: Same as the input.
    label: The label token.
  """
  if input_for_classify is not None:
    input_for_classify_tokens = jnp.array(
        [0] + list(vocabulary.encode(input_for_classify)))
  else:
    input_for_classify_tokens = jnp.array([], dtype=jnp.int32)

  return NestedMap.FromNestedDict(
      dict(
          tokens=dict(
              prefix=jnp.array([0] + list(vocabulary.encode(prefix))),
              sep=jnp.array(vocabulary.encode('SEP')),
              decode=jnp.array([-1] * decode_len),
              input_for_classify=input_for_classify_tokens,
              classify=jnp.array(vocabulary.encode('SAFETY ')),
          ),
          logits=smooth_logits(
              [0] * input_len, 0.5, len(vocab_mask), dtype=dtype
          ),  # pytype: disable=wrong-arg-types  # jax-ndarray
          vocab_mask=vocab_mask,
          label=vocabulary.encode('1')[-1],
      )
  )


def simple_decode(
    mdl_vars,
    input_onehot,
    use_decoded_mask,
    prefix_len,
    gs_params,
    prng_key,
    greedy,
    wrapped_model,
):
  # pylint: disable-next=g-doc-args
  # pylint: disable-next=g-doc-return-or-yield
  """Calls simple_decode() of the given model.

  We pass the model in a wrapper because it is unhashable so cannot be jitted.
  """
  full_mdl_vars = mdl_vars.copy()
  full_mdl_vars['params'] = {
      'embedding_model': mdl_vars['params']['lm']['softmax'],
      'model': mdl_vars['params']
  }

  with base_layer.JaxContext.new_context(
      hparams=base_layer.JaxContext.HParams(do_eval=True)):
    outputs, _ = wrapped_model.model.apply(
        full_mdl_vars,
        input_onehot,
        use_decoded_mask,
        prefix_len,
        gs_params,
        greedy,
        method=wrapped_model.model.simple_decode,
        rngs={RANDOM: prng_key},
        mutable=[DECODE_CACHE],
    )
    return outputs


def logits_loss(output_logits, final_label):
  # How closely the final logit matches the label.
  output_probs = nn.softmax(output_logits[:, -1], axis=-1)
  return -output_probs[:, final_label]


def filter_onehot(input_onehot, vocab_mask):
  # This function only works with positive inputs.
  input_onehot = jnp.where(vocab_mask[None, None, :], input_onehot, 0.0)
  return input_onehot / jnp.sum(input_onehot, axis=-1, keepdims=True)


def difference_loss(
    input_logits,
    input_onehot,
    output_logits,
    vocab_mask,
):
  """How much do the updated token probabilities differ from the predicted ones.
  """
  del input_onehot

  normalized_output_logits = jax.nn.log_softmax(output_logits, axis=-1)
  input_probability = jax.nn.softmax(input_logits, axis=-1)
  input_probability = filter_onehot(input_probability, vocab_mask)

  result_logits = input_probability * normalized_output_logits
  result_logits = jnp.sum(result_logits, -1)

  # Divide by a constant to scale the loss.
  return -result_logits / 30.0


def loss_fn(
    input_logits,
    mdl_vars,
    full_input,
    prng_key,
    input_gs_params,
    decode_gs_params,
    difference_loss_weight,
    batch_size,
    is_eval,
    wrapped_model,
    return_aux_tokens,
):
  """Computes predicted output label and compares against true label.

  The input to simple_decode is constructed from full_input.tokens, and
  input_logits.
  If there are tokens in input_for_classify, the input is
    prefix, input_logits, sep, decode. The decode tokens are placeholders.
    simple_decode uses this input to get the output tokens.
    The next input is input_for_classify, sep, decode_output, classify.
    simple_decode uses this input to get the classifier score.
    Note that only input_for_classify is given when calculating the classifier
    score, not input_logits.

  If there are no input_for_classify tokens, the input is
    prefix, input_logits, sep, decode, classify
    simple_decode gets the decoded model output, then adds the classify tokens
    to get the classify score.

  Args:
    input_logits: The adversarial logits. The gradient is computed with respect
      to these.
    mdl_vars: Model vars
    full_input: See make_inputs. Use input_logits instead of the logits from
      here.
    prng_key: Rand key.
    input_gs_params: The temp and hard for the gumbel softmax on the input.
    decode_gs_params: The temp and hard for the gumbel softmax during decode.
    difference_loss_weight: How much optimize the probability of the input.
    batch_size: batch_size
    is_eval: True to use maximum instead of sampling with gumbel softmax.
    wrapped_model: Pax model.
    return_aux_tokens: True to return additional tokens for logging.

  Returns:
    total_loss: weighted sum of adversarial and difference losses
    losses: Map of loss and difference_loss
    aux_tokens: Returned if return_aux_tokens.
  """
  input_logits_batch = replicate_batch(input_logits, batch_size)

  if is_eval:
    # Take the softmax because
    # construct_decode_input only works with positive inputs.
    input_onehot = jax.nn.softmax(input_logits_batch, axis=-1)
    assert batch_size == 1
  else:
    prng_key, gs_prng_key = jax.random.split(prng_key)
    input_onehot = gumbel_softmax(input_logits_batch, input_gs_params['temp'],
                                  input_gs_params['hard'], gs_prng_key)

  input_onehot = filter_onehot(input_onehot, full_input.vocab_mask)
  if is_eval:
    input_onehot = calc_max_onehot(input_onehot)

  # Converts the tokens to onehot representation.
  onehot_tokens = {}
  for token_type, tokens in full_input.tokens.items():
    onehot_tokens[token_type] = replicate_batch(
        jax.nn.one_hot(
            tokens, input_onehot.shape[-1], dtype=input_onehot.dtype
        ),
        input_onehot.shape[0],
    )

  # Construct the input.
  decode_prefix_onehot = jnp.concatenate(
      [onehot_tokens['prefix'], input_onehot, onehot_tokens['sep']], axis=1)
  decode_prefix_len = decode_prefix_onehot.shape[1]
  use_decoded_mask = jnp.concatenate([
      jnp.zeros(decode_prefix_onehot.shape[1]),
      jnp.ones(onehot_tokens['decode'].shape[1])
  ],
                                     axis=0)
  decode_input_onehot = jnp.concatenate(
      [decode_prefix_onehot, onehot_tokens['decode']], axis=1)

  has_input_for_classify = full_input.tokens.input_for_classify.shape[0] > 0

  if not has_input_for_classify:
    # Add the classify tokens.
    decode_input_onehot = jnp.concatenate(
        [decode_input_onehot, onehot_tokens['classify']], axis=1)
    use_decoded_mask = jnp.concatenate(
        [use_decoded_mask,
         jnp.zeros(onehot_tokens['classify'].shape[1])],
        axis=0)

  prng_key, decode_key = jax.random.split(prng_key)
  decode_out = simple_decode(  # pytype: disable=wrong-arg-types  # jnp-type
      mdl_vars,
      decode_input_onehot,
      use_decoded_mask,
      decode_prefix_len,
      decode_gs_params,
      decode_key,
      greedy=is_eval,
      wrapped_model=wrapped_model,
  )

  # Calc difference_loss
  update_start = full_input.tokens['prefix'].shape[0] - 1
  # Axis 1 not 0 because input_onehot has been replicated into multiple batches.
  update_end = update_start + input_onehot.shape[1]
  updated_decoded_logits = decode_out['logits'][:, update_start:update_end, :]
  difference_loss_value = jnp.mean(
      difference_loss(
          input_logits_batch,
          input_onehot,
          updated_decoded_logits,
          full_input.vocab_mask,
      )
  )

  if has_input_for_classify:
    # Construct the classifier input.
    classify_input_onehot = jnp.concatenate([
        onehot_tokens['input_for_classify'], onehot_tokens['sep'],
        decode_out['onehot'], onehot_tokens['classify']
    ],
                                            axis=1)
    decode_mask_classify = jnp.zeros(classify_input_onehot.shape[1])
    classify_prefix_len = len(decode_mask_classify)

    prng_key, decode_key = jax.random.split(prng_key)
    classify_out = simple_decode(  # pytype: disable=wrong-arg-types  # jnp-array
        mdl_vars,
        classify_input_onehot,
        decode_mask_classify,
        classify_prefix_len,
        decode_gs_params,
        decode_key,
        greedy=is_eval,
        wrapped_model=wrapped_model,
    )

    loss = jnp.mean(logits_loss(classify_out['logits'], full_input.label))

  else:
    loss = jnp.mean(logits_loss(decode_out['logits'], full_input.label))

  total_loss = loss + difference_loss_value * difference_loss_weight
  losses = {'loss': loss, 'difference_loss': difference_loss_value}

  if return_aux_tokens:
    aux_tokens = {
        'decode_prefix': jnp.argmax(decode_prefix_onehot, -1),
        'decode_input': jnp.argmax(decode_input_onehot, -1),
        'decode_out_onehot': jnp.argmax(decode_out['onehot'], -1),
        'decode_out_logits': jnp.argmax(decode_out['logits'], -1),
    }
    if has_input_for_classify:
      aux_tokens['classify_input'] = jnp.argmax(classify_input_onehot, -1)
    return total_loss, losses, aux_tokens
  else:
    return total_loss, losses


loss_fn_jit = jax.jit(
    loss_fn,
    static_argnames=[
        'wrapped_model',
        'batch_size',
        'is_eval',
        'return_aux_tokens',
    ],
)


def loss_grad(
    full_input,
    model_states,
    prng_key,
    input_gs_params,
    decode_gs_params,
    difference_loss_weight,
    batch_size,
    wrapped_model,
):
  """Returns the loss, and gradient of loss_fn."""
  (_, loss), grad = jax.value_and_grad(loss_fn, has_aux=True)(
      full_input.logits,
      model_states.mdl_vars,
      full_input,
      prng_key,
      input_gs_params,
      decode_gs_params,
      difference_loss_weight,
      batch_size,
      is_eval=False,
      wrapped_model=wrapped_model,
      return_aux_tokens=False,
  )
  return loss, grad


@functools.partial(
    jax.pmap,
    in_axes=(0, 0, 0, None, None, None, None, None, None),
    static_broadcasted_argnums=[7, 8],
    axis_name='batch')
# Arguments must be passed by position not keyword because of pmap
def update_input_rep_par(
    full_input,
    model_states,
    prng_key,
    lr,
    input_gs_params,
    decode_gs_params,
    difference_loss_weight,
    local_batch_size,
    wrapped_model,
):
  """Updates the input logits to minimize the loss."""
  prng_key, loss_rng = jax.random.split(prng_key)

  loss, grad = loss_grad(
      full_input,
      model_states,
      loss_rng,
      input_gs_params,
      decode_gs_params,
      difference_loss_weight,
      local_batch_size,
      wrapped_model,
  )

  grad = jax.lax.pmean(grad, axis_name='batch')
  loss = jax.lax.pmean(loss, axis_name='batch')

  optimizer = optax.adam(lr)
  updates, opt_state = optimizer.update(grad, model_states.opt_states)
  input_logits = optax.apply_updates(full_input.logits, updates)

  return input_logits, opt_state, loss, prng_key


def eval_label_prob(
    full_input,
    model_states,
    verbose,
    vocabulary,
    wrapped_model,
):
  """Computes the probability of the label after the decoding step.

  Uses the token with the highest probability for the input, and during
  decoding. It doesn't use gumbel softmax.

  Args:
    full_input: The result of make_inputs. Used for the logits, and vocab_mask.
    model_states: For the model.
    verbose: Prints the inputs and outputs if True.
    vocabulary: Model vocab.
    wrapped_model: Task including the model.

  Returns:
    The probability of the label after the decoding.
    The full tokens that are used to calculate the score. This is the
    adversarial input followed by the separator, followed by the decoded output.
  """
  _, losses, aux_tokens = loss_fn_jit(
      full_input.logits,
      model_states.mdl_vars,
      full_input,
      jax.random.PRNGKey(0),
      input_gs_params=None,
      decode_gs_params=None,
      difference_loss_weight=0.0,
      batch_size=1,
      is_eval=True,
      wrapped_model=wrapped_model,
      return_aux_tokens=True,
  )
  loss = losses['loss']

  if verbose:
    display_dict = collections.defaultdict(list)

    for key, tokens in aux_tokens.items():
      display_dict[key] = _display_tokens(tokens[0], vocabulary)

    display_dict = {k: pd.Series(v) for k, v in display_dict.items()}

    print(pd.DataFrame(display_dict).to_string())

  decode_out_tokens = aux_tokens['decode_out_onehot'][
      0, :len(full_input.tokens['decode']) + 1]
  decode_input_output = jnp.concatenate(
      [aux_tokens['decode_prefix'][0], decode_out_tokens])

  return -loss, losses['difference_loss'], decode_input_output


def make_model_input(tokens):
  num_tokens = len(tokens)
  return NestedMap.FromNestedDict({
      'ids': tokens,
      'labels': np.zeros([num_tokens], np.int32),
      'paddings': np.zeros([num_tokens], np.int32),
      'weights': np.zeros([num_tokens], np.int32),
      'segment_ids': None,
      'segment_pos': None,
  })


@functools.partial(jax.jit, static_argnames='wrapped_model')
def regular_decode(model_states, input_tokens, wrapped_model):
  """Decodes using the built in PAX decoding."""
  model_input = replicate_batch_tree(make_model_input(input_tokens), 1)
  var_weight_hparams = wrapped_model.model.abstract_init_with_metadata(
      model_input
  )
  (_, per_example_out, _), _ = trainer_lib.decode_step(
      wrapped_model.model,
      model_states.to_eval_state(),
      jax.random.PRNGKey(1234),
      var_weight_hparams,
      model_input,
      fprop_dtype=wrapped_model.model.fprop_dtype,
  )
  return per_example_out


def dec_enc(tokens,
            vocabulary):
  """Decodes the tokens with the vocab then encodes them again.

  It will usually give the same result. This is used to make sure the input is
  tokens which are possible.

  Args:
   tokens: tokens
   vocabulary: vocabulary

  Returns:
    The tokens after decoding then encoding.
  """
  return jnp.array([0] + list(vocabulary.encode(vocabulary.decode(tokens))))


def filter_after_eos(tokens):
  """Removes all tokens after the eos token (eos token has value 1)."""
  # The 0th element of where is the first occurrence.
  index = jnp.where(tokens == 1)[0]
  if index.shape[0] == 0:
    return tokens
  return tokens[:index[0]]


def eval_label_prob_reg_decode(
    full_input,
    model_states,
    verbose,
    vocabulary,
    wrapped_model,
    use_dec_enc,
):
  """Similar to eval_label_prob, but it uses the normal PAX decode algorithm.

  Uses tokens instead of a one hot encoding.
  Usually it has the same results as eval_label_prob.
  But sometimes the results are different due to floating point errors.

  Args:
    full_input: The result of make_inputs. Used for the logits, and vocab_mask.
    model_states: For the model.
    verbose: Prints the inputs and outputs if True.
    vocabulary: Model vocab.
    wrapped_model: Task including the model.
    use_dec_enc: True to apply the dec_enc to the input.

  Returns:
    The probability of the label after the decoding.
    The full tokens that are used to calculate the score. This is the
    adversarial input followed by the separator, followed by the decoded output.
  """
  input_onehot = jax.nn.softmax(full_input.logits, axis=-1)
  input_onehot = filter_onehot(input_onehot, full_input.vocab_mask)[0]
  input_tokens = jnp.argmax(input_onehot, -1)

  if use_dec_enc:
    input_tokens = dec_enc(input_tokens, vocabulary)
  full_input_tokens = jnp.concatenate(
      [full_input.tokens.prefix, input_tokens, full_input.tokens.sep], 0)
  if use_dec_enc:
    full_input_tokens = dec_enc(full_input_tokens, vocabulary)

  decode_out = regular_decode(model_states, full_input_tokens, wrapped_model)

  decode_end_i = full_input_tokens.shape[0] + full_input.tokens.decode.shape[0]
  if full_input.tokens.input_for_classify.shape[0] > 0:
    decode_end_i += 1
  decoded_tokens = decode_out['output_ids'][
      0, 0, full_input_tokens.shape[0]:decode_end_i]
  decoded_tokens = filter_after_eos(decoded_tokens)

  full_decode_tokens = jnp.concatenate([full_input_tokens, decoded_tokens], 0)

  if full_input.tokens.input_for_classify.shape[0] > 0:
    safety_classifier_input = jnp.concatenate([
        full_input.tokens.input_for_classify, full_input.tokens.sep,
        decoded_tokens, full_input.tokens.classify
    ], 0)
  else:
    safety_classifier_input = jnp.concatenate(
        [full_decode_tokens, full_input.tokens.classify], 0)

  safety_out = regular_decode(
      model_states, safety_classifier_input, wrapped_model
  )

  safety_token = safety_out['output_ids'][0, 0,
                                          safety_classifier_input.shape[0]]
  safety_token_prob = jnp.exp(
      safety_out['logprobs'][0, 0, safety_classifier_input.shape[0]])

  if verbose:
    display_dict = collections.defaultdict(list)

    display_dict['decoded_tokens'] = _display_tokens(full_decode_tokens,
                                                     vocabulary)
    display_dict['safety_out'] = _display_tokens(
        safety_out['output_ids'][0, 0, :safety_classifier_input.shape[0] + 1],
        vocabulary)

    display_dict = {k: pd.Series(v) for k, v in display_dict.items()}

    print(pd.DataFrame(display_dict).to_string())

  if safety_token == full_input.label:
    return safety_token_prob, full_decode_tokens
  else:
    return 1 - safety_token_prob, full_decode_tokens


def _display_tokens(
    tokens,
    vocabulary):
  display_vals = []
  for token in tokens:
    display_vals.append(vocabulary.decode([token]))
  return display_vals
