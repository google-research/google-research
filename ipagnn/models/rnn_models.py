# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Learned Interpreters RNN models."""

from absl import logging  # pylint: disable=unused-import
from flax import nn
import jax
import jax.numpy as jnp

from ipagnn.modules import common_modules

Embed = common_modules.Embed
StackedRNNCell = common_modules.StackedRNNCell


class StackedLSTMModel(nn.Module):
  """Applies an LSTM to form statement embeddings, and again for program embs.
  """

  def apply(self,
            example_inputs,
            info,
            config,
            train=False,
            cache=None):
    """Applies lstm model on the inputs.

    Args:
      example_inputs: input data
      info: Dataset info object
      config: config for the experiment
      train: bool: if model is training.
      cache: flax autoregressive cache for fast decoding. unused.

    Returns:
      output of a lstm decoder.
    """
    vocab_size = info.features[info._builder.key('statements')].vocab_size  # pylint: disable=protected-access
    output_token_vocabulary_size = info.output_vocab_size
    inputs = example_inputs['code_statements']
    lengths = example_inputs['code_length']
    emb_dim = config.model.hidden_size

    assert inputs.ndim == 2  # (batch_size, length)
    x = inputs.astype('int32')

    def emb_init(key, shape, dtype=jnp.float32):
      return jax.random.uniform(
          key, shape, dtype,
          -config.initialization.maxval,
          config.initialization.maxval)

    batch_size = x.shape[0]
    code_embeddings = Embed(
        x, num_embeddings=vocab_size, features=emb_dim,
        emb_init=emb_init,
        name='embed')
    # code_embeddings.shape: batch_size, length, emb_dim

    lstm1 = nn.LSTMCell.partial(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name='encoder_1',
    )
    lstm2 = nn.LSTMCell.partial(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name='encoder_2',
    )
    encoder_cells = [lstm1, lstm2]
    encoder = StackedRNNCell.partial(cells=encoder_cells)

    lstm3 = nn.LSTMCell.partial(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name='decoder_1',
    )
    lstm4 = nn.LSTMCell.partial(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name='decoder_2',
    )
    decoder = StackedRNNCell.partial(cells=[lstm3, lstm4])

    output_length = 1

    def get_logits(code_embeddings, length):
      # code_embeddings.shape: length, emb_dim

      initial_carry_e = encoder.initialize_carry(
          jax.random.PRNGKey(0), encoder_cells, (), emb_dim)

      def apply_encoder(carry, inp):
        i = carry[1]
        c1, o1 = encoder(carry[0], inp)
        return jax.tree_multimap(
            lambda x_new, x_old: jnp.where(i < length, x_new, x_old),
            ((c1, i+1), o1),
            (carry, inp)
        )

      (encoder_state, unused_i), unused_outputs = (
          jax.lax.scan(
              apply_encoder,
              (initial_carry_e, 0),
              code_embeddings
          )
      )

      decoder_inputs = jnp.zeros((output_length, emb_dim))
      unused_carry, decoder_outputs = jax.lax.scan(
          decoder, encoder_state, decoder_inputs)

      logits = nn.Dense(
          decoder_outputs,
          output_token_vocabulary_size,
          kernel_init=nn.initializers.normal(
              stddev=config.initialization.maxval),
          bias_init=nn.initializers.zeros,
          name='output_layer')
      return logits

    logits = jax.vmap(get_logits)(code_embeddings, lengths)
    return jnp.reshape(
        logits, (batch_size, output_length, output_token_vocabulary_size))


