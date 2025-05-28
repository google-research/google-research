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

"""Definition of Flax models used in STUDY."""


from typing import Any, Callable, Optional

from flax import linen as nn
from flax import struct
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np
from study_recommend import types

INPUT_FIELDS = types.ModelInputFields

Initializer = initializers.Initializer


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable[Ellipsis, Initializer] = nn.initializers.xavier_uniform()
  bias_init: Callable[Ellipsis, Initializer] = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable[[], jnp.ndarray]] = None
  dynamic_mask_randomizing: bool = True
  separator_token_value: Optional[int] = None
  num_grade_levels: Optional[int] = None


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis.

  The input is padded with a single zero on the left in the sequence dimension
  and the final element is removed. If this shifted sequence is fed into an
  autoregressive we would expect the original sequence as output.

  Args:
    x: jax.ndarray to be shifted right
    axis: The axis on which to shift

  Returns:
    The shifted jax.ndarray.
  """
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
  )
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace End of previous sequence by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= segment_ids == shift_right(segment_ids, axis=axis)
  return shifted


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Implement the sin/cosine based positional encoding from (Vaswani et al. 2017)
  Paper: https://arxiv.org/abs/1706.03762

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Initialize the Sinusoidal Position Embedding."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """

  posemb_init: Optional[Callable[[], jnp.ndarray]]
  max_len: int
  decode: bool = False

  @nn.compact
  def __call__(self, inputs, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a learned
    position embedding is desired, pass an initializer to posemb_init in the
    configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, (
        'Number of dimensions should be 3, but it is: %d' % inputs.ndim
    )
    length = inputs.shape[-2]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=self.max_len)(
          None, pos_emb_shape, None
      )
    else:
      pos_embedding = self.param(
          'pos_embedding', self.posemb_init, pos_emb_shape
      )
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable(
          'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.uint32)
      )
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """

  mlp_dim: int
  kernel_init: Callable[Ellipsis, Initializer]
  bias_init: Callable[Ellipsis, Initializer]
  dropout_rate: float
  deterministic: bool
  dtype: Any
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
    )(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
    )(x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=self.deterministic
    )
    return output


@struct.dataclass
class EncoderDecoderBlockConfig:
  num_heads: int
  dtype: Any
  qkv_dim: int
  kernel_init: Callable[Ellipsis, Initializer]
  bias_init: Callable[Ellipsis, Initializer]
  attention_dropout_rate: float
  dropout_rate: float
  mlp_dim: int
  deterministic: bool
  decode: bool


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: EncoderDecoderBlockConfig

  @nn.compact
  def __call__(self, inputs, decoder_mask=None, encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    config = self.config

    # Decoder block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=config.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
        decode=config.decode,
    )(x, decoder_mask)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )
    x = x + inputs

    # MLP block.
    z = nn.LayerNorm(dtype=config.dtype)(x)
    z = MlpBlock(
        mlp_dim=config.mlp_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        dropout_rate=config.dropout_rate,
        deterministic=config.deterministic,
        dtype=config.dtype,
    )(z)

    return x + z


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """

  config: TransformerConfig

  def validate_input(self, input_):
    assert input_.ndim == 2  # (batch, len)

  @nn.compact
  def __call__(
      self,
      inputs,
      inputs_positions=None,
      inputs_segmentation=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    config = self.config
    self.validate_input(inputs)
    # Initialize the embedding layer.

    output_embed = nn.Embed(
        num_embeddings=config.output_vocab_size,
        features=config.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
    )

    y = inputs.astype('int32')
    if not config.decode:
      # Get predictions for elements index 0 till (i - 1) where we have labels.
      # by shifting the input right.
      y = shift_inputs(y, segment_ids=inputs_segmentation)

    # Embed the inputs.
    y = output_embed(y)
    # Apply positional embedding and dropout to the embeddings.
    y = AddPositionEmbs(
        posemb_init=config.posemb_init,
        max_len=config.max_len,
        decode=config.decode,
        name='posembed_output',
    )(y, inputs_positions=inputs_positions)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=config.deterministic
    )

    y = y.astype(config.dtype)

    # The main body of the Decoder.
    layer_config = EncoderDecoderBlockConfig(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_dim=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        attention_dropout_rate=config.attention_dropout_rate,
        dropout_rate=config.dropout_rate,
        mlp_dim=config.mlp_dim,
        deterministic=config.deterministic,
        decode=config.decode,
    )

    # The main body of the transformer.
    for lyr in range(config.num_layers):
      y = EncoderDecoder1DBlock(
          config=layer_config, name=f'encoderdecoderblock_{lyr}'
      )(y, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask)
    y = nn.LayerNorm(dtype=config.dtype, name='encoderdecoder_norm')(y)

    # Compute the final logits.
    if config.logits_via_embedding:
      # Input and output emebedding weights are shared.
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          config.output_vocab_size,
          dtype=config.dtype,
          kernel_init=config.kernel_init,
          bias_init=config.bias_init,
          name='logitdense',
      )(y)
    return logits


class IndividualRecommender(nn.Module):
  """Autoregressive sequence modeling based recommender model.

  The model is a Transformer-based pure decoder stack for causal sequence
  modeling and it makes recommendations by conditioning on only the history of
  the student at hand.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs):
    """Applies IndividualRecommender on the inputs.

    Args:
      inputs: dictionary representing a batch of input data target data with
        following keys: titles - A batch of arrays representing sequences of
        titles interacted with per user. Each title is represented by the titles
        integer index. student_ids - A batch of arrays that represent the
        StudentID corresponding to the interactions recorded in titles. Multiple
        different StudentIDs in the same row denotes packed examples.
        inputs_positions - input subsequence positions for packed examples.
        Input positions for each packed example must not contain indices with
        values >= length of the packed sample. Undefined behaviour occurs if
        this is voilated.

    Returns:
      logits array from the recommender.
    """
    titles = inputs[INPUT_FIELDS.TITLES]
    input_positions = inputs.get(INPUT_FIELDS.INPUT_POSITIONS, None)
    input_segmentations = inputs.get(INPUT_FIELDS.STUDENT_IDS, None)
    config = self.config

    if isinstance(input_positions, np.ndarray):
      input_positions = jnp.array(input_positions)

    if config.decode:
      # Use no attention mask and caching for fast autogregressive decoding.
      decoder_mask = None
    else:
      # Make causal attention masks and no attention to padding.
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(titles > 0, titles > 0, dtype=config.dtype),
          nn.make_causal_mask(titles, dtype=config.dtype),
      )

    # Enforce additional block-diagonal attention masks if packing examples.
    # Example packing is when the sequences from multiple data points are
    # concatenated to form a longer sequence. When examples are packed
    # IndividualRecommender will produce output identical to processing each
    # datapoint separately and concatenating the results.
    # Packing is necessary to avoid adding large amounts of padding
    # or muliple (slow) Jax JIT compilations during training.
    if input_segmentations is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(
              input_segmentations,
              input_segmentations,
              jnp.equal,
              dtype=config.dtype,
          ),
      )
    if config.num_grade_levels is not None:
      grade_levels = inputs[INPUT_FIELDS.GRADE_LEVELS]
      # As grade is constant throughout each row and is available before
      # the first title we will not call shift_input on grade_levels.

      # We will generate and apply a set of biases. For each grade level we
      # will learn one unique bias per title.
      grade_levels_biases = nn.Embed(
          num_embeddings=config.num_grade_levels,
          features=config.output_vocab_size,
          embedding_init=nn.initializers.normal(stddev=1e-2),
          dtype=config.dtype,
          name='grade_levels_bias',
      )(grade_levels)
      # Add axis for sequence.
      grade_levels_biases = grade_levels_biases[:, jnp.newaxis, :]
    else:
      grade_levels_biases = 0
    # Apply the transformer decoder to the data with the computed mask
    # and add on grade level biases.
    logits = (
        Decoder(config=config, name='decoder')(
            titles,
            inputs_positions=input_positions,
            inputs_segmentation=input_segmentations,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=None,
        )
        + grade_levels_biases
    )

    return logits.astype(self.config.dtype)


class StudyRecommender(nn.Module):
  """Implementation of the STUDY Recommender model.

  This model will make recommendations for each user by conditioning
  on the users history as well as the history of peers.
  Paper: https://arxiv.org/abs/2306.07946
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs):
    """Applies StudyRecommender on the inputs.

    Args:
      inputs: dictionary representing a batch of input data target data with
        following keys: titles - A batch of arrays representing sequences of
        titles interacted with. Each title is represented by the titles integer
        index. student_ids - A batch of arrays that represent the StudentID
        corresponding to the interactions recorded in titles. Multiple different
        StudentIDs in the same row denotes a group of students in the same
        classroom and inference is done jointly. timestamps - timestamps of each
        interaction. This is used to enforce temporal causality in cross user
        conditioning.

    Returns:
      logits array from STUDY Recommender.
    """
    config = self.config
    titles = inputs[INPUT_FIELDS.TITLES]
    student_ids = inputs[INPUT_FIELDS.STUDENT_IDS]
    timestamps = inputs[INPUT_FIELDS.TIMESTAMPS]
    inputs_positions = None

    # Create a causal upper triangular attention mask.
    decoder_mask = nn.make_causal_mask(titles, dtype=config.dtype)

    # Combine with block-diagonal attention masks for within-student
    # interactions to enforce only within-student interactions.
    within_student = nn.make_attention_mask(
        student_ids, student_ids, jnp.equal, dtype=config.dtype
    )
    # Combining the two masks reduces by logical and.
    decoder_mask = nn.combine_masks(decoder_mask, within_student)

    # The mask timestamp_causal allows cross student attention to records that
    # have an earlier timestamps.
    timestamp_causal = nn.make_attention_mask(
        timestamps, timestamps, jnp.greater, dtype=config.dtype
    )

    decoder_mask = jnp.logical_or(
        decoder_mask,
        timestamp_causal,
    )
    # Mask out padding and separator tokens.
    not_padding = titles != 0
    not_separator = titles != config.separator_token_value
    is_title = jnp.logical_and(not_padding, not_separator)

    decoder_mask = nn.combine_masks(
        decoder_mask,
        nn.make_attention_mask(is_title, is_title, dtype=config.dtype),
    )

    if config.num_grade_levels is not None:
      grade_levels = inputs[INPUT_FIELDS.GRADE_LEVELS]
      # As grade is constant throughout each row and is available before
      # the first title we will not call shift_input on grade_levels.

      # We will generate apply a set of biases bias for each title that is
      # unique for each grade level.
      grade_levels_biases = nn.Embed(
          num_embeddings=config.num_grade_levels,
          features=config.output_vocab_size,
          embedding_init=nn.initializers.normal(stddev=1e-2),
          dtype=config.dtype,
          name='grade_levels_bias',
      )(grade_levels)
      # Add axis for sequence.
      grade_levels_biases = grade_levels_biases[:, jnp.newaxis, :]
    else:
      grade_levels_biases = 0
    # Apply the transformer decoder to the data with the computed mask
    # and add on grade level biases.

    logits = (
        Decoder(config=config, name='decoder')(
            titles,
            inputs_positions=inputs_positions,
            inputs_segmentation=None,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=None,
        )
        + grade_levels_biases
    )
    return logits.astype(self.config.dtype)


def generate_model_config(experiment_config):
  """Generate the Transformer config from the global experiment config."""
  return TransformerConfig(
      output_vocab_size=experiment_config.vocab_size,
      logits_via_embedding=experiment_config.logits_via_embedding,
      dtype=jnp.bfloat16 if experiment_config.use_bfloat16 else jnp.float32,
      emb_dim=experiment_config.emb_dim,
      num_heads=experiment_config.num_heads,
      num_layers=experiment_config.num_layers,
      qkv_dim=experiment_config.qkv_dim,
      mlp_dim=experiment_config.mlp_dim,
      max_len=max(
          experiment_config.max_target_length,
          experiment_config.max_eval_target_length,
      ),
      dropout_rate=experiment_config.dropout_rate,
      attention_dropout_rate=experiment_config.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
      separator_token_value=experiment_config.separator_token,
      num_grade_levels=experiment_config.num_grade_levels,
  )
