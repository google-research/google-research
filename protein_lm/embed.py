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
"""Embedding API for pretrained models."""

import functools
from flax.training import common_utils
import gin
import jax
from protein_lm import data
from protein_lm import models


def make_batch(protein_strings, length=None):
  """Encode a list of protein strings to a batch of tokens.

  Args:
    protein_strings: List of strings.
    length: Crop sequences to this length. Defaults to max length in batch.

  Returns:
    [batch x length] jax numpy array.
  """
  max_length = max(len(x) for x in protein_strings)

  # TODO(ddohan): Avoid adding unnecessary padding tokens.
  batch = jax.numpy.array(data.protein_domain.encode(protein_strings, pad=True))
  if length is None:
    length = max_length + 1
  batch = batch[:, :length]
  return batch


@functools.lru_cache(10)
def get_embed_fn(model=None,
                 ckpt_dir=None,
                 output_head='output_emb',
                 reduce_fn=None,
                 length=128):
  """Get a function which maps lists of strings to embeddings.

  Args:
    model: Pretrained model.
    ckpt_dir: Directory to load if model is None.
    output_head: Which model output to return. See embed.FlaxLM
    reduce_fn: Postprocessing function to apply on top of embeddings, such as
      `partial(jax.numpy.sum, axis=-2)`.
    length: If given, use fixed length batches. Otherwise is length of longest
      string in the batch.

  Returns:
    Function which accepts a list of strings, and returns batched embeddings.
  """
  if model is None:
    if ckpt_dir is None:
      raise ValueError('Must provide a loaded model or checkpoint directory.')
    model = models.load_model(ckpt_dir=ckpt_dir)
  else:
    if ckpt_dir is not None:
      raise ValueError('Provide only one of `model` or checkpoint directory.')

  def predict_fn(model_target, inputs):

    emb = models.predict_step(
        model_target,
        inputs,
        preprocess_fn=model.preprocess,
        output_head=output_head)
    if reduce_fn:
      emb = reduce_fn(emb)
    return emb

  p_predict_step = jax.pmap(predict_fn, axis_name='batch')

  def _embed(protein_sequences):
    """Encode proteins into a batch, embed, and run reduce_fn on output."""
    batch = make_batch(protein_sequences, length=length)
    batch = common_utils.shard(batch)
    result = p_predict_step(model.optimizer.target, batch)

    # Combine leading two dimensions (ndevices, batch_size / n_devices)
    result = jax.numpy.reshape(result, [-1] + list(result.shape[2:]))
    return result

  return _embed


@gin.configurable
class ProteinLMEmbedder(object):
  """Embeddings from a pretrained language model.

  Stateful wrapper around get_embed_fn.
  """

  def __init__(self,
               model=None,
               ckpt_dir=None,
               output_head='output_emb',
               reduce_fn=None,
               length=128):
    self._embed_fn = get_embed_fn(
        model=model,
        ckpt_dir=ckpt_dir,
        output_head=output_head,
        reduce_fn=reduce_fn,
        length=length)

  def _decode(self, encoded_sequences):
    """Decode encoded sequences to strings."""
    return [
        data.protein_domain.vocab.decode(s, stop_at_eos=True)
        for s in encoded_sequences
    ]

  def __call__(self, encoded_sequences):
    """Decode encoded sequences to strings them embed."""
    sequences = self._decode(encoded_sequences)
    return self._embed_fn(sequences)
