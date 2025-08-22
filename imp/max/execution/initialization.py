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

"""Custom initialization of the model weights."""

import functools
import pickle

from absl import logging
from flax import traverse_util
from jax import numpy as jnp
import tensorflow as tf

from imp.max.core import constants
from imp.max.utils import typing


# TODO(b/234949870): deprecate this and merge with checkpointing pipeline
def restore_word_embeddings(model_params,
                            tokenizer):
  """Restore word embeddings from a given path."""

  model_params = traverse_util.flatten_dict(model_params, sep='/')
  embd_key = 'params/text_raw_to_embed/txt_to_embedding/embedding'
  if embd_key not in model_params:
    raise ValueError(
        f'Initialization override function expects `{embd_key}` in '
        "params dictionary. Make sure your model's name scope matches.")

  model_embeddings = model_params[embd_key]
  d_model = model_embeddings.shape[-1]

  if tokenizer == constants.HOWTO100M_EN:
    embedding_name = 'word2vec'
  elif tokenizer == constants.BERT_EN:
    dim2size = {512: 'small', 768: 'base', 1024: 'large'}
    embedding_name = 'bert_uncased_{}'.format(dim2size[d_model])
  else:
    raise ValueError('Text tokenizer {!r} not supported!'.format(tokenizer))

  embedding_path = f'{constants.EMBEDDING_DIR}/{embedding_name}.pkl'

  with tf.io.gfile.GFile(embedding_path, 'rb') as fh:
    embedding_values = pickle.load(fh)['word_embeddings']

  # make sure the correct vocab_size has been used
  if model_embeddings.shape != embedding_values.shape:
    raise ValueError('Text embedding layer is not configured properly. '
                     f'Expected shape={embedding_values.shape}, '
                     f'but configured with shape={model_embeddings.shape}.')

  # finally replace embedding weights
  model_params[embd_key] = jnp.array(embedding_values)

  # unflatten model_params
  model_params = traverse_util.unflatten_dict(model_params, sep='/')

  logging.info('Word embeddings restored successfully from %s', embedding_path)
  return model_params


def create_init_fn(init_name):
  """Create init override function to be used in execution."""

  if not init_name:
    return

  elif init_name == 'word_embedding_howto100m_en':
    return functools.partial(
        restore_word_embeddings,
        tokenizer=constants.HOWTO100M_EN,
    )
  elif init_name == 'word_embedding_bert_uncased_en':
    return functools.partial(
        restore_word_embeddings,
        tokenizer=constants.BERT_EN,
    )
  else:
    return
