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

"""Code for sampling from the LDA generative model."""
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnums=(1,))
def sample_params(key, num_docs, doc_topic_alpha, topic_word_alpha):
  """Samples the parameters for LDA, the topic/word dists and doc/topic dists.

  Args:
    key: A JAX PRNG key.
    num_docs: The number of documents to sample.
    doc_topic_alpha: The parameter alpha for the dirichlet prior over the
      document-topic categorical distributions. Should be a vector of shape
      [num_topics].
    topic_word_alpha: The alpha parameter for the dirichlet prior over the
      topic-word categorical distributions. Should be a vector of shape
        [vocab_size].
  Returns:
    topic_params: A [num_topics, vocab_size] matrix containing parameters
      for each topic's categorical distribution over words.
    doc_params: A [num_docs, num_topics] matrix containing parameters for
      each document's categorical distribution over topics.
  """
  # Each topic is a multinomial distribution over words.
  # The topic matrix is a [num_topics, vocab_size] set of parameters for
  # num_topics different multinomial distributions over the vocabulary.
  num_topics = doc_topic_alpha.shape[0]
  k1, k2 = jax.random.split(key)
  topic_params = jax.random.dirichlet(k1, alpha=topic_word_alpha,
                                      shape=[num_topics])
  # Each document is a multinomial distribution over topics.
  # The document matrix is a [num_documents, num_topics] set of parameters for
  # num_documents different multinomial distributions over the set of topics.
  doc_params = jax.random.dirichlet(k2, alpha=doc_topic_alpha,
                                    shape=[num_docs])
  return topic_params, doc_params


@partial(jit, static_argnums=3)
def sample_docs(key, topic_params, doc_params, doc_length):
  """Samples documents given parameters.

  Args:
    key: A JAX PRNG key.
    topic_params: A [num_topics, vocab_size] matrix containing parameters
      for each topic's categorical distribution over words.
    doc_params: A [num_docs, num_topics] matrix containing parameters for
      each document's categorical distribution over topics.
    doc_length: The length of each document, a python int.
  Returns:
    doc_words: A [num_documents, doc_length] matrix containing the indices of
      each word in each document. Each index will be in [0, vocab_size).
  """
  num_documents, _ = doc_params.shape
  k1, k2 = jax.random.split(key)
  # Sample the sequence of topics for each document, a
  # [num_documents, doc_length] matrix of topic indices.
  doc_topics = jax.random.categorical(k1, doc_params,
                                      shape=(doc_length, num_documents)).T
  # Sample the sequence of words for each document, a
  # [doc_length, num_documents] matrix of word indices.
  sample_doc = lambda a: jax.random.categorical(a[1], topic_params[a[0]])
  keys = jax.random.split(k2, num=num_documents)
  doc_words = jax.lax.map(sample_doc, (doc_topics, keys))
  return doc_words


@partial(jit, static_argnums=(1, 2, 3, 4))
def sample_lda(key, num_docs, num_topics, vocab_size, doc_length):
  """Samples documents and parameters from LDA using default prior parameters.

  Samples from LDA assuming that each element of doc_topic_alpha is 1/num_topics
  and each element of topic_word_alpha is 1/vocab_size.

  Args:
    key: A JAX PRNG key.
    num_docs: The number of documents to sample, a python int.
    num_topics: The number of latent topics, a python int.
    vocab_size: The number of possible words, a python int.
    doc_length: The length of each document, a python int.
  Returns:
    doc_words: A [num_documents, doc_length] matrix containing the indices of
      each word in each document. Each index will be in [0, vocab_size).
    topic_params: A [num_topics, vocab_size] matrix containing parameters
      for each topic's categorical distribution over words.
    doc_params: A [num_docs, num_topics] matrix containing parameters for
      each document's categorical distribution over topics.
  """
  key1, key2 = jax.random.split(key)
  topic_params, doc_params = sample_params(
      key1, num_docs,
      jnp.full([num_topics], 1. / num_topics),
      jnp.full([vocab_size], 1. / vocab_size))
  doc_words = sample_docs(key2, topic_params, doc_params, doc_length)
  return doc_words, topic_params, doc_params

