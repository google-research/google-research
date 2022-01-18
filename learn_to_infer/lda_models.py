# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Transformers for performing inference in the LDA generative model.
"""
from functools import partial

from . import transformer
from . import util

import flax
from flax.deprecated import nn
import jax
import jax.numpy as jnp
import jax.random


class Embedding(nn.Module):

  def apply(self,
            inputs,
            num_embeddings,
            embedding_dim,
            weight_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embedding module.

    Args:
      inputs: input data
      num_embeddings: The number of embedding vectors.
      embedding_dim: The size of the embedding dimension.
      weight_init: embedding initializer
    Returns:
      output: embedded input data
    """
    embedding = self.param("embedding", (num_embeddings, embedding_dim),
                           weight_init)
    if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
      raise ValueError("Input type must be an integer or unsigned integer.")
    return jnp.take(embedding, inputs, axis=0)


class SequenceEmbedding(nn.Module):

  def apply(self,
            input_seq,
            num_embeddings=3000,
            embedding_dim=512,
            weight_init=nn.initializers.normal(stddev=1.)):
    embeddings = Embedding(input_seq, num_embeddings, embedding_dim,
                           weight_init=weight_init)
    return jnp.sum(embeddings, axis=-2)


class LDATopicWordInferenceMachine(nn.Module):

  def apply(self,
            documents,
            topic_params,
            num_topics=50,
            vocab_size=30000,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            qkv_dim=512,
            embedding_dim=512,
            activation_fn=flax.nn.relu,
            weight_init=jax.nn.initializers.xavier_uniform(),
            embedding_init=jax.nn.initializers.normal(stddev=1.0)):
    batch_size, num_documents, _ = documents.shape

    # [batch_size, num_documents, doc_length, embedding_dim]
    doc_word_embeddings = Embedding(
        documents, num_embeddings=vocab_size, embedding_dim=embedding_dim,
        weight_init=embedding_init)
    # [batch_size, num_documents, embedding_dim]
    doc_embeddings = jnp.sum(doc_word_embeddings, axis=-2)
    # out_topics will be [batch_size, num_topics, vocab_size]
    out_raw_topic_params = transformer.EncoderDecoderTransformer(
        doc_embeddings,
        jnp.full([batch_size], num_documents),
        jnp.full([batch_size], num_topics),
        targets=topic_params,
        target_dim=vocab_size,
        max_input_length=num_documents,
        max_target_length=num_topics,
        num_heads=num_heads,
        num_encoders=num_encoders,
        qkv_dim=qkv_dim,
        activation_fn=activation_fn,
        weight_init=weight_init)

    # softmax the last dimension of the topic params to make each vector sum
    # to one.
    out_topic_params = jax.nn.softmax(out_raw_topic_params, axis=-1)

    return out_topic_params

  @classmethod
  def loss(cls, params, key, documents, topic_params):
    batch_size, num_topics, _ = topic_params.shape
    pred_topic_params = cls.call(
        params, documents, topic_params=topic_params)

    wasserstein_dist, _ = jax.vmap(
        util.atomic_sinkhorn, in_axes=(0, None, 0, None, 0))(
            topic_params, jnp.zeros(num_topics) - jnp.log(num_topics),
            pred_topic_params, jnp.zeros(num_topics) - jnp.log(num_topics),
            jax.random.split(key, num=batch_size))

    return wasserstein_dist


class LDADocTopicInferenceMachine(nn.Module):

  def apply(self,
            documents,
            topic_params,
            doc_topic_proportions,
            num_topics=50,
            vocab_size=30000,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            qkv_dim=512,
            embedding_dim=512,
            activation_fn=flax.nn.relu,
            weight_init=jax.nn.initializers.xavier_uniform(),
            embedding_init=jax.nn.initializers.normal(stddev=1.0)):
    """
    Args:
      documents: [batch_size, doc_length] integer Tensor.
      topic_params: [batch_size, num_topics, vocab_size] float Tensor.
      doc_topic_proportions: [batch_size, num_topics] float Tensor.
    """

    batch_size, doc_length = documents.shape

    embedding = Embedding.shared(
        num_embeddings=vocab_size, embedding_dim=embedding_dim,
        weight_init=embedding_init, name="word_embedding")

    # [batch_size, doc_length, embedding_dim]
    doc_word_embeddings = embedding(documents)

    # embedded_topics will be [batch_size, num_topics, embedding_dim]
    embedded_topics = flax.nn.Dense(
        topic_params,
        features=embedding_dim,
        kernel_init=weight_init)

    # concatenate the document and topic embeddings to produce
    # [batch_size, document_length + num_topics, embedding_dim]
    doc_topic_input = jnp.concatenate(
        [doc_word_embeddings, embedded_topics], axis=1)

    # feed the doc_topic_input into a transformer to produce the
    # [batch_size, num_topics, 1] document-topic proportions.
    out_doc_topics = transformer.EncoderDecoderTransformer(
        doc_topic_input,
        jnp.full([batch_size], doc_length + num_topics),
        jnp.full([batch_size], num_topics),
        targets=doc_topic_proportions[Ellipsis, jnp.newaxis],
        target_dim=1,
        max_input_length=doc_length + num_topics,
        max_target_length=num_topics,
        num_heads=num_heads,
        num_encoders=num_encoders,
        qkv_dim=qkv_dim,
        activation_fn=activation_fn,
        weight_init=weight_init)

    return jnp.squeeze(out_doc_topics, axis=2)

  @classmethod
  def loss(cls, params, key, documents, topic_params, doc_topic_proportions):
    pred_doc_topic_proportions = cls.call(
        params, documents, topic_params=topic_params,
        doc_topic_proportions=doc_topic_proportions)

    pred_doc_topic_logprobs = jax.nn.log_softmax(pred_doc_topic_proportions,
                                                 axis=-1)
    doc_topic_loss = util.categorical_kl(doc_topic_proportions,
                                         pred_doc_topic_logprobs)
    return doc_topic_loss
