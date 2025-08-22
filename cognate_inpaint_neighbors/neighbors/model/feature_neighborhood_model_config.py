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

"""Registration and configuration of FeatureNeighborhoodModel."""

import math as maths

# pylint: disable=unused-import
import feature_neighborhood_flags
import feature_neighborhood_input as fn
import feature_neighborhood_model as mdl
import feature_neighborhood_model_trans

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import py_utils
from lingvo.tasks.mt import base_config

FLAGS = tf.flags.FLAGS


class NeighborhoodBase(base_model_params.SingleTaskModelParams):
  """Base Model for training with neighbors."""

  _use_neighbors = False
  _share_embeddings = False
  _hidden_size = 256
  _embedding_dim = 256

  def Train(self):
    p = fn.FeatureNeighborhoodInput.Params()
    p.name = "feature_neighborhood_train_set"
    p.file_buffer_size = 2
    p.file_pattern = FLAGS.feature_neighborhood_train_path
    p.file_random_seed = 42
    p.num_samples = 11000000
    p.file_buffer_size = 10000
    p.num_batcher_threads = FLAGS.batch_size
    p.bucket_upper_bound = [1024]
    p.batch_size = FLAGS.batch_size
    p.bucket_batch_limit = [FLAGS.batch_size]
    if self._share_embeddings:
      output_symbol_path = FLAGS.input_symbols
    else:
      output_symbol_path = FLAGS.output_symbols
    p.feature_neighborhood_input, _, _, = (
        fn.FeatureNeighborhoodInput.ParameterizedConfigs(
            input_symbol_path=FLAGS.input_symbols,
            output_symbol_path=output_symbol_path,
            append_eos=FLAGS.append_eos,
            max_spelling_len=FLAGS.max_spelling_len,
            max_pronunciation_len=FLAGS.max_pronunciation_len,
            max_neighbors=FLAGS.max_neighbors,
            split_output_on_space=FLAGS.split_output_on_space))
    p.use_neighbors = self._use_neighbors
    return p

  def Dev(self):
    p = fn.FeatureNeighborhoodInput.Params()
    p.name = "feature_neighborhood_dev_set"
    p.file_pattern = FLAGS.feature_neighborhood_dev_path
    p.num_samples = 20000
    p.file_buffer_size = 256
    p.num_batcher_threads = 1
    p.bucket_upper_bound = [1024]
    p.batch_size = 100
    p.bucket_batch_limit = [100]
    if self._share_embeddings:
      output_symbol_path = FLAGS.input_symbols
    else:
      output_symbol_path = FLAGS.output_symbols
    p.feature_neighborhood_input, _, _, = fn.FeatureNeighborhoodInput.ParameterizedConfigs(
        input_symbol_path=FLAGS.input_symbols,
        output_symbol_path=output_symbol_path,
        append_eos=FLAGS.append_eos,
        max_spelling_len=FLAGS.max_spelling_len,
        max_pronunciation_len=FLAGS.max_pronunciation_len,
        max_neighbors=FLAGS.max_neighbors,
        split_output_on_space=FLAGS.split_output_on_space)

    p.use_neighbors = self._use_neighbors
    return p

  def Test(self):
    p = fn.FeatureNeighborhoodInput.Params()
    p.name = "feature_neighborhood_test_set"
    p.file_pattern = FLAGS.feature_neighborhood_test_path
    p.num_samples = 20000
    p.file_buffer_size = 256
    p.num_batcher_threads = 1
    p.bucket_upper_bound = [1024]
    p.batch_size = 100
    p.bucket_batch_limit = [100]
    if self._share_embeddings:
      output_symbol_path = FLAGS.input_symbols
    else:
      output_symbol_path = FLAGS.output_symbols
    p.feature_neighborhood_input, _, _, = (
        fn.FeatureNeighborhoodInput.ParameterizedConfigs(
            input_symbol_path=FLAGS.input_symbols,
            output_symbol_path=output_symbol_path,
            append_eos=FLAGS.append_eos,
            max_spelling_len=FLAGS.max_spelling_len,
            max_pronunciation_len=FLAGS.max_pronunciation_len,
            max_neighbors=FLAGS.max_neighbors,
            split_output_on_space=FLAGS.split_output_on_space))

    p.use_neighbors = self._use_neighbors
    return p

  def Golden(self):
    p = fn.FeatureNeighborhoodInput.Params()
    p.name = "feature_neighborhood_golden_set"
    p.file_pattern = ""  # Note: Needs a valid location here.
    p.num_samples = 2008
    p.file_buffer_size = 256
    p.num_batcher_threads = 1
    p.bucket_upper_bound = [1024]
    p.batch_size = 8
    p.bucket_batch_limit = [8]
    if self._share_embeddings:
      output_symbol_path = FLAGS.input_symbols
    else:
      output_symbol_path = FLAGS.output_symbols
    p.feature_neighborhood_input, _, _, = (
        fn.FeatureNeighborhoodInput.ParameterizedConfigs(
            input_symbol_path=FLAGS.input_symbols,
            output_symbol_path=output_symbol_path,
            append_eos=FLAGS.append_eos,
            max_spelling_len=FLAGS.max_spelling_len,
            max_pronunciation_len=FLAGS.max_pronunciation_len,
            max_neighbors=FLAGS.max_neighbors,
            split_output_on_space=FLAGS.split_output_on_space))

    p.use_neighbors = self._use_neighbors
    return p

  def Task(self):
    p = mdl.FeatureNeighborhoodModel.Params()
    p.name = "feature_neighborhood_with_neighbors"
    if self._share_embeddings:
      output_symbol_path = FLAGS.input_symbols
    else:
      output_symbol_path = FLAGS.output_symbols
    _, p.input_symbols, p.output_symbols = (
        fn.FeatureNeighborhoodInput.ParameterizedConfigs(
            input_symbol_path=FLAGS.input_symbols,
            output_symbol_path=output_symbol_path,
            append_eos=FLAGS.append_eos,
            max_spelling_len=FLAGS.max_spelling_len,
            max_pronunciation_len=FLAGS.max_pronunciation_len,
            max_neighbors=FLAGS.max_neighbors,
            split_output_on_space=FLAGS.split_output_on_space))
    p.input_vocab_size = p.input_symbols.num_symbols()
    p.output_vocab_size = p.output_symbols.num_symbols()
    p.train.learning_rate = FLAGS.learning_rate
    p.max_neighbors = FLAGS.max_neighbors
    p.max_pronunciation_len = FLAGS.max_pronunciation_len
    p.max_spelling_len = FLAGS.max_spelling_len
    p.start = p.output_symbols.find("<s>")
    p.use_neighbors = self._use_neighbors
    p.enc_units = self._hidden_size
    p.embedding_dim = self._embedding_dim
    return p


@model_registry.RegisterSingleTaskModel
class RNNWithoutNeighbors(NeighborhoodBase):
  """RNN Model for training without neighbors."""


@model_registry.RegisterSingleTaskModel
class RNNWithNeighbors(NeighborhoodBase):
  """RNN Model for training with neighbors.

  Number of params: ~4,187,300.
  """

  _use_neighbors = True


@model_registry.RegisterSingleTaskModel
class RNNLargeWithoutNeighbors(NeighborhoodBase):
  """Large RNN Model for training without neighbors."""

  _hidden_size = 512


@model_registry.RegisterSingleTaskModel
class RNNLargeWithNeighbors(NeighborhoodBase):
  """Large RNN Model for training with neighbors.

  Number of params: ~8,890,000.
  """

  _use_neighbors = True
  _hidden_size = 512


@model_registry.RegisterSingleTaskModel
class RNNWithNeighborsShareEmb(RNNWithNeighbors):
  """RNN Model for training with neighbors with shared embeddings."""

  _share_embeddings = True


class TransformerBase(NeighborhoodBase):
  """Base class for Transformer Model Params."""

  _num_layers = 4
  _num_heads = 8
  _residual_dropout_prob = 0.0
  _relu_dropout_prob = 0.0
  _input_dropout_prob = 0.0
  _atten_dropout_prob = 0.0
  _aux_dropout_prob = 0.0
  _label_smoothing_uncertainty = 0.0
  _use_neigh_id_emb = False
  _grid_step = 10
  _attention_type = "AVERAGE"
  _also_shuffle_neighbors = True

  def Task(self):
    p = feature_neighborhood_model_trans.FeatureNeighborhoodModelTrans.Params()
    if self._share_embeddings:
      output_symbol_path = FLAGS.input_symbols
    else:
      output_symbol_path = FLAGS.output_symbols
    _, p.input_symbols, p.output_symbols = (
        fn.FeatureNeighborhoodInput.ParameterizedConfigs(
            input_symbol_path=FLAGS.input_symbols,
            output_symbol_path=output_symbol_path,
            append_eos=FLAGS.append_eos,
            max_spelling_len=FLAGS.max_spelling_len,
            max_pronunciation_len=FLAGS.max_pronunciation_len,
            max_neighbors=FLAGS.max_neighbors))
    p.input_vocab_size = p.input_symbols.num_symbols()
    p.output_vocab_size = p.output_symbols.num_symbols()
    p.max_neighbors = FLAGS.max_neighbors
    p.max_pronunciation_len = FLAGS.max_pronunciation_len
    p.max_spelling_len = FLAGS.max_spelling_len
    p.start = p.output_symbols.find("<s>")
    p.share_embeddings = self._share_embeddings

    if self._share_embeddings:
      vocab_size = p.input_vocab_size
    else:
      vocab_size = p.output_vocab_size

    p = base_config.SetupTransformerParams(
        p,
        name="feature_neighborhood_with_neighbors",
        vocab_size=vocab_size,
        model_dim=p.embedding_dim,
        hidden_dim=p.enc_units,
        num_heads=self._num_heads,
        num_layers=self._num_layers,
        learning_rate=3.0,
        warmup_steps=40000,
        residual_dropout_prob=self._residual_dropout_prob,
        relu_dropout_prob=self._relu_dropout_prob,
        input_dropout_prob=self._input_dropout_prob,
        atten_dropout_prob=self._atten_dropout_prob,
        label_smoothing_uncertainty=self._label_smoothing_uncertainty)
    if not self._share_embeddings:
      p.encoder.token_emb.vocab_size = p.input_vocab_size
    p.eval.samples_per_summary = 20000
    # TODO(llion): Might need to change the output vocab size to one that can
    # be sharded to run efficiently on TPUs.
    p.decoder.softmax.num_shards = 1
    p.decoder.target_seq_len = p.max_pronunciation_len

    if py_utils.use_tpu():
      p.decoder.beam_search = model_helper.ChangeToBeamSearchTpuHelper(
          p.decoder.beam_search)

    if FLAGS.neigh_use_tpu:
      for pp in [p.encoder, p.decoder]:
        pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)
      p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    p.use_neighbors = self._use_neighbors
    if self._use_neighbors:
      p.spell_encoder = base_config.SetupTransformerEncoder(
          vocab_size=p.input_vocab_size,
          model_dim=p.embedding_dim,
          hidden_dim=p.enc_units,
          num_heads=self._num_heads,
          num_layers=self._num_layers,
          residual_dropout_prob=self._residual_dropout_prob,
          relu_dropout_prob=self._relu_dropout_prob,
          input_dropout_prob=self._input_dropout_prob,
          atten_dropout_prob=self._atten_dropout_prob)
      if self._attention_type != "CONCATAVE":
        p.pron_encoder = base_config.SetupTransformerEncoder(
            vocab_size=p.output_vocab_size,
            model_dim=p.embedding_dim,
            hidden_dim=p.enc_units,
            num_heads=self._num_heads,
            num_layers=self._num_layers,
            residual_dropout_prob=self._residual_dropout_prob,
            relu_dropout_prob=self._relu_dropout_prob,
            input_dropout_prob=self._input_dropout_prob,
            atten_dropout_prob=self._atten_dropout_prob)
      else:
        if not self._share_embeddings:
          raise ValueError("Must share embeddings to concat spelling and pron.")
      if FLAGS.neigh_use_tpu:
        for pp in [p.spell_encoder, p.pron_encoder]:
          if pp:
            pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)

    p.also_shuffle_neighbors = self._also_shuffle_neighbors
    if self._use_neigh_id_emb:
      assert self._use_neighbors
      p.use_neigh_id_emb = True
      if self._attention_type == "CONCAT":
        neigh_id_emb = layers.EmbeddingLayer.Params().Set(
            vocab_size=FLAGS.max_neighbors + 1,  # +1 to include the main input
            embedding_dim=p.embedding_dim,
            max_num_shards=1,
            params_init=py_utils.WeightInit.Gaussian(
                1.0 / maths.sqrt(p.embedding_dim)),
            scale_sqrt_depth=True)
        p.encoder.task_emb = neigh_id_emb
      elif self._attention_type == "AVERAGE":
        neigh_id_emb = layers.EmbeddingLayer.Params().Set(
            vocab_size=FLAGS.max_neighbors,
            embedding_dim=p.embedding_dim,
            max_num_shards=1,
            params_init=py_utils.WeightInit.Gaussian(
                1.0 / maths.sqrt(p.embedding_dim)),
            scale_sqrt_depth=True)
        p.spell_encoder.task_emb = neigh_id_emb
        p.pron_encoder.task_emb = neigh_id_emb

    p.neigh_att_type = self._attention_type
    p.aux_dropout_prob = self._aux_dropout_prob

    return p


@model_registry.RegisterSingleTaskModel
class TransformerWithoutNeighbors(TransformerBase):
  """Transformer Model for training without neighbors."""


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighbors(TransformerBase):
  """Transformer Model for training with neighbors.

  export NAME="share_emb"
  export CKPT_LIMIT=200000

  Correct: 123014  /  138320
  Accuracy = 0.8893435511856564 (Error Rate = 0.11065644881434356 )

  Number of params: ~8,180,000.
  """

  _use_neighbors = True


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsShareEmb(TransformerWithNeighbors):
  """Transformer Model for training with neighbors with shared embeddings."""

  _share_embeddings = True


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsDropout(TransformerWithNeighbors):
  """Transformer Model for training with neighbors with dropout."""

  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsDropoutMore(TransformerWithNeighbors):
  """Transformer Model for training with neighbors with more dropout."""

  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithoutNeighborsDropout(TransformerWithoutNeighbors):
  """Transformer Model for training without neighbors with dropout."""

  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithoutNeighborsMoreDropout(TransformerWithoutNeighbors):
  """Transformer Model for training without neighbors with dropout."""

  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsNeighID(TransformerBase):
  """Transformer Model for training with neighbor IDs."""

  _use_neighbors = True
  _use_neigh_id_emb = True
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsNeighIDDropout(TransformerBase):
  """Transformer Model for training with neighbor IDs."""

  _use_neighbors = True
  _use_neigh_id_emb = True
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsConcatAll(TransformerBase):
  """Transformer Model for training with neighbor info all concatenated."""

  _use_neighbors = True
  _use_neigh_id_emb = True
  _attention_type = "CONCAT"
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsConcatAve(TransformerBase):
  """Transformer Model for training with neighbor info concatenated then ave."""

  _use_neighbors = True
  _attention_type = "CONCATAVE"
  _share_embeddings = True
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsConcatAllHigherDropout(TransformerBase):
  """Transformer Model for training with neighbor info all concatenated."""

  _use_neighbors = True
  _use_neigh_id_emb = True
  _attention_type = "CONCAT"
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsCAHDNoShuffle(TransformerBase):
  """Transformer Model for training with neighbor info all concatenated."""

  _use_neighbors = True
  _use_neigh_id_emb = True
  _attention_type = "CONCAT"
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1
  _also_shuffle_neighbors = False


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsMemoryAtt(TransformerBase):
  """Transformer Model for training with neighbor memory attention."""

  _use_neighbors = True
  _attention_type = "MEMORY"
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _input_dropout_prob = 0.1
  _atten_dropout_prob = 0.1
  _label_smoothing_uncertainty = 0.1


@model_registry.RegisterSingleTaskModel
class TransformerWithNeighborsTiny(TransformerBase):
  """A tiny helpful neighbor model for tiny datasets."""

  _use_neighbors = True
  _residual_dropout_prob = 0.1
  _relu_dropout_prob = 0.1
  _atten_dropout_prob = 0.1
  _label_smoothing_uncertainty = 0.2
  _use_neigh_id_emb = True
  _attention_type = "CONCAT"
  _also_shuffle_neighbors = True
  _num_layers = 3
  _num_heads = 2
  _hidden_size = 32
  _embedding_dim = 32
