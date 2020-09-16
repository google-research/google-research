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

"""A custom Tensor2tensor problem for a text-to-text problem on the CFQ dataset.

This file was created using the guidelines at
https://tensorflow.github.io/tensor2tensor/new_problem.html.
"""

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.problem import DatasetSplit
from tensor2tensor.layers import common_hparams
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class CFQ(text_problems.Text2TextProblem):
  """A Tensor2tensor problem for a text-to-text problem on the CFQ dataset."""

  @property
  def is_generate_per_split(self):
    # We already have split training, dev, and test.
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.TEST,
        'shards': 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, split):
    split_to_name = {
        DatasetSplit.EVAL: 'dev',
        DatasetSplit.TRAIN: 'train',
        DatasetSplit.TEST: 'test'
    }
    split_name = split_to_name[split]
    folder_name = os.path.join(data_dir, split_name)
    encode_name = os.path.join(folder_name, '%s_encode.txt' % split_name)
    decode_name = os.path.join(folder_name, '%s_decode.txt' % split_name)

    with tf.gfile.Open(encode_name) as encode_f, tf.gfile.Open(
        decode_name) as decode_f:
      for x, y in zip(encode_f, decode_f):
        yield {
            'inputs': x.strip(),
            'targets': y.strip(),
        }

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def oov_token(self):
    return '<OOV>'


@registry.register_hparams
def cfq_transformer_base():
  """Transformer hyperparameters inherited from transformer_base_single_gpu."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = 'layer'
  hparams.hidden_size = 512
  hparams.num_hidden_layers = 6
  hparams.batch_size = 1024
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate = 0.2
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_decay_scheme = 'noam'
  hparams.learning_rate_schedule = 'constant*linear_warmup*rsqrt_decay'
  hparams.learning_rate_warmup_steps = 16000
  hparams.initializer_gain = 1.0
  hparams.initializer = 'uniform_unit_scaling'
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.997
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 16

  # Add new ones like this.
  hparams.add_hparam('filter_size', 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam('num_encoder_layers', 0)
  hparams.add_hparam('num_decoder_layers', 0)
  # Attention-related flags.
  hparams.add_hparam('num_heads', 8)
  hparams.add_hparam('attention_key_channels', 0)
  hparams.add_hparam('attention_value_channels', 0)
  hparams.add_hparam('ffn_layer', 'dense_relu_dense')
  hparams.add_hparam('parameter_attention_key_channels', 0)
  hparams.add_hparam('parameter_attention_value_channels', 0)
  # All hyperparameters ending in 'dropout' are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam('attention_dropout', 0.1)
  hparams.add_hparam('attention_dropout_broadcast_dims', '')
  hparams.add_hparam('relu_dropout', 0.1)
  hparams.add_hparam('relu_dropout_broadcast_dims', '')
  hparams.add_hparam('pos', 'timing')  # timing, none
  hparams.add_hparam('nbr_decoder_problems', 1)
  hparams.add_hparam('proximity_bias', False)
  hparams.add_hparam('causal_decoder_self_attention', True)
  hparams.add_hparam('use_pad_remover', True)
  hparams.add_hparam('self_attention_type', 'dot_product')
  hparams.add_hparam('conv_first_kernel', 3)
  hparams.add_hparam('attention_variables_3d', False)
  hparams.add_hparam('use_target_space_embedding', True)
  # These parameters are only used when ffn_layer=='local_moe_tpu'
  hparams.add_hparam('moe_overhead_train', 1.0)
  hparams.add_hparam('moe_overhead_eval', 2.0)
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-3
  # If specified, use this value instead of problem name in metrics.py.
  # This is useful for programs that can automatically compare experiments side
  #   by side based on the same metric names.
  hparams.add_hparam('overload_eval_metric_name', '')
  # For making a transformer encoder unidirectional by using masked
  # attention.
  hparams.add_hparam('unidirectional_encoder', False)
  # For hard attention.
  hparams.add_hparam('hard_attention_k', 0)
  hparams.add_hparam('gumbel_noise_weight', 0.0)

  hparams.layer_preprocess_sequence = 'n'
  hparams.layer_postprocess_sequence = 'da'
  hparams.layer_prepostprocess_dropout = 0.1

  return hparams


@registry.register_hparams
def cfq_transformer():
  """Transformer hyperparameters tuned with Vizier for CFQ."""
  hparams = cfq_transformer_base()
  # The model is really small compared to a default one (512x6), but that is
  # sufficient for CFQ and makes the learning much faster.
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.num_heads = 16
  # Note: batch size is the number of tokens in one batch (not sentences).
  hparams.batch_size = 4096
  hparams.learning_rate_schedule = 'constant*linear_warmup*rsqrt_decay'
  hparams.learning_rate_constant = 0.08
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def cfq_lstm_attention_multi():
  """LSTM+attention hyperparameters tuned for CFQ."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.initializer = 'uniform_unit_scaling'
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.add_hparam('attention_layer_size', hparams.hidden_size)
  hparams.add_hparam('output_attention', True)
  hparams.add_hparam('num_heads', 1)
  hparams.add_hparam('attention_mechanism', 'bahdanau')
  hparams.num_heads = 4
  # The remaining hyperparameters were determined as described in the paper:
  hparams.batch_size = 2048
  hparams.dropout = 0.4
  hparams.hidden_size = 512
  hparams.learning_rate = 0.03
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def cfq_universal_transformer():
  """Universal Transformer hyperparameters tuned for CFQ."""
  hparams = cfq_transformer_base()
  hparams = universal_transformer.update_hparams_for_universal_transformer(
      hparams)
  # The following hyperparameters are taken from universal_transformer_tiny:
  hparams.filter_size = 512
  hparams.learning_rate_warmup_steps = 8000
  # The remaining hyperparameters were determined as described in the paper:
  hparams.batch_size = 2048
  hparams.gate_ffn_layer = 'dense'
  hparams.hidden_size = 256
  hparams.learning_rate_constant = 0.14
  hparams.learning_rate_schedule = 'constant*linear_warmup*rsqrt_decay'
  hparams.num_heads = 4
  hparams.num_hidden_layers = 6
  hparams.num_rec_steps = 4
  hparams.recurrence_type = 'basic'
  return hparams
