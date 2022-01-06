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

"""Tests for wmt_mlperf.models."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as onp

from aqt.jax import get_bounds
from aqt.jax import hlo_utils
from aqt.jax import primitives
from aqt.jax import quant_config
from aqt.jax import test_utils
from aqt.jax.quantization import QuantType
from aqt.jax.wmt_mlperf import models
from aqt.jax.wmt_mlperf import training_hparams_generator_lib


class ModelsTest(parameterized.TestCase):

  def setUp(self):
    super(ModelsTest, self).setUp()
    self.input_shape = (1, 1)
    self.target_shape = (1, 1)
    self.inputs = jnp.ones(self.input_shape, dtype=jnp.float32)
    self.target = jnp.ones(self.target_shape, dtype=jnp.float32)
    self.key = jax.random.PRNGKey(0)
    self.transformer_small_kwargs = {
        'vocab_size': 1,
        'output_vocab_size': 1,
        'max_len': 1,
        'train': False,
    }
    self.transformer_full_kwargs = {
        'vocab_size': 4,
        'output_vocab_size': 4,
        'max_len': 2,
        'train': False,
    }

  def init_model(self, transformer_kwargs):
    model = models.Transformer(
        use_bfloat16=False,
        quant_context=quant_config.QuantContext(
            collect_acts_stats=False, update_bounds=False),
        dropout_rate=.1,
        attention_dropout_rate=.1,
        should_decode=False,
        **transformer_kwargs)
    state = model.init(self.key, jnp.zeros(self.input_shape, jnp.float32),
                       jnp.zeros(self.target_shape, jnp.float32))
    return model, state

  @parameterized.named_parameters(
      dict(testcase_name='test_mlp_weight_quant_8bit', mlp_weight_prec=8),
      dict(testcase_name='test_mlp_weight_quant_4bit', mlp_weight_prec=4),
      dict(testcase_name='test_mlp_weight_quant_1bit', mlp_weight_prec=1),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_mlp_weight_quant(self, floor_with_gradient, round_with_gradient,
                            mlp_weight_prec):
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=mlp_weight_prec,
        embedding_weight_prec=None,
        attention_weight_prec=None,
        mlp_pos_inputs_prec=None,
        mlp_pos_inputs_hyper=None,
        mlp_signed_inputs_prec=None,
        mlp_signed_inputs_hyper=None,
        attention_kqv_inputs_prec=None,
        attention_kqv_inputs_hyper=None,
        attention_out_inputs_prec=None,
        attention_out_inputs_hyper=None,
        logits_inputs_prec=None,
        logits_inputs_hyper=None,
        logits_via_embeddings=True,
        attention_act_q_inputs_prec=None,
        attention_act_q_inputs_hyper=None,
        attention_act_k_inputs_prec=None,
        attention_act_k_inputs_hyper=None,
        attention_act_probs_inputs_prec=None,
        attention_act_v_inputs_prec=None,
        attention_act_v_inputs_hyper=None,
        num_layers=1,
        emb_dim=1,
        num_heads=1,
        qkv_dim=1,
        mlp_dim=1,
        quant_type=QuantType.fake_quant)
    transformer_kwargs = self.transformer_small_kwargs
    transformer_kwargs['hparams'] = hparams
    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    model, init_state = self.init_model(transformer_kwargs)
    # there are 2 MLP blocks in this model, with 2 quant ops each, so both clip
    # and round should be called 4 times each.
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 4)
    floor_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    output = model.apply(init_state, self.inputs, self.target)
    self.assertEqual(output.shape, (1, 1))
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 4)
    floor_with_gradient.assert_not_called()

  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_without_mlp_weight_quant(self, floor_with_gradient,
                                    round_with_gradient):
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=None,
        embedding_weight_prec=None,
        attention_weight_prec=None,
        mlp_pos_inputs_prec=None,
        mlp_pos_inputs_hyper=None,
        mlp_signed_inputs_prec=None,
        mlp_signed_inputs_hyper=None,
        attention_kqv_inputs_prec=None,
        attention_kqv_inputs_hyper=None,
        attention_out_inputs_prec=None,
        attention_out_inputs_hyper=None,
        logits_inputs_prec=None,
        logits_inputs_hyper=None,
        logits_via_embeddings=True,
        attention_act_q_inputs_prec=None,
        attention_act_q_inputs_hyper=None,
        attention_act_k_inputs_prec=None,
        attention_act_k_inputs_hyper=None,
        attention_act_probs_inputs_prec=None,
        attention_act_v_inputs_prec=None,
        attention_act_v_inputs_hyper=None,
        num_layers=1,
        emb_dim=1,
        num_heads=1,
        qkv_dim=1,
        mlp_dim=1,
        quant_type=QuantType.fake_quant)
    transformer_kwargs = self.transformer_small_kwargs
    transformer_kwargs['hparams'] = hparams
    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    model, init_state = self.init_model(transformer_kwargs)
    model.apply(init_state, self.inputs, self.target)
    round_with_gradient.assert_not_called()
    floor_with_gradient.assert_not_called()

  def _num_mlp_floors(self, weight_quant, pos_input_quant, neg_input_quant):
    # There are 2 MLP blocks per layer (1 encoder, 1 decoder) and 2 weight quant
    # ops per MLP block, so 4 in total per layer.
    mlp_floors_per_layer = 4 if weight_quant else 0

    # There are 2 MLP blocks per layer (1 encoder, 1 decoder) and 1 input quant
    # op per unsigned MLP block, so 2 in total per layer.
    if pos_input_quant:
      mlp_floors_per_layer = mlp_floors_per_layer + 2

    # There are 2 MLP blocks per layer (1 encoder, 1 decoder) and 1 input quant
    # op per signed MLP block, so 2 in total per layer.
    if neg_input_quant:
      mlp_floors_per_layer = mlp_floors_per_layer + 2

    return mlp_floors_per_layer

  def _num_embedding_floors(self, weight_quant, act_quant):
    # 3 embedding layers in the whole model
    embedding_floors = 3 if weight_quant else 0

    # logits activation quantization
    if act_quant:
      embedding_floors = embedding_floors + 1

    return embedding_floors

  def _num_attention_floors(self, weight_quant, kqv_input_quant,
                            out_input_quant, act_q_input_quant,
                            act_k_input_quant, act_qk_input_quant,
                            act_v_input_quant):
    # 3 attention blocks per layer (1 on encoder, 2 on decoder), each
    # attention block has 4 weight quant ops, so 12 in total per layer.
    attention_floors_per_layer = 12 if weight_quant else 0

    # 3 attention blocks per layer (1 on encoder, 2 on decoder), each
    # attention block has 3 kqv activation quant ops, so 9 in total per layer.
    if kqv_input_quant:
      attention_floors_per_layer = attention_floors_per_layer + 9

    # 3 attention blocks per layer (1 on encoder, 2 on decoder), each attention
    # block has 1 dense out activation quant op, so 3 in total per layer.
    if out_input_quant:
      attention_floors_per_layer = attention_floors_per_layer + 3

    # 3 attention blocks per layer (1 on encoder, 2 on decoder), each attention
    # block has 1 act*act activation quant op for each act per layer.
    for act_quant in [
        act_q_input_quant, act_k_input_quant, act_qk_input_quant,
        act_v_input_quant
    ]:
      if act_quant:
        attention_floors_per_layer = attention_floors_per_layer + 3

    return attention_floors_per_layer

  @parameterized.named_parameters(
      dict(
          testcase_name='test_2layers_no_quant',
          num_layers=2,
          mlp_weight_prec=None,
          mlp_pos_inputs_prec=None,
          mlp_signed_inputs_prec=None,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=None,
          embedding_weight_prec=None,
          attention_weight_prec=None,
          logits_inputs_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
      ),
      dict(
          testcase_name='test_3layers_full_quant',
          num_layers=3,
          mlp_weight_prec=4,
          mlp_pos_inputs_prec=8,
          mlp_signed_inputs_prec=8,
          attention_kqv_inputs_prec=2,
          attention_out_inputs_prec=4,
          embedding_weight_prec=4,
          attention_weight_prec=4,
          logits_inputs_prec=8,
          attention_act_q_inputs_prec=4,
          attention_act_k_inputs_prec=8,
          attention_act_probs_inputs_prec=8,
          attention_act_v_inputs_prec=4,
      ),
  )
  def test_number_of_floor_ops(
      self, num_layers, mlp_weight_prec, mlp_pos_inputs_prec,
      mlp_signed_inputs_prec, attention_kqv_inputs_prec,
      attention_out_inputs_prec, embedding_weight_prec, attention_weight_prec,
      logits_inputs_prec, attention_act_q_inputs_prec,
      attention_act_k_inputs_prec, attention_act_probs_inputs_prec,
      attention_act_v_inputs_prec):
    # Counts number of floor ops as a proxy for quantization ops.
    act_fixed_clip_bound = 3.0
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=mlp_weight_prec,
        embedding_weight_prec=embedding_weight_prec,
        attention_weight_prec=attention_weight_prec,
        mlp_pos_inputs_prec=mlp_pos_inputs_prec,
        mlp_pos_inputs_hyper=act_fixed_clip_bound,
        mlp_signed_inputs_prec=mlp_signed_inputs_prec,
        mlp_signed_inputs_hyper=act_fixed_clip_bound,
        attention_kqv_inputs_prec=attention_kqv_inputs_prec,
        attention_kqv_inputs_hyper=act_fixed_clip_bound,
        attention_out_inputs_prec=attention_out_inputs_prec,
        attention_out_inputs_hyper=act_fixed_clip_bound,
        logits_inputs_prec=logits_inputs_prec,
        logits_inputs_hyper=act_fixed_clip_bound,
        logits_via_embeddings=True,
        attention_act_q_inputs_prec=attention_act_q_inputs_prec,
        attention_act_q_inputs_hyper=act_fixed_clip_bound,
        attention_act_k_inputs_prec=attention_act_k_inputs_prec,
        attention_act_k_inputs_hyper=act_fixed_clip_bound,
        attention_act_probs_inputs_prec=attention_act_probs_inputs_prec,
        attention_act_v_inputs_prec=attention_act_v_inputs_prec,
        attention_act_v_inputs_hyper=act_fixed_clip_bound,
        num_layers=num_layers,
        emb_dim=5,
        num_heads=8,
        qkv_dim=8,
        mlp_dim=7,
        quant_type=QuantType.fake_quant)

    transformer_kwargs = self.transformer_full_kwargs
    transformer_kwargs['hparams'] = hparams
    input_shape = (2, 4)
    target_shape = input_shape
    model, init_state = self.init_model(transformer_kwargs)
    hlo_proto = hlo_utils.load_hlo_proto_from_model(model, init_state,
                                                    [input_shape, target_shape])
    floor_count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, r'floor')

    mlp_floors_per_layer = self._num_mlp_floors(
        (mlp_weight_prec is not None), (mlp_pos_inputs_prec is not None),
        (mlp_signed_inputs_prec is not None))

    attention_floors_per_layer = self._num_attention_floors(
        (attention_weight_prec is not None),
        (attention_kqv_inputs_prec is not None),
        (attention_out_inputs_prec is not None),
        (attention_act_q_inputs_prec is not None),
        (attention_act_k_inputs_prec is not None),
        (attention_act_probs_inputs_prec is not None),
        (attention_act_v_inputs_prec is not None))

    embedding_floors = self._num_embedding_floors(
        (embedding_weight_prec is not None), (logits_inputs_prec is not None))

    expected_floor_count = num_layers * (
        mlp_floors_per_layer + attention_floors_per_layer) + embedding_floors
    self.assertEqual(floor_count, expected_floor_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='test_2layers_att8bit_weight_quant',
          num_layers=2,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=None,
          attention_weight_prec=8,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_3layers_att8bit_kqv_inputs_quant',
          num_layers=3,
          attention_kqv_inputs_prec=8,
          attention_out_inputs_prec=None,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_3layers_att8bit_act_q_inputs_quant',
          num_layers=3,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=None,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=8,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_att8bit_act_k_inputs_quant',
          num_layers=2,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=None,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=8,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_3layers_att8bit_act_qk_inputs_quant',
          num_layers=3,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=None,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=4,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_att8bit_act_v_inputs_quant',
          num_layers=2,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=None,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=8,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_3layers_att8bit_kqv_inputs_auto_quant',
          num_layers=3,
          attention_kqv_inputs_prec=8,
          attention_out_inputs_prec=None,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=False,
      ),
      dict(
          testcase_name='test_2layers_att8bit_out_inputs_quant',
          num_layers=2,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=8,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_att8bit_out_inputs_auto_quant',
          num_layers=2,
          attention_kqv_inputs_prec=None,
          attention_out_inputs_prec=8,
          attention_weight_prec=None,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=False,
      ),
      dict(
          testcase_name='test_2layers_att_weight_kqv_out_quant',
          num_layers=2,
          attention_kqv_inputs_prec=8,
          attention_out_inputs_prec=4,
          attention_weight_prec=2,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_att_weight_kqv_out_act_quant',
          num_layers=2,
          attention_kqv_inputs_prec=8,
          attention_out_inputs_prec=4,
          attention_weight_prec=2,
          attention_act_q_inputs_prec=4,
          attention_act_k_inputs_prec=8,
          attention_act_probs_inputs_prec=4,
          attention_act_v_inputs_prec=2,
          inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_att_weight_kqv_out_auto_quant',
          num_layers=2,
          attention_kqv_inputs_prec=8,
          attention_out_inputs_prec=4,
          attention_weight_prec=2,
          attention_act_q_inputs_prec=None,
          attention_act_k_inputs_prec=None,
          attention_act_probs_inputs_prec=None,
          attention_act_v_inputs_prec=None,
          inputs_hyper_is_float=False,
      ),
      dict(
          testcase_name='test_3layers_att_weight_kqv_out_act_auto_quant',
          num_layers=3,
          attention_kqv_inputs_prec=8,
          attention_out_inputs_prec=4,
          attention_weight_prec=2,
          attention_act_q_inputs_prec=8,
          attention_act_k_inputs_prec=4,
          attention_act_probs_inputs_prec=4,
          attention_act_v_inputs_prec=2,
          inputs_hyper_is_float=False,
      ),
  )
  def test_number_of_floor_ops_attention(
      self,
      num_layers,
      attention_kqv_inputs_prec,
      attention_out_inputs_prec,
      attention_weight_prec,
      attention_act_q_inputs_prec,
      attention_act_k_inputs_prec,
      attention_act_probs_inputs_prec,
      attention_act_v_inputs_prec,
      inputs_hyper_is_float,
  ):
    # Counts number of floor ops as a proxy for quantization ops.
    if inputs_hyper_is_float:
      inputs_hyper = 6.0
    else:
      inputs_hyper = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=None,
        embedding_weight_prec=None,
        attention_weight_prec=attention_weight_prec,
        mlp_pos_inputs_prec=None,
        mlp_pos_inputs_hyper=None,
        mlp_signed_inputs_prec=None,
        mlp_signed_inputs_hyper=None,
        attention_kqv_inputs_prec=attention_kqv_inputs_prec,
        attention_kqv_inputs_hyper=inputs_hyper,
        attention_out_inputs_prec=attention_out_inputs_prec,
        attention_out_inputs_hyper=inputs_hyper,
        logits_inputs_prec=None,
        logits_inputs_hyper=None,
        logits_via_embeddings=True,
        attention_act_q_inputs_prec=attention_act_q_inputs_prec,
        attention_act_q_inputs_hyper=inputs_hyper,
        attention_act_k_inputs_prec=attention_act_k_inputs_prec,
        attention_act_k_inputs_hyper=inputs_hyper,
        attention_act_probs_inputs_prec=attention_act_probs_inputs_prec,
        attention_act_v_inputs_prec=attention_act_v_inputs_prec,
        attention_act_v_inputs_hyper=inputs_hyper,
        num_layers=num_layers,
        emb_dim=5,
        num_heads=8,
        qkv_dim=8,
        mlp_dim=7,
        quant_type=QuantType.fake_quant)

    transformer_kwargs = self.transformer_full_kwargs
    transformer_kwargs['hparams'] = hparams
    input_shape = (2, 4)
    target_shape = input_shape
    model, init_state = self.init_model(transformer_kwargs)
    hlo_proto = hlo_utils.load_hlo_proto_from_model(model, init_state,
                                                    [input_shape, target_shape])
    floor_count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, r'floor')

    attention_floors_per_layer = self._num_attention_floors(
        (attention_weight_prec is not None),
        (attention_kqv_inputs_prec is not None),
        (attention_out_inputs_prec is not None),
        (attention_act_q_inputs_prec is not None),
        (attention_act_k_inputs_prec is not None),
        (attention_act_probs_inputs_prec is not None),
        (attention_act_v_inputs_prec is not None))

    expected_floor_count = num_layers * attention_floors_per_layer
    self.assertEqual(floor_count, expected_floor_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='test_3layers_mlp8bit_weight_quant',
          num_layers=3,
          mlp_weight_prec=8,
          mlp_pos_inputs_prec=None,
          mlp_pos_inputs_hyper_is_float=True,
          mlp_signed_inputs_prec=None,
          mlp_signed_inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_pos_inputs_quant',
          num_layers=2,
          mlp_weight_prec=None,
          mlp_pos_inputs_prec=8,
          mlp_pos_inputs_hyper_is_float=True,
          mlp_signed_inputs_prec=None,
          mlp_signed_inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_pos_inputs_auto_quant',
          num_layers=2,
          mlp_weight_prec=None,
          mlp_pos_inputs_prec=8,
          mlp_pos_inputs_hyper_is_float=False,
          mlp_signed_inputs_prec=None,
          mlp_signed_inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_pos_inputs_weights_quant',
          num_layers=2,
          mlp_weight_prec=8,
          mlp_pos_inputs_prec=8,
          mlp_pos_inputs_hyper_is_float=True,
          mlp_signed_inputs_prec=None,
          mlp_signed_inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_neg_inputs_quant',
          num_layers=2,
          mlp_weight_prec=None,
          mlp_pos_inputs_prec=None,
          mlp_pos_inputs_hyper_is_float=True,
          mlp_signed_inputs_prec=8,
          mlp_signed_inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_neg_inputs_auto_quant',
          num_layers=2,
          mlp_weight_prec=None,
          mlp_pos_inputs_prec=None,
          mlp_pos_inputs_hyper_is_float=True,
          mlp_signed_inputs_prec=8,
          mlp_signed_inputs_hyper_is_float=False,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_all_inputs_weights_quant',
          num_layers=2,
          mlp_weight_prec=8,
          mlp_pos_inputs_prec=8,
          mlp_pos_inputs_hyper_is_float=True,
          mlp_signed_inputs_prec=8,
          mlp_signed_inputs_hyper_is_float=True,
      ),
      dict(
          testcase_name='test_2layers_mlp8bit_all_inputs_weights_auto_quant',
          num_layers=2,
          mlp_weight_prec=8,
          mlp_pos_inputs_prec=8,
          mlp_pos_inputs_hyper_is_float=False,
          mlp_signed_inputs_prec=8,
          mlp_signed_inputs_hyper_is_float=False,
      ),
  )
  def test_number_of_floor_ops_mlp(self, num_layers, mlp_weight_prec,
                                   mlp_pos_inputs_prec,
                                   mlp_pos_inputs_hyper_is_float,
                                   mlp_signed_inputs_prec,
                                   mlp_signed_inputs_hyper_is_float):
    # Counts number of floor ops as a proxy for quantization ops.
    if mlp_pos_inputs_hyper_is_float:
      mlp_pos_inputs_hyper = 6.0
    else:
      mlp_pos_inputs_hyper = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    if mlp_signed_inputs_hyper_is_float:
      mlp_pos_inputs_hyper = 6.0
    else:
      mlp_pos_inputs_hyper = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=mlp_weight_prec,
        embedding_weight_prec=None,
        attention_weight_prec=None,
        mlp_pos_inputs_prec=mlp_pos_inputs_prec,
        mlp_pos_inputs_hyper=mlp_pos_inputs_hyper,
        mlp_signed_inputs_prec=mlp_signed_inputs_prec,
        mlp_signed_inputs_hyper=mlp_pos_inputs_hyper,
        attention_kqv_inputs_prec=None,
        attention_kqv_inputs_hyper=None,
        attention_out_inputs_prec=None,
        attention_out_inputs_hyper=None,
        logits_inputs_prec=None,
        logits_inputs_hyper=None,
        logits_via_embeddings=True,
        attention_act_q_inputs_prec=None,
        attention_act_q_inputs_hyper=None,
        attention_act_k_inputs_prec=None,
        attention_act_k_inputs_hyper=None,
        attention_act_probs_inputs_prec=None,
        attention_act_v_inputs_prec=None,
        attention_act_v_inputs_hyper=None,
        num_layers=num_layers,
        emb_dim=5,
        num_heads=8,
        qkv_dim=8,
        mlp_dim=7,
        quant_type=QuantType.fake_quant)

    transformer_kwargs = self.transformer_full_kwargs
    transformer_kwargs['hparams'] = hparams
    input_shape = (2, 4)
    target_shape = input_shape
    model, init_state = self.init_model(transformer_kwargs)
    hlo_proto = hlo_utils.load_hlo_proto_from_model(model, init_state,
                                                    [input_shape, target_shape])
    floor_count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, r'floor')

    mlp_floors_per_layer = self._num_mlp_floors(
        (mlp_weight_prec is not None), (mlp_pos_inputs_prec is not None),
        (mlp_signed_inputs_prec is not None))

    expected_floor_count = num_layers * mlp_floors_per_layer
    self.assertEqual(floor_count, expected_floor_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='test_3layers_embedding8bit_weight_quant',
          num_layers=3,
          embedding_weight_prec=8,
          logits_inputs_prec=None,
          logits_inputs_hyper_is_float=True,
          logits_via_embeddings=True,
      ),
      dict(
          testcase_name='test_2layers_embedding8bit_inputs_auto_quant',
          num_layers=2,
          embedding_weight_prec=None,
          logits_inputs_prec=8,
          logits_inputs_hyper_is_float=False,
          logits_via_embeddings=True,
      ),
      dict(
          testcase_name='test_2layers_embedding8bit_inputs_weights_quant_fixed',
          num_layers=2,
          embedding_weight_prec=8,
          logits_inputs_prec=8,
          logits_inputs_hyper_is_float=True,
          logits_via_embeddings=True,
      ),
      dict(
          testcase_name='test_2layers_embedding8bit_inputs_weights_auto_quant',
          num_layers=2,
          embedding_weight_prec=8,
          logits_inputs_prec=8,
          logits_inputs_hyper_is_float=False,
          logits_via_embeddings=True,
      ),
      dict(
          testcase_name='test_2layers_embedding8bit_without_logit_sharing',
          num_layers=2,
          embedding_weight_prec=8,
          logits_inputs_prec=8,
          logits_inputs_hyper_is_float=False,
          logits_via_embeddings=False,
      ),
  )
  def test_number_of_floor_ops_embedding(self, num_layers,
                                         embedding_weight_prec,
                                         logits_inputs_prec,
                                         logits_inputs_hyper_is_float,
                                         logits_via_embeddings):
    # Counts number of floor ops as a proxy for quantization ops.
    if logits_inputs_hyper_is_float:
      logits_inputs_hyper = 6.0
    else:
      logits_inputs_hyper = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)

    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=None,
        embedding_weight_prec=embedding_weight_prec,
        attention_weight_prec=None,
        mlp_pos_inputs_prec=None,
        mlp_pos_inputs_hyper=None,
        mlp_signed_inputs_prec=None,
        mlp_signed_inputs_hyper=None,
        attention_kqv_inputs_prec=None,
        attention_kqv_inputs_hyper=None,
        attention_out_inputs_prec=None,
        attention_out_inputs_hyper=None,
        logits_inputs_prec=logits_inputs_prec,
        logits_inputs_hyper=logits_inputs_hyper,
        logits_via_embeddings=logits_via_embeddings,
        attention_act_q_inputs_prec=None,
        attention_act_q_inputs_hyper=None,
        attention_act_k_inputs_prec=None,
        attention_act_k_inputs_hyper=None,
        attention_act_probs_inputs_prec=None,
        attention_act_v_inputs_prec=None,
        attention_act_v_inputs_hyper=None,
        num_layers=num_layers,
        emb_dim=5,
        num_heads=8,
        qkv_dim=8,
        mlp_dim=7,
        quant_type=QuantType.fake_quant)

    transformer_kwargs = self.transformer_full_kwargs
    transformer_kwargs['hparams'] = hparams
    input_shape = (2, 4)
    target_shape = input_shape
    model, init_state = self.init_model(transformer_kwargs)
    hlo_proto = hlo_utils.load_hlo_proto_from_model(model, init_state,
                                                    [input_shape, target_shape])
    floor_count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, r'floor')

    embedding_floor_ops = self._num_embedding_floors(
        (embedding_weight_prec is not None), (logits_inputs_prec is not None))

    self.assertEqual(floor_count, embedding_floor_ops)

  def test_padding_mask(self):
    # Fuzzing test to make sure activation statistics aren't affected by padding
    # tokens.
    #
    # This tests works by changing the embedding of the padding token (token
    # with id '0') and making sure all the stats stay the same.
    #
    # It also tests that the stats *do* change when the embedding of a
    # non-padding token changes.
    inputs_hyper = get_bounds.GetBounds.Hyper(
        initial_bound=6.0,
        stddev_coeff=3.0,
        absdev_coeff=2.0,
        mix_coeff=0.5,
        granularity=quant_config.QuantGranularity.per_channel)
    # Set logits_via_embedding to false so that the embedding of the padding
    # token doesn't affect the logits calculation at the end of the decoder.
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=8,
        embedding_weight_prec=None,
        attention_weight_prec=8,
        mlp_pos_inputs_prec=8,
        mlp_pos_inputs_hyper=inputs_hyper,
        mlp_signed_inputs_prec=8,
        mlp_signed_inputs_hyper=inputs_hyper,
        attention_kqv_inputs_prec=8,
        attention_kqv_inputs_hyper=inputs_hyper,
        attention_out_inputs_prec=8,
        attention_out_inputs_hyper=inputs_hyper,
        logits_inputs_prec=8,
        logits_inputs_hyper=inputs_hyper,
        logits_via_embeddings=False,
        attention_act_q_inputs_prec=8,
        attention_act_q_inputs_hyper=inputs_hyper,
        attention_act_k_inputs_prec=8,
        attention_act_k_inputs_hyper=inputs_hyper,
        attention_act_probs_inputs_prec=8,
        attention_act_v_inputs_prec=8,
        attention_act_v_inputs_hyper=inputs_hyper,
        num_layers=2,
        emb_dim=5,
        num_heads=2,
        qkv_dim=4,
        mlp_dim=4,
        quant_type=QuantType.fake_quant)
    module = models.Transformer(
        hparams=hparams,
        quant_context=quant_config.QuantContext(
            update_bounds=True, collect_acts_stats=True),
        vocab_size=3,
        output_vocab_size=3,
        max_len=10,
        train=False,
        use_bfloat16=False,
        dropout_rate=.1,
        attention_dropout_rate=.1,
        should_decode=False)
    key = jax.random.PRNGKey(0)
    # Mark the first token of the target and last token of the inputs as padding
    # tokens.
    targets = onp.array([[0, 2]])
    inputs = onp.array([[1, 0]])
    initial_state = module.init(key, inputs=inputs, targets=targets)
    # Change the embedding of the padding token.
    initial_state = initial_state.unfreeze()
    initial_state['params']['shared_embedding']['embedding'] = initial_state[
        'params']['shared_embedding']['embedding'].at[0, :].set(10.0)
    module.train = True
    _, state1 = module.apply(
        flax.core.freeze(initial_state),
        inputs=inputs,
        targets=targets,
        mutable=True,
        rngs={'dropout': key})
    initial_state['params']['shared_embedding']['embedding'] = initial_state[
        'params']['shared_embedding']['embedding'].at[0, :].set(20.0)
    _, state2 = module.apply(
        flax.core.freeze(initial_state),
        inputs=inputs,
        targets=targets,
        mutable=True,
        rngs={'dropout': key})
    # This tests the statistics in both the GetBounds and StatsTag modules.
    test_utils.assert_stats_are_equal(state1, state2)

    # Now we repeat the test, but changing the embedding of a non-padding token
    # (token with ID 1 here). We expect to see the stats change.
    # print(initial_state)
    initial_state['params']['shared_embedding']['embedding'] = initial_state[
        'params']['shared_embedding']['embedding'].at[1, :].set(10.0)
    _, state1 = module.apply(
        flax.core.freeze(initial_state),
        inputs=inputs,
        targets=targets,
        mutable=True,
        rngs={'dropout': key})
    initial_state['params']['shared_embedding']['embedding'] = initial_state[
        'params']['shared_embedding']['embedding'].at[1, :].set(200.0)
    _, state2 = module.apply(
        flax.core.freeze(initial_state),
        inputs=inputs,
        targets=targets,
        mutable=True,
        rngs={'dropout': key})
    print(initial_state['get_bounds']['encoder']['encoderblock_0']
          ['enc_self_att']['K']['bounds'])
    print(state1['get_bounds']['encoder']['encoderblock_0']['enc_self_att']['K']
          ['bounds'])
    print(state2['get_bounds']['encoder']['encoderblock_0']['enc_self_att']['K']
          ['bounds'])
    print('')
    test_utils.assert_stats_are_unequal(state1, state2)

  def test_hparams_without_logits_when_logits_not_shared_raises_error(self):
    # Create hparams without logits hparams by passing in
    # logits_via_embeddings=True.
    inputs_hyper = get_bounds.GetBounds.Hyper(
        initial_bound=6.0,
        stddev_coeff=3.0,
        absdev_coeff=2.0,
        mix_coeff=0.5,
        granularity=quant_config.QuantGranularity.per_channel)
    hparams = training_hparams_generator_lib.create_base_transformer_hparams(
        mlp_weight_prec=8,
        embedding_weight_prec=None,
        attention_weight_prec=8,
        mlp_pos_inputs_prec=8,
        mlp_pos_inputs_hyper=inputs_hyper,
        mlp_signed_inputs_prec=8,
        mlp_signed_inputs_hyper=inputs_hyper,
        attention_kqv_inputs_prec=8,
        attention_kqv_inputs_hyper=inputs_hyper,
        attention_out_inputs_prec=8,
        attention_out_inputs_hyper=inputs_hyper,
        logits_inputs_prec=8,
        logits_inputs_hyper=inputs_hyper,
        logits_via_embeddings=True,
        attention_act_q_inputs_prec=8,
        attention_act_q_inputs_hyper=inputs_hyper,
        attention_act_k_inputs_prec=8,
        attention_act_k_inputs_hyper=inputs_hyper,
        attention_act_probs_inputs_prec=8,
        attention_act_v_inputs_prec=8,
        attention_act_v_inputs_hyper=inputs_hyper,
        num_layers=2,
        emb_dim=5,
        num_heads=2,
        qkv_dim=4,
        mlp_dim=4,
        quant_type=QuantType.fake_quant)

    self.assertIsNone(hparams.decoder.logits)

    # Now set logits_via_embedding in the model hparams to False.
    hparams.logits_via_embedding = False
    module = models.Transformer(
        hparams=hparams,
        quant_context=quant_config.QuantContext(
            update_bounds=True, collect_acts_stats=True),
        vocab_size=3,
        output_vocab_size=3,
        max_len=10,
        use_bfloat16=False,
        train=False,
        dropout_rate=.1,
        attention_dropout_rate=.1,
        should_decode=False)
    key = jax.random.PRNGKey(0)
    # Mark the first token of the target and last token of the inputs as padding
    # tokens.
    targets = onp.array([[0, 2]])
    inputs = onp.array([[1, 0]])
    # Because the model is not sharing logits with embeddings, but the logits
    # hparams are missing, it should raise an error.
    with self.assertRaises(ValueError):
      module.init(key, inputs=inputs, targets=targets)


if __name__ == '__main__':
  absltest.main()
