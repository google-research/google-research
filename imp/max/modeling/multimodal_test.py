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

"""Tests for multimodal modules."""
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from imp.max.core import constants
from imp.max.modeling import multimodal
from imp.max.utils import sharding


Modality = constants.Modality
_RAW2EMBED_NAMES = {
    Modality.WAVEFORM: 'wav_to_embedding',
    Modality.SPECTROGRAM: 'spc_to_embedding',
    Modality.VISION: 'rgb_to_embedding',
}
_SPECIAL_TOKENS = {
    Modality.WAVEFORM: 'waveform_embedding',
    Modality.SPECTROGRAM: 'spectrogram_embedding',
    Modality.VISION: 'vision_embedding',
    Modality.TEXT: 'text_embedding',
}


@dataclasses.dataclass
class Shardings:
  pos_encode_embed = ('model', None)
  pos_encode_layernorm = (None,)
  token_raws = ('data', None, None, 'model')
  token_embeds = ('data', None, None, None)
  token_raw_to_embed_kernel = ('model', None)
  conv_raw_to_embed_kernel = {
      Modality.VISION: ('model', None, None, None, None),
      Modality.SPECTROGRAM: (None, 'model', None, None),
      Modality.WAVEFORM: ('model', None, None),
  }
  conv_raw_to_embed_bias = {Modality.VISION: (None,),
                            Modality.SPECTROGRAM: (None,),
                            Modality.WAVEFORM: (None,)}
  special_token_embedding = (None, None)
  classification_kernel = (None, 'model')


def _create_global_mesh():
  return jax.sharding.Mesh(
      sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                      dcn_mesh_shape=(1, 1)),
      ['data', 'model'],
  )


class RawToEmbedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'waveform_deterministic',
          'd_model': 7,
          'modality': Modality.WAVEFORM,
          'patch_size': 3,
          'pos_buckets': 4,
          'dropout_rate': 0.1,
          'droptoken_rate': 0.5,
          'deterministic': True,
          'input_shape': (1, 5, 12, 1),
          'embedding_shape': (1, 5, 4, 7),
          'output_shape': (1, 5, 4, 7)
      },
      {
          'testcase_name': 'waveform',
          'd_model': 7,
          'modality': Modality.WAVEFORM,
          'patch_size': 3,
          'pos_buckets': 4,
          'dropout_rate': 0.1,
          'droptoken_rate': 0.5,
          'deterministic': False,
          'input_shape': (1, 5, 12, 1),
          'embedding_shape': (1, 5, 4, 7),
          'output_shape': (1, 5, 2, 7)
      },
      {
          'testcase_name': 'spectrogram_deterministic',
          'd_model': 7,
          'modality': Modality.SPECTROGRAM,
          'patch_size': (5, 16),
          'pos_buckets': (3, 2),
          'dropout_rate': 0.1,
          'droptoken_rate': 0.5,
          'deterministic': True,
          'input_shape': (1, 5, 15, 32, 1),
          'embedding_shape': (1, 5, 3, 2, 7),
          'output_shape': (1, 5, 6, 7)
      },
      {
          'testcase_name': 'spectrogram',
          'd_model': 7,
          'modality': Modality.SPECTROGRAM,
          'patch_size': (5, 16),
          'pos_buckets': (3, 2),
          'dropout_rate': 0.1,
          'droptoken_rate': 0.5,
          'deterministic': False,
          'input_shape': (1, 5, 15, 32, 1),
          'embedding_shape': (1, 5, 3, 2, 7),
          'output_shape': (1, 5, 3, 7)
      },
      {
          'testcase_name': 'vision_deterministic',
          'd_model': 10,
          'modality': Modality.VISION,
          'patch_size': (4, 16, 16),
          'pos_buckets': (4, 2, 2),
          'dropout_rate': 0.1,
          'droptoken_rate': 0.2,
          'deterministic': True,
          'input_shape': (1, 4, 16, 32, 32, 3),
          'embedding_shape': (1, 4, 4, 2, 2, 10),
          'output_shape': (1, 4, 16, 10)
      },
      {
          'testcase_name': 'vision',
          'd_model': 10,
          'modality': Modality.VISION,
          'patch_size': (4, 16, 16),
          'pos_buckets': (4, 2, 2),
          'dropout_rate': 0.1,
          'droptoken_rate': 0.2,
          'deterministic': False,
          'input_shape': (1, 4, 16, 32, 32, 3),
          'embedding_shape': (1, 4, 4, 2, 2, 10),
          'output_shape': (1, 4, 12, 10)
      },
  )
  def test_raw_to_embed(self, d_model, modality, patch_size,
                        pos_buckets, dropout_rate, droptoken_rate,
                        deterministic, input_shape, embedding_shape,
                        output_shape):

    shardings = Shardings()
    raw_to_embed_kernel_shardings = shardings.conv_raw_to_embed_kernel[modality]
    raw_to_embed_bias_shardings = shardings.conv_raw_to_embed_bias[modality]
    embedder = multimodal.RawToEmbed(
        d_model=d_model,
        modality=modality,
        patch_size=patch_size,
        pos_buckets=pos_buckets,
        dropout_rate=dropout_rate,
        droptoken_rate=droptoken_rate,
        raw_to_embed_kernel_shardings=raw_to_embed_kernel_shardings,
        raw_to_embed_bias_shardings=raw_to_embed_bias_shardings,
        pos_encode_embed_shardings=shardings.pos_encode_embed,
        pos_encode_layernorm_shardings=shardings.pos_encode_layernorm,
    )

    @jax.jit
    def _run_forward(inputs):
      variables = embedder.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      outputs = embedder.apply(
          variables=variables,
          rngs={
              'dropout': jax.random.key(2),
              'droptoken': jax.random.key(3)
          },
          inputs=inputs,
          deterministic=deterministic)
      return outputs, variables

    inputs = jnp.ones(input_shape, dtype=jnp.int32)
    with _create_global_mesh():
      outputs, variables = _run_forward(inputs)

    chex.assert_shape(outputs[0], output_shape)
    chex.assert_equal(outputs[1], embedding_shape)

    # Assert shardings are propagated properly
    r2e_name = _RAW2EMBED_NAMES[modality]
    self.assertEqual(variables['params'][r2e_name]['kernel'].names,
                     raw_to_embed_kernel_shardings)
    self.assertEqual(variables['params'][r2e_name]['bias'].names,
                     raw_to_embed_bias_shardings)

  def test_undefined_modality(self):

    @jax.jit
    def _run_forward():
      embedded = multimodal.RawToEmbed(d_model=2, modality='undefined')
      embedded.init(rngs={'params': jax.random.key(1)})

    with self.assertRaises(ValueError):
      _run_forward()

  def test_text_not_implemented(self):

    @jax.jit
    def _run_forward():
      embedded = multimodal.RawToEmbed(d_model=2,
                                       modality=Modality.TEXT)
      embedded.init(rngs={'params': jax.random.key(1)})

    with self.assertRaises(NotImplementedError):
      _run_forward()


class TokenRawToEmbedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'waveform_1d',
          'd_model': 4,
          'd_raw': 6,
          'modality': Modality.WAVEFORM,
          'pos_buckets': 5,
          'dropout_rate': 0.1,
          'deterministic': True,
      },
      {
          'testcase_name': 'spectrogram_2d',
          'd_model': 4,
          'd_raw': 3,
          'modality': Modality.SPECTROGRAM,
          'pos_buckets': (3, 2),
          'dropout_rate': 0.1,
          'deterministic': True,
      },
      {
          'testcase_name': 'vision_3d',
          'd_model': 4,
          'd_raw': 5,
          'modality': Modality.VISION,
          'pos_buckets': (2, 3, 3),
          'dropout_rate': 0.1,
          'deterministic': True,
      },
      {
          'testcase_name': 'vision_1d',
          'd_model': 4,
          'd_raw': 5,
          'modality': Modality.VISION,
          'pos_buckets': 16,
          'dropout_rate': 0.1,
          'deterministic': True,
      },
  )
  def test_raw_patch_to_embed(self,
                              d_model,
                              d_raw,
                              modality,
                              pos_buckets,
                              dropout_rate,
                              deterministic):
    shardings = Shardings()
    embedder = multimodal.TokenRawToEmbed(
        d_model=d_model,
        modality=modality,
        pos_buckets=pos_buckets,
        dropout_rate=dropout_rate,
        raw_to_embed_kernel_shardings=shardings.token_raw_to_embed_kernel)

    @jax.jit
    def _run_forward(inputs):
      variables = embedder.init(
          rngs={'params': jax.random.key(1)}, inputs=inputs)
      outputs = embedder.apply(
          variables=variables,
          rngs={
              'dropout': jax.random.key(2),
              'droptoken': jax.random.key(3)
          },
          inputs=inputs,
          deterministic=deterministic)
      return outputs, variables

    input_shape = (1, 2, np.prod(pos_buckets), d_raw)
    output_shape = input_shape[:-1] + (d_model,)
    with _create_global_mesh():
      inputs = jnp.ones(input_shape, dtype=jnp.int32)
      inputs = sharding.shard_array(inputs, shardings.token_raws)
      outputs, variables = _run_forward(inputs)

    # Assert output shape
    chex.assert_shape(outputs, output_shape)

    # Assert shardings are propagated properly
    r2e_name = _RAW2EMBED_NAMES[modality]
    self.assertEqual(variables['params'][r2e_name]['kernel'].names,
                     shardings.token_raw_to_embed_kernel)


class PerModalitySpecialTokenTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'left_vision',
          'features': 1,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.VISION,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': None,
          'attention_bias_shape': None
      },
      {
          'testcase_name': 'left_vision_mask_only',
          'features': 1,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.VISION,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': [4, 5, 6],
          'attention_bias_shape': None
      },
      {
          'testcase_name': 'left_vision_bias_only',
          'features': 1,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.VISION,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': None,
          'attention_bias_shape': [1, 6, 6]
      },
      {
          'testcase_name': 'left_vision_bias_and_mask',
          'features': 1,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.VISION,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': [4, 5, 6],
          'attention_bias_shape': [1, 6, 6]
      },
      {
          'testcase_name': 'right_vision',
          'features': 2,
          'extension': constants.Extension.APPEND,
          'modality': Modality.VISION,
          'input_shape': (2, 3, 4, 2),
          'output_shape': (2, 3, 5, 2),
          'token_mask_shape': [2, 3, 4],
          'attention_bias_shape': [2, 4, 4]
      },
      {
          'testcase_name': 'left_vision_no_mask',
          'features': 2,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.VISION,
          'input_shape': (2, 3, 4, 2),
          'output_shape': (2, 3, 5, 2),
          'token_mask_shape': None,
          'attention_bias_shape': None
      },
      {
          'testcase_name': 'left_waveform_no_bias',
          'features': 4,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.WAVEFORM,
          'input_shape': (2, 3, 4, 4),
          'output_shape': (2, 3, 5, 4),
          'token_mask_shape': [2, 3, 4],
          'attention_bias_shape': None
      },
      {
          'testcase_name': 'right_waveform',
          'features': 1,
          'extension': constants.Extension.APPEND,
          'modality': Modality.WAVEFORM,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': [4, 5, 6],
          'attention_bias_shape': [3, 6, 6]
      },
      {
          'testcase_name': 'left_waveform_no_mask',
          'features': 20,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.WAVEFORM,
          'input_shape': (1, 1, 4, 20),
          'output_shape': (1, 1, 5, 20),
          'token_mask_shape': None,
          'attention_bias_shape': [3, 4, 4]
      },
      {
          'testcase_name': 'left_spectrogram_no_bias',
          'features': 4,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.SPECTROGRAM,
          'input_shape': (2, 3, 4, 4),
          'output_shape': (2, 3, 5, 4),
          'token_mask_shape': [2, 3, 4],
          'attention_bias_shape': None
      },
      {
          'testcase_name': 'right_spectrogram',
          'features': 1,
          'extension': constants.Extension.APPEND,
          'modality': Modality.SPECTROGRAM,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': [4, 5, 6],
          'attention_bias_shape': [3, 6, 6]
      },
      {
          'testcase_name': 'left_spectrogram_no_mask',
          'features': 20,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.SPECTROGRAM,
          'input_shape': (1, 1, 4, 20),
          'output_shape': (1, 1, 5, 20),
          'token_mask_shape': None,
          'attention_bias_shape': [3, 4, 4]
      },
      {
          'testcase_name': 'left_text',
          'features': 4,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.TEXT,
          'input_shape': (2, 3, 4, 4),
          'output_shape': (2, 3, 5, 4),
          'token_mask_shape': [2, 3, 4],
          'attention_bias_shape': [1, 4, 4]
      },
      {
          'testcase_name': 'right_text_bias_only',
          'features': 1,
          'extension': constants.Extension.APPEND,
          'modality': Modality.TEXT,
          'input_shape': (4, 5, 6, 1),
          'output_shape': (4, 5, 7, 1),
          'token_mask_shape': None,
          'attention_bias_shape': [3, 6, 6]
      },
      {
          'testcase_name': 'left_text_no_mask',
          'features': 20,
          'extension': constants.Extension.PREPEND,
          'modality': Modality.TEXT,
          'input_shape': (1, 1, 4, 20),
          'output_shape': (1, 1, 5, 20),
          'token_mask_shape': None,
          'attention_bias_shape': None
      },
  )
  def test_special_token(self, features, extension, modality, input_shape,
                         output_shape, token_mask_shape,
                         attention_bias_shape):
    shardings = Shardings()
    special_token = multimodal.PerModalitySpecialToken(
        features=features,
        extension=extension,
        embedding_shardings=shardings.special_token_embedding,
        activation_shardings=shardings.token_embeds,
    )

    @jax.jit
    def _run_forward(inputs, token_mask, attention_bias):
      variables = special_token.init(
          rngs={'params': jax.random.key(1)},
          inputs=inputs,
          modality=modality,
          token_mask=token_mask,
          attention_bias=attention_bias)
      outputs = special_token.apply(
          variables=variables,
          rngs={},
          inputs=inputs,
          modality=modality,
          token_mask=token_mask,
          attention_bias=attention_bias)
      return outputs, variables

    inputs = jnp.ones(input_shape)
    if token_mask_shape:
      token_mask = jnp.ones(token_mask_shape)
      output_token_mask_shape = token_mask_shape
      output_token_mask_shape[-1] += 1
    else:
      token_mask = None
    if attention_bias_shape:
      attention_bias = jnp.ones(attention_bias_shape)
      output_attention_bias_shape = attention_bias_shape
      output_attention_bias_shape[-2] += 1
      output_attention_bias_shape[-1] += 1
    else:
      attention_bias = None

    with _create_global_mesh():
      outputs, variables = _run_forward(inputs, token_mask, attention_bias)
    chex.assert_shape(outputs[0], output_shape)

    if token_mask_shape:
      chex.assert_shape(outputs[1], output_token_mask_shape)  # pylint: disable=undefined-variable
    else:
      self.assertIsNone(outputs[1])

    if attention_bias_shape:
      chex.assert_shape(outputs[2], output_attention_bias_shape)  # pylint: disable=undefined-variable
    else:
      self.assertIsNone(outputs[2])

    # Assert shardings are propagated properly
    spc_name = _SPECIAL_TOKENS[modality]
    self.assertEqual(
        variables['params'][spc_name].names, shardings.special_token_embedding)

  @parameterized.named_parameters(
      ('undefined_extension', Modality.VISION, None, None, ValueError,
       'undefined'),
      ('undefined_modality', 'undefined', None, None, KeyError),
      ('4d_token_mask', Modality.VISION,
       (2, 3, 4, 4), None, ValueError),
      ('2d_token_mask', Modality.VISION,
       (2, 3), None, ValueError),
      ('4d_attention_bias', Modality.VISION, None,
       (1, 1, 4, 4), ValueError),
      ('2d_attention_bias', Modality.VISION, None,
       (1, 1), ValueError),
      ('mismatch_token_mask', Modality.VISION,
       (1, 1, 2), None, ValueError),
      ('mismatch_attention_bias', Modality.VISION, None,
       (1, 2, 2), ValueError),
      ('mismatch_attention_bias_q', Modality.VISION, None,
       (1, 2, 4), ValueError),
      ('mismatch_attention_bias_kv', Modality.VISION, None,
       (1, 4, 2), ValueError))
  def test_undefined_behavior(self,
                              modality,
                              token_mask_shape,
                              attention_bias_shape,
                              error_type,
                              extension=constants.Extension.PREPEND):

    @jax.jit
    def _run_forward():
      inputs = jnp.ones((2, 3, 4, 2))
      if token_mask_shape:
        token_mask = jnp.ones(token_mask_shape)
      else:
        token_mask = None
      if attention_bias_shape:
        attention_bias = jnp.ones(attention_bias_shape)
      else:
        attention_bias = None
      token = multimodal.PerModalitySpecialToken(features=2,
                                                 extension=extension)
      variables = token.init(
          rngs={'params': jax.random.key(1)},
          inputs=inputs,
          modality=modality,
          token_mask=token_mask,
          attention_bias=attention_bias)
      token.apply(
          variables=variables,
          rngs={},
          inputs=inputs,
          modality=modality,
          token_mask=token_mask,
          attention_bias=attention_bias)

    with self.assertRaises(error_type):
      _run_forward()

  def test_per_modality_classifier(self):
    shardings = Shardings()
    per_modality_classifier = multimodal.PerModalityCLS(
        vision_classes=(('in', 5), ('jft', 6)),
        waveform_classes=np.prod([1, 7]),
        spectrogram_classes=np.prod([1, 2]),
        text_classes=(('c4', 8),),
        predictions_key=constants.DataFeatureName.LOGITS,
        kernel_shardings=shardings.classification_kernel,
    )
    inputs = jnp.ones((2, 3, 4), dtype=jnp.float32)
    rngs = {'params': jax.random.key(10)}

    @jax.jit
    def _run_forward():
      all_params = {}
      all_outputs = {}
      for modality in (Modality.VISION,
                       Modality.WAVEFORM,
                       Modality.SPECTROGRAM,
                       Modality.TEXT):
        variables = per_modality_classifier.init(
            rngs=rngs, inputs=inputs, modality=modality)
        outputs = per_modality_classifier.apply(
            variables, inputs=inputs, modality=modality)
        all_params[modality] = variables['params']
        all_outputs[modality] = outputs
      all_params = traverse_util.flatten_dict(all_params, sep='/')
      all_outputs = traverse_util.flatten_dict(all_outputs, sep='/')
      return all_outputs, all_params

    outputs, params = _run_forward()

    # Assert variables contain all modalities and their specific heads
    cls_sh = shardings.classification_kernel
    self.assertEqual(params['vision/vision_cls_in/kernel'].value.shape, (4, 5))
    self.assertEqual(params['vision/vision_cls_in/kernel'].names, cls_sh)
    self.assertEqual(params['vision/vision_cls_in/bias'].value.shape, (5,))
    self.assertEqual(params['vision/vision_cls_jft/kernel'].value.shape, (4, 6))
    self.assertEqual(params['vision/vision_cls_jft/kernel'].names, cls_sh)
    self.assertEqual(params['vision/vision_cls_jft/bias'].value.shape, (6,))
    self.assertEqual(params['waveform/waveform_cls/kernel'].value.shape, (4, 7))
    self.assertEqual(params['waveform/waveform_cls/kernel'].names, cls_sh)
    self.assertEqual(params['waveform/waveform_cls/bias'].value.shape, (7,))
    self.assertEqual(params['spectrogram/spectrogram_cls/kernel'].value.shape,
                     (4, 2))
    self.assertEqual(params['spectrogram/spectrogram_cls/kernel'].names, cls_sh)
    self.assertEqual(params['spectrogram/spectrogram_cls/bias'].value.shape,
                     (2,))
    self.assertEqual(params['text/text_cls_c4/kernel'].value.shape, (4, 8))
    self.assertEqual(params['text/text_cls_c4/kernel'].names, cls_sh)
    self.assertEqual(params['text/text_cls_c4/bias'].value.shape, (8,))

    # Assert outputs contain all modalities and their specific logits
    self.assertEqual(outputs['vision/logits_in'].shape, (2, 3, 5))
    self.assertEqual(outputs['vision/logits_jft'].shape, (2, 3, 6))
    self.assertEqual(outputs['waveform/logits'].shape, (2, 3, 7))
    self.assertEqual(outputs['spectrogram/logits'].shape, (2, 3, 2))
    self.assertEqual(outputs['text/logits_c4'].shape, (2, 3, 8))

  def test_per_modality_temperature(self):
    per_modality_temperature = multimodal.PerModalityTemperature(
        init_value=0.01,
        modalities=('text', 'vision'),
    )
    variables = per_modality_temperature.init(
        rngs={'params': jax.random.key(1)})
    temperature = per_modality_temperature.apply(variables)

    self.assertEqual(temperature['text_vision'], 0.01)

if __name__ == '__main__':
  absltest.main()
