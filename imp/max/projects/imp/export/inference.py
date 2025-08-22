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

"""Utilities to run inference on IMP.

Example inference:

```
batch_size = 32

video = np.ones([batch_size, 16, 256, 256, 3])
text = np.array([f'example text {i}' for i in range(batch_size)])
audio = np.ones([batch_size, 2048, 1])

model_fn = get_model('imp-moe-b')
preds = model_fn(video=video, text=text, audio=audio)

# Per-token embeddings
video_embeddings = predictions['encoder']['vision']['token_raw']['features']
text_embeddings = predictions['encoder']['text']['token_id']['features']
audio_embeddings = (
    predictions['encoder']['spectrogram']['token_raw']['features'])

# Examples of other types of features
mean_pooled_video_embeddings = (
    predictions['encoder']['vision']['token_raw']['features_agg'])
zero_shot_text2vision = (
    predictions['common_space']['text']['token_id']['vision'])
```
"""

import functools
from typing import Any, Callable

import flax
from flax import traverse_util
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from imp.max.config import registry
from imp.max.core import constants
from imp.max.core import utils
from imp.max.data import processing
from imp.max.data import tokenizers
from imp.max.execution import config as exec_config
from imp.max.execution import executors
from imp.max.export import function_setup
from imp.max.modeling import multimodal
from imp.max.optimization import objectives
from imp.max.projects.imp.config import data as imp_data_config
from imp.max.projects.imp.config import experiment as exp_config
from imp.max.utils import typing


Registrar = registry.Registrar
Registrar = registry.Registrar


DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
TaskFlowName = constants.TaskFlowName
Modality = constants.Modality


def _ensure_compatible_checkpoint(params):
  """Renames keys in params to avoid checkpoint breakages."""
  params = flax.traverse_util.flatten_dict(params, sep='/')

  def _rename_key(k):
    return k.replace('layer_scan_sparse', 'layer_scan_moe')

  params = {_rename_key(k): v for k, v in params.items()}
  params = flax.traverse_util.unflatten_dict(params, sep='/')
  return params


def _preprocess_vision(vision, config):
  """Preprocesses the vision input for IMP."""
  patch_sizes = config.model.vision_patch_size
  temporal_patch_size = patch_sizes[0]
  vision_shapes = vision.shape
  num_frames = vision_shapes[1]
  vision = jnp.reshape(vision,
                       (vision_shapes[0], -1, num_frames, vision_shapes[-3],
                        vision_shapes[-2], vision_shapes[-1]))
  if any([p > 1 for p in patch_sizes]):
    if temporal_patch_size > num_frames:
      vision = jnp.tile(vision, [1, 1, temporal_patch_size, 1, 1, 1])
  vision_full = multimodal.extract_volume_patches(
      vision, patch_sizes, flatten=False)
  vision_flattened = multimodal.extract_volume_patches(
      vision, patch_sizes, flatten=True)

  _, _, f, h, w, _ = vision_full.shape

  # Extract spatio-temporal positional encoding
  patch_3d_coordinates = utils.construct_3d_positions(
      f, h, w, normalize=True)
  # Add token ID
  patch_1d_ids = utils.construct_1d_positions(
      vision_flattened.shape[2], normalize=False)

  data_collection_type = DataFeatureType.INPUTS
  data_collection_route = DataFeatureRoute.ENCODER

  get_feature_name = functools.partial(
      processing.get_flattened_key,
      ftype=data_collection_type,
      froute=data_collection_route,
      modality=Modality.VISION)

  token_raw_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_RAW)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)

  return traverse_util.unflatten_dict({
      token_raw_feature_name: vision_flattened,
      token_coordinate_feature_name: patch_3d_coordinates,
      token_position_id_feature_name: patch_1d_ids,
  }, sep='/')


def _preprocess_text(text):
  """Preprocesses the text input for IMP."""

  seq_len = text.shape[-1]
  text = np.reshape(text, (-1, 1, seq_len))

  # Add padding mask for the padded locations
  token_mask = jnp.where(text == 0, 0, 1)

  # Extract positional encoding
  token_coordinate = utils.construct_1d_positions(
      seq_len,
      normalize=True)

  # Add token position ID
  token_position_id = utils.construct_1d_positions(
      seq_len,
      normalize=False)

  data_collection_type = DataFeatureType.INPUTS
  data_collection_route = DataFeatureRoute.ENCODER

  get_feature_name = functools.partial(
      processing.get_flattened_key,
      ftype=data_collection_type,
      froute=data_collection_route,
      modality=Modality.TEXT)

  token_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_ID)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)
  token_mask_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_MASK)

  return traverse_util.unflatten_dict({
      token_id_feature_name: text,
      token_coordinate_feature_name: token_coordinate,
      token_position_id_feature_name: token_position_id,
      token_mask_feature_name: token_mask,
  }, sep='/')


def _imp_postprocess(
    outputs):
  """Default postprocess function for IMP saved model."""

  # Including inputs/hyperparams is not necessary for saved model.
  outputs = outputs[DataFeatureType.OUTPUTS]

  # Remove instance dimension.
  outputs = jax.tree.map(lambda x: x[:, 0], outputs)

  return outputs


def infer_imp_vision_with_params(
    vision,
    model_params,
    config,
    executor,
    postprocess_fn = None):
  """Returns inference function that only receives a vision tensor."""
  postprocess_fn = postprocess_fn or _imp_postprocess
  modalities = [Modality.VISION]
  target_modalities = [Modality.VISION, Modality.TEXT]

  inputs = {
      **_preprocess_vision(vision, config),
      DataFeatureType.METADATA: imp_data_config.get_contrastive_metadata(
          modalities, target_modalities=target_modalities)
  }

  model_params = _ensure_compatible_checkpoint(model_params)
  mutables = {}

  return function_setup.infer_model_with_params(
      inputs, model_params, mutables, executor,
      postprocess_fn=postprocess_fn)


def infer_imp_text_with_params(
    text,
    model_params,
    config,
    executor,
    postprocess_fn = None):
  """Returns inference function that only receives a vision tensor."""
  del config
  postprocess_fn = postprocess_fn or _imp_postprocess
  modalities = [Modality.TEXT]
  target_modalities = [Modality.VISION, Modality.TEXT]

  inputs = {
      **_preprocess_text(text),
      DataFeatureType.METADATA: imp_data_config.get_contrastive_metadata(
          modalities, target_modalities=target_modalities)
  }

  model_params = _ensure_compatible_checkpoint(model_params)
  mutables = {}

  return function_setup.infer_model_with_params(
      inputs, model_params, mutables, executor,
      postprocess_fn=postprocess_fn)


def process_text(
    text,
    max_num_tokens = 32,
    tokenizer = None,
):
  """Preprocesses the text input for IMP."""
  data_collection_type = DataFeatureType.INPUTS
  data_collection_route = DataFeatureRoute.ENCODER

  get_feature_name = functools.partial(
      processing.get_flattened_key,
      ftype=data_collection_type,
      froute=data_collection_route,
      modality=Modality.TEXT)

  raw_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  token_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_ID)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)
  token_mask_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_MASK)

  text = tf.constant(text, dtype=tf.string)

  if len(text.shape) < 1:
    text = text[tf.newaxis]

  features = {
      raw_feature_name: text,
  }

  if tokenizer is None:
    tokenizer = tokenizers.get_tokenizer()
    tokenizer.initialize()

  keep_raw_string = False
  prepend_bos = False
  append_eos = False
  tokenized = processing.tokenize_raw_string(
      features,
      tokenizer=tokenizer,
      raw_feature_name=raw_feature_name,
      token_id_feature_name=token_id_feature_name,
      token_coordinate_feature_name=token_coordinate_feature_name,
      token_position_id_feature_name=token_position_id_feature_name,
      token_mask_feature_name=token_mask_feature_name,
      keep_raw_string=keep_raw_string,
      prepend_bos=prepend_bos,
      append_eos=append_eos,
      max_num_tokens=max_num_tokens,
      max_num_sentences=text.shape[0])

  return tokenized


def process_vision(video,
                   patch_size = (4, 16, 16),
                   normalize = False,
                   dtype = tf.float32):
  """Preprocesses the vision input for IMP."""
  data_collection_type = DataFeatureType.INPUTS
  data_collection_route = DataFeatureRoute.ENCODER

  get_feature_name = functools.partial(
      processing.get_flattened_key,
      ftype=data_collection_type,
      froute=data_collection_route,
      modality=Modality.VISION)

  raw_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  token_raw_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_RAW)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)

  video = tf.cast(video, dtype)

  if normalize:
    video = processing.normalize_image(video, zero_centering_image=True)

  video = {
      raw_feature_name: video,
  }

  video = processing.tokenize_raw_rgb(
      video,
      raw_feature_name=raw_feature_name,
      token_raw_feature_name=token_raw_feature_name,
      token_coordinate_feature_name=token_coordinate_feature_name,
      token_position_id_feature_name=token_position_id_feature_name,
      temporal_patch_size=patch_size[0],
      spatial_patch_size=patch_size[1:],
      spatio_temporal_token_coordinate=True)

  return video


def process_audio(
    wav,
    patch_size = (16, 16),
    sample_rate = 16000,
    spectrogram_type = 'logmf',
    frame_length = 2048,
    frame_step = 1024,
    num_features = 128,
    lower_edge_hertz = 80.0,
    upper_edge_hertz = 7600.0,
    preemphasis = None,
    normalize_audio = False,
):
  """Preprocesses the audio input for IMP."""
  data_collection_type = DataFeatureType.INPUTS
  data_collection_route = DataFeatureRoute.ENCODER

  get_feature_name = functools.partial(
      processing.get_flattened_key,
      ftype=data_collection_type,
      froute=data_collection_route,
      modality=Modality.SPECTROGRAM)
  raw_spectrogram_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  token_raw_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_RAW)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)

  wav = tf.constant(wav)

  # Note: this step can take a lot of processing time.
  spectrogram = processing.compute_audio_spectrogram(
      wav,
      sample_rate=sample_rate,
      spectrogram_type=spectrogram_type,
      frame_length=frame_length,
      frame_step=frame_step,
      num_features=num_features,
      lower_edge_hertz=lower_edge_hertz,
      upper_edge_hertz=upper_edge_hertz,
      normalize=normalize_audio,
      preemphasis=preemphasis)

  audio = {
      raw_spectrogram_feature_name: spectrogram,
  }

  audio = processing.tokenize_raw_spectrogram(
      audio,
      raw_feature_name=raw_spectrogram_feature_name,
      token_raw_feature_name=token_raw_feature_name,
      token_coordinate_feature_name=token_coordinate_feature_name,
      token_position_id_feature_name=token_position_id_feature_name,
      temporal_patch_size=patch_size[0],
      spectoral_patch_size=patch_size[1])

  return audio


def preprocess_input(
    video = None,
    text = None,
    audio = None,
    tokenizer = None,
    normalize_video = False,
    video_patch_size = (4, 16, 16),
    max_num_text_tokens = 32,
    audio_params = None):
  """Preprocesses the input for IMP with the given modalities."""

  inputs = {}
  modalities = []
  if audio_params is None:
    audio_params = {}

  if video is not None:
    inputs.update(
        process_vision(
            video, normalize=normalize_video, patch_size=video_patch_size))
    modalities.append(Modality.VISION)
  if text is not None:
    inputs.update(process_text(
        text, tokenizer=tokenizer, max_num_tokens=max_num_text_tokens))
    modalities.append(Modality.TEXT)
  if audio is not None:
    inputs.update(process_audio(audio, **audio_params))
    modalities.append(Modality.SPECTROGRAM)
  inputs = jax.tree.map(lambda x: x.numpy()[:, np.newaxis], inputs)

  inputs = traverse_util.unflatten_dict(inputs, sep='/')
  inputs[DataFeatureType.METADATA] = imp_data_config.get_contrastive_metadata(
      modalities=modalities,
      target_modalities=(Modality.VISION, Modality.TEXT, Modality.SPECTROGRAM))

  return inputs


def run_model(inputs,
              model_fn,
              params,
              prng_keys):
  """Runs prediction on the given model and input."""

  predictions = model_fn(params, inputs, False, rngs=prng_keys)
  predictions = jax.tree.map(lambda x: x[:, 0],
                             predictions['outputs']['encoder'])
  return predictions


def imp_moe_b():
  """Config definition for IMP-MoE-B."""

  config_path = 'path/to/checkpoint'  # pylint: disable=unused-variable
  return exp_config.SparseImpMoeBV1TrainExperiment(path=config_path)




MODEL_CONFIGS = {
    'imp-moe-b': imp_moe_b,
}


def available_models():
  """A list of all available models."""
  return list(MODEL_CONFIGS.keys())


def get_model_config(name):
  """Returns a config from the given model name."""
  return MODEL_CONFIGS[name]()


def get_model(
    name,
    normalize_video = False,
    video_patch_size = (4, 16, 16),
    max_num_text_tokens = 32,
    audio_params = None,
):
  """Returns a callable pretrained model from the given name."""

  config = get_model_config(name)
  model = Registrar.get_class_by_name(config.model.name)(  # pylint: disable=attribute-error
      **config.model.as_dict())  # pylint: disable=attribute-error
  params = flax.training.checkpoints.restore_checkpoint(
      config.path, target=None, step=0)

  params = _ensure_compatible_checkpoint(params)

  _, prng_keys = executors.get_rngs(model.get_rng_keys(), add_params_rngs=True)
  model_jit_fn = jax.jit(model.apply, static_argnums=(2, 3))

  tokenizer = tokenizers.get_tokenizer()
  tokenizer.initialize()

  def _model_fn(video=None, text=None, audio=None):
    inputs = preprocess_input(
        video=video,
        text=text,
        audio=audio,
        tokenizer=tokenizer,
        normalize_video=normalize_video,
        video_patch_size=video_patch_size,
        max_num_text_tokens=max_num_text_tokens,
        audio_params=audio_params,
    )

    predictions = model_jit_fn(params, inputs, False, rngs=prng_keys)
    predictions = jax.tree.map(
        lambda x: x[:, 0], predictions[DataFeatureType.OUTPUTS]
    )
    return predictions

  return _model_fn


def retrieve(
    predictions, from_modality, to_modality
):
  """Retrieves the embeddings to predict zero-shot inference."""
  return objectives.l2_normalize(
      predictions[DataFeatureRoute.COMMON_SPACE][
          from_modality][DataFeatureName.TOKEN_RAW][to_modality]
  )


def zero_shot_predict(
    inputs,
    values,
    from_modality,
    to_modality,
):
  """Returns a scoring matrix comparing the values which match the inputs."""
  a2b = retrieve(inputs, from_modality, to_modality)
  b2a = retrieve(values, from_modality, to_modality)
  return jnp.einsum('bd,cd->bc', a2b, b2a)
