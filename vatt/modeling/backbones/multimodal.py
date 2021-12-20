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

# Lint as: python3
"""Main multimodal components of vatt, e.g. Loss Functions, Models, etc.."""

import tensorflow as tf

from vatt.modeling.backbones.audio import factory as aud_factory
from vatt.modeling.backbones.text import factory as txt_factory
from vatt.modeling.backbones.unified import factory as uvatt_factory
from vatt.modeling.backbones.video import factory as vid_factory


class AudioTextVideoFusion(tf.keras.layers.Layer):
  """Module to fuse audio, text and video for joint embedding learning."""

  def __init__(
      self,
      # Audio parameters.
      audio_backbone="resnet18",
      audio_model_kwargs=None,
      # Language parameters.
      text_backbone="linear",
      text_model_kwargs=None,
      # video parameters.
      video_backbone="resnet50",
      video_model_kwargs=None,
      name="audio_text_video_model",
      **kwargs):
    """Initialize the AudioTextVideoFusion class.

    Args:
      audio_backbone: Backbone for audio.
      audio_model_kwargs: Other specific parameters to pass to the audio module.
      text_backbone: The base language model name.
      text_model_kwargs: Other specific parameters to pass to the text module.
      video_backbone: The 3D CNN backbone.
      video_model_kwargs: Other specific parameters to pass to the video module.
      name: graph name.
      **kwargs: additional model args
    """
    super(AudioTextVideoFusion, self).__init__(name=name)
    # Audio parameters.
    self._audio_backbone = audio_backbone
    self._audio_model_kwargs = audio_model_kwargs or {}

    # Language parameters.
    self._text_backbone = text_backbone
    self._text_model_kwargs = text_model_kwargs or {}

    # video parameters.
    self._video_backbone = video_backbone
    self._video_model_kwargs = video_model_kwargs or {}

    # Defining all modules
    # first define backbones
    self.vid_backbone = vid_factory.build_model(
        backbone=self._video_backbone,
        override_params=self._video_model_kwargs,
        mode="embedding",
        )

    self.aud_backbone = aud_factory.build_model(
        backbone=self._audio_backbone,
        override_params=self._audio_model_kwargs
        )

    self.txt_backbone = txt_factory.build_model(
        backbone=self._text_backbone,
        override_params=self._text_model_kwargs
        )

  def call(self,
           video,
           audio,
           word_ids,
           training=True):
    """Computes video, text and audio embeddings.

    Args:
      video: The videos tensor of shape [B1, T, H, W, 3] where B1 is the batch
        size, T is the number of frames per clip, H the height, W the width
        and 3 the rgb channels.
      audio: The audio tensor of shape [B2, T', F] where B2 is the
        batch size, T' is the number of temporal frames, F is the number of
        frequency frames.
      word_ids: If words_embeddings is set to None, it will use the word indices
        input instead so that we can compute the word embeddings within the
        model graph. The expected shape is [B3, N, D] where B3 is the batch size
        and N the maximum number of words per sentence.
      training: Whether or not to activate the graph in training mode.

    Returns:
      video: a dict containing the video embeddings of shape
        [B1, T1', H', W', d_vid] and [B1, d_vid].
      audio: a dict containing the audio embeddings of shape
        [B2, T2', S', d_aud] and [B2, d_aud].
      text: a dict containing the text embeddings of shape
        [B3, T3', d_txt] and [B3, d_txt].
    """
    # Computes the video representation.
    backbone_outputs = self.vid_backbone(video,
                                         training=training)

    video_outputs = {"features": backbone_outputs["features"],
                     "features_pooled": backbone_outputs["features_pooled"]}

    # Computes the audio representation.
    backbone_outputs = self.aud_backbone(audio,
                                         training=training)

    audio_outputs = {"features": backbone_outputs["features"],
                     "features_pooled": backbone_outputs["features_pooled"]}

    # Computes the sentence representation.
    txt_mask = tf.where(word_ids == 0, 0, 1)
    backbone_outputs = self.txt_backbone(word_ids,
                                         training=training,
                                         attention_mask=txt_mask)

    text_outputs = {"features": backbone_outputs["word_embeddings"],
                    "features_pooled": backbone_outputs["sentence_embeddings"],
                    "word_ids": word_ids,
                    "attention_mask": txt_mask}

    return {
        "video": video_outputs,
        "audio": audio_outputs,
        "text": text_outputs,
    }


class UnifiedFusion(tf.keras.layers.Layer):
  """Module to fuse audio, text and video for joint embedding learning."""

  def __init__(
      self,
      # Audio parameters.
      unified_backbone="uvatt",
      unified_model_kwargs=None,
      name="unified_fusion_model",
      **kwargs):
    """Initialize the UnifiedFusion class.

    Args:
      unified_backbone: The unified shared backbone for all modalities.
      unified_model_kwargs: Other specific parameters to pass to the module.
      name: graph name.
      **kwargs: additional model args
    """
    super(UnifiedFusion, self).__init__(name=name)
    # Audio parameters.
    self._unified_backbone = unified_backbone
    self._unified_model_kwargs = unified_model_kwargs or {}

    self.unified_backbone = uvatt_factory.build_model(
        backbone=self._unified_backbone,
        override_params=self._unified_model_kwargs,
        )

  def call(self,
           video,
           audio,
           word_ids,
           training=True):
    """Computes video, text and audio embeddings.

    Args:
      video: The videos tensor of shape [B1, T, H, W, 3] where B1 is the batch
        size, T is the number of frames per clip, H the height, W the width
        and 3 the rgb channels.
      audio: The audio tensor of shape [B2, T', F] where B2 is the
        batch size, T' is the number of temporal frames, F is the number of
        frequency frames.
      word_ids: If words_embeddings is set to None, it will use the word indices
        input instead so that we can compute the word embeddings within the
        model graph. The expected shape is [B3, N, D] where B3 is the batch size
        and N the maximum number of words per sentence.
      training: Whether or not to activate the graph in training mode.

    Returns:
      video: a dict containing the video embeddings of shape
        [B1, T1', H', W', d_vid] and [B1, d_vid].
      audio: a dict containing the audio embeddings of shape
        [B2, T2', S', d_aud] and [B2, d_aud].
      text: a dict containing the text embeddings of shape
        [B3, T3', d_txt] and [B3, d_txt].
    """
    # Computes the video representation.
    txt_attn_mask = tf.where(word_ids == 0, 0, 1)
    backbone_outputs = self.unified_backbone(video,
                                             audio,
                                             word_ids,
                                             txt_attn_mask,
                                             training=training)

    backbone_outputs["text"].update({
        "word_ids": word_ids,
        "attention_mask": txt_attn_mask
    })

    return backbone_outputs
