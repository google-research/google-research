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

"""Data processing modules."""

import functools
import json
from typing import Sequence, Type

from absl import logging
from dmvr import builders
from dmvr import processors
from flax import traverse_util
import tensorflow as tf
import tensorflow_probability as tfp

from imp.max.core import constants
from imp.max.data import tokenizers
from imp.max.data.datasets import prompts
from imp.max.utils import typing


Modality = constants.Modality
ParsingFeatureName = constants.ParsingFeatureName
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
VOCABULARY = tokenizers.VOCABULARY


# ----------------------------------------------------------------------
# --------- Constant feature name compilation helper functions ---------
# ----------------------------------------------------------------------
def get_flattened_key(ftype,
                      froute,
                      fname,
                      modality,
                      sep = '/'):
  """Compiles the key for features in the features dictionary."""
  return sep.join([ftype, froute, modality, fname])


def has_token(name):
  """Returns True if the input feature name is a token name."""
  token_names = (DataFeatureName.TOKEN_RAW, DataFeatureName.TOKEN_ID)
  return any(token_name in name for token_name in token_names)


# ----------------------------------------------------------------------
# ------------------ Tensor-level Processing functions -----------------
# ----------------------------------------------------------------------
# Auxiliary processing fns from DMVR
sample_sequence = processors.sample_sequence
sample_linspace_sequence = processors.sample_linspace_sequence
sample_or_pad_non_sorted_sequence = processors.sample_or_pad_non_sorted_sequence
normalize_image = processors.normalize_image
resize_smallest = processors.resize_smallest
crop_image = processors.crop_image
random_flip_left_right = processors.random_flip_left_right
color_default_augm = processors.color_default_augm
set_shape = processors.set_shape
crop_or_pad_words = processors.crop_or_pad_words
tokenize = processors.tokenize


def get_min_resize_value(crop_size):
  """Returns the nearest multiple of 8 to 256/224 to be used as min_resize."""
  return int(crop_size * 1.15 // 8 * 8)


def decode_image(image_string, channels = 0):
  """Decodes image raw bytes string into a RGB uint8 tensor.

  Args:
    image_string: A tensor of type strings with the raw image bytes where the
      first dimension is timesteps. Supports JPEG, PNG, GIF, and BMP formats.
      For GIFs, the only the first frame is used.
    channels: Number of channels of the image. Allowed values are 0, 1 and
      3. If 0, the number of channels will be calculated at runtime and no
      static shape is set.

  Returns:
    A `tf.Tensor` of shape [T, H, W, C] of type `tf.uint8` with the decoded
    images.
  """
  decode_fn = functools.partial(
      tf.image.decode_image,
      channels=channels,
      expand_animations=False)
  return tf.map_fn(
      decode_fn,
      image_string,
      back_prop=False,
      dtype=tf.uint8)


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def maybe_densify_tensor(
    tensor):
  if isinstance(tensor, tf.sparse.SparseTensor):
    tensor = tf.sparse.to_dense(tensor)
  return tensor


def construct_1d_positions(inputs,
                           normalize = True):
  """Constructs 1D positional encoding IDs for image/video.

  Args:
    inputs: A tensor with shape [instance, time, ...].
    normalize: A bool indicating whether the position values should be
      normalized to the maximum length or not. If this is set to True, the
      resulting tensor with the position values will have a dtype=tf.float32.

  Returns:
    A tensor with shape [1, time] that encodes positional IDs corresponding to
    each temporal position. If normalize=True, this tensor will have
    dtype=tf.float32. Otherwise, it will be dtype=tf.int32.
  """
  t = inputs.shape[1]
  temporal_ids = tf.range(t)  # (t,)
  pos_ids = temporal_ids[tf.newaxis, :]  # (1, t)

  if normalize:
    max_positions = tf.cast(t, dtype=tf.float32)
    pos_ids = tf.cast(pos_ids, dtype=tf.float32) / max_positions

  return pos_ids


def construct_2d_positions(inputs,
                           normalize = True):
  """Constructs 2D positional encoding IDs for 2D signals.

  Args:
    inputs: A tensor with shape [instance, time, feature, channel].
    normalize: A bool indicating whether the position values should be
      normalized to the maximum length or not. If this is set to True, the
      resulting tensor with the position values will have a dtype=tf.float32.

  Returns:
    A tensor with shape [1, time * feature, 2] that encodes positional
    IDs corresponding to each spectro-temporal position. If normalize=True, this
    tensor will have dtype=tf.float32. Otherwise, it will be dtype=tf.int32.
  """
  t, s = inputs.shape[1:3]
  temporal_ids, spectoral_ids = tf.meshgrid(
      tf.range(t), tf.range(s), indexing='ij')

  # (t, s, 2)
  pos_ids = tf.stack([temporal_ids, spectoral_ids], axis=2)
  pos_ids = tf.reshape(pos_ids, [1, -1, 2])  # (1, t*s, 2)

  if normalize:
    max_positions = tf.constant([[[t, s]]], dtype=tf.float32)
    pos_ids = tf.cast(pos_ids, dtype=tf.float32) / max_positions

  return pos_ids


def construct_3d_positions(inputs,
                           normalize = True):
  """Constructs 3D positional encoding IDs for image/video.

  Args:
    inputs: A tensor with shape [instance, time, height, width, channel].
    normalize: A bool indicating whether the position values should be
      normalized to the maximum length or not. If this is set to True, the
      resulting tensor with the position values will have a dtype=tf.float32.

  Returns:
    A tensor with shape [1, time * height * width, 3] that encodes positional
    IDs corresponding to each spatio-temporal position. If normalize=True, this
    tensor will have dtype=tf.float32. Otherwise, it will be dtype=tf.int32.
  """
  t, h, w = inputs.shape[1:4]
  temporal_ids, vertical_ids, horizontal_ids = tf.meshgrid(
      tf.range(t), tf.range(h), tf.range(w), indexing='ij')

  # (t, h, w, 3)
  pos_ids = tf.stack([temporal_ids, vertical_ids, horizontal_ids], axis=3)
  pos_ids = tf.reshape(pos_ids, [1, -1, 3])  # (1, t*h*w, 3)

  if normalize:
    max_positions = tf.constant([[[t, h, w]]], dtype=tf.float32)
    pos_ids = tf.cast(pos_ids, dtype=tf.float32) / max_positions

  return pos_ids


def sample_drop_idx(length,
                    drop_rate):
  """Randomly (with uniform distribution) samples dropping indices.

  Args:
    length: A positive integer indicating the fixed length of the full inputs.
    drop_rate: The dropping rate, which should be a positive number in (0, 1).

  Returns:
    A tuple of two 1D tensors containing the indices of the tokens that are
    kept/dropped after sampling.
  """
  if not 0. < drop_rate < 1.:
    raise ValueError('The drop rate should be a positive number in (0, 1). '
                     f'Instead, received {drop_rate=}.')

  if length < 1:
    raise ValueError('Length should be a positive integer. Instead, received '
                     f'{length=}.')

  max_drop_rate = 1 - 1. / length
  if drop_rate > max_drop_rate:
    raise ValueError(
        f'The configured dropping {drop_rate=} leads to full drop of the entire'
        f' tokens. This specific input contains {length} tokens. Please provide'
        f' a rate in the range of (0., {max_drop_rate}]')

  num_tokens_to_keep = int((1 - drop_rate) * length)
  token_idx = tf.range(length)
  token_idx_shuffled = tf.random.shuffle(token_idx)
  keep_idx = tf.sort(token_idx_shuffled[:num_tokens_to_keep])
  drop_idx = tf.sort(token_idx_shuffled[num_tokens_to_keep:])

  return keep_idx, drop_idx


def extend_waveform_dim(raw_audio,
                        num_windows = 1):
  """Extends the last dimension of the raw waveform as a single channel.

  Args:
    raw_audio: The tensor containing raw 1D waveform signal.
    num_windows: Represents number of windows in the original sampling method.

  Returns:
    The tensor whose last dimension is expanded to represent the channel dim.
  """
  if num_windows > 1:
    raw_audio = tf.reshape(raw_audio, [num_windows, -1])
  return tf.expand_dims(raw_audio, axis=-1)


def label_smoothing(labels,
                    alpha = 0.1):
  """Applies label smoothing to data labels.

  Args:
    labels: a tensor with classes in the last dimension.
    alpha: the smoothing rate.

  Returns:
    the smoothed labels.
  """
  num_classes = get_shape(labels)[-1]
  return (1 - alpha) * labels + alpha / num_classes


def label_normalization(labels):
  """Applies normalization to data labels.

  Args:
    labels: a tensor with classes in the last dimension.

  Returns:
    labels tensor with classes rescaled to sum to 1.
  """
  return labels / tf.reduce_sum(labels, axis=-1, keepdims=True)


def random_crop_resize(frames, output_h, output_w,
                       aspect_ratio,
                       area_range):
  """First crops clip with jittering and then resizes to (output_h, output_w).

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    output_h: Resized image height.
    output_w: Resized image width.
    aspect_ratio: Float tuple with the aspect range for cropping.
    area_range: Float tuple with the area range for cropping.

  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype.
  """
  shape = get_shape(frames)
  seq_len, _, _, num_channels = shape[0], shape[1], shape[2], shape[3]
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  factor = output_w / output_h
  aspect_ratio = (aspect_ratio[0] * factor, aspect_ratio[1] * factor)
  sample_distorted_bbox = tf.image.sample_distorted_bounding_box(
      shape[1:],
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio,
      area_range=area_range,
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bbox
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  size = tf.convert_to_tensor(
      (seq_len, target_height, target_width, num_channels))
  offset = tf.convert_to_tensor((0, offset_y, offset_x, 0))
  frames = tf.slice(frames, offset, size)
  frames = tf.cast(tf.image.resize(frames, (output_h, output_w)), frames.dtype)
  frames.set_shape((seq_len, output_h, output_w, num_channels))
  return frames


def multi_crop_image(frames, target_height,
                     target_width):
  """3 crops the image sequence of images.

  Follows 3-crop implementation introduced in https://arxiv.org/abs/1812.03982.

  Args:
    frames: A Tensor of dimension [timesteps, in_height, in_width,
      num_channels].
    target_height: Target cropped image height.
    target_width: Target cropped image width.

  Returns:
    A Tensor of shape [timesteps, out_height, out_width, num_channels] of type
    uint8
    with the cropped images.
  """
  # Three-crop evaluation.
  seq_len, height, width, num_channels = get_shape(frames)

  size = tf.convert_to_tensor(
      (seq_len, target_height, target_width, num_channels))

  offset_1 = tf.broadcast_to([0, 0, 0, 0], [4])
  # pylint:disable=g-long-lambda
  offset_2 = tf.cond(
      tf.greater_equal(height, width),
      true_fn=lambda: tf.broadcast_to(
          [0, tf.cast(height, tf.float32) / 2 - target_height // 2, 0, 0], [4]),
      false_fn=lambda: tf.broadcast_to(
          [0, 0, tf.cast(width, tf.float32) / 2 - target_width // 2, 0], [4]))
  offset_3 = tf.cond(
      tf.greater_equal(height, width),
      true_fn=lambda: tf.broadcast_to(
          [0, tf.cast(height, tf.float32) - target_height, 0, 0], [4]),
      false_fn=lambda: tf.broadcast_to(
          [0, 0, tf.cast(width, tf.float32) - target_width, 0], [4]))
  # pylint:disable=g-long-lambda

  crops = []
  for offset in [offset_1, offset_2, offset_3]:
    offset = tf.cast(tf.math.round(offset), tf.int32)
    crops.append(tf.slice(frames, offset, size))
  frames = tf.concat(crops, axis=0)

  return frames


def crop_frames(
    frames,
    crop_resize_style,
    crop_size,
    min_resize,
    min_aspect_ratio = None,
    max_aspect_ratio = None,
    min_area_ratio = None,
    max_area_ratio = None,
    random_crop = False,
    multi_crop = False,
    is_flow = False,
    state = None,
):
  """Crops the frames in the given sequence.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    crop_resize_style: The style of Crop+Resize procedure. 'Inception' or 'VGG'.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    min_resize: Frames are resized so that `min(height, width)` is `min_resize`.
    min_aspect_ratio: The minimum aspect range for cropping.
    max_aspect_ratio: The maximum aspect range for cropping.
    min_area_ratio: The minimum area range for cropping.
    max_area_ratio: The maximum area range for cropping.
    random_crop: Whether to perform random cropping or not.
    multi_crop: Whether to perform 3-view crop or not. This is only enabled in
      evaluation mode. If is_training=True, this is ignored.
    is_flow: If is flow, will modify the raw values to account for the resize.
      For example, if the flow image is resized by a factor k, we need to
      multiply the flow values by the same factor k since one pixel displacement
      in the resized image corresponds to only 1/k pixel displacement in the
      original image.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'crop_offset_proportion' as key with metadata useful for cropping.
      It will be modified with added metadata if needed. This can be used to
      keep consistency between cropping of different sequences of images.

  Returns:
    A tensor of shape [timesteps, crop_size, crop_size, channels] of same type
    as input with the cropped frames.
  """
  if random_crop:
    if crop_resize_style == constants.CropStyle.INCEPTION:
      # Inception-style image crop: random crop -> resize.
      # TODO(hassanak): add stateful support
      frames = random_crop_resize(
          frames=frames,
          output_h=crop_size,
          output_w=crop_size,
          aspect_ratio=(min_aspect_ratio, max_aspect_ratio),
          area_range=(min_area_ratio, max_area_ratio))
    else:
      # VGG-style image crop: resize -> random crop.
      frames = resize_smallest(
          frames=frames,
          min_resize=min_resize,
          is_flow=is_flow)
      frames = crop_image(
          frames=frames,
          height=crop_size,
          width=crop_size,
          random=True,
          state=state)
  else:
    # Resize images (resize happens only if necessary to save compute).
    frames = resize_smallest(
        frames=frames,
        min_resize=min_resize,
        is_flow=is_flow)
    # Crop images, either a 3-view crop or a central crop
    if multi_crop:
      # Multi crop of the frames.
      frames = multi_crop_image(
          frames=frames,
          target_height=crop_size,
          target_width=crop_size)
    else:
      # Central crop of the frames.
      frames = crop_image(
          frames=frames,
          height=crop_size,
          width=crop_size,
          random=False)

  return frames


def build_prompts_from_templates_and_labels(
    templates,
    label,
    modality_instances = None,
    quantifiers = None,
    max_sentences = None):
  """Constructs sentences from the given sentence prompt templates.

  Args:
    templates: a sequence of template sentences.
    label: the target label string.
    modality_instances: a sequence of modality candidate names ('photo',
    'drawing', etc.).
    quantifiers: a sequence of quantifier candidate names ('a', 'the', etc.).
    max_sentences: the maximum number of sentences to sample from the list.

  Returns:
    a sequence of sentences with template placeholders replaced.
  """
  max_sentences = max_sentences or len(templates)

  sentences = templates

  sentences = [
      tf.strings.regex_replace(sentence, '{label}', label)
      for sentence in sentences
  ]

  if modality_instances is not None:
    modality_name_choices = tf.random.uniform(
        [max_sentences], 0, len(modality_instances), dtype=tf.int32)
    modality_instances = tf.gather(
        modality_instances, modality_name_choices, axis=0)
    modality_instances = [
        x[0] for x in tf.split(modality_instances, max_sentences)
    ]
    sentences = [
        tf.strings.regex_replace(sentence, '{instance}', instance)
        for sentence, instance in zip(sentences, modality_instances)
    ]

  if quantifiers is not None:
    quantifier_choices = tf.random.uniform(
        [max_sentences], 0, len(quantifiers), dtype=tf.int32)
    quantifiers = tf.gather(quantifiers, quantifier_choices, axis=0)
    quantifiers = [x[0] for x in tf.split(quantifiers, max_sentences)]
    sentences = [
        tf.strings.regex_replace(sentence, '{quantifier}', quantifier)
        for sentence, quantifier in zip(sentences, quantifiers)
    ]

  sentences = [
      tf.strings.regex_replace(sentence, '"', '') for sentence in sentences
  ]

  return tf.stack(sentences)


def label_to_sentence(
    label,
    modality,
    max_num_sentences = None,
    is_training = False,
    train_image_promtps = prompts.TRAIN_IMAGE_PROMPTS,
    train_video_promtps = prompts.TRAIN_VIDEO_PROMPTS,
    train_audio_promtps = prompts.TRAIN_AUDIO_PROMPTS,
    eval_image_promtps = prompts.EVAL_IMAGE_PROMPTS,
    eval_video_promtps = prompts.EVAL_VIDEO_PROMPTS,
    eval_audio_promtps = prompts.EVAL_AUDIO_PROMPTS,
    ):
  """Constructs sentences from the given label and modality.

  Args:
    label: the target label string.
    modality: the modality that the label is describing.
    max_num_sentences: the maximum number of sentences to sample from the list.
    is_training: if True, shuffles all sentences for training.
    train_image_promtps: a list of prompt templates for train images.
    train_video_promtps: a list of prompt templates for train videos.
    train_audio_promtps: a list of prompt templates for train audio.
    eval_image_promtps: a list of prompt templates for eval images.
    eval_video_promtps: a list of prompt templates for eval videos.
    eval_audio_promtps: a list of prompt templates for eval audio.

  Returns:
    a sequence of sentences with template placeholders replaced for the given
    label and modality.
  """

  joined_label = tf.strings.reduce_join(label, separator=' and ')

  if is_training:
    if modality == Modality.IMAGE:
      sentences = build_prompts_from_templates_and_labels(
          train_image_promtps,
          joined_label,
          prompts.IMAGE_MODALITY_INSTANCES,
          prompts.QUANTIFIER_INSTANCES)
    elif modality == Modality.VIDEO:
      sentences = build_prompts_from_templates_and_labels(
          train_video_promtps,
          joined_label,
          prompts.VIDEO_MODALITY_INSTANCES,
          prompts.QUANTIFIER_INSTANCES)
    elif modality in (Modality.AUDIO, Modality.WAVEFORM, Modality.SPECTROGRAM):
      sentences = build_prompts_from_templates_and_labels(
          train_audio_promtps,
          joined_label,
          prompts.AUDIO_MODALITY_INSTANCES,
          prompts.QUANTIFIER_INSTANCES)
    else:
      raise NotImplementedError
  else:
    if modality == Modality.IMAGE:
      sentences = build_prompts_from_templates_and_labels(
          eval_image_promtps,
          joined_label)
    elif modality == Modality.VIDEO:
      sentences = build_prompts_from_templates_and_labels(
          eval_video_promtps,
          joined_label)
    elif modality in (Modality.AUDIO, Modality.WAVEFORM, Modality.SPECTROGRAM):
      sentences = build_prompts_from_templates_and_labels(
          eval_audio_promtps,
          joined_label)
    else:
      raise NotImplementedError

  if is_training:
    sentences = tf.random.shuffle(sentences)

  if max_num_sentences is not None:
    sentences = sentences[:max_num_sentences]

  return sentences


def split_by_uppercase(string):
  split = tf.strings.split(
      tf.strings.regex_replace(string, r'([A-Z])', r' \1')
      )
  joined = tf.strings.reduce_join(split, axis=-1, separator=' ')
  return joined


def strip_special_char(string, special_char):
  stripped = tf.strings.regex_replace(string, special_char, ' ')
  return stripped


def extract_context_sentences(inputs,
                              max_num_sentences):
  """Extracts and pads the context sentences from a collection of sentences.

  Args:
    inputs: A string tensor containing all sentences along its first dimension.
      This tensor is expected to have a shape of (num_all_sentences,) in which
      the first dimension contains variable-length strings.
    max_num_sentences: An integer indicating the number of sentences to be
      fetched. if 'max_num_sentences > all_available_sentences', the rest will
      be padded with empty sentences. If 'max_num_sentences <
      all_available_sentences', the first max_num_sentences will be fetched.

  Returns:
    A tensor with shape (max_num_sentences,) containing 'max_num_sentences'
    sentences.
  """
  num_available_sentences = tf.shape(input=inputs)[0]
  paddings = [[0, tf.maximum(0, max_num_sentences - num_available_sentences)]]
  sentences = tf.pad(
      tensor=inputs[:max_num_sentences],
      paddings=paddings,
      constant_values=b'')
  return sentences


def patchify_raw_rgb(inputs,
                     temporal_patch_size,
                     spatial_patch_size):
  """Tokenizes raw RGB frames by spatio-temporal patching.

  Args:
    inputs: The raw sequence of images/frames with a shape of
      (instance, time, height, width, channels). These frames will be
      patched along `time`, `height`, and `width` dimensions and stacked along
      the `channels` dimension.
    temporal_patch_size: Patch size along the temporal axis. This should be a
      single integer.
    spatial_patch_size: Patch size along the spatila axis. This is expected to
      be a tuple of two integers (height_patch, width_patch).

  Returns:
    The tokenized images/frames containing the raw pixels along the last
    dimension.
  """
  patch_sizes = (1, temporal_patch_size) + spatial_patch_size + (1,)
  tokens_raw = tf.extract_volume_patches(input=inputs,
                                         ksizes=patch_sizes,
                                         strides=patch_sizes,
                                         padding='SAME')
  return tokens_raw


def patchify_raw_waveform(inputs,
                          temporal_patch_size):
  """Tokenizes raw waveform by temporal patching.

  Args:
    inputs: The raw sequence of waveform samples with a shape of
      (instance, time, channels). These samples will be patched along the `time`
      dimension and stacked along the `channels` dimension.
    temporal_patch_size: Patch size along the temporal axis. This should be a
      single integer.

  Returns:
    The tokenized waveform signal containing the raw samples along the last
    dimension.
  """
  # Extend along the input sequence's batch dimension to use tf.image for
  # patching
  inputs = inputs[tf.newaxis, :]
  patch_sizes = (1, 1, temporal_patch_size, 1)
  tokens_raw = tf.image.extract_patches(images=inputs,
                                        sizes=patch_sizes,
                                        strides=patch_sizes,
                                        rates=[1, 1, 1, 1],
                                        padding='SAME')
  # Reverse the input sequence's batch extension
  tokens_raw = tokens_raw[0, :]
  return tokens_raw


def patchify_raw_spectrogram(inputs,
                             temporal_patch_size,
                             spectoral_patch_size):
  """Tokenizes raw spectrogram by spectro-temporal patching.

  Args:
    inputs: The raw sequence of waveform samples with a shape of
      (instance, time, channels). These samples will be patched along the `time`
      dimension and stacked along the `channels` dimension.
    temporal_patch_size: Patch size along the temporal axis. This should be a
      single integer.
    spectoral_patch_size: Patch size along the spectoral axis. This should be a
      single integer.

  Returns:
    The tokenized waveform signal containing the raw samples along the last
    dimension.
  """
  patch_sizes = (1, temporal_patch_size, spectoral_patch_size, 1)
  tokens_raw = tf.image.extract_patches(images=inputs,
                                        sizes=patch_sizes,
                                        strides=patch_sizes,
                                        rates=[1, 1, 1, 1],
                                        padding='SAME')
  return tokens_raw


def compute_audio_spectrogram(
    waveform,
    sample_rate = 48000,
    spectrogram_type = 'logmf',
    frame_length = 2048,
    frame_step = 1024,
    num_features = 80,
    lower_edge_hertz = 80.0,
    upper_edge_hertz = 7600.0,
    preemphasis = None,
    normalize = False,
    squared_magnitude = False,
    ):
  """Computes audio spectrograms.

  Args:
    waveform: The raw waveform with shape [instance, samples, channels].
    sample_rate: The sample rate of the input audio.
    spectrogram_type: The type of the spectrogram to be extracted from the
      waveform. Can be either `spectrogram`, `logmf`, and `mfcc`.
    frame_length: The length of each spectroram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    lower_edge_hertz: Lowest frequency to consider.
    upper_edge_hertz: Highest frequency to consider.
    preemphasis: The strength of pre-emphasis on the waveform. If None, no
      pre-emphasis will be applied.
    normalize: Whether to normalize the waveform or not.
    squared_magnitude: Whether to output the squared magnitude of the fft.

  Returns:
    The spectrogram with shape [instance, samples, spectrum, channels].

  Raises:
    ValueError: if `spectrogram_type` is one of `spectrogram`, `logmf`, or
      `mfcc`.
  """
  if spectrogram_type not in ['spectrogram', 'logmf', 'mfcc']:
    raise ValueError('Spectrogram type should be one of `spectrogram`, '
                     f'`logmf`, or `mfcc`, got {spectrogram_type}')

  waveform = tf.cast(waveform, dtype=tf.float32)
  if normalize:
    waveform /= (
        tf.reduce_max(tf.abs(waveform), axis=-2, keepdims=True) + 1e-8)

  if preemphasis is not None:
    waveform = tf.concat(
        [waveform[:1], waveform[1:] - preemphasis * waveform[:-1]], axis=0)

  def _extract_spectrogram(
      waveform,
      spectrogram_type):
    stfts = tf.signal.stft(waveform,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           fft_length=frame_length,
                           window_fn=tf.signal.hann_window,
                           pad_end=True)
    if squared_magnitude:
      stfts = tf.square(stfts)
    spectrograms = tf.abs(stfts)

    if spectrogram_type == 'spectrogram':
      return spectrograms[Ellipsis, :num_features]

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_features, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    if spectrogram_type == 'logmf':
      return log_mel_spectrograms

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[Ellipsis, :13]
    return mfccs

  # Perform per-channel spectrogram
  waveform_channels = tf.unstack(
      waveform, num=waveform.shape[-1], axis=-1)
  spectrogram_fn = functools.partial(
      _extract_spectrogram, spectrogram_type=spectrogram_type)
  spectrogram_split = [spectrogram_fn(wav) for wav in waveform_channels]
  spectrogram = tf.stack(spectrogram_split, axis=-1)
  return spectrogram


def construct_padding_mask(inputs, pad_token_id):
  """Constructs a padding mask in locations where input contains pad token."""
  return tf.where(inputs == pad_token_id, 0, 1)


def create_label_map_table(
    file_path,
    num_labels = None):
  """Creates a StaticHashTable mapping a key to a label.

  Args:
    file_path: the path to a .json file with either string keys and values or
      all integer keys in ascending order for index remapping.
    num_labels: optional expected number of labels.

  Returns:
    a table mapping string keys to string labels.
  """
  with tf.io.gfile.GFile(file_path) as f:
    mapping = json.loads(f.read())

  keys = list(mapping.keys())
  labels = list(mapping.values())

  if num_labels is not None and len(labels) != num_labels:
    raise ValueError(
        f'Actual number of labels {len(labels)} in file {file_path}  does not '
        f'match expected labels {num_labels}')

  try:
    # json assumes string keys, so if all keys can be parsed as integers, then
    # this can be safely handled as integer keys.
    keys = [int(key) for key in keys]
    keys_dtype = tf.int32
  except ValueError:
    keys_dtype = tf.string

  if isinstance(labels[0], int):
    labels_dtype = tf.int32
    default_value = 0
  else:
    labels_dtype = tf.string
    default_value = 'Unknown'

  keys_tensor = tf.constant(keys, dtype=keys_dtype)
  labels_tensor = tf.constant(labels, dtype=labels_dtype)

  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, labels_tensor),
      default_value)


def label_id_to_name_map(
    label_id,
    label_map):
  """Performs a label map table lookup on label index keys.

  Args:
    label_id: A tensor containing the label indices.
    label_map: A table mapping integers to label strings.

  Returns:
    The string label names.
  """
  label_id = tf.cast(tf.argmax(label_id, axis=-1), dtype=tf.int32)
  return label_map.lookup(label_id)


def label_name_to_name_remap(
    label_name,
    label_map):
  """Performs a label map table lookup on label names.

  Args:
    label_name: A tensor containing the label names (or mid strings).
    label_map: A table re-mapping string to strings.

  Returns:
    The string label names.
  """
  return label_map.lookup(label_name)


# ----------------------------------------------------------------------
# ---------------- Dictionary-level Processing functions ---------------
# ----------------------------------------------------------------------
def apply_fn_on_features_dict(fn,
                              feature_name):
  """Transforms a function to be applied on features dictionary."""
  def transformed_fn(features_dict, *args, **kwargs):
    features_dict[feature_name] = fn(
        features_dict[feature_name], *args, **kwargs)
    return features_dict
  return transformed_fn


def batched_mixup(features_dict,
                  feature_name,
                  label_feature_name,
                  alpha = 5,
                  beta = 2,
                  mixup_labels = True):
  """Mixup processing function as in https://arxiv.org/pdf/1710.09412.pdf.

  Args:
    features_dict: The single-level dictionary of batched features.
    feature_name: The key for the feature-of-interest to be mixed up.
    label_feature_name: The key for the labels corresponding to the mixed-up
      feature. If `mixup_labels` is True, the corresponding labels with this
      key will be mixed up accordingly.
    alpha: The alpha parameter in the Beta distribution. See the corresponding
      documentations in `tfp.distributions.Beta` for more details.
    beta: The beta parameter in the Beta distribution. See the corresponding
      documentations in `tfp.distributions.Beta` for more details.
    mixup_labels: If True, the corresponding labels will be mixed up too.

  Returns:
    The single-level dictionary of batched features, whose tensors corresponding
    to `feature_name` and `label_feature_name` are mixed up along their batch
    dimension using indices sampled from a Beta distribution with parameters
    `alpha` and `beta`.

  Raises:
    KeyError: if either of `feature_name` or `label_feature_name` do not exist
      in `features_dict`.
  """

  # get features
  features = features_dict[feature_name]
  feature_shape = get_shape(features)
  batch_size = feature_shape[0]
  # flatten features to a 1d signal
  features = tf.reshape(features, [batch_size, -1])  # (bs, seq_len)
  seq_len = get_shape(features)[1]

  # create random indices to fetch random samples
  seq_idx = tf.range(seq_len)[None, :]
  seq_idx = tf.tile(seq_idx, [batch_size, 1])
  batch_idx = tf.range(batch_size)
  batch_idx = tf.random.shuffle(batch_idx)[:, None]
  batch_idx_ext = tf.tile(batch_idx, [1, seq_len])
  gather_idx = tf.stack([batch_idx_ext, seq_idx], axis=2)
  shuffled_features = tf.gather_nd(features, gather_idx)  # (bs, seq_len)

  # sample lambda per-sample
  beta_dist = tfp.distributions.Beta(alpha, beta)
  lmbda = beta_dist.sample([batch_size, 1])  # (bs, 1)
  # mixup signals
  mixed_up_features = lmbda * features + (1 - lmbda) * shuffled_features
  features_dict[feature_name] = tf.reshape(mixed_up_features, feature_shape)

  if mixup_labels:
    if label_feature_name not in features_dict:
      raise KeyError(
          f'{label_feature_name} not available, but label mixup requested.')

    labels = features_dict[label_feature_name]
    shuffled_labels = tf.gather_nd(labels, batch_idx)  # (bs, n_class)
    mixed_up_labels = lmbda * labels + (1 - lmbda) * shuffled_labels
    features_dict[label_feature_name] = mixed_up_labels

  return features_dict


def tokenize_raw_rgb(
    features_dict,
    raw_feature_name,
    token_raw_feature_name,
    token_coordinate_feature_name,
    token_position_id_feature_name,
    temporal_patch_size,
    spatial_patch_size,
    spatio_temporal_token_coordinate,
):
  """Tokenizes the raw RGB pixels by patchifying.

  Args:
    features_dict: The single-level dictionary of batched features.
    raw_feature_name: The key corresponding to the raw features in the features
      dictionary.
    token_raw_feature_name: The key corresponding to the tokenized raw features
      in the features dictionary.
    token_coordinate_feature_name: The key corresponding to the coordinate of
      the tokenized features in the features dictionary.
    token_position_id_feature_name: The key corresponding to the position id of
      the tokenized features in the features dictionary.
    temporal_patch_size: Patch size along the temporal axis. This should be a
      single integer.
    spatial_patch_size: Patch size along the spatial axis. This is expected to
      be a tuple of two integers (height_patch, width_patch).
    spatio_temporal_token_coordinate: Whether the token coordinates are in a
      3D spatio-temporal space or a 1D flattened space.

  Returns:
    The features dictionary containing the tokens and their corresponding
    metadata.
  """
  raw_rgb = features_dict[raw_feature_name]
  if temporal_patch_size > raw_rgb.shape[1]:
    # replicate along the temporal axis to avoid zero-padding when patching
    raw_rgb = tf.tile(raw_rgb, [1, temporal_patch_size, 1, 1, 1])

  patched_rgb = patchify_raw_rgb(
      inputs=raw_rgb,
      temporal_patch_size=temporal_patch_size,
      spatial_patch_size=spatial_patch_size)

  # Flatten all tokens
  flattened_patched_rgb = tf.reshape(
      patched_rgb, (patched_rgb.shape[0], -1, patched_rgb.shape[-1]))
  features_dict[token_raw_feature_name] = flattened_patched_rgb

  if spatio_temporal_token_coordinate:
    # Extract spatio-temporal positional encoding
    patch_3d_coordinates = construct_3d_positions(patched_rgb, normalize=True)
    features_dict[token_coordinate_feature_name] = patch_3d_coordinates
  else:
    # Extract flattened positional encoding
    patch_1d_coordinates = construct_1d_positions(flattened_patched_rgb,
                                                  normalize=True)
    features_dict[token_coordinate_feature_name] = patch_1d_coordinates

  # Add token ID
  patch_1d_ids = construct_1d_positions(flattened_patched_rgb, normalize=False)
  features_dict[token_position_id_feature_name] = patch_1d_ids

  return features_dict


def tokenize_raw_waveform(
    features_dict,
    raw_feature_name,
    token_raw_feature_name,
    token_coordinate_feature_name,
    token_position_id_feature_name,
    temporal_patch_size,
):
  """Tokenizes the raw waveform samples by patchifying.

  Args:
    features_dict: The single-level dictionary of batched features.
    raw_feature_name: The key corresponding to the raw features in the features
      dictionary.
    token_raw_feature_name: The key corresponding to the tokenized raw features
      in the features dictionary.
    token_coordinate_feature_name: The key corresponding to the coordinate of
      the tokenized features in the features dictionary.
    token_position_id_feature_name: The key corresponding to the position id of
      the tokenized features in the features dictionary.
    temporal_patch_size: Patch size along the temporal axis. This should be a
      single integer.

  Returns:
    The features dictionary containing the tokens and their corresponding
    metadata.
  """
  # Patch the raw waveform
  raw_waveform = features_dict[raw_feature_name]
  patched_waveform = patchify_raw_waveform(raw_waveform, temporal_patch_size)
  features_dict[token_raw_feature_name] = patched_waveform

  # Extract positional encoding
  patch_1d_coordinates = construct_1d_positions(patched_waveform,
                                                normalize=True)
  features_dict[token_coordinate_feature_name] = patch_1d_coordinates

  # Add token position ID
  patch_1d_ids = construct_1d_positions(patched_waveform, normalize=False)
  features_dict[token_position_id_feature_name] = patch_1d_ids

  return features_dict


def tokenize_raw_spectrogram(
    features_dict,
    raw_feature_name,
    token_raw_feature_name,
    token_coordinate_feature_name,
    token_position_id_feature_name,
    temporal_patch_size,
    spectoral_patch_size
):
  """Tokenizes the raw spectrogram features by patchifying.

  Args:
    features_dict: The single-level dictionary of batched features.
    raw_feature_name: The key corresponding to the raw features in the features
      dictionary.
    token_raw_feature_name: The key corresponding to the tokenized raw features
      in the features dictionary.
    token_coordinate_feature_name: The key corresponding to the coordinate of
      the tokenized features in the features dictionary.
    token_position_id_feature_name: The key corresponding to the position id of
      the tokenized features in the features dictionary.
    temporal_patch_size: Patch size along the temporal axis. This should be a
      single integer.
    spectoral_patch_size: Patch size along the spectoral axis. This should be a
      single integer.

  Returns:
    The features dictionary containing the tokens and their corresponding
    metadata.
  """
  # Patch the raw spectrogram
  raw_spectrogram = features_dict[raw_feature_name]
  patched_spectrogram = patchify_raw_spectrogram(
      inputs=raw_spectrogram,
      temporal_patch_size=temporal_patch_size,
      spectoral_patch_size=spectoral_patch_size)

  # Extract positional encoding
  patch_2d_coordinates = construct_2d_positions(patched_spectrogram,
                                                normalize=True)
  features_dict[token_coordinate_feature_name] = patch_2d_coordinates

  # Flatten all tokens
  flattened_patched_spectrogram = tf.reshape(
      patched_spectrogram,
      (patched_spectrogram.shape[0], -1, patched_spectrogram.shape[-1]))
  features_dict[token_raw_feature_name] = flattened_patched_spectrogram

  # Add token ID
  patch_2d_ids = construct_2d_positions(flattened_patched_spectrogram,
                                        normalize=False)
  features_dict[token_position_id_feature_name] = patch_2d_ids

  return features_dict


def tokenize_raw_string(
    features_dict,
    tokenizer,
    raw_feature_name,
    token_id_feature_name,
    token_coordinate_feature_name,
    token_position_id_feature_name,
    token_mask_feature_name,
    keep_raw_string,
    prepend_bos,
    append_eos,
    max_num_tokens,
    max_num_sentences,
):
  """Tokenizes the raw spectrogram features by patchifying.

  Args:
    features_dict: The single-level dictionary of batched features.
    tokenizer: Either a string containing the tokenizer name, or an instance
      of the tokenizer. If a string is provided and the requested tokenizer is
      supported, a tokenizer is instantiated and initialized upon calling this
      method. Hence, if this function is intended to be called multiple times
      upon the graph construction, it is better to instantiate the tokenizer
      outside the function and pass it on to the functions.
    raw_feature_name: The key corresponding to the raw features in the features
      dictionary.
    token_id_feature_name: The key corresponding to the tokenized discrete
      features in the features dictionary.
    token_coordinate_feature_name: The key corresponding to the coordinate of
      the tokenized features in the features dictionary.
    token_position_id_feature_name: The key corresponding to the position id of
      the tokenized features in the features dictionary.
    token_mask_feature_name: The key corresponding to the 0/1 mask of the
      tokenized features in the features dictionary.
    keep_raw_string: Whether to keep the raw string after tokenization.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    max_num_tokens: Maximum number of tokens to keep from the tokenized text.
    max_num_sentences: Maximum number of the expected text instances.

  Returns:
    The features dictionary containing the tokens and their corresponding
    metadata.
  """
  if isinstance(tokenizer, str):
    # Fetch and initialize the tokenizer
    tokenizer = tokenizers.get_tokenizer(tokenizer)
    tokenizer.initialize()

  # Tokenize the sentence.
  features_dict = tokenize(
      features=features_dict,
      tokenizer=tokenizer,
      raw_string_name=raw_feature_name,
      tokenized_name=token_id_feature_name,
      prepend_bos=prepend_bos,
      append_eos=append_eos,
      max_num_tokens=max_num_tokens,
      keep_raw_string=keep_raw_string,
  )

  # Pad or crop words to max_num_words.
  features_dict[token_id_feature_name] = tokenizers.crop_or_pad_words(
      words=features_dict[token_id_feature_name],
      max_num_words=max_num_tokens,
      pad_value=tokenizer.pad_token)

  # Set text shape.
  features_dict[token_id_feature_name] = set_shape(
      inputs=features_dict[token_id_feature_name],
      shape=(max_num_sentences, max_num_tokens))

  # Add padding mask for the padded locations
  features_dict[token_mask_feature_name] = construct_padding_mask(
      inputs=features_dict[token_id_feature_name],
      pad_token_id=tokenizer.pad_token)

  # Extract positional encoding
  features_dict[token_coordinate_feature_name] = construct_1d_positions(
      inputs=features_dict[token_id_feature_name],
      normalize=True)

  # Add token position ID
  features_dict[token_position_id_feature_name] = construct_1d_positions(
      inputs=features_dict[token_id_feature_name],
      normalize=False)

  return features_dict


def add_drop_token(
    features_dict,
    drop_rate,
    token_feature_name,
    token_coordinate_feature_name,
    drop_coordinate_feature_name,
    token_position_id_feature_name,
    drop_position_id_feature_name,
):
  """Applies DropToken to a token sequence (https://arxiv.org/abs/2104.11178).

  It is assumed that ALL features in the features_dict are pre-batch features,
  which indicates that axis=1 contains the `length` axis.

  Args:
    features_dict: The mutable dictionary of features in the input processing
      pipeline.
    drop_rate: A float in range (0, 1), indicating the rate of which tokens are
      dropped with.
    token_feature_name: The key for the tokens in the inputs dictionary.
    token_coordinate_feature_name: The key for the position of tokens in the
      original space from which the raw inputs where tokenized.
    drop_coordinate_feature_name: The key for the position of tokens in that
      were dropped as the outcome of DropToken. This position corresponds with
      the original space from which the raw inputs where tokenized.
    token_position_id_feature_name: The key for the absolute position of the
      tokens in the flattened form.
    drop_position_id_feature_name: The key for the absolute position of the
      dropped tokens in the flattened form.

  Returns:
    The features dictionary updated with the truncated tokens, their positions,
    and the positions where DropToken has been applied.
  """

  # Fetch tokens and their positions (in the original space)
  tokens = features_dict[token_feature_name]
  token_coordinates = features_dict[token_coordinate_feature_name]

  # Sample drop IDs
  length = tokens.shape[1]
  keep_idx, drop_idx = sample_drop_idx(length, drop_rate)

  # Fetch tokens and their positions according to drop IDs
  truncated_tokens = tf.gather(tokens, keep_idx, axis=1)
  truncated_token_coordinates = tf.gather(token_coordinates, keep_idx, axis=1)
  drop_coordinates = tf.gather(token_coordinates, drop_idx, axis=1)

  # Update the features dictionary
  features_dict[token_feature_name] = truncated_tokens
  features_dict[token_coordinate_feature_name] = truncated_token_coordinates
  features_dict[drop_coordinate_feature_name] = drop_coordinates
  features_dict[token_position_id_feature_name] = keep_idx[tf.newaxis, :]
  features_dict[drop_position_id_feature_name] = drop_idx[tf.newaxis, :]
  return features_dict


def add_padding_mask(
    features_dict,
    feature_name,
    mask_feature_name,
    pad_value = 0.0,
    reduction_axis = None,
    dtype = tf.float32,
):
  """Constructs and adds padding mask depending on the input norm/value."""
  features = features_dict[feature_name]
  if reduction_axis is not None:
    features = tf.norm(features, axis=reduction_axis)
  padding_mask = tf.cast(tf.where(features == pad_value, 0, 1), dtype=dtype)
  features_dict[mask_feature_name] = padding_mask
  return features_dict


def stochastic_token_id_masking(
    features_dict,
    token_id_feature_name,
    token_mask_feature_name,
    token_mask_id,
    mask_rate,
    stochastic_rate = False,
    alpha = 5.0,
    beta = 2.0):
  """Randomly masks tokens along the temporal axis.

  This function also supports stochastic rates. More specifically, rates are
  sampled from a Beta distribution (if stochastic_rate = True), resulting in
  dynamic-rate masking.

  Args:
    features_dict: The mutable dictionary of features in the input processing
      pipeline containing the tokens-of-interest.
    token_id_feature_name: The key corresponding to the tokenized discrete
      features in the features dictionary.
    token_mask_feature_name: The key corresponding to the 0/1 mask of the
      tokenized features in the features dictionary.
    token_mask_id: The masking token_id that replaces the original token_ids.
    mask_rate: The masking rate, which should be a positive number in (0, 1).
    stochastic_rate: If True, the effective maksing rate would be sampled
      from a Beta distribution with `Mode=rate * Mode(Beta)`.
    alpha: The `Alpha` parameter in the Beta distibution.
    beta: The `Beta` parameter in the Beta distribution.

  Returns:
    The features dictionary updated with the masked tokens and the corresponding
    0/1 mask.
  """
  if not 0. < mask_rate < 1.:
    raise ValueError('The masking rate should be a positive number in (0, 1). '
                     f'Instead, received {mask_rate=}.')

  # Fetch token_ids
  token_ids = features_dict[token_id_feature_name]
  length = token_ids.shape[1]

  # Perform verification checks
  if length < 1:
    raise ValueError('Length should be a positive integer. Instead, received '
                     f'{length=}.')

  max_mask_rate = 1 - 1. / length
  if mask_rate > max_mask_rate:
    raise ValueError(
        f'The configured {mask_rate=} leads to full masking of the entire'
        f' tokens. This specific input contains {length} tokens. Please provide'
        f' a rate in the range of (0., {max_mask_rate}]')

  mask_rate = tf.convert_to_tensor(mask_rate, dtype=tf.float32)
  if stochastic_rate:
    allowed_config_msg = (
        'Either use `alpha > 1 & beta > 1` to yield a semi-bell-shape PDF or '
        'use `alpha > 1 & beta <= 1 to yield a Mode-1 Bimodial PDF.')
    if alpha <= 1 and beta > 1:
      raise ValueError(
          f'The configured {alpha=} and {beta=} results in a Mode-0 Bimodial '
          f'distribution. {allowed_config_msg}')
    if alpha == 1 and beta == 1:
      raise ValueError(
          f'The configured {alpha=} and {beta=} results in a Uniform '
          f'distribution. {allowed_config_msg}')
    mask_rate_multiplier = tfp.distributions.Beta(alpha, beta).sample(())
    mask_rate *= mask_rate_multiplier

  num_mask_indices = tf.cast(length * mask_rate, dtype=token_ids.dtype)
  mask_indices = tf.random.experimental.index_shuffle(
      index=tf.range(num_mask_indices),
      seed=tf.random.get_global_generator().make_seeds()[:, 0],
      max_index=length-1)
  mask = tf.scatter_nd(
      mask_indices[:, tf.newaxis],
      updates=tf.ones(shape=(num_mask_indices,), dtype=token_ids.dtype),
      shape=(length,))

  token_mask = tf.tile(mask[tf.newaxis, :], [token_ids.shape[0], 1])
  token_ids = (1 - token_mask) * token_ids + token_mask * token_mask_id
  features_dict[token_mask_feature_name] = token_mask
  features_dict[token_id_feature_name] = token_ids
  return features_dict


def integrate_context_sentences(
    features_dict,
    raw_feature_name,
    context_raw_feature_name):
  """Fetches raw main and context sentences and concatenates them.

  Args:
    features_dict: The mutable dictionary of features in the input processing
      pipeline containing the sentences-of-interest.
    raw_feature_name: The key corresponding to the raw string features in the
      features dictionary.
    context_raw_feature_name: The key corresponding to the raw context string
      features in the features dictionary.

  Returns:
    The dictionary of features with only the corresponding features of
    `raw_feature_name` that contains a concatenation of the main raw sentences
    and the context sentences. The features corresponding with the key
    `context_raw_feature_name` are removed.
  """
  features_dict[raw_feature_name] = tf.concat([
      features_dict[raw_feature_name], features_dict[context_raw_feature_name]
  ], axis=0)
  del features_dict[context_raw_feature_name]
  return features_dict


def remove_key(features_dict,
               key):
  """Removes an arbitrary key from the dictionary of features."""
  del features_dict[key]
  return features_dict


def remove_key_with_prefix(
    features_dict,
    key_prefix):
  """Removes all keys with certain prefix from the dictionary of features."""
  feature_keys = list(features_dict.keys())
  for key in feature_keys:
    if key.startswith(key_prefix):
      del features_dict[key]
  return features_dict


def add_new_feature(features_dict,
                    feature,
                    feature_name):
  """Adds a new feature with the given key name to the features dictionary."""
  features_dict.update({feature_name: feature})
  return features_dict


def copy_feature(features_dict,
                 source_feature_name,
                 target_feature_name):
  """Copies features of a key over to another key in the features dictionary."""
  source_feature = features_dict[source_feature_name]
  features_dict.update({target_feature_name: source_feature})
  return features_dict


def unflatten_features_dict(features_dict,
                            sep = '/'):
  """Unflattens the features dictionary.

  By default, multiple modalities create features with a unique prefix. For
  example, all features in the pre-processing stage for the vision modality
  are represented as 'vision/FEATURE_NAME'. This function unflattens them.

  Args:
    features_dict: The flattened single-level dictionary of features.
    sep: A string separtor for unflattening the dictionary.

  Returns:
    The unflattened two-level features_dict.
  """
  return traverse_util.unflatten_dict(features_dict, sep=sep)


# ----------------------------------------------------------------------
# ------ Tools for integrating reading, decoding and processing --------
# ----------------------------------------------------------------------
def parse_features(
    parser_builder,
    parsing_feature_name,
    parsed_feature_name,
    feature_type,
    dtype,
    shape = None,
    is_context = None,
    **kwargs,
):
  """Generic parsing method for parsing any type of acceptable features.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    parsed_feature_name: Name of the feature in the dictionary which contains
      all parsed features.
    feature_type: The expected type of the features to be parsed. Acceptable
      feature types are `tf.io.VarLenFeature`, `tf.io.FixedLenFeature`, and
      `tf.io.FixedLenSequenceFeature`.
    dtype: The dtype of the parsed features.
    shape: The optional shape of the parsed features. Only applicable if the
      features have fixed length.
    is_context: Whether the parsed feature is a context feature. This is used
      for auxiliary features along with the main features-of-interest.
    **kwargs: Catch irrelevant builders.
  """
  del kwargs
  extra_kwargs = {}
  if feature_type in (tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature):
    feature_type = functools.partial(feature_type, shape=shape)
  if is_context is not None:
    # Some parser builders accept `is_context` arg
    extra_kwargs['is_context'] = is_context

  parser_builder.parse_feature(
      feature_name=parsing_feature_name,
      feature_type=feature_type(dtype=dtype),
      output_name=parsed_feature_name,
      **extra_kwargs)


def sample_sequence_features(
    sampler_builder,
    feature_name,
    num_samples,
    stride = 1,
    linspace_size = 1,
    random_sampling = False,
    **kwargs,
):
  """Generic sampling method for sequence features.

  Args:
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    feature_name: Name of the feature in the dictionary which contains
      all parsed features.
    num_samples: Number of features to be sampled.
    stride: The stride with which the features are sampled.
    linspace_size: If larger than 1, the sampling span would be divided to
      linear portions with this size.
    random_sampling: Whether random sampling should be applied.
    **kwargs: Catch irrelevant builders.
  """
  del kwargs
  if random_sampling and linspace_size > 1:
    logging.info('Linear sampling is ignored since `random_sampling` is true.')

  if random_sampling:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: sample_sequence(
            x, num_samples, True, stride, state=s),
        # pylint: enable=g-long-lambda
        feature_name=feature_name,
        fn_name=f'{feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=True)
  else:
    if linspace_size > 1:
      # Sample linspace clips.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: sample_linspace_sequence(x, linspace_size, num_samples,
                                                stride),
          # pylint: enable=g-long-lambda
          feature_name=feature_name,
          fn_name=f'{feature_name}_linspace_sample')
    else:
      # Sample middle clip.
      sampler_builder.add_fn(
          fn=lambda x: sample_sequence(x, num_samples, False, stride),
          feature_name=feature_name,
          fn_name=f'{feature_name}_middle_sample')


def decode_vision_features(
    decoder_builder,
    feature_name,
    is_rgb = True,
    is_flow = False,
    **kwargs,
):
  """Decoding method for raw vision features.

  Args:
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    feature_name: Name of the feature in the dictionary which contains
      all sampled features.
    is_rgb: If `True`, the number of channels in the image is 3, if False, 1. If
      is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    **kwargs: Catch irrelevant builders.
  """
  del kwargs
  # Decode image string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the pre
  # Hence the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: decode_image(x, channels=num_raw_channels),
      feature_name=feature_name,
      fn_name=f'{feature_name}_decode_image')


def decode_sparse_features(
    decoder_builder,
    feature_name,
    **kwargs,
):
  """Decoding method for raw text features.

  Args:
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    feature_name: Name of the feature in the dictionary which contains
      all sampled features.
    **kwargs: Catch irrelevant builders.
  """
  del kwargs
  # Densify text tensor.
  decoder_builder.add_fn(
      maybe_densify_tensor,
      feature_name=feature_name,
      fn_name=f'{feature_name}_sparse_to_dense')


def add_vision(
    parser_builder,
    sampler_builder,
    decoder_builder,
    preprocessor_builder,
    postprocessor_builder,
    parsing_feature_name = ParsingFeatureName.IMAGE_ENCODED,
    data_collection_type = DataFeatureType.INPUTS,
    data_collection_route = DataFeatureRoute.ENCODER,
    keep_raw_rgb = False,
    is_training = True,
    num_frames = 32,
    stride = 1,
    num_test_clips = 1,
    min_resize = 224,
    crop_size = 200,
    multi_crop = False,
    spatial_patch_size = (1, 1),
    temporal_patch_size = 1,
    spatio_temporal_token_coordinate = True,
    zero_centering_image = False,
    crop_resize_style = constants.CropStyle.INCEPTION,
    min_aspect_ratio = 0.5,
    max_aspect_ratio = 2,
    min_area_ratio = 0.5,
    max_area_ratio = 1.0,
    color_augmentation = False,
    random_flip = False,
    sync_random_state = True,
    is_rgb = True,
    is_flow = False,
    token_drop_rate = 0.,
    dtype = 'float32',
    **kwargs):
  """Custom vision reader & processor based on DMVR logic.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_rgb: Whether to keep raw RGB pixels.
    is_training: Whether or not in training mode. If `True`, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip. For single images, use 1.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that `min(height, width)` is `min_resize`.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    multi_crop: Whether to perform 3-view crop or not. This is only enabled in
      evaluation mode. If is_training=True, this is ignored.
    spatial_patch_size: Pixels are sampled in `spatial_patch_size[0] x
      spatial_patch_size[1]` regions
    temporal_patch_size: Frames are sampled in temporal_patch_size regions
    spatio_temporal_token_coordinate: If True, the token coordinate associated
      with the patches will be 3D. Otherwise, it will be the coordinate
      associated with the flattened patches.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    crop_resize_style: The style of Crop+Resize procedure. 'Inception' or 'VGG'.
    min_aspect_ratio: The minimum aspect range for cropping.
    max_aspect_ratio: The maximum aspect range for cropping.
    min_area_ratio: The minimum area range for cropping.
    max_area_ratio: The maximum area range for cropping.
    color_augmentation: Whether to jitter color or not.
    random_flip: Whether to apply random left/right flips.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
    is_rgb: If `True`, the number of channels in the image is 3, if False, 1. If
      is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    token_drop_rate: The rate of which tokens are dropped.
    dtype: the dtype to cast the output to after batching.
    **kwargs: Catch irrelevant builders.

  Returns:
    tuple of added modality names.
  """
  del kwargs

  # Validate parameters.
  if is_training is None:
    raise AttributeError('`is_training` was not propagagted properly.')

  if is_flow and is_rgb is not None:
    raise ValueError('`is_rgb` should be `None` when requesting flow.')

  if is_flow and not zero_centering_image:
    raise ValueError('Flow contains displacement values that can be negative, '
                     'but `zero_centering_image` was set to `False`.')

  is_patching = any([
      p > 1 for p in spatial_patch_size + (temporal_patch_size,)])
  if not keep_raw_rgb and not is_patching:
    raise ValueError('Either `keep_raw_rgb` should be True or patching '
                     'should be specified with non-one patch sizes. Instead, '
                     f'{keep_raw_rgb=}, {spatial_patch_size=}, '
                     f'{temporal_patch_size=}.')

  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)

  get_feature_name = functools.partial(get_flattened_key,
                                       ftype=data_collection_type,
                                       froute=data_collection_route,
                                       modality=Modality.VISION)
  raw_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  token_raw_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_RAW)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  drop_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.DROP_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)
  drop_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.DROP_POSITION_ID)

  # Parse frames or single image.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parse_features(
        parser_builder=parser_builder,
        parsing_feature_name=parsing_feature_name,
        parsed_feature_name=raw_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature,
        dtype=tf.string,
        shape=())
  elif (
      isinstance(parser_builder, builders.ExampleParserBuilder)
  ):
    parse_features(
        parser_builder=parser_builder,
        parsing_feature_name=parsing_feature_name,
        parsed_feature_name=raw_feature_name,
        feature_type=tf.io.FixedLenFeature,
        dtype=tf.string,
        shape=())
    # Expand dimensions so single images have the same structure as videos.
    sampler_builder.add_fn(
        fn=lambda x: tf.expand_dims(x, axis=0),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_expand_dims')
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  sample_sequence_features(
      sampler_builder=sampler_builder,
      feature_name=raw_feature_name,
      num_samples=num_frames,
      stride=stride,
      linspace_size=num_test_clips,
      random_sampling=is_training)

  decode_vision_features(
      decoder_builder=decoder_builder,
      feature_name=raw_feature_name,
      is_rgb=is_rgb,
      is_flow=is_flow)

  if is_flow:
    # Cast the flow to `tf.float32`, normalizing between [-1.0, 1.0].
    preprocessor_builder.add_fn(
        fn=lambda x: normalize_image(x, zero_centering_image=True),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_normalize')

  preprocessor_builder.add_fn(
      fn=lambda x, s=None: crop_frames(  # pylint: disable=g-long-lambda
          x, state=s, crop_resize_style=crop_resize_style, crop_size=crop_size,
          min_aspect_ratio=min_aspect_ratio, max_aspect_ratio=max_aspect_ratio,
          min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
          min_resize=min_resize, random_crop=is_training, multi_crop=multi_crop,
          is_flow=is_flow),
      feature_name=raw_feature_name,
      fn_name=f'{raw_feature_name}_crop_frames',
      # Use state to keep coherence between modalities if requested.
      stateful=sync_random_state,
      )

  # Random flip
  if is_training and random_flip:
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: random_flip_left_right(
            x, state=s, is_flow=is_flow),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_random_flip',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)

  if is_flow:
    # Keep only two channels for the flow: horizontal and vertical displacement.
    preprocessor_builder.add_fn(
        fn=lambda x: x[:, :, :, :2],
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_extract_flow_channels')

    # Clip the flow to stay between [-1.0 and 1.0]
    preprocessor_builder.add_fn(
        fn=lambda x: tf.clip_by_value(x, -1.0, 1.0),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_clip_flow')
  else:
    # Cast the frames to `tf.float32`, normalizing according to
    # `zero_centering_image`.
    preprocessor_builder.add_fn(
        fn=lambda x: normalize_image(x, zero_centering_image),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_normalize')

    # Also apply color augmentation (if specified)
    if color_augmentation and is_rgb and is_training:
      # Random color jitter
      preprocessor_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x, s=None: color_default_augm(
              x,
              zero_centering_image=zero_centering_image,
              prob_color_augment=0.8,
              prob_color_drop=0.0,
          ),
          feature_name=raw_feature_name,
          fn_name=f'{raw_feature_name}_color_jitter',
      )

  # accumulate all sampled clips along the first dimension
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
          x, (-1, num_frames, x.shape[1], x.shape[2], x.shape[3])),
      feature_name=raw_feature_name,
      fn_name=f'{raw_feature_name}_reshape')

  if is_patching:
    preprocessor_builder.add_fn(
        fn=lambda x: tokenize_raw_rgb(  # pylint: disable=g-long-lambda
            x, raw_feature_name=raw_feature_name,
            token_raw_feature_name=token_raw_feature_name,
            token_coordinate_feature_name=token_coordinate_feature_name,
            token_position_id_feature_name=token_position_id_feature_name,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
            spatio_temporal_token_coordinate=spatio_temporal_token_coordinate),
        fn_name=f'{raw_feature_name}_tokenize')

    # Drop tokens
    if (token_drop_rate > 0.) and is_training:
      preprocessor_builder.add_fn(
          fn=lambda x: add_drop_token(  # pylint: disable=g-long-lambda
              x, token_drop_rate, token_raw_feature_name,
              token_coordinate_feature_name, drop_coordinate_feature_name,
              token_position_id_feature_name, drop_position_id_feature_name,),
          fn_name=f'{token_raw_feature_name}_drop')

  # Note: dataset preprocessing is much faster on float32, so we move
  # casting to bfloat16 so it can be done in a single pass after batching.
  if keep_raw_rgb:
    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_cast')
  else:
    preprocessor_builder.add_fn(
        fn=lambda x: remove_key(x, raw_feature_name),
        fn_name=f'{raw_feature_name}_remove')
  postprocessor_builder.add_fn(
      fn=lambda x: tf.cast(x, dtype),
      feature_name=token_raw_feature_name,
      fn_name=f'{token_raw_feature_name}_cast')
  postprocessor_builder.add_fn(
      fn=lambda x: tf.cast(x, dtype),
      feature_name=token_coordinate_feature_name,
      fn_name=f'{token_coordinate_feature_name}_cast')
  if (token_drop_rate > 0.) and is_training:
    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=drop_coordinate_feature_name,
        fn_name=f'{drop_coordinate_feature_name}_cast')
  return (Modality.VISION,)


def add_waveform(
    parser_builder,
    sampler_builder,
    preprocessor_builder,
    postprocessor_builder,
    parsing_feature_name = ParsingFeatureName.WAVEFORM,
    data_collection_type = DataFeatureType.INPUTS,
    data_collection_route = DataFeatureRoute.ENCODER,
    keep_raw_waveform = False,
    is_training = True,
    num_samples = 30720,
    stride = 1,
    temporal_patch_size = 1,
    num_test_clips = 1,
    token_drop_rate = 0.,
    dtype = 'float32',
    **kwargs):
  """Adds functions to process audio waveform to builders.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different audio features within a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_waveform: Whether to keep raw waveform signal.
    is_training: Whether or not in training mode. If `True`, random sample is
      used.
    num_samples: Number of samples per subclip.
    stride: Temporal stride to sample audio signal.
    temporal_patch_size: The patching window size for waveform tokenization.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggreagated in the batch dimension.
    token_drop_rate: The rate of which tokens are dropped.
    dtype: the dtype to cast the output to after batching.
    **kwargs: Catch irrelevant builders.

  Returns:
    tuple of added modality names.
  """
  del kwargs

  # Validate parameters.
  if is_training is None:
    raise AttributeError('`is_training` was not propagagted properly.')

  is_patching = temporal_patch_size > 1
  if not keep_raw_waveform and not is_patching:
    raise ValueError('Either `keep_raw_waveform` should be True or patching '
                     'should be specified with non-one patch sizes. Instead, '
                     f'{keep_raw_waveform=}, {temporal_patch_size=}.')

  get_feature_name = functools.partial(get_flattened_key,
                                       ftype=data_collection_type,
                                       froute=data_collection_route,
                                       modality=Modality.WAVEFORM)
  raw_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  token_raw_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_RAW)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  drop_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.DROP_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)
  drop_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.DROP_POSITION_ID)

  parse_features(
      parser_builder=parser_builder,
      parsing_feature_name=parsing_feature_name,
      parsed_feature_name=raw_feature_name,
      feature_type=tf.io.VarLenFeature,
      dtype=tf.float32)

  # Densify.
  sampler_builder.add_fn(
      fn=lambda x: tf.reshape(tf.sparse.to_dense(x), (-1,)),
      feature_name=raw_feature_name,
      fn_name=f'{raw_feature_name}_sparse_to_dense')

  sample_sequence_features(
      sampler_builder=sampler_builder,
      feature_name=raw_feature_name,
      num_samples=num_samples,
      stride=stride,
      linspace_size=num_test_clips,
      random_sampling=is_training)

  # accumulate all sampled clips along the first dimension and extend the last
  # dimension
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, (-1, num_samples, 1)),
      feature_name=raw_feature_name,
      fn_name=f'{raw_feature_name}_reshape')

  if temporal_patch_size > 1:
    preprocessor_builder.add_fn(
        fn=lambda x: tokenize_raw_waveform(  # pylint: disable=g-long-lambda
            x, raw_feature_name=raw_feature_name,
            token_raw_feature_name=token_raw_feature_name,
            token_coordinate_feature_name=token_coordinate_feature_name,
            token_position_id_feature_name=token_position_id_feature_name,
            temporal_patch_size=temporal_patch_size),
        fn_name=f'{raw_feature_name}_patching')

    # Drop tokens
    if (token_drop_rate > 0.) and is_training:
      preprocessor_builder.add_fn(
          fn=lambda x: add_drop_token(  # pylint: disable=g-long-lambda
              x, token_drop_rate, token_raw_feature_name,
              token_coordinate_feature_name, drop_coordinate_feature_name,
              token_position_id_feature_name, drop_position_id_feature_name),
          fn_name=f'{token_raw_feature_name}_drop')

      postprocessor_builder.add_fn(
          fn=lambda x: tf.cast(x, dtype),
          feature_name=drop_coordinate_feature_name,
          fn_name=f'{drop_coordinate_feature_name}_cast')

    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=token_raw_feature_name,
        fn_name=f'{token_raw_feature_name}_cast')
    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=token_coordinate_feature_name,
        fn_name=f'{token_coordinate_feature_name}_cast')

  # Note: dataset preprocessing is much faster on float32, so we move
  # casting to bfloat16 so it can be done in a single pass after batching.
  if keep_raw_waveform:
    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_cast')
  else:
    preprocessor_builder.add_fn(
        fn=lambda x: remove_key(x, raw_feature_name),
        fn_name=f'{raw_feature_name}_remove')

  return (Modality.WAVEFORM,)


def add_spectrogram(
    parser_builder,
    sampler_builder,
    preprocessor_builder,
    postprocessor_builder,
    parsing_feature_name = ParsingFeatureName.WAVEFORM,
    data_collection_type = DataFeatureType.INPUTS,
    data_collection_route = DataFeatureRoute.ENCODER,
    keep_waveform_features = False,
    keep_raw_waveform = False,
    keep_raw_spectrogram = False,
    is_training = True,
    num_raw_waveform_samples = 30720,
    waveform_stride = 1,
    waveform_temporal_patch_size = 1,
    temporal_patch_size = 1,
    spectoral_patch_size = 1,
    sample_rate = 48000,
    spectrogram_type = 'logmf',
    frame_length = 2048,
    frame_step = 1024,
    num_features = 80,
    lower_edge_hertz = 80.0,
    upper_edge_hertz = 7600.0,
    preemphasis = None,
    normalize_audio = False,
    num_test_clips = 1,
    token_drop_rate = 0.,
    dtype = 'float32',
    **kwargs):
  """Adds functions to process audio spectrogram feature to builders.

  Note that this function does not extract and parse audio feature. Instead, it
  should be used after a `add_audio` function. The output spectrogram is of the
  shape [batch_size, num_frames, num_features].

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different audio features within a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_waveform_features: Whether to keep the features related to waveform.
    keep_raw_waveform: Whether to keep raw waveform signals.
    keep_raw_spectrogram: Whether to keep raw spectrogram features.
    is_training: Whether or not in training mode. If `True`, random sample is
      used.
    num_raw_waveform_samples: Number of the raw waveform samples per subclip.
    waveform_stride: Temporal stride to sample raw waveform signal.
    waveform_temporal_patch_size: The patching window size for waveform
      tokenization.
    temporal_patch_size: The temporal patching window size for spectrogram
      tokenization.
    spectoral_patch_size: The spectoral patching window size for spectrogram
      tokenization.
    sample_rate: The sample rate of the input audio.
    spectrogram_type: The type of the spectrogram to be extracted from the
      waveform. Can be either `spectrogram`, `logmf`, and `mfcc`.
    frame_length: The length of each spectrogram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    lower_edge_hertz: Lowest frequency to consider.
    upper_edge_hertz: Highest frequency to consider.
    preemphasis: The strength of pre-emphasis on the waveform. If None, no
      pre-emphasis will be applied.
    normalize_audio: Whether to normalize the waveform or not.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggregated in the batch dimension.
    token_drop_rate: The rate of which tokens are dropped.
    dtype: the dtype to cast the output to after batching.
    **kwargs: Catch irrelevant builders.

  Returns:
    tuple of added modality names.
  """
  del kwargs

  if not keep_waveform_features and keep_raw_waveform:
    raise ValueError('Cannot keep raw waveform when `keep_waveform_features` '
                     'is True.')

  # Get feature names
  get_feature_name = functools.partial(get_flattened_key,
                                       ftype=data_collection_type,
                                       froute=data_collection_route,
                                       modality=Modality.SPECTROGRAM)
  raw_waveform_feature_name = get_flattened_key(
      ftype=data_collection_type,
      froute=data_collection_route,
      modality=Modality.WAVEFORM,
      fname=DataFeatureName.RAW)
  raw_spectrogram_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  token_raw_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_RAW)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  drop_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.DROP_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)
  drop_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.DROP_POSITION_ID)

  # Extract raw waveform
  add_waveform(
      parser_builder=parser_builder,
      sampler_builder=sampler_builder,
      preprocessor_builder=preprocessor_builder,
      postprocessor_builder=postprocessor_builder,
      parsing_feature_name=parsing_feature_name,
      data_collection_type=data_collection_type,
      data_collection_route=data_collection_route,
      keep_raw_waveform=True,
      is_training=is_training,
      num_samples=num_raw_waveform_samples,
      stride=waveform_stride,
      temporal_patch_size=waveform_temporal_patch_size,
      num_test_clips=num_test_clips,
      sync_random_state=True,
      token_drop_rate=token_drop_rate,
      dtype=dtype,
  )

  # Extract audio spectrograms.
  preprocessor_builder.add_fn(
      fn=lambda x: add_new_feature(  # pylint: disable=g-long-lambda
          x, compute_audio_spectrogram(
              x[raw_waveform_feature_name],
              sample_rate=sample_rate,
              spectrogram_type=spectrogram_type,
              frame_length=frame_length,
              frame_step=frame_step,
              num_features=num_features,
              lower_edge_hertz=lower_edge_hertz,
              upper_edge_hertz=upper_edge_hertz,
              normalize=normalize_audio,
              preemphasis=preemphasis),
          feature_name=raw_spectrogram_feature_name),
      fn_name=f'{raw_spectrogram_feature_name}_construct')

  if temporal_patch_size > 1 or spectoral_patch_size > 1:
    preprocessor_builder.add_fn(
        fn=lambda x: tokenize_raw_spectrogram(  # pylint: disable=g-long-lambda
            x, raw_feature_name=raw_spectrogram_feature_name,
            token_raw_feature_name=token_raw_feature_name,
            token_coordinate_feature_name=token_coordinate_feature_name,
            token_position_id_feature_name=token_position_id_feature_name,
            temporal_patch_size=temporal_patch_size,
            spectoral_patch_size=spectoral_patch_size),
        fn_name=f'{raw_spectrogram_feature_name}_patching')

    # Drop tokens
    if (token_drop_rate > 0.) and is_training:
      preprocessor_builder.add_fn(
          fn=lambda x: add_drop_token(  # pylint: disable=g-long-lambda
              x, token_drop_rate, token_raw_feature_name,
              token_coordinate_feature_name, drop_coordinate_feature_name,
              token_position_id_feature_name, drop_position_id_feature_name),
          fn_name=f'{token_raw_feature_name}_drop')
      postprocessor_builder.add_fn(
          fn=lambda x: tf.cast(x, dtype),
          feature_name=drop_coordinate_feature_name,
          fn_name=f'{drop_coordinate_feature_name}_cast')

    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=token_raw_feature_name,
        fn_name=f'{token_raw_feature_name}_cast')
    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=token_coordinate_feature_name,
        fn_name=f'{token_coordinate_feature_name}_cast')

  # Note: dataset preprocessing is much faster on float32, so we move
  # casting to bfloat16 so it can be done in a single pass after batching.
  if keep_raw_spectrogram:
    postprocessor_builder.add_fn(
        fn=lambda x: tf.cast(x, dtype),
        feature_name=raw_spectrogram_feature_name,
        fn_name=f'{raw_spectrogram_feature_name}_cast')
  else:
    preprocessor_builder.add_fn(
        fn=lambda x: remove_key(x, raw_spectrogram_feature_name),
        fn_name=f'{raw_spectrogram_feature_name}_remove')

  if not keep_raw_waveform:
    postprocessor_builder.remove_fn(f'{raw_waveform_feature_name}_cast')
    preprocessor_builder.add_fn(
        fn=lambda x: remove_key(x, raw_waveform_feature_name),
        fn_name=f'{raw_waveform_feature_name}_remove')

  if keep_waveform_features:
    return (Modality.SPECTROGRAM, Modality.WAVEFORM)
  else:
    postprocessor_builder.add_fn(
        fn=lambda x: remove_key_with_prefix(x, Modality.WAVEFORM),
        fn_name=f'all_{Modality.WAVEFORM}_remove')
    return (Modality.SPECTROGRAM,)


def add_text(
    parser_builder,
    decoder_builder,
    preprocessor_builder,
    postprocessor_builder,
    tokenizer,
    parsing_feature_name = ParsingFeatureName.CAPTION,
    parsing_context_feature_name = ParsingFeatureName.CONTEXT_INPUT,
    parsing_language_feature_name = (
        ParsingFeatureName.EXAMPLE_LABEL_LANGUAGE),
    data_collection_type = DataFeatureType.INPUTS,
    data_collection_route = DataFeatureRoute.ENCODER,
    keep_raw_string = False,
    is_training = True,
    prepend_bos = False,
    append_eos = False,
    max_num_sentences = 1,
    max_num_tokens = 16,
    sync_random_state = False,
    max_context_sentences = 0,
    is_multi_language = False,
    dtype = 'float32',
    **kwargs):
  """Adds functions to process text feature to builders.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    tokenizer: An instance of a tokenizer.
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different text features within a single dataset.
    parsing_context_feature_name: Name of the context feature in the input
      features dictionary. Only needed if max_context_sentences>0.
    parsing_language_feature_name: Name of the language feature in the input
      table to be parsed. This is Optional and is only used if is_multi_language
      is True.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_string: Whether to keep raw string.
    is_training: Whether or not in training mode. This will be used to randomly
      sample the captions.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    max_num_sentences: Maximum number of captions to keep. If there are more
      captions in the proto, only the first `max_num_sentences` will be returned
      if `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_sentences` will be randomly sampled. Finally if the proto
      contains less than `max_num_sentences`, we pad with empty srings to make
      sure there are `max_num_sentences` in total.
    max_num_tokens: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations used for sampling
      the captions.
    max_context_sentences: Maximum number of temporal neighboring sentences to
       keep.
    is_multi_language: Whether to a language feature indicating the language of
      the text.
    dtype: the dtype to cast the float outputs to after batching.
    **kwargs: Catch irrelevant builders.

  Returns:
    tuple of added modality names.
  """
  del kwargs

  # Validate parameters.
  if is_training is None:
    raise AttributeError('`is_training` was not propagagted properly.')

  if parsing_language_feature_name is None and is_multi_language:
    raise ValueError(
        'When is_multi_language is True, `parsing_language_feature_name` '
        'should be provided.')

  get_feature_name = functools.partial(get_flattened_key,
                                       ftype=data_collection_type,
                                       froute=data_collection_route,
                                       modality=Modality.TEXT)
  raw_feature_name = get_feature_name(
      fname=DataFeatureName.RAW)
  context_raw_feature_name = get_feature_name(
      fname='context_'+DataFeatureName.RAW)
  token_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_ID)
  token_coordinate_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_COORDINATE)
  token_position_id_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_POSITION_ID)
  token_mask_feature_name = get_feature_name(
      fname=DataFeatureName.TOKEN_MASK)
  language_feature_name = get_feature_name(
      fname=DataFeatureName.LANGUAGE)

  # Parse text indices.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parse_features(
        parser_builder=parser_builder,
        parsing_feature_name=parsing_feature_name,
        parsed_feature_name=raw_feature_name,
        feature_type=tf.io.VarLenFeature,
        dtype=tf.string,
        is_context=True)
    # Add context sentences.
    if max_context_sentences > 0:
      parse_features(
          parser_builder=parser_builder,
          parsing_feature_name=parsing_context_feature_name,
          parsed_feature_name=context_raw_feature_name,
          feature_type=tf.io.VarLenFeature,
          dtype=tf.string,
          is_context=True)
  elif (
      isinstance(parser_builder, builders.ExampleParserBuilder)
  ):
    parse_features(
        parser_builder=parser_builder,
        parsing_feature_name=parsing_feature_name,
        parsed_feature_name=raw_feature_name,
        feature_type=tf.io.VarLenFeature,
        dtype=tf.string)
    # Add language annotation.
    if is_multi_language:
      parse_features(
          parser_builder=parser_builder,
          parsing_feature_name=parsing_language_feature_name,
          parsed_feature_name=language_feature_name,
          feature_type=tf.io.VarLenFeature,
          dtype=tf.string)

  decode_sparse_features(
      decoder_builder=decoder_builder,
      feature_name=raw_feature_name)

  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      lambda x, s=None: sample_or_pad_non_sorted_sequence(
          x, max_num_sentences, b'', random=is_training, state=s),
      # pylint: enable=g-long-lambda
      feature_name=raw_feature_name,
      fn_name=f'{raw_feature_name}_sample_captions',
      # Use state to keep coherence between modalities if requested.
      stateful=sync_random_state)

  # Process context sentences.
  if max_context_sentences > 0:
    # Decode context sentences.
    decoder_builder.add_fn(tf.sparse.to_dense, context_raw_feature_name)

    # Keep only a subset of context sentences.
    preprocessor_builder.add_fn(
        fn=lambda x: extract_context_sentences(x, max_context_sentences),
        feature_name=context_raw_feature_name,
        fn_name=f'{context_raw_feature_name}_extract_sentences')

    # Concatenate context sentences with the main sentences.
    preprocessor_builder.add_fn(
        fn=lambda x: integrate_context_sentences(  # pylint: disable=g-long-lambda
            x, raw_feature_name, context_raw_feature_name),
        fn_name=f'{context_raw_feature_name}_integrate')

  preprocessor_builder.add_fn(
      fn=lambda x: tokenize_raw_string(  # pylint: disable=g-long-lambda
          x, tokenizer=tokenizer, raw_feature_name=raw_feature_name,
          token_id_feature_name=token_id_feature_name,
          token_coordinate_feature_name=token_coordinate_feature_name,
          token_position_id_feature_name=token_position_id_feature_name,
          token_mask_feature_name=token_mask_feature_name,
          keep_raw_string=keep_raw_string,
          prepend_bos=prepend_bos,
          append_eos=append_eos,
          max_num_tokens=max_num_tokens,
          max_num_sentences=(max_num_sentences + max_context_sentences)),
      fn_name=f'{raw_feature_name}_tokenization')

  postprocessor_builder.add_fn(
      fn=lambda x: tf.cast(x, dtype),
      feature_name=token_coordinate_feature_name,
      fn_name=f'{token_coordinate_feature_name}_cast')
  return (Modality.TEXT,)


def add_text_from_label(
    parser_builder,
    decoder_builder,
    preprocessor_builder,
    postprocessor_builder,
    tokenizer,
    parsing_label_name_feature_name = (
        ParsingFeatureName.CLIP_LABEL_TEXT),
    data_collection_type = DataFeatureType.INPUTS,
    data_collection_route = DataFeatureRoute.ENCODER,
    keep_raw_string = False,
    is_training = True,
    is_multi_label = False,
    add_label_name_in_sentence = False,
    modality = Modality.IMAGE,
    prepend_bos = False,
    append_eos = False,
    max_num_sentences = 1,
    max_num_tokens = 16,
    dtype = 'float32',
    label_remap_dictionary_path = None,
    **kwargs):
  """Adds functions to process label names to builders.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    tokenizer: An instance of a tokenizer.
    parsing_label_name_feature_name: Name of the label name feature in the input
      `tf.train.Example` or `tf.train.SequenceExample`. Exposing this as an
      argument allows using this function for different label features within
      a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_string: Whether to keep raw string.
    is_training: Whether or not in training mode. This will be used to randomly
      sample the captions.
    is_multi_label: Whether raw data contains multiple labels per example.
    add_label_name_in_sentence: Return a complete sentence with label name.
    modality: The modality whose labels are being turned to sentences.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    max_num_sentences: Maximum number of sentences to keep. If there are more
      captions in the proto, only the first `max_num_sentences` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_sentences` will be randomly sampled. Finally if the proto
      contains less than `max_num_sentences`, we pad with empty srings to make
      sure there are `max_num_sentences` in total.
    max_num_tokens: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id.
    dtype: the dtype to cast the output to after batching.
    label_remap_dictionary_path: if not None, creates a table from the label map
      path that maps keys to label strings.
    **kwargs: Catch irrelevant builders.

  Returns:
    tuple of added modality names.
  """
  del kwargs

  get_feature_name = functools.partial(get_flattened_key,
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

  # Parse label.
  # TODO(b/242550126): make this arg mandatory
  if parsing_label_name_feature_name is not None:
    if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
      parse_features(
          parser_builder=parser_builder,
          parsing_feature_name=parsing_label_name_feature_name,
          parsed_feature_name=raw_feature_name,
          feature_type=tf.io.VarLenFeature,
          dtype=tf.string,
          is_context=True)
    elif (
        isinstance(parser_builder, builders.ExampleParserBuilder)
    ):
      parse_features(
          parser_builder=parser_builder,
          parsing_feature_name=parsing_label_name_feature_name,
          parsed_feature_name=raw_feature_name,
          feature_type=tf.io.VarLenFeature,
          dtype=tf.string)
    else:
      raise ValueError('`parser_builder` has an unexpected type.')

  if label_remap_dictionary_path:
    if not parsing_label_name_feature_name:
      raise ValueError(f'If `{label_remap_dictionary_path=}`, '
                       '`parsing_label_name_feature_name` should be specified')

    label_map = create_label_map_table(label_remap_dictionary_path)
    preprocessor_builder.add_fn(
        fn=lambda x: label_name_to_name_remap(  # pylint: disable=g-long-lambda
            label_name=x, label_map=label_map,
        ),
        feature_name=raw_feature_name,
        fn_name=f'{parsing_label_name_feature_name}_label_mapping')

  # Densify labels tensor (if sparse) in order to support multi label case.
  decode_sparse_features(
      decoder_builder=decoder_builder,
      feature_name=raw_feature_name)

  if not is_multi_label:
    preprocessor_builder.add_fn(
        fn=lambda x: processors.set_shape(x, (1,)),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_set_shape')

  if add_label_name_in_sentence:
    # iterate over labels and replace them with full sentences
    preprocessor_builder.add_fn(
        fn=lambda x: label_to_sentence(  # pylint: disable=g-long-lambda
            x, modality, max_num_sentences, is_training),
        feature_name=raw_feature_name,
        fn_name=f'{raw_feature_name}_to_sentence')

  # Tokenize the label/sentence.
  preprocessor_builder.add_fn(
      fn=lambda x: tokenize(  # pylint: disable=g-long-lambda
          x, tokenizer, raw_feature_name, token_id_feature_name,
          prepend_bos, append_eos, max_num_tokens, keep_raw_string),
      fn_name=f'{raw_feature_name}_tokenization')

  if not add_label_name_in_sentence:
    # Override max_num_sentences to 1, because we don't have any sentences
    max_num_sentences = 1

  # Set text shape.
  preprocessor_builder.add_fn(
      fn=lambda x: set_shape(x, (max_num_sentences, max_num_tokens)),
      feature_name=token_id_feature_name,
      fn_name=f'{token_id_feature_name}_set_shape')

  # Extract positional encoding
  preprocessor_builder.add_fn(
      fn=lambda x: add_new_feature(  # pylint: disable=g-long-lambda
          x, construct_1d_positions(x[token_id_feature_name], normalize=True),
          feature_name=token_coordinate_feature_name),
      fn_name=f'{token_id_feature_name}_position_encoding')

  # Add token position ID
  preprocessor_builder.add_fn(
      fn=lambda x: add_new_feature(  # pylint: disable=g-long-lambda
          x, construct_1d_positions(x[token_id_feature_name], normalize=False),
          feature_name=token_position_id_feature_name),
      fn_name=f'{token_id_feature_name}_id')

  # Add attention mask for the padded locations
  preprocessor_builder.add_fn(
      fn=lambda x: add_new_feature(  # pylint: disable=g-long-lambda
          x, tf.where(x[token_id_feature_name] == tokenizer.pad_token, 0, 1),
          feature_name=token_mask_feature_name),
      fn_name=f'{token_id_feature_name}_attention_mask')
  postprocessor_builder.add_fn(
      fn=lambda x: tf.cast(x, dtype),
      feature_name=token_coordinate_feature_name,
      fn_name=f'{token_coordinate_feature_name}_cast')
  return (Modality.TEXT,)


def add_label(
    parser_builder,
    decoder_builder,
    preprocessor_builder,
    parsing_label_index_feature_name = (
        ParsingFeatureName.CLIP_LABEL_INDEX),
    data_collection_type = DataFeatureType.TARGETS,
    data_collection_route = DataFeatureRoute.ENCODER,
    modality = '',
    is_multi_label = False,
    one_hot_label = True,
    num_classes = None,
    smoothing = 0.0,
    normalize = False,
    **kwargs):
  """Adds functions to process label feature to builders.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    parsing_label_index_feature_name: Name of the label index feature in the
      input `tf.train.Example` or `tf.train.SequenceExample`. Exposing this as
      an argument allows using this function for different label features within
      a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    modality: The modality under which the labels are being added.
    is_multi_label: Whether raw data contains multiple labels per example.
    one_hot_label: Return labels as one hot tensors. If `is_multi_label` is
      `True`, one hot tensor might have multiple ones.
    num_classes: Total number of classes in the dataset. It has to be provided
      if `one_hot_label` is `True`.
    smoothing: Label smoothing alpha value for one-hot labels.
    normalize: If `True` and `add_label_index=True`, one-hot label indices
      will be normalized so that the channels sum to 1.
    **kwargs: Catch irrelevant builders.
  """
  del kwargs

  # Validate parameters.
  if one_hot_label and not num_classes:
    raise ValueError(
        '`num_classes` must be given when requesting one hot label.')

  if not modality:
    raise ValueError(
        '`modality` should be specified when adding label annotation.')

  if is_multi_label and not one_hot_label:
    logging.warning(
        'Multi label indices will be returned as variable size tensors.')

  label_feature_name = get_flattened_key(ftype=data_collection_type,
                                         froute=data_collection_route,
                                         fname=DataFeatureName.LABEL,
                                         modality=modality)
  # Parse label.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parse_features(
        parser_builder=parser_builder,
        parsing_feature_name=parsing_label_index_feature_name,
        parsed_feature_name=label_feature_name,
        feature_type=tf.io.VarLenFeature,
        dtype=tf.int64,
        is_context=True)
  elif (
      isinstance(parser_builder, builders.ExampleParserBuilder)
  ):
    parse_features(
        parser_builder=parser_builder,
        parsing_feature_name=parsing_label_index_feature_name,
        parsed_feature_name=label_feature_name,
        feature_type=tf.io.VarLenFeature,
        dtype=tf.int64)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Densify labels tensor (if sparse) in order to support multi label case.
  decode_sparse_features(
      decoder_builder=decoder_builder,
      feature_name=label_feature_name)

  if one_hot_label:
    # Replace label index by one hot representation.
    preprocessor_builder.add_fn(
        fn=lambda x: tf.reduce_sum(  # pylint: disable=g-long-lambda
            input_tensor=tf.one_hot(x, num_classes),
            axis=0),
        feature_name=label_feature_name,
        fn_name=f'{label_feature_name}_one_hot')
    if smoothing > 0:
      preprocessor_builder.add_fn(
          fn=lambda x: label_smoothing(x, smoothing),
          feature_name=label_feature_name,
          fn_name=f'{label_feature_name}_smooth')
    if normalize:
      preprocessor_builder.add_fn(
          fn=label_normalization,
          feature_name=label_feature_name,
          fn_name=f'{label_feature_name}_normalize')

  elif not is_multi_label:
    preprocessor_builder.add_fn(
        fn=lambda x: processors.set_shape(x, (1,)),
        feature_name=label_feature_name,
        fn_name=f'{label_feature_name}_set_shape')

  # Set add a placeholder instance dimension.
  preprocessor_builder.add_fn(
      fn=lambda x: x[tf.newaxis, :],
      feature_name=label_feature_name,
      fn_name=f'{label_feature_name}_instance_expand')
