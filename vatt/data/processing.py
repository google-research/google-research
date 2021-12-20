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
"""Data processing modules for dataloaders."""

import os
from typing import List, Tuple, Optional

from dmvr import builders
from dmvr import modalities
from dmvr import processors
from dmvr import tokenizers
import tensorflow as tf
import tensorflow_probability as tfp


_VOCAB_BASE_DIR = "./misc/"


def get_vocab_path(vocab = "howto100m_en"):
  """Return the vocabulary path for a given data & language."""

  all_vocabs = ["howto100m_en", "bert_uncased_en"]
  if vocab not in all_vocabs:
    raise ValueError("Available vocabularies are: %s." % all_vocabs)
  vocab_path = os.path.join(_VOCAB_BASE_DIR, vocab + ".txt")
  return vocab_path


def get_sharded_files(table_path):
  """Get the final list of sharded files."""

  table_pattern = table_path.split("@")
  if len(table_pattern) == 2 and table_pattern[1].isdigit():
    shard_pattern = table_pattern[0] + "*-of-*"
  else:
    shard_pattern = os.path.join(table_path, "*-of-*")

  shards = tf.io.matching_files(shard_pattern).numpy()
  shards_list = [shard.decode("utf-8") for shard in shards]

  return shards_list


def get_tokenizer(tokenizer = "howto100m_en"):
  vocabulary_path = get_vocab_path(tokenizer)
  if "howto100m" in tokenizer:
    return tokenizers.WordTokenizer(vocabulary_path)

  elif "bert" in tokenizer:
    return tokenizers.BertTokenizer(vocabulary_path)

  else:
    raise ValueError(f"Tokenizer {tokenizer} not supported.")


# Default feature names
class FeatureNames:
  """Collection of feature names for dataloaders."""

  VISION = "vision"
  AUDIO = "audio"
  AUDIO_MEL = "audio_mel"
  AUDIO_MASK = "audio_mask"
  TEXT_STRING = "text_string"
  TEXT_INDEX = "text"
  TEXT_MASK = "text_mask"
  CONTEXT_TEXT_STRING = "context_text_string"
  CONTEXT_TEXT_INDEX = "context_text_index"
  LABEL_STRING = "label_string"
  LABEL_INDEX = "label"


# ----------------------------------------------------------------------
# ------------------------ Processing functions ------------------------
# ----------------------------------------------------------------------

# Auxiliary processing fns from DMVR
sample_sequence = processors.sample_sequence
sample_linspace_sequence = processors.sample_linspace_sequence
sample_or_pad_non_sorted_sequence = processors.sample_or_pad_non_sorted_sequence
decode_jpeg = processors.decode_jpeg
normalize_image = processors.normalize_image
resize_smallest = processors.resize_smallest
crop_image = processors.crop_image
random_flip_left_right = processors.random_flip_left_right
color_default_augm = processors.color_default_augm
scale_jitter_augm = processors.scale_jitter_augm
set_shape = processors.set_shape
crop_or_pad_words = processors.crop_or_pad_words
tokenize = processors.tokenize


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def remove_key(features_dict, key):
  del features_dict[key]
  return features_dict


def remove_label(features_dict):
  if FeatureNames.LABEL_INDEX in features_dict:
    del features_dict[FeatureNames.LABEL_INDEX]
  if FeatureNames.LABEL_STRING in features_dict:
    del features_dict[FeatureNames.LABEL_STRING]
  return features_dict


def remove_vision(features_dict):
  if FeatureNames.VISION in features_dict:
    del features_dict[FeatureNames.VISION]
  return features_dict


def remove_audio(features_dict):
  if FeatureNames.AUDIO in features_dict:
    del features_dict[FeatureNames.AUDIO]
  return features_dict


def add_gaussian(inputs, gamma):
  std = gamma * tf.reduce_max(tf.math.abs(inputs))
  inputs = inputs + tf.random.normal(tf.shape(inputs), stddev=std)
  return inputs


def extend_waveform_dim(raw_audio, num_windows=1):
  if num_windows > 1:
    raw_audio = tf.reshape(raw_audio, [num_windows, -1])
  return tf.expand_dims(raw_audio, axis=-1)


def label_smoothing(inputs,
                    alpha=0.1,
                    multi_label=False):
  """Smoothing data labels."""

  # assert data has labels
  assert FeatureNames.LABEL_INDEX in inputs
  labels = inputs[FeatureNames.LABEL_INDEX]
  num_classes = get_shape(labels)[-1]
  if multi_label:
    # multi-label is not supported for now
    raise NotImplementedError
  else:
    smoothed_labels = (1 - alpha) * labels + alpha / num_classes
  inputs[FeatureNames.LABEL_INDEX] = smoothed_labels

  return inputs


def batched_mixup(inputs,
                  feature_name,
                  alpha=5,
                  beta=2,
                  mixup_labels=True):
  """Mixup processing function as in https://arxiv.org/pdf/1710.09412.pdf."""

  # get features
  features = inputs[feature_name]
  feature_shape = get_shape(features)
  batch_size = feature_shape[0]
  # flatten inputs to a 1d signal
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
  inputs[feature_name] = tf.reshape(mixed_up_features, feature_shape)

  if mixup_labels:
    assert FeatureNames.LABEL_INDEX in inputs
    labels = inputs[FeatureNames.LABEL_INDEX]
    shuffled_labels = tf.gather_nd(labels, batch_idx)  # (bs, n_class)
    mixed_up_labels = lmbda * labels + (1 - lmbda) * shuffled_labels
    inputs[FeatureNames.LABEL_INDEX] = mixed_up_labels

  return inputs


def linearize(inputs, feature_name):
  inputs[feature_name] = tf.reshape(inputs[feature_name], [-1])
  return inputs


def random_crop_resize(frames,
                       output_h,
                       output_w,
                       num_frames,
                       num_channels,
                       aspect_ratio,
                       area_range):
  """First crops clip with jittering and then resizes to (output_h, output_w).

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    output_h: Resized image height.
    output_w: Resized image width.
    num_frames: Number of input frames per clip.
    num_channels: Number of channels of the clip.
    aspect_ratio: Float tuple with the aspect range for cropping.
    area_range: Float tuple with the area range for cropping.
  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype.
  """
  shape = tf.shape(frames)
  seq_len, _, _, channels = shape[0], shape[1], shape[2], shape[3]
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
  size = tf.convert_to_tensor((
      seq_len, target_height, target_width, channels))
  offset = tf.convert_to_tensor((
      0, offset_y, offset_x, 0))
  frames = tf.slice(frames, offset, size)
  frames = tf.cast(
      tf.image.resize(frames, (output_h, output_w)),
      frames.dtype)
  frames.set_shape((num_frames, output_h, output_w, num_channels))
  return frames


def multi_crop_image(frames,
                     target_height,
                     target_width):
  """3 crops the image sequence of images.

  If requested size is bigger than image size, image is padded with 0.

  Args:
    frames: A Tensor of dimension [timesteps, in_height, in_width, channels].
    target_height: Target cropped image height.
    target_width: Target cropped image width.

  Returns:
    A Tensor of shape [timesteps, out_height, out_width, channels] of type uint8
    with the cropped images.
  """
  # Three-crop evaluation.
  shape = tf.shape(frames)
  static_shape = frames.shape.as_list()
  seq_len = shape[0] if static_shape[0] is None else static_shape[0]
  height = shape[1] if static_shape[1] is None else static_shape[1]
  width = shape[2] if static_shape[2] is None else static_shape[2]
  channels = shape[3] if static_shape[3] is None else static_shape[3]

  size = tf.convert_to_tensor(
      (seq_len, target_height, target_width, channels))

  offset_1 = tf.broadcast_to([0, 0, 0, 0], [4])
  # pylint:disable=g-long-lambda
  offset_2 = tf.cond(
      tf.greater_equal(height, width),
      true_fn=lambda: tf.broadcast_to([
          0, tf.cast(height, tf.float32) / 2 - target_height // 2, 0, 0
      ], [4]),
      false_fn=lambda: tf.broadcast_to([
          0, 0, tf.cast(width, tf.float32) / 2 - target_width // 2, 0
      ], [4]))
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


def add_audio_text_if_empty(features_dict,
                            has_valid_text,
                            has_valid_audio,
                            num_audio_samples,
                            max_context_sentences,
                            max_num_words):
  """."""
  if has_valid_audio:
    features_dict[FeatureNames.AUDIO_MASK] = tf.ones(shape=(), dtype=tf.float32)
  else:
    aud_shape = [num_audio_samples,]
    features_dict[FeatureNames.AUDIO] = tf.zeros(aud_shape, dtype=tf.float32)
    features_dict[FeatureNames.AUDIO_MASK] = tf.zeros(shape=(),
                                                      dtype=tf.float32)

  if has_valid_text:
    features_dict[FeatureNames.TEXT_MASK] = tf.ones(shape=(), dtype=tf.float32)
  else:
    txt_shape = [max_context_sentences+1, max_num_words]
    features_dict[FeatureNames.TEXT_INDEX] = tf.zeros(txt_shape, dtype=tf.int32)
    features_dict[FeatureNames.TEXT_MASK] = tf.zeros(shape=(), dtype=tf.float32)
  return features_dict


def space_to_depth(inputs,
                   temporal_block_size,
                   spatial_block_size,
                   feature_name):
  """Performs per batch space to depth."""

  inputs[feature_name] = processors.batched_space_to_depth(
      frames=inputs[feature_name],
      temporal_block_size=temporal_block_size,
      spatial_block_size=spatial_block_size,
      )

  return inputs


def make_spectrogram(audio,
                     stft_length=2048,
                     stft_step=1024,
                     stft_pad_end=True,
                     use_mel=True,
                     mel_lower_edge_hertz=80.,
                     mel_upper_edge_hertz=7600.,
                     mel_sample_rate=48000.,
                     mel_num_bins=40,
                     use_log=True,
                     log_eps=1.,
                     log_scale=10000.):
  """Computes (mel) spectrograms for signals t."""
  stfts = tf.signal.stft(audio,
                         frame_length=stft_length,
                         frame_step=stft_step,
                         fft_length=stft_length,
                         pad_end=stft_pad_end)
  spectrogram = tf.abs(stfts)
  if use_mel:
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        mel_num_bins, num_spectrogram_bins, mel_sample_rate,
        mel_lower_edge_hertz, mel_upper_edge_hertz)
    spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    spectrogram.set_shape(spectrogram.shape[:-1] +
                          linear_to_mel_weight_matrix.shape[-1:])

  if use_log:
    spectrogram = tf.log(log_eps + log_scale * spectrogram)
  return spectrogram


def raw_audio_to_spectrogram(feat_dict,
                             sample_rate=48000,
                             stft_length=0.032,
                             stft_step=0.016,
                             mel_bins=80,
                             rm_audio=False,
                             num_windows=1):
  """Computes audio spectrogram and eventually removes raw audio."""
  raw_audio = feat_dict[FeatureNames.AUDIO]
  if num_windows > 1:
    raw_audio = tf.reshape(raw_audio, [num_windows, -1])
  stft_length = int(sample_rate * stft_length)
  stft_step = int(sample_rate * stft_step)
  mel = make_spectrogram(audio=raw_audio,
                         mel_sample_rate=sample_rate,
                         stft_length=stft_length,
                         stft_step=stft_step,
                         mel_num_bins=mel_bins,
                         use_mel=True)
  mel = tf.expand_dims(mel, axis=-1)
  # Adding a channel dimension.
  feat_dict[FeatureNames.AUDIO_MEL] = mel
  if rm_audio:
    del feat_dict[FeatureNames.AUDIO]
  return feat_dict


def normalize_spectrogram(spectrogram):
  """Normalize spectrogram within [0, 1]."""
  max_val = tf.reduce_max(spectrogram, axis=[-3, -2, -1], keepdims=True)
  min_val = tf.reduce_min(spectrogram, axis=[-3, -2, -1], keepdims=True)
  diff = max_val - min_val
  return tf.math.divide_no_nan(spectrogram - min_val, diff)


def get_audio_shape(params, fps, sr):
  """Calculate exact audio input shape given input parameters."""
  # check if it is video-text-only setting
  # pass [1, 1] to avoid unnecessary TPU occupation
  if params.split == "train":
    if params.name == "howto100m" and not params.use_howto100m_audio:
      return [1, 1]

  n_audio_secs = params.num_frames / fps
  n_waveform_samples = int(n_audio_secs * sr)
  if params.raw_audio:
    audio_shape = [n_waveform_samples, 1]
  else:
    n_stft_steps = int(params.stft_step * sr)
    n_stft_samples = int(n_waveform_samples / n_stft_steps)
    audio_shape = [n_stft_samples, params.mel_bins, 1]

  return audio_shape


def get_video_shape(params, is_space_to_depth=False):
  """Returns exact video shape as model's input."""
  if is_space_to_depth:
    video_shape = [params.num_frames // 2,
                   params.frame_size // 2,
                   params.frame_size // 2,
                   24]
  else:
    video_shape = [params.num_frames,
                   params.frame_size,
                   params.frame_size,
                   3]
  return video_shape


# ----------------------------------------------------------------------
# ------ Tools for integrating reading, decoding and processing --------
# ----------------------------------------------------------------------

# directly pointing to dmvr tools
add_label = modalities.add_label


# modifying dmvr's add_image to add support for multi-view crop and
# Inception-style crop+resize
def add_vision(
    parser_builder,
    sampler_builder,
    decoder_builder,
    preprocessor_builder,
    postprocessor_builder,
    input_feature_name = "image/encoded",
    output_feature_name = FeatureNames.VISION,
    is_training = True,
    # Video related parameters.
    num_frames = 32,
    stride = 1,
    num_test_clips = 1,
    min_resize = 224,
    crop_size = 200,
    multi_crop = False,
    zero_centering_image = False,
    crop_resize_style = "Inception",
    min_aspect_ratio = 0.5,
    max_aspect_ratio = 2,
    min_area_ratio = 0.5,
    max_area_ratio = 1.0,
    color_augmentation = False,
    sync_random_state = True,
    is_rgb = True,
    is_flow = False):
  """Custom vision reader & processor based on DMVR logic."""

  modalities.add_image(
      parser_builder=parser_builder,
      sampler_builder=sampler_builder,
      decoder_builder=decoder_builder,
      preprocessor_builder=preprocessor_builder,
      postprocessor_builder=postprocessor_builder,
      input_feature_name=input_feature_name,
      output_feature_name=output_feature_name,
      is_training=is_training,
      num_frames=num_frames,
      stride=stride,
      num_test_clips=num_test_clips,
      min_resize=min_resize,
      crop_size=crop_size,
      zero_centering_image=zero_centering_image,
      sync_random_state=sync_random_state,
      is_rgb=is_rgb,
      is_flow=is_flow,
      )

  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  if is_training:
    if crop_resize_style == "Inception":
      # remove the default VGG-style crop
      preprocessor_builder.remove_fn(f"{output_feature_name}_resize_smallest")
      preprocessor_builder.remove_fn(f"{output_feature_name}_random_crop")

      # add Inception-style image crop: random crop -> resize.
      preprocessor_builder.add_fn(
          fn=lambda x: random_crop_resize(  # pylint: disable=g-long-lambda
              x, crop_size, crop_size, num_frames, num_raw_channels,
              (min_aspect_ratio, max_aspect_ratio),
              (min_area_ratio, max_area_ratio)),
          feature_name=output_feature_name,
          fn_name=f"{output_feature_name}_random_crop_resize")

  else:
    if multi_crop:
      # remove the default central crop fn
      preprocessor_builder.remove_fn(f"{output_feature_name}_central_crop")

      # add multi crop of the frames.
      preprocessor_builder.add_fn(
          fn=lambda x: multi_crop_image(x, crop_size, crop_size),
          feature_name=output_feature_name,
          fn_name=f"{output_feature_name}_multi_crop")

  if not is_flow:
    # apply color augmentation if specified
    if color_augmentation and is_rgb and is_training:
      # Random color jitter
      preprocessor_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x, s=None: color_default_augm(
              x, zero_centering_image=zero_centering_image,
              prob_color_augment=0.8,
              prob_color_drop=0.0,
              ),
          feature_name=output_feature_name,
          fn_name=f"{output_feature_name}_color_jitter",
          )


def add_audio(
    parser_builder,
    sampler_builder,
    preprocessor_builder,
    postprocessor_builder,
    input_feature_name = "WAVEFORM/feature/floats",
    output_feature_name = FeatureNames.AUDIO,
    is_training = True,
    # Audio related parameters.
    num_samples = 30720,
    stride = 1,
    sample_rate = 48000,
    target_sample_rate = None,
    num_test_clips = 1,
    sync_random_state = True):
  """Adds functions to process audio feature to builders."""

  modalities.add_audio(
      parser_builder=parser_builder,
      sampler_builder=sampler_builder,
      postprocessor_builder=postprocessor_builder,
      preprocessor_builder=preprocessor_builder,
      input_feature_name=input_feature_name,
      output_feature_name=output_feature_name,
      is_training=is_training,
      num_samples=num_samples,
      stride=stride,
      sample_rate=sample_rate,
      target_sample_rate=target_sample_rate,
      num_test_clips=num_test_clips,
      sync_random_state=sync_random_state,
    )

  if num_test_clips > 1 and not is_training:
    postprocessor_builder.remove_fn(f"{output_feature_name}_reshape")

    # In this case, multiple clips are merged together in batch dimenstion which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(x, (-1, num_samples)),
        feature_name=output_feature_name,
        fn_name=f"{output_feature_name}_reshape")


def add_text(
    parser_builder,
    decoder_builder,
    preprocessor_builder,
    tokenizer,
    is_training = True,
    input_feature_name = "caption/string",
    output_raw_string_name = FeatureNames.TEXT_STRING,
    output_feature_name = FeatureNames.TEXT_INDEX,
    # Text related parameters.
    prepend_bos = False,
    append_eos = False,
    keep_raw_string = False,
    max_num_sentences = 1,
    max_num_tokens = 16,
    sync_random_state = False):
  """Adds functions to process text feature to builders."""

  modalities.add_text(
      parser_builder=parser_builder,
      decoder_builder=decoder_builder,
      preprocessor_builder=preprocessor_builder,
      tokenizer=tokenizer,
      is_training=is_training,
      input_feature_name=input_feature_name,
      output_raw_string_name=output_raw_string_name,
      output_feature_name=output_feature_name,
      prepend_bos=prepend_bos,
      append_eos=append_eos,
      keep_raw_string=keep_raw_string,
      max_num_captions=max_num_sentences,
      max_num_tokens=max_num_tokens,
      sync_random_state=sync_random_state,
      )
