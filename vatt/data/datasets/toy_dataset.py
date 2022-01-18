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

# Lint as: python3
"""Moments in Time action classification factories."""

import abc

from vatt.data import loading
from vatt.data import processing


class ToyFactory(loading.BaseDMVRFactory, abc.ABC):
  """Base class for a toy factory."""

  _BASE_DIR = 'PATH/TO/YOUR/TFRECORD'
  _NUM_CLASSES = 1000

  _TABLES = {
      'train': 'train@10',
      'test': 'test@1',
  }

  def __init__(self, subset = 'train', split = 1):
    """Constructor of ToyFactory."""

    del split
    if subset not in ToyFactory._TABLES:
      raise ValueError('Invalid subset "{}". The available subsets are: {}'
                       .format(subset, ToyFactory._TABLES.keys()))

    super().__init__(ToyFactory._BASE_DIR,
                     table=f'{subset}',
                     source='tfrecord',
                     raw_data_format='tf_sequence_example')

  def _build(self,
             is_training = True,
             # Video related parameters.
             num_frames = 32,
             stride = 1,
             num_test_clips = 1,
             min_resize = 256,
             crop_size = 224,
             multi_crop = False,
             crop_resize_style = 'Inception',
             min_aspect_ratio = 0.5,
             max_aspect_ratio = 2,
             min_area_ratio = 0.08,
             max_area_ratio = 1.0,
             zero_centering_image = False,
             color_augmentation = True,
             # Text related parameters.
             max_num_words = 16,
             max_context_sentences = 1,
             tokenizer = 'howto100m_en',
             prepend_bos = False,
             append_eos = False,
             keep_raw_string = False,
             # Audio related parameters.
             num_samples = 153600,  # 48000 (Hz) * 32 / 10 (fps)
             audio_stride = 1,
             sync_audio_and_image = True,
             # Label related parameters.
             one_hot_label = True,
             output_label_string = False,
             **kwargs):
    """Default build for this dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_test_clips: number of test clip (1 by default). If more than one,
        this will sample multiple linearly spaced clips within each video at
        test time. If 1, then a single clip in the middle of the video is
        sampled.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      multi_crop: if 3-view crop should be performed.
      crop_resize_style: The style of Crop+Resize. 'Inception' or 'VGG'.
      min_aspect_ratio: The minimum aspect range for cropping.
      max_aspect_ratio: The maximum aspect range for cropping.
      min_area_ratio: The minimum area range for cropping.
      max_area_ratio: The maximum area range for cropping.
      zero_centering_image: whether to have images between [-1, 1] or [0, 1].
      color_augmentation: Whether to jitter color or not.
      max_num_words: maximum number of words to keep from the text.
      max_context_sentences: number of temporal neighboring sentences to sample.
      tokenizer: defining which tokenizer in what language should be used.
      prepend_bos: prepend BOS token in the tokenizer.
      append_eos: append EOS token in the tokenizer.
      keep_raw_string: keep the raw string or not.
      num_samples: number of audio samples.
      audio_stride: temporal stride for audio samples.
      sync_audio_and_image: sample audio and image in sync.
      one_hot_label: whether or not to return one hot version of labels.
      output_label_string: whether or not to return label as text.
      **kwargs: additional args
    """

    del kwargs

    processing.add_vision(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        postprocessor_builder=self.postprocessor_builder,
        input_feature_name='image/encoded',
        output_feature_name=processing.FeatureNames.VISION,
        is_training=is_training,
        num_frames=num_frames,
        stride=stride,
        num_test_clips=num_test_clips,
        min_resize=min_resize,
        crop_size=crop_size,
        multi_crop=multi_crop,
        crop_resize_style=crop_resize_style,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        zero_centering_image=zero_centering_image,
        color_augmentation=color_augmentation,
        sync_random_state=False,
        is_rgb=True,
        is_flow=False
        )

    processing.add_audio(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        preprocessor_builder=self.preprocessor_builder,
        postprocessor_builder=self.postprocessor_builder,
        input_feature_name='waveform/floats',
        output_feature_name=processing.FeatureNames.AUDIO,
        is_training=is_training,
        num_samples=num_samples,
        stride=audio_stride,
        num_test_clips=num_test_clips,
        sync_random_state=sync_audio_and_image
        )

    # Use the HowTo100M tokenizer by default.
    self.tokenizer = processing.get_tokenizer(tokenizer)

    processing.add_text(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        tokenizer=self.tokenizer,
        is_training=False,
        input_feature_name='caption/string',
        output_raw_string_name=processing.FeatureNames.TEXT_STRING,
        output_feature_name=processing.FeatureNames.TEXT_INDEX,
        prepend_bos=prepend_bos,
        append_eos=append_eos,
        keep_raw_string=keep_raw_string,
        max_num_sentences=max_context_sentences,
        max_num_tokens=max_num_words,
        sync_random_state=False,
        )

    processing.add_label(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        input_label_index_feature_name='label/index',
        input_label_name_feature_name='label/text',
        output_label_index_feature_name=processing.FeatureNames.LABEL_INDEX,
        output_label_name_feature_name=processing.FeatureNames.LABEL_STRING,
        is_multi_label=False,
        one_hot_label=one_hot_label,
        num_classes=self._NUM_CLASSES,
        add_label_name=output_label_string,
        )

  def tables(self):
    return self._TABLES
