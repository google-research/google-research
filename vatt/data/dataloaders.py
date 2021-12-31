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
"""Main experiments script for pre-training MMV model OR MMRL."""

import functools
from typing import Any, Optional

import tensorflow as tf

from vatt.data import loading
from vatt.data import processing
from vatt.data.datasets import factory as ds_fctr


ExpConfig = Any
FeatureNames = processing.FeatureNames
SELF_SUP_DS = ['howto100m', 'audioset']
VID_CLS_DS = ['kinetics400',
              'kinetics600',
              'kinetics700',
              'mit',
              'hmdb51',
              'ucf101']
AUD_CLS_DS = ['audioset', 'esc50']
IMG_CLS_DS = ['imagenet']
CLS_DS = {'hmdb51': {'num_classes': 51,
                     'splits': [1, 2, 3],
                     'total_steps': 10000},
          'ucf101': {'num_classes': 101,
                     'splits': [1, 2, 3],
                     'total_steps': 10000},
          'esc50': {'num_classes': 50,
                    'splits': [1, 2, 3, 4, 5],
                    'total_steps': 10000},
          'kinetics400': {'num_classes': 400},
          'kinetics600': {'num_classes': 600},
          'kinetics700': {'num_classes': 700},
          'mit': {'num_classes': 339},
          'imagenet': {'num_classes': 1000},
          'audioset': {'num_classes': 527},
          }
TEXT_DS = {'howto100m': {'num_clips': 10000},
           'youcook2': {'num_clips': 3320},
           'msrvtt': {'num_clips': 1000},
           'msrvtt1000': {'num_clips': 1000}}
AUDIO_DS = ['audioset', 'esc50']

DEFAULT_FPS = 25  # default fps of all datasets
DEFAULT_SR = 48000  # default sampling rate of all datasets
REF_FPS = 10  # reference fps used during pre-training
REF_SR = 48000  # reference sampling rate used during pre-training


class PreTrainLoader(loading.BaseLoader):
  """Constructs the dataloader for pre-train."""

  def __init__(self, dataset_id, params):
    # Generic parameters
    input_params = params.train.input
    self._num_frames = input_params.num_frames
    self._frame_size = input_params.frame_size
    self._video_stride = input_params.video_stride
    self._raw_audio = input_params.raw_audio
    self._stft_length = input_params.stft_length
    self._stft_step = input_params.stft_step
    self._mel_bins = input_params.mel_bins
    self._zero_centering_image = input_params.zero_centering_image
    self._max_num_words = input_params.max_num_words
    self._max_context_sentences = input_params.max_context_sentences
    self._space_to_depth = input_params.space_to_depth
    self._linearize_vision = input_params.linearize_vision

    # Augmentation parameters
    self._min_resize = input_params.min_resize
    self._min_area_ratio = input_params.min_area_ratio
    self._max_area_ratio = input_params.max_area_ratio
    self._min_aspect_ratio = input_params.min_aspect_ratio
    self._max_aspect_ratio = input_params.max_aspect_ratio
    self._crop_resize_style = input_params.crop_resize_style
    self._scale_jitter = input_params.scale_jitter
    self._audio_noise = input_params.audio_noise
    self._audio_mixup = input_params.audio_mixup
    self._mixup_alpha = input_params.mixup_alpha
    self._mixup_beta = input_params.mixup_beta

    ds_names = dataset_id.split('+')
    assert 'howto100m' in dataset_id, 'Only HT+ is supported'
    ds_factories = []
    for ds_name in ds_names:
      params_factory = {
          'is_training': True,
          'num_frames': self._num_frames,
          'stride': self._video_stride,
          'crop_size': self._frame_size,
          'min_resize': self._min_resize,
          'zero_centering_image': self._zero_centering_image,
          'min_area_ratio': self._min_area_ratio,
          'max_area_ratio': self._max_area_ratio,
          'min_aspect_ratio': self._min_aspect_ratio,
          'max_aspect_ratio': self._max_aspect_ratio,
          'crop_resize_style': self._crop_resize_style,
      }

      fps = REF_FPS if ds_name == 'howto100m' else DEFAULT_FPS
      n_audio_secs = self._num_frames / REF_FPS
      stride = self._video_stride * int(fps // REF_FPS)
      params_factory['stride'] = stride
      self._num_audio_samples = int(REF_SR * n_audio_secs)
      params_factory['num_samples'] = self._num_audio_samples

      if ds_name == 'howto100m':
        params_factory.update({
            'output_audio': True,
            'max_num_words': self._max_num_words,
            'max_context_sentences': self._max_context_sentences,
        })

      # Get the factory.
      factory_args = {'subset': 'train'}
      factory_class = ds_fctr.get_ds_factory(
          dataset_name=ds_name,
          )(**factory_args)

      ds_factory = factory_class.configure(**params_factory)

      # Add zeros to audio and/or text if the dataset does not have audio
      # or text already. Also add a boolean to whether audio and/or text
      # are valid and should be used
      ds_factory.sampler_builder.add_fn(
          functools.partial(
              processing.add_audio_text_if_empty,
              has_valid_text=(ds_name == 'howto100m'),
              has_valid_audio=True,
              num_audio_samples=self._num_audio_samples,
              max_context_sentences=self._max_context_sentences,
              max_num_words=self._max_num_words,
          ))

      # Remove labels from inputs
      if ds_name == 'audioset':
        ds_factory.postprocessor_builder.add_fn(processing.remove_label)

      # Add audio preprocessing.
      if self._audio_noise > 0.:
        # Add gaussian noise
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.add_gaussian,
                gamma=self._audio_noise,
                ),
            feature_name=FeatureNames.AUDIO,
            fn_name='volume_gaussian'
            )

      if self._raw_audio:
        ds_factory.preprocessor_builder.add_fn(
            processing.extend_waveform_dim,
            feature_name=FeatureNames.AUDIO,
            fn_name='extend_waveform',
            )
      else:
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.raw_audio_to_spectrogram,
                sample_rate=REF_SR,
                stft_length=self._stft_length,
                stft_step=self._stft_step,
                mel_bins=self._mel_bins,
                rm_audio=True
                )
            )
        ds_factory.preprocessor_builder.add_fn(
            processing.normalize_spectrogram,
            feature_name=FeatureNames.AUDIO_MEL,
            fn_name='normalize_mel',
            )

      # Extra data augmentation on video.
      if self._scale_jitter and self._crop_resize_style == 'VGG':
        # scale jitter is applied only when crop+resize is VGG-style
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.scale_jitter_augm,
                prob=0.8,
                ),
            feature_name=FeatureNames.VISION,
            fn_name=f'{FeatureNames.VISION}_jitter_scale',
            add_before_fn_name=f'{FeatureNames.VISION}_resize_smallest'
            )

      ds_factories.append(ds_factory)

    # Add batch-level data-agnostic post-processing functions
    postprocess_fns = []

    if self._space_to_depth:
      postprocess_fns.append(
          functools.partial(
              processing.space_to_depth,
              temporal_block_size=2,
              spatial_block_size=2,
              feature_name=FeatureNames.VISION,
              )
          )

    if self._linearize_vision:
      postprocess_fns.append(
          functools.partial(
              processing.linearize,
              feature_name=FeatureNames.VISION,
              )
          )

    if self._audio_mixup:
      feat_name = FeatureNames.AUDIO if self._raw_audio else FeatureNames.AUDIO_MEL
      postprocess_fns.append(
          functools.partial(
              processing.batched_mixup,
              feature_name=feat_name,
              alpha=self._mixup_alpha,
              beta=self._mixup_beta,
              mixup_labels=False,
              )
          )

    num_post_processors = len(postprocess_fns)
    if num_post_processors == 0:
      postprocess_fns = None

    super(PreTrainLoader, self).__init__(
        dmvr_factory=ds_factories,
        params=input_params,
        postprocess_fns=postprocess_fns,
        num_epochs=-1,
        mode='train',
        name=dataset_id,
        )


class EvalLoader(loading.BaseLoader):
  """Gets parameters, split, and data name and returns data_loader instance."""

  def __init__(self,
               dataset_id,
               subset,
               params,
               split = None,
               ):
    # Generic parameters
    input_params = params.eval.input
    self._num_frames = input_params.num_frames
    self._frame_size = input_params.frame_size
    self._video_stride = input_params.video_stride
    self._audio_stride = input_params.audio_stride
    self._min_resize = input_params.frame_size
    self._raw_audio = input_params.raw_audio
    self._stft_length = input_params.stft_length
    self._stft_step = input_params.stft_step
    self._mel_bins = input_params.mel_bins
    self._multi_crop = input_params.multi_crop
    self._zero_centering_image = input_params.zero_centering_image
    self._max_num_words = input_params.max_num_words
    self._space_to_depth = input_params.space_to_depth

    if subset == 'train':
      self._mode = 'train'
      self._is_training = True
      self._num_epochs = input_params.num_augmentation
      self._color_augment = input_params.color_augment
      self._audio_mixup = input_params.audio_mixup
      self._num_windows_test = 1
      if self._num_epochs == 1:
        self._is_training = False
        self._color_augment = False
        self._audio_mixup = False
      else:
        self._min_area_ratio = input_params.min_area_ratio
        self._max_area_ratio = input_params.max_area_ratio
        self._min_aspect_ratio = input_params.min_aspect_ratio
        self._max_aspect_ratio = input_params.max_aspect_ratio
        self._mixup_alpha = input_params.mixup_alpha
        self._mixup_beta = input_params.mixup_beta
    else:
      self._mode = 'test'
      self._is_training = False
      self._num_epochs = 1
      self._num_windows_test = input_params.num_windows_test

    params_factory = {
        'is_training': self._is_training,
    }

    ref_fps = REF_FPS  # assume all train_ds were used
    if dataset_id in AUD_CLS_DS:
      sample_rate = DEFAULT_SR
      n_audio_secs = self._num_frames / ref_fps
      num_audio_samples = int(sample_rate * n_audio_secs)
      params_factory['num_samples'] = num_audio_samples
      params_factory['audio_stride'] = self._audio_stride

    else:
      params_factory.update({
          'num_frames': self._num_frames,
          'stride': self._video_stride,
          'min_resize': self._min_resize,
          'crop_size': self._frame_size,
          'zero_centering_image': self._zero_centering_image
      })

    if dataset_id in TEXT_DS:
      params_factory['max_num_words'] = self._max_num_words

    if self._mode == 'test':
      params_factory['num_test_clips'] = self._num_windows_test
      if dataset_id in VID_CLS_DS:
        params_factory['multi_crop'] = self._multi_crop

    # add augmentation-related parameters
    if self._is_training and dataset_id not in AUD_CLS_DS:
      params_factory.update({
          'crop_resize_style': 'Inception',
          'min_area_ratio': self._min_area_ratio,
          'max_area_ratio': self._max_area_ratio,
          'min_aspect_ratio': self._min_aspect_ratio,
          'max_aspect_ratio': self._max_aspect_ratio,
      })

    factory_args = {'subset': subset}
    if dataset_id in CLS_DS:
      factory_args['split'] = split

    factory_class = ds_fctr.get_ds_factory(
        dataset_name=dataset_id,
        )(**factory_args)
    ds_factory = factory_class.configure(**params_factory)

    if dataset_id in AUD_CLS_DS:
      if self._raw_audio:
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.extend_waveform_dim,
                num_windows=self._num_windows_test,
                ),
            feature_name=FeatureNames.AUDIO,
            fn_name='extend_waveform',
            )
      else:
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.raw_audio_to_spectrogram,
                sample_rate=DEFAULT_SR,
                stft_length=self._stft_length,
                stft_step=self._stft_step,
                mel_bins=self._mel_bins,
                num_windows=self._num_windows_test,
                specaugment=None,
                rm_audio=False,
                )
            )
        ds_factory.preprocessor_builder.add_fn(
            processing.normalize_spectrogram,
            feature_name=FeatureNames.AUDIO_MEL,
            fn_name='normalize_mel',
            )

    # Add batch-level data-agnostic post-processing functions
    postprocess_fns = []

    if self._space_to_depth:
      postprocess_fns.append(
          functools.partial(
              processing.space_to_depth,
              temporal_block_size=2,
              spatial_block_size=2,
              feature_name=FeatureNames.VISION,
              )
          )

    if self._is_training:
      if self._audio_mixup and dataset_id in AUD_CLS_DS:
        postprocess_fns.append([
            functools.partial(
                processing.batched_mixup,
                feature_name=(FeatureNames.AUDIO
                              if self._raw_audio else FeatureNames.AUDIO_MEL),
                alpha=self._mixup_alpha,
                beta=self._mixup_beta,
                mixup_labels=False,
            )
        ])

    split = '0' if split is None else str(split)
    name = dataset_id + '@' + split

    super(EvalLoader, self).__init__(
        dmvr_factory=ds_factory,
        params=input_params,
        postprocess_fns=postprocess_fns,
        num_epochs=self._num_epochs,
        mode=self._mode,
        name=name,
        )


class VisionFineTuneLoader(loading.BaseLoader):
  """Gets parameters, split, and data name and returns data_loader instance."""

  def __init__(self, dataset_id, params):
    # Generic parameters
    input_params = params.train.input
    assert input_params.has_data, 'Please provide a dataset name.'
    self._frame_size = input_params.frame_size
    self._zero_centering_image = input_params.zero_centering_image
    self._space_to_depth = input_params.space_to_depth
    self._linearize_vision = input_params.linearize_vision

    if dataset_id in VID_CLS_DS:
      self._num_frames = input_params.num_frames
      self._video_stride = input_params.video_stride

    # Augmentation parameters
    self._mixup = input_params.mixup
    self._mixup_alpha = input_params.mixup_alpha
    self._min_area_ratio = input_params.min_area_ratio
    self._max_area_ratio = input_params.max_area_ratio
    self._min_aspect_ratio = input_params.min_aspect_ratio
    self._max_aspect_ratio = input_params.max_aspect_ratio
    self._color_augment = input_params.color_augment
    self._label_smoothing = input_params.label_smoothing

    params_factory = {
        'is_training': True,
        'crop_size': self._frame_size,
        'crop_resize_style': 'Inception',
        'min_area_ratio': self._min_area_ratio,
        'max_area_ratio': self._max_area_ratio,
        'min_aspect_ratio': self._min_aspect_ratio,
        'max_aspect_ratio': self._max_aspect_ratio,
        'zero_centering_image': self._zero_centering_image,
    }
    if dataset_id in VID_CLS_DS:
      params_factory['num_frames'] = self._num_frames
      params_factory['stride'] = self._video_stride

    # Get the factory.
    factory_args = {'subset': 'train'}
    factory_class = ds_fctr.get_ds_factory(
        dataset_name=dataset_id,
        )(**factory_args)

    ds_factory = factory_class.configure(**params_factory)
    ds_factory.postprocessor_builder.add_fn(processing.remove_audio)

    # Add batch-level data-agnostic post-processing functions
    postprocess_fns = []
    if self._label_smoothing > 0.0:
      alpha = self._label_smoothing
      assert alpha <= 1.0, 'Please provide a valid smoothing factor'
      postprocess_fns.append(
          functools.partial(
              processing.label_smoothing,
              alpha=alpha,
              multi_label=False,
              )
          )

    if self._mixup:
      postprocess_fns.append(
          functools.partial(
              processing.batched_mixup,
              feature_name=FeatureNames.VISION,
              alpha=self._mixup_alpha,
              beta=self._mixup_alpha,
              mixup_labels=True,
              )
          )

    if self._space_to_depth and dataset_id in VID_CLS_DS:
      postprocess_fns.append(
          functools.partial(
              processing.space_to_depth,
              temporal_block_size=2,
              spatial_block_size=2,
              feature_name=FeatureNames.VISION,
              )
          )

    if self._linearize_vision:
      postprocess_fns.append(
          functools.partial(
              processing.linearize,
              feature_name=FeatureNames.VISION,
              )
          )

    num_post_processors = len(postprocess_fns)
    if num_post_processors == 0:
      postprocess_fns = None

    super(VisionFineTuneLoader, self).__init__(
        dmvr_factory=ds_factory,
        params=input_params,
        postprocess_fns=postprocess_fns,
        num_epochs=-1,
        mode='train',
        name=dataset_id,
        )


class VisionEvalLoader(loading.BaseLoader):
  """Gets parameters, split, and data name and returns data_loader instance."""

  def __init__(self, dataset_id, params):
    # Generic parameters
    input_params = params.eval.input
    assert input_params.has_data, 'Please provide a dataset name.'
    self._frame_size = input_params.frame_size
    self._zero_centering_image = input_params.zero_centering_image
    self._space_to_depth = input_params.space_to_depth
    self._linearize_vision = input_params.linearize_vision

    if dataset_id in VID_CLS_DS:
      self._num_frames = input_params.num_frames
      self._video_stride = input_params.video_stride
      self._multi_crop = input_params.multi_crop
      self._num_windows_test = input_params.num_windows_test

    with tf.name_scope('input_{}_test'.format(dataset_id)):
      params_factory = {
          'is_training': False,
          'min_resize': self._frame_size,
          'crop_size': self._frame_size,
          'zero_centering_image': self._zero_centering_image
      }

      if dataset_id in VID_CLS_DS:
        params_factory['num_frames'] = self._num_frames
        params_factory['stride'] = self._video_stride
        params_factory['num_test_clips'] = self._num_windows_test
        params_factory['multi_crop'] = self._multi_crop

      factory_args = {'subset': 'test'}

      if dataset_id.lower().startswith('kinetics'):
        factory_args['subset'] = 'valid'

      factory_class = ds_fctr.get_ds_factory(
          dataset_name=dataset_id,
          )(**factory_args)
      ds_factory = factory_class.configure(**params_factory)
      ds_factory.postprocessor_builder.add_fn(processing.remove_audio)

    postprocess_fns = []

    if self._space_to_depth and dataset_id in VID_CLS_DS:
      postprocess_fns.append(
          functools.partial(
              processing.space_to_depth,
              temporal_block_size=2,
              spatial_block_size=2,
              feature_name=FeatureNames.VISION,
              )
          )

    if self._linearize_vision:
      postprocess_fns.append(
          functools.partial(
              processing.linearize,
              feature_name=FeatureNames.VISION,
              )
          )

    num_post_processors = len(postprocess_fns)
    if num_post_processors == 0:
      postprocess_fns = None

    super(VisionEvalLoader, self).__init__(
        dmvr_factory=ds_factory,
        params=input_params,
        postprocess_fns=postprocess_fns,
        num_epochs=1,
        mode='eval',
        name=dataset_id,
        )


class AudioFineTuneLoader(loading.BaseLoader):
  """Gets parameters, split, and data name and returns data_loader instance."""

  def __init__(self, dataset_id, params):
    # Generic parameters
    input_params = params.train.input
    assert input_params.has_data, 'Please provide a dataset name.'
    self._num_frames = input_params.num_frames
    self._video_stride = input_params.video_stride
    self._audio_stride = input_params.audio_stride
    self._raw_audio = input_params.raw_audio
    self._stft_length = input_params.stft_length
    self._stft_step = input_params.stft_step
    self._mel_bins = input_params.mel_bins
    n_audio_secs = self._num_frames / REF_FPS
    self._num_samples = int(REF_SR * n_audio_secs)

    # Augmentation parameters
    self._audio_noise = input_params.audio_noise
    self._mixup = input_params.mixup
    self._mixup_alpha = input_params.mixup_alpha

    params_factory = {
        'is_training': True,
        'num_samples': self._num_samples,
        'stride': self._video_stride,
        'audio_stride': self._audio_stride,
    }

    with tf.name_scope('input_{}_train'.format(dataset_id)):
      # Get the factory.
      factory_args = {'subset': 'train'}
      factory_class = ds_fctr.get_ds_factory(
          dataset_name=dataset_id,
          )(**factory_args)

      ds_factory = factory_class.configure(**params_factory)

      ds_factory.postprocessor_builder.add_fn(processing.remove_vision)
      # Add audio preprocessing.
      if self._audio_noise > 0.:
        # Add gaussian noise
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.add_gaussian,
                gamma=self._audio_noise,
                ),
            feature_name=FeatureNames.AUDIO,
            fn_name='volume_gaussian'
            )

      if not self._raw_audio:
        ds_factory.preprocessor_builder.add_fn(
            functools.partial(
                processing.raw_audio_to_spectrogram,
                sample_rate=REF_SR,
                stft_length=self._stft_length,
                stft_step=self._stft_step,
                mel_bins=self._mel_bins,
                rm_audio=True
                )
            )
        ds_factory.preprocessor_builder.add_fn(
            processing.normalize_spectrogram,
            feature_name=FeatureNames.AUDIO_MEL,
            fn_name='normalize_mel',
            )

      postprocess_fns = []
      if self._mixup:
        postprocess_fns.append(
            functools.partial(
                processing.batched_mixup,
                feature_name=(FeatureNames.AUDIO
                              if self._raw_audio else FeatureNames.AUDIO_MEL),
                alpha=self._mixup_alpha,
                beta=self._mixup_alpha,
                mixup_labels=True,
                )
            )

    num_post_processors = len(postprocess_fns)
    if num_post_processors == 0:
      postprocess_fns = None

    super(AudioFineTuneLoader, self).__init__(
        dmvr_factory=ds_factory,
        params=input_params,
        postprocess_fns=postprocess_fns,
        num_epochs=-1,
        mode='train',
        name=dataset_id,
        )


class AudioEvalLoader(loading.BaseLoader):
  """Gets parameters, split, and data name and returns data_loader instance."""

  def __init__(self, dataset_id, params):
    # Generic parameters
    input_params = params.eval.input
    assert input_params.has_data, 'Please provide a dataset name.'
    self._num_frames = input_params.num_frames
    self._video_stride = input_params.video_stride
    self._audio_stride = input_params.audio_stride
    self._raw_audio = input_params.raw_audio
    self._stft_length = input_params.stft_length
    self._stft_step = input_params.stft_step
    self._mel_bins = input_params.mel_bins
    self._video_stride = self._video_stride * int(DEFAULT_FPS // REF_FPS)
    n_audio_secs = self._num_frames / REF_FPS
    self._num_samples = int(REF_SR * n_audio_secs)
    self._num_windows_test = input_params.num_windows_test

    params_factory = {
        'is_training': False,
        'num_samples': self._num_samples,
        'stride': self._video_stride,
        'audio_stride': self._audio_stride,
        'num_test_clips': self._num_windows_test,
    }

    factory_args = {'subset': 'test'}
    factory_class = ds_fctr.get_ds_factory(
        dataset_name=dataset_id,
        )(**factory_args)
    ds_factory = factory_class.configure(**params_factory)

    ds_factory.postprocessor_builder.add_fn(processing.remove_vision)

    # Add audio preprocessing.
    if not self._raw_audio:
      ds_factory.preprocessor_builder.add_fn(
          functools.partial(
              processing.raw_audio_to_spectrogram,
              sample_rate=REF_SR,
              stft_length=self._stft_length,
              stft_step=self._stft_step,
              mel_bins=self._mel_bins,
              rm_audio=True
              )
          )
      ds_factory.preprocessor_builder.add_fn(
          processing.normalize_spectrogram,
          feature_name=FeatureNames.AUDIO_MEL,
          fn_name='normalize_mel',
          )

    super(AudioEvalLoader, self).__init__(
        dmvr_factory=ds_factory,
        params=input_params,
        postprocess_fns=None,
        num_epochs=1,
        mode='eval',
        name=dataset_id,
        )
