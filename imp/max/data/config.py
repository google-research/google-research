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

"""Configuration of the data pipeline."""

import dataclasses
from typing import Any

import fiddle as fdl
from flax import struct

from imp.max.config import base
from imp.max.config import validators
from imp.max.core import constants
from imp.max.data import processing as proc_utils
from imp.max.utils import typing


ParsingFeatureName = constants.ParsingFeatureName
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
Bundle = fdl.Config


# ----------------------------------------------------------------------
# ---------------------------- Processing ------------------------------
# ----------------------------------------------------------------------


@dataclasses.dataclass
class Modality(base.Config):
  """BaseClass for Modality specification.

  Attributes:
   is_training: Whether or not in training mode.
  """
  is_training: bool | None = None


@dataclasses.dataclass
class CustomModality(base.Config):
  """Configuration of a modality with fully customizable modules.

  Attributes:
    parsing: An instance of a (sequence of) configured Bundle(s) that parse
      certain features from a data table. These Bundles should carry functions
      that accept an instance of a ParserBuilder as input.
    sampling: An instance of a (sequence of) configured Bundle(s) that sample
      the parsed features. These Bundles should carry functions that accept an
      instance of a SamplerBuilder as input.
    decoding: An instance of a (sequence of) configured Bundle(s) that decode
      the sampled features. These Bundles should carry functions that accept an
      instance of a DecoderBuilder as input.
    preprocessing: An instance of a (sequence of) configured Bundle(s) that
      process the raw features in a specific order. These Bundles should
      carry functions that operate on features dictionary.
    preprocessing: An instance of a (sequence of) configured Bundle(s) that
      post-process the features in a specific order. These Bundles should
      carry functions that operate on features dictionary.
  """
  parsing: Bundle | tuple[Bundle, Ellipsis] | None = None
  sampling: Bundle | tuple[Bundle, Ellipsis] | None = None
  decoding: Bundle | tuple[Bundle, Ellipsis] | None = None
  preprocessing: Bundle | tuple[Bundle, Ellipsis] | None = None
  postprocessing: Bundle | tuple[Bundle, Ellipsis] | None = None


@dataclasses.dataclass
class Tokenizer(base.Config):
  """Configuration if a tokenizer is required.

  Attributes:
    name: Tokenizer to use.
    initialize: Whether the tokenizer should be initialized.
  """
  name: str = constants.T5_EN
  initialize: bool = False


@dataclasses.dataclass
class Label(base.Config):
  """Configuration if a label reader and processor is desired.

  Attributes:
    parsing_label_index_feature_name: Name of the label index feature in the
      input `tf.train.Example` or `tf.train.SequenceExample`. Exposing this as
      an argument allows using this function for different label features within
      a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    is_multi_label: Whether raw data contains multiple labels per example.
    one_hot_label: Return labels as one hot tensors. If `is_multi_label` is
      `True`, one hot tensor might have multiple ones.
    num_classes: Total number of classes in the dataset. It has to be provided
      if `one_hot_label` is `True`.
    smoothing: Label smoothing alpha value for one-hot labels.
    normalize: If `True` and `add_label_index=True`, one-hot label indices
      will be normalized so that the channels sum to 1.
  """
  parsing_label_index_feature_name: str = ParsingFeatureName.CLIP_LABEL_INDEX
  data_collection_type: str = DataFeatureType.TARGETS
  data_collection_route: str = DataFeatureRoute.LABEL_CLASSIFIER
  is_multi_label: bool = False
  one_hot_label: bool = True
  num_classes: int | None = None
  smoothing: float = 0.0
  normalize: bool = False


@dataclasses.dataclass
class Annotation(base.Config):
  """BaseClass for Annotations specification.

  Attributes:
   label: Configuration of label classes.
   bounding_box: Configuration of bounding boxes (used for vision modality).
   segmentation: Configuration of any segmentation mask (either temporal or
     spatial).
  """
  label: Label | None = None
  bounding_box: Any | None = None
  segmentation: Any | None = None


@dataclasses.dataclass
class Vision(Modality):
  """Configuration if a vision reader and processor is desired.

  Attributes:
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_rgb: Whether to keep raw RGB pixels.
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
    temporal_patch_size: Frames are sampled in temporal_patch_size regions.
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
    is_rgb: If `True`, the number of channels in the JPEG is 3, if False, 1. If
      is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    droptoken_rate: The rate of which tokens are dropped.
    annotation: The configuration of annotations.
    dtype: the dtype to cast the output to.
  """
  parsing_feature_name: str = ParsingFeatureName.IMAGE_ENCODED
  data_collection_type: str = DataFeatureType.INPUTS
  data_collection_route: str = DataFeatureRoute.ENCODER
  keep_raw_rgb: bool = False
  is_training: bool | None = None
  temporal_patch_size: int = 8
  spatial_patch_size: tuple[int, int] = (16, 16)
  spatio_temporal_token_coordinate: bool = True
  min_resize: int = 256
  crop_size: int = 224
  crop_resize_style: str = constants.CropStyle.INCEPTION
  min_aspect_ratio: float = 0.5
  max_aspect_ratio: float = 2
  min_area_ratio: float = 0.5
  max_area_ratio: float = 1.0
  zero_centering_image: bool = True
  color_augmentation: bool = True
  random_flip: bool = True
  num_test_clips: int = 1
  num_frames: int = 1
  stride: int = 1
  multi_crop: bool = False
  sync_random_state: bool = True
  is_rgb: bool = True
  is_flow: bool = False
  token_drop_rate: float = 0.
  annotation: Annotation | None = None
  dtype: str = 'float32'


@dataclasses.dataclass
class Video(Vision):
  num_frames: int = 32


@dataclasses.dataclass
class Waveform(Modality):
  """Configuration if an audio waveform reader and processor is desired.

  Attributes:
    parsing_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different audio features within a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_waveform: Whether to keep raw waveform signal.
    num_samples: Number of samples per subclip.
    stride: Temporal stride to sample audio signal.
    temporal_patch_size: Waveforms are sampled in temporal_patch_size regions
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggreagated in the batch dimension.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
    token_drop_rate: The rate of which tokens are dropped.
    annotation: The configuration of annotations.
    dtype: the dtype to cast the output to.
  """
  parsing_feature_name: str = ParsingFeatureName.WAVEFORM
  data_collection_type: str = DataFeatureType.INPUTS
  data_collection_route: str = DataFeatureRoute.ENCODER
  keep_raw_waveform: bool = False
  num_samples: int = 153600  # 48000 (Hz) * 32 / 10 (fps)
  stride: int = 1
  temporal_patch_size: int = 256
  num_test_clips: int = 1
  sync_random_state: bool = True
  token_drop_rate: float = 0.
  annotation: Annotation | None = None
  dtype: str = 'float32'


@dataclasses.dataclass
class Spectrogram(Modality):
  """Configuration if an audio waveform reader and processor is desired.

  Attributes:
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
    stride: Temporal stride to sample raw waveform signal.
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
  """
  parsing_feature_name: str = ParsingFeatureName.WAVEFORM
  data_collection_type: str = DataFeatureType.INPUTS
  data_collection_route: str = DataFeatureRoute.ENCODER
  keep_waveform_features: bool = True
  keep_raw_waveform: bool = False
  keep_raw_spectrogram: bool = False
  is_training: bool = True
  num_raw_waveform_samples: int = 153600  # 48000 (Hz) * 32 / 10 (fps)
  waveform_stride: int = 1
  waveform_temporal_patch_size: int = 256
  temporal_patch_size: int = 4
  spectoral_patch_size: int = 5
  sample_rate: int = 48000
  spectrogram_type: str = 'logmf'
  frame_length: int = 2048
  frame_step: int = 1024
  num_features: int = 80
  lower_edge_hertz: float = 80.0
  upper_edge_hertz: float = 7600.0
  preemphasis: float | None = None
  normalize_audio: bool = False
  num_test_clips: int = 1
  token_drop_rate: float = 0.
  annotation: Annotation | None = None
  dtype: str = 'float32'


@dataclasses.dataclass
class Text(Modality):
  """Configuration if a text reader and processor is desired.

  Attributes:
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
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    max_num_sentences: Maximum number of captions to keep. If there are more
      captions in the proto, only the first `max_num_sentences` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_sentences` will be randomly sampled. Finally if the proto
      contains less than `max_num_sentences`, we pad with empty srings to make
      sure there are `max_num_sentences` in total.
    max_num_tokens: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id.
    max_context_sentences: Maximum number of temporal neighboring sentences to
       keep.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations used for sampling
      the captions.
    is_multi_language: Whether to add a language feature indicating the language
      of the text.
    tokenizer_specs: Tokenizer to use for text tokenization.
    annotation: The configuration of annotations.
    dtype: the dtype to cast the output to.
  """
  parsing_feature_name: str = ParsingFeatureName.CAPTION
  parsing_context_feature_name: str = ParsingFeatureName.CONTEXT_INPUT
  parsing_language_feature_name: str | None = (
      ParsingFeatureName.EXAMPLE_LABEL_LANGUAGE)
  data_collection_type: str = DataFeatureType.INPUTS
  data_collection_route: str = DataFeatureRoute.ENCODER
  keep_raw_string: bool = False
  prepend_bos: bool = False
  append_eos: bool = False
  max_num_sentences: int = 1
  max_num_tokens: int = 16
  max_context_sentences: int = 0
  sync_random_state: bool = True
  is_multi_language: bool = False
  tokenizer_specs: Tokenizer = dataclasses.field(default_factory=Tokenizer)
  annotation: Annotation | None = None
  dtype: str = 'float32'


@dataclasses.dataclass
class TextFromLabel(Modality):
  """Configuration if a label reader and processor is desired.

  Attributes:
    parsing_label_name_feature_name: Name of the label name feature in the input
      `tf.train.Example` or `tf.train.SequenceExample`. Exposing this as an
      argument allows using this function for different label features within
      a single dataset.
    data_collection_type: The type of the collection in which the features will
      be stored.
    data_collection_route: The routing collection in which the features will be
      stored.
    keep_raw_string: Whether to keep raw string.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    is_training: Whether or not in training mode. This will be used to randomly
      sample the captions.
    is_multi_label: Whether raw data contains multiple labels per example.
    add_label_name_in_sentence: Return a complete sentence with label name.
    modality: The modality whose labels are being turned to sentences.
    max_num_sentences: Maximum number of sentences to keep. If there are more
      captions in the proto, only the first `max_num_sentences` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_sentences` will be randomly sampled. Finally if the proto
      contains less than `max_num_sentences`, we pad with empty srings to make
      sure there are `max_num_sentences` in total.
    max_num_tokens: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id.
    label_remap_dictionary_path: if not None, creates a table from the label map
      path that maps keys to label strings.
    tokenizer_specs: Tokenizer to use for text tokenization.
    annotation: The configuration of annotations.
    keep_labels: if False, remove the label indices from the data. This is
      useful for batching multiple datasets on the text field in the case of
      multiple incompatible labels. Only applicable to custom factories that map
      label indices to names.
    dtype: the dtype to cast the output to.
  """

  parsing_label_name_feature_name: str | None = (
      ParsingFeatureName.CLIP_LABEL_TEXT)
  data_collection_type: str = DataFeatureType.INPUTS
  data_collection_route: str = DataFeatureRoute.ENCODER
  keep_raw_string: bool = False
  prepend_bos: bool = False
  append_eos: bool = False
  is_training: bool = True
  is_multi_label: bool = False
  add_label_name_in_sentence: bool = True
  modality: str = constants.Modality.IMAGE
  max_num_sentences: int = 1
  max_num_tokens: int = 16
  label_remap_dictionary_path: str | None = None
  tokenizer_specs: Tokenizer = dataclasses.field(default_factory=Tokenizer)
  annotation: Annotation | None = None
  keep_labels: bool = True
  dtype: str = 'float32'


@dataclasses.dataclass
class Modalities:
  """Configuration of all desired modalities for a dataset.

  Attributes:
    audio: Configuration for audio modality.
    vision:  Configuration for vision modality.
    text: Configuration for text modality.
  """
  audio: Waveform | Spectrogram | CustomModality | None = None
  vision: Vision | CustomModality | None = None
  text: Text | TextFromLabel | CustomModality | None = None


# ----------------------------------------------------------------------
# ---------------------------- DataLoaders -----------------------------
# ----------------------------------------------------------------------


@dataclasses.dataclass
class BaseDataDefinition(base.Config):
  """The base configuration for data retrieval."""
  table: str
  prop_data: float = 1.0
  prop_seed: int | None = None
  num_shards: int | None = None
  shard_index: int | None = None


@dataclasses.dataclass
class ThirdPartyDataDefinition(BaseDataDefinition):
  """Configuration for data prepared using ThirdPartyBase."""
  base_dir: str | None = None
  tables: dict[str, str] | None = None
  num_examples: dict[str, int] | None = None
  source: str = constants.TFRECORD
  raw_data_format: str = constants.TF_SEQUENCE_EXAMPLE




@dataclasses.dataclass
class TfdsDataDefinition(BaseDataDefinition):
  """Configuration for data prepared using SeqioData."""

  dataset_name: str = ''


@dataclasses.dataclass
class SeqioDataDefinition(BaseDataDefinition):
  """Configuration for data prepared using SeqioData."""

  task_name: str = ''


DataLoaderT = (
    ThirdPartyDataDefinition | TfdsDataDefinition | SeqioDataDefinition
)


@dataclasses.dataclass
class Dataset(base.Config):
  """The base configuration for retrieving and processing datasets."""

  name: str
  data: DataLoaderT
  modalities: Modalities
  factory: str = constants.DataFactory.THIRDPARTY
  # TODO(b/241179079): remove is_training and only keep in `Loader` below
  is_training: bool = False


@struct.dataclass
class Metadata:
  """A structured metadata to guide model and objectives how to route data.

  Attributes:
    dataflow: A sequence of datapasses each representing how each forward pass
      should cosume the input data. These datapasses also include how the model
      outputs and their corresponding targets should be structured. These are
      also used later in the objective functions to dictate how the input/output
      /target triplet is utilized for calculating the loss value.
    taskflow: A sequence of taskpasses each representing certain metadata about
      the task being solved in each datapass. This should either have the same
      number of elements as dataflow, or be empty.
  """
  dataflow: tuple[typing.Data, Ellipsis] = struct.field(pytree_node=False)
  taskflow: tuple[typing.Data, Ellipsis] = struct.field(pytree_node=False)

  def __post_init__(self):
    if ((self.dataflow and self.taskflow)
        and len(self.dataflow) != len(self.taskflow)):
      raise ValueError(
          '`dataflow` and `taskflow` should have the same number of elements. '
          f'Instead, received {len(self.dataflow)} and {len(self.taskflow)}.')


@dataclasses.dataclass
class Loader(base.Config):
  """The base configuration for the DataLoader."""

  dataset: Dataset | tuple[Dataset, Ellipsis]
  metadata: Metadata | None = None
  sampling_weights: tuple[float, Ellipsis] = ()
  serving: str | tuple[str, Ellipsis] = constants.ServingStrategy.PRETRAIN
  interval: int = 1
  batch_size: int | None = None
  microbatch_splits: int | None = None
  shuffle: bool | None = None
  is_training: bool | None = None
  seed: int | None = None
  num_epochs: int | None = None
  use_data_service: bool = False
  data_service_address: str | None = None
  data_service_sharding_policy: int | None = None
  shuffle_buffer_multiplier: float = 1.
  prefetch: int = 2
  tuning: str = constants.DataTuning.FAST

  def __post_init__(self):
    """Propagates loader constraints to dataset."""

    if isinstance(self.dataset, tuple):
      if len(self.dataset) != len(self.sampling_weights):
        raise ValueError(
            'Number of datasets does not match number of sampling rates: '
            f'{len(self.dataset)} vs. {len(self.sampling_weights)}.')

      if self.is_training is not None:
        for dataset in self.dataset:
          dataset.is_training = self.is_training

    elif self.is_training is not None:
      self.dataset.is_training = self.is_training


@validators.lock
@dataclasses.dataclass
class ExperimentData(base.Config):
  """Experiment data configuration."""

  # These are all canonical config params. If any of them is specified by
  # user, all corresponding loader-specific params will be overridden.
  vision_spatial_size: tuple[int, int] | None = None
  vision_spatial_patch_size: tuple[int, int] | None = None
  vision_temporal_size: int | None = None
  vision_temporal_patch_size: int | None = None
  waveform_temporal_size: int | None = None
  waveform_temporal_patch_size: int | None = None
  spectrogram_temporal_patch_size: int | None = None
  spectrogram_spectoral_patch_size: int | None = None
  text_size: int | None = None
  is_training: bool | None = None
  shuffle: bool | None = None
  num_epochs: int | None = None
  batch_size: int | None = None
  microbatch_splits: int | None = None
  tokenizer: Tokenizer | None = None
  loaders: tuple[Loader, Ellipsis] = ()
  checkpointing: bool = False
  dtype: str = 'float32'

  def __post_init__(self):
    """Propagates experiment constraints to each loader."""

    # Loaders should be copied to avoid overriding multiple configs that
    # share a reference to the same loader config.
    self.loaders: tuple[Loader, Ellipsis] = tuple(
        loader.copy() for loader in self.loaders)

    # override loader-specific configs if canonical_params is provided
    for loader in self.loaders:
      if isinstance(loader.dataset, tuple):
        loader_datasets = loader.dataset
      else:
        loader_datasets = (loader.dataset,)

      if self.batch_size is not None:
        loader.batch_size = self.batch_size

      if self.microbatch_splits is not None:
        loader.microbatch_splits = self.microbatch_splits

      if self.shuffle is not None:
        loader.shuffle = self.shuffle

      if self.num_epochs is not None:
        loader.num_epochs = self.num_epochs

      if self.is_training is not None:
        loader.is_training = self.is_training
        for dataset in loader_datasets:
          dataset.is_training = self.is_training
          if dataset.modalities.audio:
            dataset.modalities.audio.is_training = self.is_training
          if dataset.modalities.text:
            dataset.modalities.text.is_training = self.is_training
          if dataset.modalities.vision:
            dataset.modalities.vision.is_training = self.is_training

      for dataset in loader_datasets:
        self._update_modalites(dataset)
        self._update_tokenizers(dataset)

    self._validate_configs()

  def _update_modalites(self, dataset):
    self._update_vision_params(dataset.modalities.vision, dataset.is_training)
    self._update_audio_params(dataset.modalities.audio)
    self._update_text_params(dataset.modalities.text)

  def _update_vision_params(self, params, is_training):
    """Propagates experiment-wide vision constraints."""
    if params is None:
      return
    if self.vision_spatial_size is not None:
      params.crop_size = self.vision_spatial_size[0]
      if is_training:
        # adjust min_resize accordingly. we calculate min_resize by finding
        # the nearest multiple of 8 to (256/224 * crop_size) ; which is an
        # arguable convention in vision
        params.min_resize = proc_utils.get_min_resize_value(params.crop_size)
      else:
        # otherwise, we force min_resize to be the same as crop_size
        params.min_resize = params.crop_size

    if self.vision_spatial_patch_size is not None:
      params.spatial_patch_size = self.vision_spatial_patch_size

    if self.vision_temporal_size is not None:
      params.num_frames = self.vision_temporal_size

    if self.vision_temporal_patch_size is not None:
      params.temporal_patch_size = self.vision_temporal_patch_size

    params.dtype = self.dtype

  def _update_audio_params(self, params):
    """Propagates experiment-wide audio constraints."""
    if params is None:
      return
    if isinstance(params, Waveform):
      if self.waveform_temporal_size is not None:
        params.num_samples = self.waveform_temporal_size

      if self.waveform_temporal_patch_size is not None:
        params.temporal_patch_size = self.waveform_temporal_patch_size

    elif isinstance(params, Spectrogram):
      if self.waveform_temporal_size is not None:
        params.num_raw_waveform_samples = self.waveform_temporal_size

      if self.waveform_temporal_patch_size is not None:
        params.waveform_temporal_patch_size = self.waveform_temporal_patch_size

      if self.spectrogram_temporal_patch_size is not None:
        params.temporal_patch_size = self.spectrogram_temporal_patch_size

      if self.spectrogram_spectoral_patch_size is not None:
        params.spectoral_patch_size = self.spectrogram_spectoral_patch_size

    params.dtype = self.dtype

  def _update_text_params(self, params):
    """Propagates experiment-wide text constraints."""
    if params is None:
      return
    if self.text_size is not None:
      params.max_num_tokens = self.text_size

    params.dtype = self.dtype

  def _update_tokenizers(self, dataset):
    """Propagates experiment-wide tokenizer to the loader."""
    if self.tokenizer is None:
      return
    if dataset.modalities.text is not None:
      dataset.modalities.text.tokenizer_specs = self.tokenizer

  def _validate_configs(self):
    for loader in self.loaders:
      if isinstance(loader.dataset, tuple):
        loader_datasets = loader.dataset
      else:
        loader_datasets = (loader.dataset,)

      dataset_names = [dataset.name for dataset in loader_datasets]
      if any(dataset_name is None for dataset_name in dataset_names):
        raise ValueError('Dataset name should be a valid string.')
      else:
        name = '_'.join(dataset_names)

      if loader.is_training is None:
        raise ValueError('is_training should be set in Loader, is None for '
                         f'{loader}')

      # In order to improve efficiency on distributed systems, we require
      # a multiple of 8 for batch sizes across all datasets
      if (loader.batch_size is None
          or loader.batch_size % 8 != 0
          or loader.batch_size == 0):
        raise ValueError(
            '`batch_size` should always be a multiple of 8. Configured as '
            f'{loader.batch_size} for dataset {name}.')

      if loader.microbatch_splits is None:
        loader.microbatch_splits = 1
      elif loader.microbatch_splits <= 0:
        raise ValueError(
            '`microbatch_splits` should be configured as a positive integer. '
            f'Configured as {loader.microbatch_splits} for dataset {name}.')

      if loader.shuffle is None:
        raise ValueError(
            '`shuffle` should be configured as True/False. Configured as '
            f'{loader.shuffle} for dataset {name}.')

      if loader.num_epochs is None or (loader.num_epochs == 0):
        raise ValueError(
            '`num_epochs` should be configured as a non-zero int. Configured '
            f'as {loader.num_epochs} for dataset {name}.')

      if any(dataset.is_training is None for dataset in loader_datasets):
        raise ValueError(
            '`is_training` should be configured as True/False. Configured as '
            f'{[dataset.is_training for dataset in loader_datasets]} '
            f'for dataset {name}.')

  def update_data_service_address(self, address):
    """Updates the data service address for all dataloaders."""
    for loader in self.loaders:
      loader.data_service_address = address
      if not address:
        loader.use_data_service = False
