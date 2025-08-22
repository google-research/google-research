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

"""Definition of constants needed to handle data."""

import enum

# Tokenizers
TOKENIZER_ATTRIBUTE = 'tokenizer'
BERT = 'bert'
HOWTO100M = 'howto100m'

# Text Vocabulary
BERT_EN = 'bert_uncased_en'
HOWTO100M_EN = 'howto100m_en'
T5_EN = 't5_en'
ALL_VOCABULARIES = [HOWTO100M_EN, BERT_EN, T5_EN]
VOCABULARY_DIR = 'path/to/vocabulary/'
EMBEDDING_DIR = 'path/to/embedding/'
T5_DEFAULT_SPM_PATH = 'path/to/t5_default_spm.model'

# Data files
TXT_SUFFIX = '.txt'
TFRECORD = 'tfrecord'
TF_EXAMPLE = 'tf_example'
TF_SEQUENCE_EXAMPLE = 'tf_sequence_example'


class Mode(enum.Enum):
  """Experiment mode for training or evaluation."""
  TRAIN: str = 'train'
  EVAL: str = 'eval'


class Task(enum.Enum):
  """Experiment task for pre-/training or finetuning."""
  TRAIN: str = 'train'
  PRETRAIN: str = 'pretrain'
  FINETUNE: str = 'finetune'


class TextVocabSize:
  """Default vocab size of the default dictionaries."""
  T5_EN: int = 32100


class LabelReMap:
  """Available label maps."""
  DATASET_OF_CHOICE = 'path/to/json'


# Crop Styles
class CropStyle:
  """Available crop styles."""
  INCEPTION: str = 'Inception'
  VGG: str = 'VGG'


# Modalities
class Modality:
  """Available modalities."""
  VIDEO: str = 'video'
  IMAGE: str = 'image'
  WAVEFORM: str = 'waveform'
  SPECTROGRAM: str = 'spectrogram'
  VISION: str = 'vision'
  AUDIO: str = 'audio'
  TEXT: str = 'text'


# Feature names
class ParsingFeatureName:
  """Feature names of keys in the input data table to be parsed."""
  IMAGE_ENCODED: str = 'image/encoded'
  WAVEFORM: str = 'WAVEFORM/feature/floats'
  CAPTION: str = 'caption/string'
  CONTEXT_INPUT: str = 'context/negative/string'
  CLIP_LABEL_INDEX: str = 'clip/label/index'
  CLIP_LABEL_STRING: str = 'clip/label/string'
  CLIP_LABEL_TEXT: str = 'clip/label/text'
  EXAMPLE_ANNOTATION_STRING: str = 'example/annotation/string'
  EXAMPLE_LABEL_INDEX: str = 'example/label/index'
  EXAMPLE_LABEL_STRING: str = 'example/label/string'
  EXAMPLE_LABEL_STRING_MID: str = 'example/label/string_mid'
  EXAMPLE_LABEL_TITLE: str = 'example/label/title'
  EXAMPLE_LABEL_CAPTION: str = 'example/label/caption'
  EXAMPLE_LABEL_CAPTION_TRANSLATED: str = 'example/label/caption/translated'
  EXAMPLE_LABEL_LANGUAGE: str = 'example/label/language'
  TEXT: str = 'text'


class DataFeatureType:
  """The type/category of the data features."""
  INPUTS: str = 'inputs'
  OUTPUTS: str = 'outputs'
  TARGETS: str = 'targets'
  CONDITIONS: str = 'conditions'
  METADATA: str = 'metadata'
  HYPERPARAMS: str = 'hyperparams'


class DataFeatureRoute:
  """The type/category of the data routing."""
  ENCODER: str = 'encoder'
  DECODER: str = 'decoder'
  LABEL_CLASSIFIER: str = 'label_classifier'
  TARGET_CLASSIFIER: str = 'target_classifier'
  COMMON_SPACE: str = 'common_space'


class DataFeatureName:
  """Common names of input/target/model features."""
  # Input/Target features
  RAW: str = 'raw'
  LABEL: str = 'label'
  TOKEN_RAW: str = 'token_raw'
  TOKEN_ID: str = 'token_id'
  TOKEN_EMBED: str = 'token_embed'
  TOKEN_MASK: str = 'token_mask'
  TOKEN_COORDINATE: str = 'token_coordinate'
  TOKEN_POSITION_ID: str = 'token_position_id'
  TOKEN_SEGMENT_ID: str = 'token_segment_id'
  DROP_COORDINATE: str = 'drop_coordinate'
  DROP_POSITION_ID: str = 'drop_position_id'
  LANGUAGE: str = 'language'
  # Model features
  FEATURES: str = 'features'
  FEATURES_AGG: str = 'features_agg'
  FEATURES_ALL: str = 'features_all'
  FEATURE_MAPS: str = 'feature_maps'
  LOGITS: str = 'logits'
  TEMPERATURE: str = 'temperature'


class MetadataName:
  """Common names of the metadata entries."""
  DATAFLOW: str = 'dataflow'
  TASKFLOW: str = 'taskflow'


class TaskFlowName:
  """Common names of the taskflow entries."""
  DECODING_MODE: str = 'decoding_mode'


class DataFeatureRank:
  """Feature ranks for each modality."""

  class Common:
    """Common feature ranks."""
    LABEL: int = 3  # (batch, instance, classes)
    TOKEN_RAW: int = 4  # (batch, instance, tokens, pixels/samples)
    TOKEN_ID: int = 3  # (batch, instance, tokens)
    TOKEN_MASK: int = 3  # (batch, instance, tokens)
    TOKEN_POSITION_ID: int = 3  # (batch, instance, tokens)
    TOKEN_SEGMENT_ID: int = 3  # (batch, instance, tokens)
    DROP_COORDINATE: int = 4  # (batch, instance, dropped_tokens, coordinates)
    DROP_POSITION_ID: int = 3  # (batch, instance, dropped_tokens)

  class Vision(Common):
    """Vision feature ranks."""
    RAW: int = 6  # (batch, instance, frames, height, width, channels)
    TOKEN_COORDINATE: int = 4  # (batch, instance, tokens, coordinates)

  class Waveform(Common):
    """Waveform feature ranks."""
    RAW: int = 4  # (batch, instance, samples, channels)
    TOKEN_COORDINATE: int = 3  # (batch, instance, tokens)

  class Spectrogram(Common):
    """Spectrogram feature ranks."""
    RAW: int = 5  # (batch, instance, samples, spectrum, channels)
    TOKEN_COORDINATE: int = 4  # (batch, instance, tokens, coordinates)

  class Text(Common):
    """Text feature ranks."""
    RAW: int = 2  # (batch, instance)
    LANGUAGE: int = 2  # (batch, instance)
    TOKEN_COORDINATE: int = 3  # (batch, instance, tokens)


class ProbeDataRank:
  """Probe data ranks for each modality."""
  IMAGE: int = 4  # (instance, height, width, channels)
  WAVEFORM: int = 3  # (instance, samples, channels)
  TEXT: int = 2  # (instance, length)
  SCALAR: int = 0  # ()
  HISTOGRAM: int = 1  # (instance,)


class DataFactory:
  """Available data factories."""
  SEQIO = 'seqio'
  THIRDPARTY = 'thirdparty'
  TFDS = 'tfds'


class ServingStrategy:
  """Serving strategies, useful for routing each dataset to a proper runner."""
  CLASSIFICATION = 'classification'
  PRETRAIN = 'pretrain'
  RETRIEVAL = 'retrieval'
  BULK_ZS_RETRIEVAL = 'bulk_zero_shot_retrieval'
  BULK_TEST_PREDICT_CLS = 'bulk_test_predict_classification'
  BULK_LINEAR_CLS = 'bulk_linear_classification'
  BULK_ZS_CLS = 'bulk_zero_shot_classification'
  BULK_TEST_ZS_CLS = 'bulk_test_zero_shot_classification'
  ONLINE_LINEAR_CLS = 'online_linear_classification'


class Replace:
  LATEST = '{latest}'


class FlaxCollection:
  """All collections supported under MAX."""
  PARAMS: str = 'params'
  PARAMS_AXES: str = 'params_axes'
  INTERMEDIATES: str = 'intermediates'
  CACHE: str = 'cache'
  BATCH_STATS: str = 'batch_stats'
  PROBES: str = 'probes'
  AUX_LOSS: str = 'aux_loss'


class TransformAxisName:
  """All axis names used in transforms."""
  LAYERS: str = 'layers'
  STACK: str = 'stack'


class SequenceLength:
  """Max sequence length constants for batches containing text."""
  TINY: int = 16
  SMALL: int = 64
  MEDIUM: int = 256


class CommonSpace:
  """The type of common space for computing contrastive loss."""
  DISJOINT: str = 'disjoint'
  JOINT: str = 'joint'
  FINE_AND_COARSE: str = 'fine_and_coarse'


class AggregationType:
  """Aggregation type for transformer outputs."""
  NONE: str = 'none'
  SPECIAL_TOKEN: str = 'special_token'
  GLOBAL_AVERAGE_POOL: str = 'global_average_pool'
  GLOBAL_MAX_POOL: str = 'global_max_pool'
  GLOBAL_SUM_POOL: str = 'global_sum_pool'
  MULTI_HEAD_ATTENTION_POOL: str = 'multi_head_attention_pool'


class Activation:
  """Ativation types across the modeling modules."""
  RELU: str = 'relu'
  GELU: str = 'gelu'
  SWISH: str = 'swish'


class Normalization:
  """Normalization types across the modeling modules."""
  BATCH_NORM: str = 'batch_norm'
  LAYER_NORM: str = 'layer_norm'


class ProbeType:
  """All probe types supported for model probing."""
  IMAGE: str = 'image'
  WAVEFORM: str = 'waveform'
  TEXT: str = 'text'
  SCALAR: str = 'scalar'
  HISTOGRAM: str = 'histogram'


class Extension:
  """Different types of array extension."""
  PREPEND: str = 'prepend'
  APPEND: str = 'append'


class DecodingMode:
  """All decoding modes."""
  MAE: str = 'masked_auto_encoder'
  AR: str = 'autoregressive'
  DIFFUSION: str = 'diffusion'
  MASK_GIT: str = 'mask_git'


class Objective:
  """Objective types for loss calculations."""
  CROSS_MODAL_NCE = 'cross_modal_nce'
  SOFTMAX_CROSS_ENTROPY = 'softmax_cross_entropy'
  SIGMOID_BINARY_CROSS_ENTROPY = 'sigmoid_binary_cross_entropy'
  MEAN_SQUARED_ERROR = 'mean_squared_error'
  OBJECTIVE_AGGREGATOR = 'objective_aggregator'


class ObjectiveAggregation:
  """Aggregation type for objectives."""
  SUM: str = 'sum'
  MEAN: str = 'mean'


class Schedule:
  """Learning rate schedule types."""
  CONSTANT_LR = 'constant_lr'
  COSINE_DECAY_LR = 'cosine_decay_lr'
  WARMUP_COSINE_DECAY_LR = 'warmup_cosine_decay_lr'
  PRE_WARMUP_COSINE_DECAY_LR = 'pre_warmup_cosine_decay_lr'


class Optimizer:
  """Optimizer types."""
  SGD = 'sgd'
  ADAM = 'adam'
  ADAM_W = 'adam_w'
  ADAFACTOR = 'adafactor'
  GALORE_ADAM_W = 'galore_adam_w'


class DataTuning:
  """Tuning profiles for dataloaders."""

  FAST = 'fast'
  EFFICIENT = 'efficient'


class Tokenizer:
  T5_DEFAULT_EXTRA_IDS: int = 100
  T5_DEFAULT_SPM_PATH: str = T5_DEFAULT_SPM_PATH
