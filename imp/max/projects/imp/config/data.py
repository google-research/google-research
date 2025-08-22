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

from typing import Any, Sequence

from imp.max.core import constants
from imp.max.data import config as data_config
from imp.max.data import processing
from imp.max.data.datasets import config as datasets_config


DataTuning = constants.DataTuning
SequenceLength = constants.SequenceLength
ServingStrategy = constants.ServingStrategy
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
TaskFlowName = constants.TaskFlowName
DecodingMode = constants.DecodingMode
INPUTS = DataFeatureType.INPUTS
TARGETS = DataFeatureType.TARGETS
OUTPUTS = DataFeatureType.OUTPUTS
ENCODER = DataFeatureRoute.ENCODER
DECODER = DataFeatureRoute.DECODER
COMMON_SPACE = DataFeatureRoute.COMMON_SPACE
LABEL_CLASSIFIER = DataFeatureRoute.LABEL_CLASSIFIER
TARGET_CLASSIFIER = DataFeatureRoute.TARGET_CLASSIFIER
VISION = constants.Modality.VISION
WAVEFORM = constants.Modality.WAVEFORM
SPECTROGRAM = constants.Modality.SPECTROGRAM
TEXT = constants.Modality.TEXT
TOKEN_RAW = DataFeatureName.TOKEN_RAW
TOKEN_ID = DataFeatureName.TOKEN_ID
FEATURES = DataFeatureName.FEATURES
LABEL = DataFeatureName.LABEL
LOGITS = DataFeatureName.LOGITS
DECODING_MODE = TaskFlowName.DECODING_MODE

num_vision_classes = datasets_config.num_vision_classes
num_audio_classes = datasets_config.num_audio_classes


BASE_TRAIN_BATCH_SIZE = 1024 * 8
BASE_VIDEO_TRAIN_BATCH_SIZE = BASE_TRAIN_BATCH_SIZE // 4
BASE_EVAL_BATCH_SIZE = 1024
VIDEO_EVAL_BATCH_SIZE = 128
LARGE_EVAL_BATCH_SIZE = 4096
TINY_EVAL_BATCH_SIZE = 256  # For datasets with a small number of examples

# Define all input sizes.
VIDEO_PATCH_SIZE = (4, 16, 16)
BASE_VIDEO_INPUT_SIZE = (16, 256, 256, 3)
BASE_IMAGE_INPUT_SIZE = (4, 256, 256, 3)
LOWRES_VIDEO_INPUT_SIZE = (16, 128, 128, 3)

# Audio params.
# Effective sample rate = 48k / 3 = 16k
# Spectrogram shape: (128, 128) = 64 tokens
AUDIO_SAMPLE_RATE = 48000
AUDIO_STRIDE = 3
AUDIO_NUM_SECONDS = 8.192
WAVEFORM_PATCH_SIZE = 256
SPECTROGRAM_FRAME_LENGTH = 2048
SPECTROGRAM_FRAME_STEP = 1024
BASE_WAVEFORM_SAMPLES = int(
        AUDIO_NUM_SECONDS * AUDIO_SAMPLE_RATE / AUDIO_STRIDE)
SPECTROGRAM_PATCH_SIZE = (16, 16)
SPECTROGRAM_FEATURES = 128

VIDEO_TOKEN_DROP_RATE = 0.75

VIDEO_STRIDE = 2
NUM_TEST_CLIPS = 4


def _vision_override(video_input_size,
                     token_drop_rate = 0.,
                     stride = None):
  """Creates a vision config override."""
  return {
      'num_frames': video_input_size[0],
      'crop_size': video_input_size[1],
      'min_resize': processing.get_min_resize_value(
          video_input_size[1]),
      'token_drop_rate': token_drop_rate,
      'stride': stride or VIDEO_STRIDE,
  }


def _audio_override(num_samples,
                    token_drop_rate = 0.):
  """Creates an audio config override."""
  return {
      'sample_rate': AUDIO_SAMPLE_RATE,
      'waveform_stride': AUDIO_STRIDE,
      'num_raw_waveform_samples': num_samples,
      'frame_length': SPECTROGRAM_FRAME_LENGTH,
      'frame_step': SPECTROGRAM_FRAME_STEP,
      'num_features': SPECTROGRAM_FEATURES,
      'temporal_patch_size': SPECTROGRAM_PATCH_SIZE[0],
      'spectoral_patch_size': SPECTROGRAM_PATCH_SIZE[1],
      'token_drop_rate': token_drop_rate,
      'keep_waveform_features': False,
  }


def get_plain_encoder_features_metadata(
    modalities):
  """Constructs the simple encoder-only metadata."""
  dataflow = ()
  for modality in modalities:
    if modality == TEXT:
      token_feature_name = TOKEN_ID
    else:
      token_feature_name = TOKEN_RAW

    datapass = (
        {
            INPUTS: {
                ENCODER: {
                    modality: {
                        token_feature_name: None,
                    },
                },
            },
            OUTPUTS: {
                ENCODER: {
                    modality: {
                        token_feature_name: FEATURES,
                    },
                },
            },
        },
    )
    dataflow += datapass
  default_metadata = data_config.Metadata(
      dataflow=dataflow,
      taskflow=(),
  )
  return default_metadata


def get_contrastive_metadata(
    modalities,
    target_modalities = ()):
  """Constructs the multimodal contrastive learning metadata."""
  dataflow = ()
  target_modalities = target_modalities or modalities
  for modality in modalities:
    common_space_targets = tuple(sorted(set(target_modalities) - {modality}))
    if modality == TEXT:
      token_feature_name = TOKEN_ID
    else:
      token_feature_name = TOKEN_RAW

    datapass = (
        {
            INPUTS: {
                ENCODER: {
                    modality: {
                        token_feature_name: None,
                    },
                },
            },
            OUTPUTS: {
                ENCODER: {
                    modality: {
                        token_feature_name: FEATURES,
                    },
                },
                COMMON_SPACE: {
                    modality: {
                        token_feature_name: common_space_targets,
                    },
                },
            },
        },
    )
    dataflow += datapass
  default_metadata = data_config.Metadata(
      dataflow=dataflow,
      taskflow=(),
  )
  return default_metadata


def get_classification_metadata(
    modality,
    logits_key = LOGITS,
):
  """Constructs the unimodal classification metadata."""
  dataflow = (
      {
          INPUTS: {
              ENCODER: {
                  modality: {
                      TOKEN_RAW: None,
                  },
              },
          },
          OUTPUTS: {
              ENCODER: {
                  modality: {
                      TOKEN_RAW: FEATURES,
                  },
              },
              LABEL_CLASSIFIER: {
                  modality: {
                      TOKEN_RAW: logits_key,
                  },
              },
          },
          TARGETS: {
              LABEL_CLASSIFIER: {
                  modality: LABEL,
              },
          },
      },
  )
  default_metadata = data_config.Metadata(
      dataflow=dataflow,
      taskflow=(),
  )
  return default_metadata


def get_bert_metadata():
  """Constructs the BERT-style text understanding metadata."""
  dataflow = (
      {
          INPUTS: {
              ENCODER: {
                  TEXT: {
                      TOKEN_ID: None,
                  },
              },
          },
          OUTPUTS: {
              ENCODER: {
                  TEXT: {
                      TOKEN_ID: FEATURES,
                  },
              },
              TARGET_CLASSIFIER: {
                  TEXT: {
                      TOKEN_ID: LOGITS,
                  },
              },
          },
          TARGETS: {
              TARGET_CLASSIFIER: {
                  TEXT: TOKEN_ID,
              },
          },
      },
  )
  default_metadata = data_config.Metadata(
      dataflow=dataflow,
      taskflow=(),
  )
  return default_metadata


def get_mae_metadata(
    modalities,
    token_feature_name = TOKEN_RAW,
):
  """Constructs the standard Masked Auto Encoding metadata."""
  dataflow = ()
  taskflow = ()
  for modality in modalities:
    if modality == TEXT:
      raise ValueError('MAE for text modality is not supported.')
    taskpass = (
        {
            DECODING_MODE: DecodingMode.MAE,
        },
    )
    datapass = (
        {
            INPUTS: {
                ENCODER: {
                    modality: {
                        token_feature_name: None,
                    },
                },
            },
            OUTPUTS: {
                ENCODER: {
                    modality: {
                        token_feature_name: FEATURES,
                    },
                },
                DECODER: {
                    modality: {
                        token_feature_name: FEATURES,
                    },
                },
                TARGET_CLASSIFIER: {
                    modality: {
                        TOKEN_RAW: LOGITS,
                    },
                },
            },
            TARGETS: {
                TARGET_CLASSIFIER: {
                    modality: TOKEN_RAW,
                },
            },
        },
    )
    dataflow += datapass
    taskflow += taskpass
  default_metadata = data_config.Metadata(
      dataflow=dataflow,
      taskflow=taskflow,
  )
  return default_metadata


def get_t5_metadata(modalities):
  """Constructs the T5-style text understanding metadata."""
  dataflow = ()
  taskflow = ()
  for modality in modalities:
    taskpass = (
        {
            DECODING_MODE: DecodingMode.AR,
        },
    )
    datapass = (
        {
            INPUTS: {
                ENCODER: {
                    modality: {
                        TOKEN_ID: None,
                    },
                },
                DECODER: {
                    modality: {
                        TOKEN_ID: None,
                    },
                },
            },
            OUTPUTS: {
                ENCODER: {
                    modality: {
                        TOKEN_ID: FEATURES,
                    },
                },
                DECODER: {
                    modality: {
                        TOKEN_ID: FEATURES,
                    },
                },
                TARGET_CLASSIFIER: {
                    modality: {
                        TOKEN_ID: LOGITS,
                    },
                },
            },
            TARGETS: {
                TARGET_CLASSIFIER: {
                    modality: TOKEN_ID,
                },
            },
        },
    )
    dataflow += datapass
    taskflow += taskpass
  default_metadata = data_config.Metadata(
      dataflow=dataflow,
      taskflow=taskflow,
  )
  return default_metadata


# ----------------------------------------------------------------------
# --------------------- EXAMPLE TRAIN DATA CONFIG ----------------------
# ----------------------------------------------------------------------
VISION_CLASSES = (
    ('example_3',
     num_vision_classes(
         datasets_config.EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT)),
    ('example_7',
     num_vision_classes(
         datasets_config.EXAMPLE_7_VIDEO_CLASSIFICATION_WITH_TEXT)),
)
AUDIO_CLASSES = (
    ('example_5',
     num_audio_classes(
         datasets_config.EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT)),
)
TEXT_CLASSES = None

IMAGE_PERCEPTION_PRETRAIN_LOADERS = (
    # -----------------------------
    # Image-Text Loaders
    # -----------------------------
    data_config.Loader(
        interval=1,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        prefetch=4,
        metadata=get_contrastive_metadata((VISION, TEXT)),
        dataset=datasets_config.EXAMPLE_2_IMAGE_TEXT_THIRDPARTY.copy_and_override({
            'modalities': {
                'vision': {
                    **_vision_override(BASE_IMAGE_INPUT_SIZE),
                },
                'text': {
                    'max_num_tokens': SequenceLength.TINY,
                },
            },
        })),
    # -----------------------------
    # Image Classification Loaders
    # -----------------------------
    data_config.Loader(
        interval=1,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        prefetch=4,
        metadata=get_classification_metadata(VISION, f'{LOGITS}_example_3'),
        dataset=datasets_config.EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT.copy_and_override({
            'name': f'{datasets_config.EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT.name}/cls',
            'modalities': {
                'vision': {
                    **_vision_override(BASE_IMAGE_INPUT_SIZE),
                    'annotation': {'label': {'normalize': True}},
                },
                'text': None,
            },
        })),
)
VIDEO_PERCEPTION_PRETRAIN_LOADERS = (
    # -----------------------------
    # Video-Text Loaders
    # -----------------------------
    data_config.Loader(
        interval=12,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        use_data_service=True,
        metadata=get_contrastive_metadata((VISION, TEXT)),
        dataset=datasets_config.EXAMPLE_4_VIDEO_AUDIO_TEXT.copy_and_override({
            'modalities': {
                'audio': None,
                'vision': {
                    **_vision_override(BASE_VIDEO_INPUT_SIZE,
                                       VIDEO_TOKEN_DROP_RATE),
                },
                'text': {
                    'max_num_tokens': SequenceLength.TINY,
                },
            },
        })),
    # -----------------------------
    # Video Classification Loaders
    # -----------------------------
    data_config.Loader(
        interval=5,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_VIDEO_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        use_data_service=True,
        metadata=get_classification_metadata(VISION, f'{LOGITS}_example_7'),
        dataset=datasets_config.EXAMPLE_7_VIDEO_CLASSIFICATION_WITH_TEXT.copy_and_override({
            'name': f'{datasets_config.EXAMPLE_7_VIDEO_CLASSIFICATION_WITH_TEXT.name}/cls',
            'modalities': {
                'audio': None,
                'vision': {
                    **_vision_override(BASE_IMAGE_INPUT_SIZE),
                },
                'text': None,
            },
        })),
)
AUDIO_PERCEPTION_PRETRAIN_LOADERS = (
    # -----------------------------
    # Video-Audio-Text Loaders
    # -----------------------------
    data_config.Loader(
        interval=12,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        use_data_service=True,
        metadata=get_contrastive_metadata((VISION, SPECTROGRAM, TEXT)),
        dataset=datasets_config.EXAMPLE_4_VIDEO_AUDIO_TEXT.copy_and_override({
            'modalities': {
                'audio': {
                    **_audio_override(BASE_WAVEFORM_SAMPLES),
                },
                'vision': {
                    **_vision_override(BASE_VIDEO_INPUT_SIZE,
                                       VIDEO_TOKEN_DROP_RATE),
                },
                'text': {
                    'max_num_tokens': SequenceLength.TINY,
                },
            },
        })),
    # -----------------------------
    # Audio Classification Loaders
    # -----------------------------
    data_config.Loader(
        interval=8,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_VIDEO_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        use_data_service=True,
        metadata=get_classification_metadata(
            SPECTROGRAM, f'{LOGITS}_example_5',
        ),
        dataset=datasets_config.EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT.copy_and_override({
            'name': f'{datasets_config.EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT.name}/cls',
            'modalities': {
                'audio': {
                    **_audio_override(BASE_WAVEFORM_SAMPLES),
                },
                'text': None,
            },
        })),
)

ALL_PERCEPTION_PRETRAIN_LOADERS = (
        IMAGE_PERCEPTION_PRETRAIN_LOADERS +
        VIDEO_PERCEPTION_PRETRAIN_LOADERS +
        AUDIO_PERCEPTION_PRETRAIN_LOADERS
)

TEXT_UNDERSTANDING_PRETRAIN_LOADERS = (
    data_config.Loader(
        interval=1,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        metadata=get_bert_metadata(),
        dataset=datasets_config.EXAMPLE_1_TEXT_SEQIO),
)
IMAGE_GENERATION_PRETRAIN_LOADERS = (
    data_config.Loader(
        interval=1,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        prefetch=1,
        tuning=DataTuning.EFFICIENT,
        metadata=get_mae_metadata((VISION,), TOKEN_RAW),
        dataset=datasets_config.EXAMPLE_2_IMAGE_TEXT_THIRDPARTY.copy_and_override({
            'modalities': {
                'vision': {
                    'token_drop_rate': 0.75,
                },
                'text': None,  # Remove text
            },
        })),
)
TEXT_GENERATION_PRETRAIN_LOADERS = (
    data_config.Loader(
        interval=1,
        num_epochs=-1,
        is_training=True,
        batch_size=BASE_TRAIN_BATCH_SIZE,
        microbatch_splits=1,
        metadata=get_t5_metadata((TEXT,)),
        dataset=datasets_config.EXAMPLE_1_TEXT_SEQIO),
)

# Bulk evaluation loaders (i.e. those that can be performed in one big loop)
BULK_EVAL_LOADERS = (
    # -----------------------------
    # Linear + Zero-Shot Classification
    # -----------------------------

    data_config.Loader(
        num_epochs=1,
        is_training=False,
        batch_size=VIDEO_EVAL_BATCH_SIZE,
        microbatch_splits=1,
        prefetch=1,
        tuning=DataTuning.EFFICIENT,
        metadata=get_contrastive_metadata((SPECTROGRAM, TEXT)),
        dataset=datasets_config.EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT.copy_and_override(
            {'modalities': {
                'audio': {
                    **_audio_override(BASE_WAVEFORM_SAMPLES),
                }
            }}),
        serving=(ServingStrategy.BULK_LINEAR_CLS,
                 ServingStrategy.BULK_ZS_CLS)),
    data_config.Loader(
        num_epochs=1,
        is_training=False,
        batch_size=VIDEO_EVAL_BATCH_SIZE,
        microbatch_splits=1,
        prefetch=1,
        tuning=DataTuning.EFFICIENT,
        metadata=get_contrastive_metadata((SPECTROGRAM, TEXT)),
        dataset=datasets_config.EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT.copy_and_override({
            'modalities': {
                'audio': {
                    **_audio_override(BASE_WAVEFORM_SAMPLES),
                    'num_test_clips': NUM_TEST_CLIPS,
                }
            },
            'data': {  # Default table was train, so here we choose test
                'table': 'test'
            }
        }),
        serving=(ServingStrategy.BULK_LINEAR_CLS,
                 ServingStrategy.BULK_ZS_CLS)),

    # -----------------------------
    # Cross-modality Retrieval (video-text example below)
    # -----------------------------

    data_config.Loader(
        num_epochs=1,
        is_training=False,
        batch_size=VIDEO_EVAL_BATCH_SIZE,
        microbatch_splits=1,
        prefetch=1,
        tuning=DataTuning.EFFICIENT,
        metadata=get_contrastive_metadata((VISION, TEXT)),
        dataset=datasets_config.EXAMPLE_4_VIDEO_AUDIO_TEXT.copy_and_override({
            'modalities': {
                'vision': {
                    'num_test_clips': NUM_TEST_CLIPS,
                    **_vision_override(BASE_VIDEO_INPUT_SIZE),
                },
                'audio': None,
            },
            'data': {
                'table': 'test'
            },
        }),
        serving=ServingStrategy.BULK_ZS_RETRIEVAL),
)

# Online classification loaders (i.e. those that need gradual online train/eval)
IMAGENET_ONLINE_EVAL_LOADERS = (
    data_config.Loader(
        num_epochs=-1,
        is_training=True,
        batch_size=LARGE_EVAL_BATCH_SIZE,
        microbatch_splits=1,
        metadata=get_plain_encoder_features_metadata((VISION,)),
        dataset=datasets_config.EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT.copy_and_override({
            'modalities': {
                'vision': {
                    **_vision_override(BASE_IMAGE_INPUT_SIZE),
                },
                'text': None,
            },
        }),
        serving=ServingStrategy.ONLINE_LINEAR_CLS),
    data_config.Loader(
        num_epochs=1,
        is_training=False,
        batch_size=LARGE_EVAL_BATCH_SIZE,
        microbatch_splits=1,
        metadata=get_plain_encoder_features_metadata((VISION,)),
        dataset=datasets_config.EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT.copy_and_override({
            'data': {
                'table': 'test'
            },
            'modalities': {
                'vision': {
                    **_vision_override(BASE_IMAGE_INPUT_SIZE),
                },
                'text': None,
            },
        }),
        serving=ServingStrategy.ONLINE_LINEAR_CLS),
)
K400_ONLINE_EVAL_LOADERS = (
    # Put your Kinetics400 loader here (see above examples)
)

# ----------------------------------------------------------------------
# -------------------------- END OF EXAMPLES ---------------------------
# ----------------------------------------------------------------------

