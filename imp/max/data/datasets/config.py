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

"""Settings of the datasets."""

import tensorflow as tf

from imp.max.core import constants
from imp.max.data import config as data_config
from imp.max.data import processing

# Constants
ParsingFeatureName = constants.ParsingFeatureName
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
Modality = constants.Modality
SequenceLength = constants.SequenceLength
LabelReMap = constants.LabelReMap
DataFactory = constants.DataFactory

# Data config modules
Modalities = data_config.Modalities
Vision = data_config.Vision
Video = data_config.Video
Spectrogram = data_config.Spectrogram
Text = data_config.Text
TextFromLabel = data_config.TextFromLabel
Annotation = data_config.Annotation
Label = data_config.Label
CustomModality = data_config.CustomModality
Bundle = data_config.Bundle
Dataset = data_config.Dataset
SeqioDataDefinition = data_config.SeqioDataDefinition
ThirdPartyDataDefinition = data_config.ThirdPartyDataDefinition


INPUT_ENCODER_TEXT_RAW_KEY = processing.get_flattened_key(
    ftype=DataFeatureType.INPUTS,
    froute=DataFeatureRoute.ENCODER,
    fname=DataFeatureName.RAW,
    modality=Modality.TEXT,
)
TARGET_DECODER_TEXT_TOKEN_ID_KEY = processing.get_flattened_key(
    ftype=DataFeatureType.TARGETS,
    froute=DataFeatureRoute.DECODER,
    fname=DataFeatureName.TOKEN_ID,
    modality=Modality.TEXT,
)
TARGET_DECODER_TEXT_TOKEN_COORDINATE_KEY = processing.get_flattened_key(
    ftype=DataFeatureType.TARGETS,
    froute=DataFeatureRoute.DECODER,
    fname=DataFeatureName.TOKEN_COORDINATE,
    modality=Modality.TEXT,
)
TARGET_DECODER_TEXT_TOKEN_POSITION_ID_KEY = processing.get_flattened_key(
    ftype=DataFeatureType.TARGETS,
    froute=DataFeatureRoute.DECODER,
    fname=DataFeatureName.TOKEN_POSITION_ID,
    modality=Modality.TEXT,
)
TARGET_DECODER_TEXT_TOKEN_MASK_KEY = processing.get_flattened_key(
    ftype=DataFeatureType.TARGETS,
    froute=DataFeatureRoute.DECODER,
    fname=DataFeatureName.TOKEN_MASK,
    modality=Modality.TEXT,
)


def num_vision_classes(config):
  """Returns the number of vision classes in the dataset."""
  if isinstance(config.modalities.vision, CustomModality):
    raise ValueError('Label cannot be fetched from `CustomModality`.')
  elif (config.modalities.vision is not None and
        config.modalities.vision.annotation is not None and
        config.modalities.vision.annotation.label is not None and
        config.modalities.vision.annotation.label.num_classes is not None):
    return config.modalities.vision.annotation.label.num_classes
  else:
    raise ValueError(f'Label not defined in {config}')


def num_audio_classes(config):
  """Returns the number of audio classes in the dataset."""
  if isinstance(config.modalities.audio, CustomModality):
    raise ValueError('Label cannot be fetched from `CustomModality`.')
  elif (config.modalities.audio is not None and
        config.modalities.audio.annotation is not None and
        config.modalities.audio.annotation.label is not None and
        config.modalities.audio.annotation.label.num_classes is not None):
    return config.modalities.audio.annotation.label.num_classes
  else:
    raise ValueError(f'Label not defined in {config}')


EXAMPLE_1_TEXT_SEQIO = Dataset(
    name='simple_seqio_task',
    data=SeqioDataDefinition(
        task_name='seqio_task_name_defined_under_"seqio_tasks.py"',
        table='train',
    ),
    modalities=Modalities(
        text=Text(
            parsing_feature_name=ParsingFeatureName.TEXT,
            max_num_tokens=SequenceLength.MEDIUM,
        ),
    ),
    factory=constants.DataFactory.SEQIO,
)

EXAMPLE_2_IMAGE_TEXT_THIRDPARTY = Dataset(
    name='example-2-image-text-thirdparty',
    data=ThirdPartyDataDefinition(
        base_dir='/path/to/tfrecord_dir',
        # We use the default practice of sharded tfrecord pattern here as an
        # example, but user can use any pattern they have stored their records
        # with; as far as it follows a glob pattern (i.e. '*' and/or '?').
        tables={'train': 'train_split_filename-*of*',
                'test': 'test_split_filename-*of*'},
        table='train',
        source=constants.TFRECORD,
        raw_data_format=constants.TF_EXAMPLE,
    ),
    modalities=Modalities(
        vision=Vision(),
        text=Text(
            parsing_feature_name='image/captions',
            max_num_sentences=5,
            max_num_tokens=SequenceLength.SMALL,
        ),
    )
)

EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT = Dataset(
    name='example-3-image-classification-with-text',
    data=ThirdPartyDataDefinition(
        base_dir='/path/to/tfrecord_dir',
        tables={'train': 'train_split_filename-*of*',
                'test': 'test_split_filename-*of*'},
        table='train',
        num_examples={
            'train': 1000,
            'test': 50,
        },
        source=constants.TFRECORD,
        raw_data_format=constants.TF_EXAMPLE,
    ),
    modalities=Modalities(
        vision=Vision(
            annotation=Annotation(
                label=Label(
                    parsing_label_index_feature_name='image/class/label',
                    num_classes=1000,
                ),
            ),
        ),
        text=TextFromLabel(
            parsing_label_name_feature_name='image/class/text',
            max_num_tokens=SequenceLength.SMALL,
        ),
    ),
)

EXAMPLE_4_VIDEO_AUDIO_TEXT = Dataset(
    name='example-4-video-audio-text',
    data=ThirdPartyDataDefinition(
        base_dir='/path/to/tfrecord_dir',
        tables={'train': 'train_split_filename-*of*',
                'test': 'test_split_filename-*of*'},
        table='train',
        source=constants.TFRECORD,
        raw_data_format=constants.TF_SEQUENCE_EXAMPLE,
    ),
    modalities=Modalities(
        vision=Video(
            color_augmentation=False,  # captions contain color names
            random_flip=False,  # captions can contain OCR
        ),
        audio=Spectrogram(),
        text=Text(
            parsing_feature_name=ParsingFeatureName.CLIP_LABEL_STRING,
            max_num_tokens=SequenceLength.MEDIUM),
    ),
)

EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT = Dataset(
    name='example-5-audio-classification-thirdparty',
    data=ThirdPartyDataDefinition(
        base_dir='/path/to/tfrecord_dir',
        tables={'train': 'train_split_filename-*of*',
                'test': 'test_split_filename-*of*'},
        table='train',
        source=constants.TFRECORD,
        raw_data_format=constants.TF_SEQUENCE_EXAMPLE,
    ),
    modalities=Modalities(
        audio=Spectrogram(parsing_feature_name='audio/WAVEFORM/floats',
                          annotation=Annotation(
                              label=Label(num_classes=50))),
        text=TextFromLabel(
            modality=Modality.AUDIO,
            max_num_tokens=SequenceLength.TINY),
    ),
    factory=DataFactory.ESC
)

EXAMPLE_6_IMAGE_TEXT_THIRDPARTY_WITH_BUNDLES = Dataset(
    name='example-6-image-text-thirdparty-with-bundles',
    data=ThirdPartyDataDefinition(
        base_dir='/path/to/tfrecord_dir',
        tables={
            'train': 'train.tfrecord-*of*',
        },
        table='train',
        source=constants.TFRECORD,
        raw_data_format=constants.TF_EXAMPLE,
    ),
    modalities=Modalities(
        vision=Vision(
            parsing_feature_name='image',
            color_augmentation=False,  # captions contain color names
            random_flip=False,  # captions can contain OCR
        ),
        text=CustomModality(
            parsing=Bundle(
                processing.parse_features,
                parsing_feature_name='texts',
                parsed_feature_name=INPUT_ENCODER_TEXT_RAW_KEY,
                feature_type=tf.io.VarLenFeature,
                dtype=tf.string,
            ),
            decoding=Bundle(
                processing.decode_sparse_features,
                feature_name=INPUT_ENCODER_TEXT_RAW_KEY,
            ),
            preprocessing=(
                Bundle(
                    processing.apply_fn_on_features_dict(
                        fn=processing.sample_or_pad_non_sorted_sequence,
                        feature_name=INPUT_ENCODER_TEXT_RAW_KEY,
                    ),
                    max_num_steps=1,
                    pad_value=b'',
                    random=True,
                ),
                Bundle(
                    processing.tokenize_raw_string,
                    tokenizer=constants.T5_EN,
                    raw_feature_name=INPUT_ENCODER_TEXT_RAW_KEY,
                    token_id_feature_name=TARGET_DECODER_TEXT_TOKEN_ID_KEY,
                    token_coordinate_feature_name=TARGET_DECODER_TEXT_TOKEN_COORDINATE_KEY,
                    token_position_id_feature_name=TARGET_DECODER_TEXT_TOKEN_POSITION_ID_KEY,
                    token_mask_feature_name=TARGET_DECODER_TEXT_TOKEN_MASK_KEY,
                    keep_raw_string=False,
                    prepend_bos=False,
                    append_eos=False,
                    max_num_tokens=SequenceLength.SMALL,
                    max_num_sentences=1,
                ),
            )
        ),
    ),
)

EXAMPLE_7_VIDEO_CLASSIFICATION_WITH_TEXT = Dataset(
    name='example-7-video-classification-with-text',
    data=ThirdPartyDataDefinition(
        base_dir='/path/to/tfrecord_dir',
        tables={'train': 'train_split_filename-*of*',
                'test': 'test_split_filename-*of*'},
        table='train',
        source=constants.TFRECORD,
        raw_data_format=constants.TF_SEQUENCE_EXAMPLE,
    ),
    modalities=Modalities(
        vision=Video(
            annotation=Annotation(
                label=Label(num_classes=700),
            ),
        ),
        text=TextFromLabel(
            parsing_label_name_feature_name='clip/label/str',
            modality=Modality.VIDEO,
            max_num_tokens=SequenceLength.SMALL,
        ),
    )
)

ALL_DATASETS = (
    EXAMPLE_1_TEXT_SEQIO,
    EXAMPLE_2_IMAGE_TEXT_THIRDPARTY,
    EXAMPLE_3_IMAGE_CLASSIFICATION_WITH_TEXT,
    EXAMPLE_4_VIDEO_AUDIO_TEXT,
    EXAMPLE_5_AUDIO_CLASSIFICATION_WITH_TEXT,
    EXAMPLE_6_IMAGE_TEXT_THIRDPARTY_WITH_BUNDLES,
    EXAMPLE_7_VIDEO_CLASSIFICATION_WITH_TEXT,
)


DATASET_REGISTRY = {dataset.name: dataset for dataset in ALL_DATASETS}
