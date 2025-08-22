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

"""Defines utility functions for seqio tasks."""

from typing import Any

from flax import traverse_util
import seqio
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf

from imp.max.core import constants
from imp.max.data import processing

FeaturesDict = dict[str, Any]
Modality = constants.Modality
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName

VOCABULARY = processing.VOCABULARY

PAD_TOKEN_ID = 0

SEQIO_OUTPUT_FEATURES_BERT = {
    'inputs': seqio.Feature(
        vocabulary=VOCABULARY,
        add_eos=False,
        required=False),
    'targets': seqio.Feature(
        vocabulary=VOCABULARY,
        add_eos=False),
}

SEQIO_OUTPUT_FEATURES_WITH_WEIGHTS_BERT = {
    **SEQIO_OUTPUT_FEATURES_BERT,
    'targets_weights': seqio.Feature(
        vocabulary=seqio.PassThroughVocabulary(0),
        required=False,
        dtype=tf.float32),
}

SEQIO_OUTPUT_FEATURES_T5 = {
    'inputs': seqio.Feature(
        vocabulary=VOCABULARY,
        add_eos=False,
        required=False),
    'targets': seqio.Feature(
        vocabulary=VOCABULARY,
        add_eos=True),
}

SEQIO_OUTPUT_FEATURES_WITH_WEIGHTS_T5 = {
    **SEQIO_OUTPUT_FEATURES_T5,
    'targets_weights': seqio.Feature(
        vocabulary=seqio.PassThroughVocabulary(0),
        required=False,
        dtype=tf.float32),
}


def mask_tokens_for_bert(
    dataset,
    sequence_length,
    output_features,
    **unused_kwargs):
  """Applies random input chunking and BERT-style masking."""
  dataset = t5_preprocessors.select_random_chunk(
      dataset,
      output_features=output_features,
      feature_key='targets',
      max_length=65536)
  dataset = t5_preprocessors.reduce_concat_tokens(
      dataset, feature_key='targets', batch_size=128)
  dataset = t5_preprocessors.split_tokens_to_inputs_length(
      dataset, output_features=output_features, sequence_length=sequence_length)
  dataset = t5_preprocessors.denoise(
      dataset,
      output_features,
      inputs_fn=t5_preprocessors.noise_token_to_random_token_or_sentinel,
      targets_fn=None,
      noise_density=0.15,
      noise_mask_fn=t5_preprocessors.iid_noise_mask
  )
  return dataset


@seqio.map_over_dataset
def normalize_text(features, key):
  """Joins and reshapes text to a scalar."""
  features[key] = tf.reshape(tf.strings.reduce_join(features[key]), [])
  return features


@seqio.map_over_dataset
def flatten_dataset(features):
  """Flattens the dataset keys."""
  return traverse_util.flatten_dict(features, sep='/')


def filter_by_language(dataset,
                       language_key = 'page_lang',
                       language = 'en'):
  """Filters the dataset by the given language."""
  return dataset.filter(lambda x: x[language_key] == language)


@seqio.map_over_dataset
def trim_to_sequence_length_bert(
    features,
    sequence_length):
  """Trims/pads the inputs and targets to their sequence lengths for BERT."""
  for key in ('inputs', 'targets'):
    features[key] = features[key][Ellipsis, :sequence_length[key]]
    pad_amount = sequence_length[key] - tf.shape(features[key])[-1]
    features[key] = tf.pad(features[key], [(0, pad_amount)])
  return features


@seqio.map_over_dataset
def trim_to_sequence_length(
    features,
    sequence_length):
  """Trims/pads the inputs and targets to their sequence lengths with EOS."""
  inputs = features['inputs'][Ellipsis, :sequence_length['inputs']]
  pad_amount = sequence_length['inputs'] - tf.shape(inputs)[-1]
  features['inputs'] = tf.pad(inputs, [(0, pad_amount)])

  targets = features['targets'][Ellipsis, :sequence_length['targets'] - 1]
  targets = tf.concat([targets, [VOCABULARY.eos_id]], axis=0)
  pad_amount = sequence_length['targets'] - tf.shape(targets)[-1]
  features['targets'] = tf.pad(targets, [(0, pad_amount)])

  return features


@seqio.map_over_dataset
def prepend_bos_token(example,
                      keys = ('targets',),
                      bos = 0):
  """Prepends a beginning-of-sequence (BOS) token to all examples."""
  for key in keys:
    example[key] = tf.concat([[bos], example[key]], axis=0)
  return example


@seqio.map_over_dataset
def token_mask_bert(example):
  """Adds a BERT mask for all targets that differ from inputs."""
  example[DataFeatureName.TOKEN_MASK] = tf.cast(
      example['inputs'] != 0, tf.float32)
  mask = tf.cast(example['inputs'] != example['targets'], tf.float32)
  example['targets_token_mask'] = mask
  return example


@seqio.map_over_dataset
def token_mask(example):
  """Adds a mask for all nonzero inputs and targets."""
  example[DataFeatureName.TOKEN_MASK] = tf.cast(
      example['inputs'] != 0, tf.float32)
  target_mask = tf.cast(example['targets'] != 0, tf.float32)
  # Make sure the first position (bos) is not masked out in case it is 0
  target_mask = tf.concat([[1.], target_mask[1:]], axis=0)
  example['targets_token_mask'] = target_mask
  return example


def add_instance_axis(features):
  """Extends the first dimension as the instance axis."""
  flattened_features = traverse_util.flatten_dict(features, sep='/')
  for feature_name in flattened_features:
    expanded_feature = flattened_features[feature_name][tf.newaxis, :]
    flattened_features[feature_name] = expanded_feature

  return traverse_util.unflatten_dict(flattened_features, sep='/')


@seqio.map_over_dataset
def seqio_to_dmvr_format(features):
  """Converts a SeqIO output dict to DMVR format."""
  # TODO(b/229771812): add support for other modalities.
  features = add_instance_axis(features)
  return {
      DataFeatureType.INPUTS: {
          DataFeatureRoute.ENCODER: {
              Modality.TEXT: {
                  DataFeatureName.TOKEN_ID: features['inputs'],
                  DataFeatureName.TOKEN_MASK: features[
                      DataFeatureName.TOKEN_MASK
                  ],
              },
          },
      },
      DataFeatureType.TARGETS: {
          DataFeatureRoute.DECODER: {
              Modality.TEXT: {
                  DataFeatureName.TOKEN_ID: features['targets'],
                  DataFeatureName.TOKEN_MASK: features['targets_token_mask'],
              },
          },
      },
  }
