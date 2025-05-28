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

"""Beam pipeline for running inference."""
import copy
import json
import os
import re
import string
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import ml_collections
from ml_collections import config_flags
import numpy as np
import six
import tensorflow as tf

from mave.benchmark.data import data_utils

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs.py',
    help_string='Training configuration.',
    lock_config=True,
)
flags.mark_flags_as_required(['config'])

_MODEL_TYPE = flags.DEFINE_string(
    'model_type', default=None, help='The model type.', required=True)

_DATA_TYPE = flags.DEFINE_string(
    'data_type',
    default=None,
    help='The data type. If None, follows --model_type.')

_SAVED_MODEL_DIR = flags.DEFINE_string(
    'saved_model_dir',
    default=None,
    help='The TF saved model dir.',
    required=True)

_MODEL_ID = flags.DEFINE_string(
    'model_id',
    default=None,
    help=('The TF saved model dir. If not set, it will be derived from '
          '--saved_model_dir'))

_INPUT_JSON_LINES_FILEPATTERN = flags.DEFINE_string(
    'input_json_lines_filepattern',
    default=None,
    help=('The input JSON Lines file pattern of the examples. Files within a'
          ' same dir will be inferenced together.'),
    required=True)

_OUTPUT_INFERENCE_RESULTS_FILEPATTERN = flags.DEFINE_string(
    'output_inference_results_filepattern',
    default=None,
    help=('The output inference results TF Records filepattern. If not set, '
          'the output results will be in the same dir as the input JSON Lines '
          'files, with filepattern: '
          '"input_filename_predictions/<model_id>/mave_tfrecords@*".'))

_THRESHOLD = flags.DEFINE_float(
    'threshold', default=0.5, help='The token level answer score threshold.')

_BUCKET_MEASURE = flags.DEFINE_enum(
    'bucket_measure',
    default='',
    enum_values=['num_words', 'num_paragraphs', ''],
    help=('The bucket measures to split the predictions. If empty, not '
          'performing bucket splitting.'))

_BUCKET_SIZE = flags.DEFINE_integer(
    'bucket_size', default=128, help='The size of the bucket.')

_MAX_NUM_BUCKETS = flags.DEFINE_integer(
    'max_num_buckets', default=16, help='The max number of buckets.')

_DEBUG = flags.DEFINE_boolean(
    'debug',
    default=True,
    help='Whether the ooutput TF Records contain dsebug info.')

_Example = Any
_A_A = 'a_a'
_A_B = 'a_b'
_A_N = 'a_n'
_N_A = 'n_a'
_N_N = 'n_n'
_MEASURE_STRS = (_A_A, _A_B, _A_N, _N_A, _N_N)


def _get_config():
  """Returns a frozen config dict by updating dynamic fields."""
  config = ml_collections.ConfigDict(_CONFIG.value)
  config.model_type = _MODEL_TYPE.value
  data_type = _DATA_TYPE.value or _MODEL_TYPE.value
  if data_type == 'bert':
    config.data.use_category = True
    config.data.use_attribute_key = True
    config.data.use_cls = True
    config.data.use_sep = True
  elif data_type == 'bilstm_crf':
    config.data.use_category = False
    config.data.use_attribute_key = False
    config.data.use_cls = False
    config.data.use_sep = False
  config.data.debug = _DEBUG.value
  config = ml_collections.FrozenConfigDict(config)
  logging.info('%s', config)
  return config


class FlattenAttributesFn(beam.DoFn):
  """DoFn to flatten attributes."""

  def __init__(self, *unused_args, **unused_kwargs):
    self._num_input_examples = beam.metrics.Metrics.counter(
        self.__class__, 'num-input-examples')
    self._num_flattened_examples = beam.metrics.Metrics.counter(
        self.__class__, 'num-flattened-examples')

  def process(self, example, *args, **kwargs):
    self._num_input_examples.inc()
    for attribute in example['attributes']:
      self._num_flattened_examples.inc()
      yield {
          'id': example['id'],
          'category': example['category'],
          'paragraphs': copy.deepcopy(example['paragraphs']),
          'attributes': [copy.deepcopy(attribute)],
      }


def _apply_global_mask_to_long(global_mask, long_mask,
                               long_global_ids):
  """Returns global to long span mask through a `long_global_ids`."""
  global_ids = np.arange(len(global_mask))
  g2l_mask = (global_ids[:, None] == long_global_ids[None, :])
  return long_mask & np.dot(global_mask, g2l_mask)


def _get_span_indexes(span_mask):
  """Yields span segment indexes given a span_mask.

  For example, span_mask is [1, 0, 1, 1, 0, 1],
  the yielded span_segments are [0], [2, 3], [5].

  Args:
    span_mask: <int|bool>[seq_len], a mask of spans.
  """
  mask_shift_right = np.concatenate([[0], span_mask[:-1]])
  indexes, _ = np.where(span_mask > 0)
  if indexes.size == 0:
    return
  segments = np.cumsum(span_mask > mask_shift_right)[indexes]
  for segment in range(1, segments.max() + 1):
    yield indexes[segments == segment]


def _detokenize(tokens):
  tokens = [six.ensure_str(t) for t in tokens]
  text = ' '.join(tokens)
  text = text.replace(' ##', '').replace('##', '')
  return text


def _normalize(text):
  text = text.lower().strip()
  text = ' '.join(text.split())
  text = re.sub(r'\s*([{}])\s*'.format(string.punctuation),
                lambda m: m.group(1), text)
  return text


def _normalize_text_squad(answer_text):
  """Text normalization from SQuAD 2.0 eval script."""

  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, u' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(answer_text))))


def get_measure_str(true_evidences,
                    top_prediction):
  """Returns a string representing the eval result."""
  normalized_true_evidences = set()
  for evidence in true_evidences:
    normalized_evidence = _normalize_text_squad(evidence['value'])
    if normalized_evidence:
      normalized_true_evidences.add(normalized_evidence)

  normalized_top_prediction = _normalize_text_squad(top_prediction)

  if not normalized_true_evidences and not normalized_top_prediction:
    return _N_N
  elif not normalized_true_evidences:
    return _N_A
  elif not normalized_top_prediction:
    return _A_N
  elif normalized_top_prediction in normalized_true_evidences:
    return _A_A
  else:
    return _A_B


def _get_num_paragraphs(json_example):
  return len(json_example['paragraphs'])


def _get_num_words(json_example):
  return sum(len(p['text'].split()) for p in json_example['paragraphs'])


def _get_bucket_str(bucket, bucket_size, max_num_buckets):
  low = bucket_size * bucket
  high = bucket_size * (bucket + 1 if bucket < max_num_buckets - 1 else np.inf)
  return f'{bucket:02d}-{low}-{high}'


def build_bucket_fn(measure_method, bucket_size,
                    max_num_buckets):
  """Returns a bucket function."""
  if measure_method == 'num_paragraphs':
    measure_fn = _get_num_paragraphs
  elif measure_method == 'num_words':
    measure_fn = _get_num_words
  else:
    raise ValueError(f'Unimplemented measure method {measure_method!r}')

  def _bucket_fn(json_example):
    measure = measure_fn(json_example)
    if max_num_buckets:
      truncated_measure = min(measure, bucket_size * max_num_buckets - 1)
    else:
      truncated_measure = measure
    bucket = truncated_measure // bucket_size
    return _get_bucket_str(bucket, bucket_size, max_num_buckets)

  return _bucket_fn


def get_all_bucket_strs():
  """Returns all possible bucket strs."""
  if not _BUCKET_MEASURE.value:
    return ['']
  return [
      _get_bucket_str(bucket, _BUCKET_SIZE.value, _MAX_NUM_BUCKETS.value)
      for bucket in range(_MAX_NUM_BUCKETS.value)
  ]


class RunInferenceFn(beam.DoFn):
  """DoFn to run inference."""

  def __init__(self, config,
               saved_model_dir, threshold, bucket_measure,
               bucket_size, max_num_buckets, *unused_args,
               **unused_kwargs):
    self._config = config
    self._saved_model_dir = saved_model_dir
    self._threshold = threshold
    self._bucket_measure = bucket_measure
    self._bucket_size = bucket_size
    self._max_num_buckets = max_num_buckets

    self._num_examples = beam.metrics.Metrics.counter(self.__class__,
                                                      'num-examples')
    # Metrics
    self._a_a = beam.metrics.Metrics.counter(self.__class__, 'metrics-a-a')
    self._a_b = beam.metrics.Metrics.counter(self.__class__, 'metrics-a-b')
    self._a_n = beam.metrics.Metrics.counter(self.__class__, 'metrics-a-n')
    self._n_a = beam.metrics.Metrics.counter(self.__class__, 'metrics-n-a')
    self._n_n = beam.metrics.Metrics.counter(self.__class__, 'metrics-n-n')

  def setup(self):
    self._converter = data_utils.get_tf_record_converter(self._config)
    self._saved_model = tf.saved_model.load(self._saved_model_dir)
    self._serving_fn = self._saved_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    if self._bucket_measure:
      self._bucket_fn = build_bucket_fn(self._bucket_measure, self._bucket_size,
                                        self._max_num_buckets)
    else:
      self._bucket_fn = None

  def _get_inputs(self,
                  tf_example):

    def _create_int_tensor(feature_name):
      int64_list = tf_example.features.feature[feature_name].int64_list.value
      return tf.constant(int64_list, dtype=tf.int32)[None, :]

    if self._config.model_type == 'bert':
      inputs = {
          'input_word_ids': _create_int_tensor('input_ids'),
          'input_mask': _create_int_tensor('input_mask'),
          'input_type_ids': _create_int_tensor('segment_ids'),
      }
    elif self._config.model_type == 'bilstm_crf':
      inputs = {
          'input_word_ids': _create_int_tensor('input_ids'),
          'input_mask': _create_int_tensor('input_mask'),
      }
    elif self._config.model_type == 'etc':
      inputs = {
          'global_token_ids': _create_int_tensor('global_token_ids'),
          'global_breakpoints': _create_int_tensor('global_breakpoints'),
          'global_token_type_ids': _create_int_tensor('global_token_type_ids'),
          'long_token_ids': _create_int_tensor('long_token_ids'),
          'long_breakpoints': _create_int_tensor('long_breakpoints'),
          'long_token_type_ids': _create_int_tensor('long_token_type_ids'),
          'long_paragraph_ids': _create_int_tensor('long_paragraph_ids'),
      }
    else:
      raise ValueError(f'Invalid model type: {self._config.model_type}.')

    return inputs

  def _get_tokens(self, tf_example):
    if self._config.model_type in ['bert', 'bilstm_crf']:
      tokens = np.array(
          tf_example.features.feature['tokens'].bytes_list.value, dtype=object)
    elif self._config.model_type == 'etc':
      tokens = np.array(
          tf_example.features.feature['long_tokens'].bytes_list.value,
          dtype=object)
    else:
      raise ValueError(f'Invalid model type: {self._config.model_type}.')
    return tokens

  def _get_scores_and_span_mask(
      self, inputs,
      outputs):
    """Returns predicted span mask on tokens."""
    if self._config.model_type in ['bert', 'bilstm_crf']:
      input_mask = tf.squeeze(inputs['input_mask']).numpy().astype(bool)
      scores = tf.squeeze(list(outputs.values())[0]).numpy()
      span_mask = (scores > self._threshold) & input_mask
    elif self._config.model_type == 'etc':
      global_input_mask = (
          tf.squeeze(inputs['global_token_ids']).numpy().astype(bool))
      long_input_mask = (
          tf.squeeze(inputs['long_token_ids']).numpy().astype(bool))
      long_paragraph_ids = tf.squeeze(inputs['long_paragraph_ids']).numpy()
      global_scores = tf.squeeze(outputs['global']).numpy()
      global_span_mask = (global_scores > self._threshold) & global_input_mask
      long_scores = tf.squeeze(outputs['long']).numpy()
      long_span_mask = (long_scores > self._threshold) & long_input_mask
      scores = long_scores
      if self._config.etc.filter_by_global:
        span_mask = _apply_global_mask_to_long(global_span_mask, long_span_mask,
                                               long_paragraph_ids)
      else:
        span_mask = long_span_mask
    else:
      raise ValueError(f'Invalid model type: {self._config.model_type}.')
    return scores, span_mask

  def _create_inference_output(
      self, json_example,
      tf_example):
    inputs = self._get_inputs(tf_example)
    outputs = self._serving_fn(**inputs)
    tokens = self._get_tokens(tf_example)
    scores, span_mask = self._get_scores_and_span_mask(inputs, outputs)
    pred_evidences = []
    for indexes in _get_span_indexes(span_mask):
      pred_span_text = _normalize(_detokenize(tokens[indexes]))
      if pred_span_text:
        pred_span_score = max(scores[indexes])
        pred_evidences.append((pred_span_score, pred_span_text))
    if pred_evidences:
      top_score, top_prediction = sorted(pred_evidences, reverse=True)[0]
    else:
      top_score, top_prediction = 0.0, ''
    assert len(json_example['attributes']) == 1, 'Attributes not flattened.'
    true_evidences = json_example['attributes'][0].get('evidences', [])
    measure_str = get_measure_str(true_evidences, top_prediction)
    beam.metrics.Metrics.counter(self.__class__, f'{measure_str}').inc()

    # Outputs a TF Exmaple containing predictions and debug fields.
    output = tf.train.Example()
    output.CopyFrom(tf_example)
    output.features.feature['measure_str'].bytes_list.value.append(
        measure_str.encode())
    output.features.feature['json_example'].bytes_list.value.append(
        json.dumps(json_example).encode())
    output.features.feature['top_prediction'].bytes_list.value.append(
        f'{top_prediction}({top_score})'.encode())
    output.features.feature[
        'normalized_top_prediction'].bytes_list.value.append(
            _normalize_text_squad(top_prediction).encode())
    normalized_true_evidences = [
        _normalize_text_squad(e['value']) for e in true_evidences
    ]
    output.features.feature[
        'normalized_true_evidences'].bytes_list.value.append(
            json.dumps(normalized_true_evidences).encode())
    return output, measure_str

  def process(self, json_example, *args,
              **kwargs):
    self._num_examples.inc()
    tf_example = next(self._converter.convert(json_example))
    output, measure_str = (
        self._create_inference_output(json_example, tf_example))
    if self._bucket_measure:
      bucket_str = self._bucket_fn(json_example)
      yield beam.pvalue.TaggedOutput(f'{bucket_str}_{measure_str}', output)
    else:
      yield beam.pvalue.TaggedOutput(f'_{measure_str}', output)


def pattern_insert(filepattern,
                   text,
                   *,
                   pattern = r'(@\*|@\d+)$',
                   keep_pattern = True):
  """Returns a new filepattern by inserting 'text' into the filepattern."""
  m = re.search(pattern, filepattern)
  if m is not None:
    output_filepattern = re.sub(pattern, '_' + text, filepattern, count=1)
    if keep_pattern:
      output_filepattern += m.group(1)
  else:
    output_filepattern = (filepattern + '_' + text)
  return output_filepattern


def pipeline(root):
  """Beam pipeline to run."""

  config = _get_config()

  dirname_pattern = os.path.dirname(_INPUT_JSON_LINES_FILEPATTERN.value)
  basename_pattern = os.path.basename(_INPUT_JSON_LINES_FILEPATTERN.value)

  dirnames = tf.io.gfile.glob(dirname_pattern)
  logging.info('Num input dir names: %s', len(dirnames))
  logging.info('%s', '\n'.join(dirnames))

  if _MODEL_ID.value:
    model_id = _MODEL_ID.value
  else:
    # The parent dir name of the model folder.
    model_id = os.path.basename(os.path.dirname(_SAVED_MODEL_DIR.value))

  for index, dirname in enumerate(dirnames):
    input_filepattern = os.path.join(dirname, basename_pattern)

    all_outputs = (
        root
        | f'{index}_ReadJsonLines' >>
        beam.io.textio.ReadFromText(input_filepattern)
        | f'{index}_JSONloads' >> beam.Map(json.loads)
        | f'{index}_FlattenAttributes' >> beam.ParDo(FlattenAttributesFn())
        | f'{index}_RunInference' >> beam.ParDo(
            RunInferenceFn(config, _SAVED_MODEL_DIR.value, _THRESHOLD.value,
                           _BUCKET_MEASURE.value, _BUCKET_SIZE.value,
                           _MAX_NUM_BUCKETS.value)).with_outputs())

    for bucket_str in get_all_bucket_strs():

      if _OUTPUT_INFERENCE_RESULTS_FILEPATTERN.value:
        output_filepattern = _OUTPUT_INFERENCE_RESULTS_FILEPATTERN.value
      else:
        output_filepattern = os.path.join(dirname, 'predictions', model_id,
                                          _BUCKET_MEASURE.value, bucket_str,
                                          'mave_tfrecord@*')

      for measure_str in _MEASURE_STRS:
        split_str = f'{bucket_str}_{measure_str}'
        outputs = all_outputs[split_str]
        _ = (
            outputs
            | f'{index}_{split_str}_WriteInferenceResults' >>
            beam.io.tfrecordio.WriteToTFRecord(
                pattern_insert(output_filepattern, measure_str),
                coder=beam.coders.ProtoCoder(tf.train.Example)))
        _ = (
            outputs
            | f'{index}_{split_str}_CountTFRecords' >>
            beam.combiners.Count.Globally()
            | f'{index}_{split_str}_JsonDDumps' >>
            beam.Map(lambda x: json.dumps(x, indent=2))
            | f'{index}_{split_str}_WriteCounts' >> beam.io.WriteToText(
                pattern_insert(
                    output_filepattern,
                    f'{measure_str}_counts',
                    keep_pattern=False),
                shard_name_template='',  # To force unsharded output.
            ))


def main(unused_argv):
  # To enable distributed workflows, follow instructions at
  # https://beam.apache.org/documentation/programming-guide/
  # to set pipeline options.
  with beam.Pipeline() as p:
    pipeline(p)

if __name__ == '__main__':
  app.run(main)
