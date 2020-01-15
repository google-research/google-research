# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Implements data loaders and metrics for the SQuAD dataset."""
import collections
import json
import os
import re
import numpy as np
from tensor2tensor.data_generators import text_encoder
import tensorflow.compat.v1 as tf
from qanet import squad_data
from qanet import squad_helper
from qanet.util import configurable
from qanet.util import tokenizer_util
from tensorflow.contrib import data as contrib_data

try:
  unicode        # Python 2
except NameError:
  unicode = str  # Python 3

_ENCODERS = dict()
# char ids 0-255 come from utf-8 encoding bytes
# assign 256-300 to special chars
_BOS_CHAR_ID = 256  # <begin sentence>
_EOS_CHAR_ID = 257  # <end sentence>
_BOW_CHAR_ID = 258  # <begin word>
_EOW_CHAR_ID = 259  # <end word>
_PAD_CHAR_ID = 260  # <padding>
_DEFAULT_CHAR_MAXLEN = 50
_BOS = "<S>"
_EOS = "</S>"
_PAD = "<PAD>"


def load_encoder(vocab_path):
  if vocab_path not in _ENCODERS:
    _ENCODERS[vocab_path] = text_encoder.SubwordTextEncoder(vocab_path)
  return _ENCODERS[vocab_path]


class SQUADDatasetPipeline(configurable.Configurable):
  """Stanford Question Answering Dataset."""

  def __init__(self, mode, config=None, data_format='squad'):
    super(SQUADDatasetPipeline, self).__init__(config=config)
    self.mode = mode
    self.is_training = self.mode == 'train'
    self.data_format = data_format

  def _get_answer(self, context, context_ids, answer_start, answer_end):
    encoder = load_encoder(self.config.vocab_path)
    subtokens = [
        encoder._subtoken_id_to_subtoken_string(s) for s in context_ids
    ]

    if not isinstance(subtokens[0], unicode):
      subtokens = [x.decode('utf-8') for x in subtokens]
    if not isinstance(context, unicode):
      context = context.decode('utf-8')
    assert isinstance(context, unicode)
    assert isinstance(subtokens[0], unicode)

    spans = tokenizer_util.match_subtokens_to_string(context, subtokens)

    start = spans[answer_start][0]
    end = spans[answer_end][1]  # + 1
    text = context[start:end]
    return text

  def get_answer_op(self, features, answer_pred_start, answer_pred_end,
                    has_answer):
    if self.config.tokenizer == 'subword':
      return tf.py_func(
          squad_data._enum_fn(self._get_answer), [
              features['context'], features['context_ids'], answer_pred_start,
              answer_pred_end
          ], 'string')
    else:
      return squad_data.get_answer_op(
          features['context'], features['context_words'], answer_pred_start,
          answer_pred_end, has_answer)

  @classmethod
  def get_input_fn(cls, mode, config, data_format='squad'):
    """Returns an input_fn suitable for tf.learn Estimators."""
    config = cls.build_config(**config)  # Merge in overrides
    tf.logging.info(config)

    def input_fn(params=None):
      del params
      dataset_instance = cls(mode=mode, config=config, data_format=data_format)
      return dataset_instance()(dataset_instance.config)

    return input_fn

  @property
  def batch_size(self):
    """Return the batch size."""
    return self.config.batch_size

  @property
  def size(self):
    # If we return None, then we must manually specify train steps.
    # eval will run until OutOfRange is raised.
    return None

  @staticmethod
  def _config():
    return dict(
        train_split='train',
        eval_split='dev',
        load_tfrecord=False,  # If false, generate on the fly
        # Number of times to repeat dataset per call.
        # If 0, repeat indefinitely.
        num_repeats=1,
        train_shuffle=True,
        cache=True,
        max_length=0,
        data_path='',
        vocab_path='',
        train_batch_size=8,
        eval_batch_size=8,
        resample_too_long=False,
        legacy_rename=True,
        data_format='',
        bytes_per_word=50,
        sort_by_length=True,  # Whether to sort by context length
        tokenizer='word')

  def __call__(self, is_tpu=False):
    """Construct train and eval inputs_fn."""
    cfg = self.config
    load_tfrecord = cfg.load_tfrecord
    if is_tpu:
      load_tfrecord = True
    # TODO(ddohan): Share the common args more clearly
    if not cfg.data_path:
      raise ValueError('Must specify a base data directory.')
    vocab_path = cfg.vocab_path or os.path.join(cfg.data_path, 'vocab.vec')
    if self.mode == 'train':
      train_input = get_input_fn(
          split=cfg.train_split,
          max_length=cfg.max_length,
          # TPUs don't handle OutOfRange exceptions from data pipelines, so we
          # repeat indefinitely and handle setting number of training steps
          # manually. This is handled by the tpu.steps_per_epoch setting.
          # On a GPU, we are able to be more exact about the exact boundary
          # between epochs and avoid reasoning in terms of step counts.
          # If 0, repeat indefinitely. Otherwise repeat N times.
          num_repeats=0 if is_tpu else cfg.num_repeats,
          shuffle=cfg.train_shuffle,
          cache=cfg.cache,
          limit=None,
          data_path=cfg.data_path,
          vocab_path=vocab_path,
          is_tpu=is_tpu,
          use_generator=not load_tfrecord,
          resample_too_long=cfg.resample_too_long,
          is_training=True,
          legacy_rename=cfg.legacy_rename,
          bytes_per_word=cfg.bytes_per_word,
          sort_by_length=cfg.sort_by_length,
          tokenizer=cfg.tokenizer)
      return train_input
    else:
      eval_input = get_input_fn(
          split=cfg.eval_split,
          max_length=None,  # Never do any filtering at eval
          limit=None,
          num_repeats=1,
          shuffle=False,
          cache=cfg.cache,
          data_path=cfg.data_path,
          vocab_path=vocab_path,
          is_tpu=False,  # Never eval on TPU because of py_func
          use_generator=not load_tfrecord,
          is_training=False,
          legacy_rename=cfg.legacy_rename,
          bytes_per_word=cfg.bytes_per_word,
          sort_by_length=cfg.sort_by_length,
          tokenizer=cfg.tokenizer)
      return eval_input


def word_tokenize(text):
  """Split on whitespace and punctuation."""
  return re.findall(r'\w+|[^\w\s]', text, re.UNICODE), None


def build_subword_tokenizer(vocab_path):
  encoder = text_encoder.SubwordTextEncoder(vocab_path)

  def encode(x):
    ids = encoder.encode(x)
    subtokens = [encoder._subtoken_id_to_subtoken_string(s) for s in ids]
    return subtokens, ids

  return encode


def build_nltk_tokenizer():
  nltk_tokenizer = tokenizer_util.NltkAndPunctTokenizer()

  def encode(text):
    return nltk_tokenizer.tokenize_paragraph_flat(text), None

  return encode


def utf_encode_list(text):
  """utf encode every element of a list."""
  return [x.encode('utf-8') for x in text]


# Global state to do some basic caching - avoid reloading
_GLOBAL_VOCAB_CACHE = dict()


def get_pretrained_embeddings_cache(embeddings_path):
  """Get pretrained vocab embeddings."""
  if embeddings_path in _GLOBAL_VOCAB_CACHE:
    return _GLOBAL_VOCAB_CACHE[embeddings_path]
  else:
    tf.logging.info('Loading pretrained embeddings from %s', embeddings_path)
    embeddings, size = squad_helper.get_emb_by_name(embeddings_path)
    embeddings['UNK'] = [0.0] * size

    # OrderedDict, so keys and values are ordered.
    assert isinstance(embeddings, collections.OrderedDict)
  _GLOBAL_VOCAB_CACHE[embeddings_path] = embeddings
  return _GLOBAL_VOCAB_CACHE[embeddings_path]


def get_answer_index(context, context_tokens, answer_start, answer):
  assert isinstance(answer, unicode)
  assert isinstance(context, unicode)
  assert isinstance(context_tokens[0], unicode)
  spans = tokenizer_util.match_subtokens_to_string(context, context_tokens)

  answer_end = answer_start + len(answer)
  word_answer_start = None
  word_answer_end = None
  for word_idx, (start, _) in enumerate(spans):
    if (start <= answer_start and
        # Check that we aren't a part of the same token
        (word_answer_start is None or spans[word_answer_start][0] != start)):
      word_answer_start = word_idx
    if start < answer_end:
      word_answer_end = word_idx
  assert word_answer_start <= word_answer_end, (context, context_tokens,
                                                answer_start, answer)
  return word_answer_start, word_answer_end


def squad_generator(path,
                    tokenizer_fn=word_tokenize,
                    sort_by_length=False,
                    is_subword=False):
  """Generate SQuAD data from the raw json file."""

  with tf.gfile.GFile(path, 'r') as f:
    squad = json.load(f)

  examples = []
  for article in squad['data']:

    for paragraph in article['paragraphs']:
      context = paragraph['context'].strip()
      context_enc = context.encode('utf-8')

      context_tokens, context_ids = tokenizer_fn(context)
      for qa in paragraph['qas']:
        question = qa['question'].strip()
        id_ = qa['id']

        answers = [answer['text'].strip() for answer in qa['answers']]
        answer_starts = [answer['answer_start'] for answer in qa['answers']]
        answer_ends = [
            start + len(answer)
            for start, answer in zip(answer_starts, answers)
        ]

        feats = {}
        feats['id'] = id_
        feats['answers'] = utf_encode_list(answers)
        feats['num_answers'] = len(answers)

        feats['context'] = context_enc
        feats['context_tokens'] = context_tokens
        if context_ids:
          feats['context_ids'] = context_ids
        feats['context_length'] = len(context_tokens)

        question_tokens, question_ids = tokenizer_fn(question)
        feats['question'] = question.encode('utf-8')
        feats['question_tokens'] = utf_encode_list(question_tokens)
        if question_ids:
          feats['question_ids'] = question_ids
        feats['question_length'] = len(feats['question_tokens'])

        starts = []
        ends = []
        if is_subword:
          for answer_start, answer in zip(answer_starts, answers):
            # start, end = get_span(spans, answer_start, answer_end)
            start, end = get_answer_index(
                context=context,
                context_tokens=feats['context_tokens'],
                answer_start=answer_start,
                answer=answer)
            starts.append(start)
            ends.append(end)
        else:
          spans = tokenizer_util.convert_to_spans(context,
                                                  feats['context_tokens'])
          starts = []
          ends = []
          for answer_start, answer_end in zip(answer_starts, answer_ends):
            start, end = get_span(spans, answer_start, answer_end)
            starts.append(start)
            ends.append(end)

        feats['answers_start_token'] = starts
        feats['answers_end_token'] = ends
        feats['context_tokens'] = utf_encode_list(feats['context_tokens'])
        examples.append(feats)

  if sort_by_length:
    examples = sorted(examples, key=lambda x: len(x['context_tokens']))
  for example in examples:
    yield example


def get_span(spans, answer_start, answer_end):
  """Get the start/end index that contains the (start,end) interval.

  Args:
    spans: [List of (start, end) tuples]
    answer_start: Start index
    answer_end: End index

  Returns:
    tuple of (start, end) indices into spans such that
    spans[start][0] <= answer_start <= answer_end <= spans[end][1]

  Raises:
    ValueError: if either the start or end position is not found.
  """
  word_answer_start = None
  word_answer_end = None
  for word_idx, span in enumerate(spans):
    if span[0] <= answer_start <= span[1]:
      word_answer_start = word_idx
    if span[0] <= answer_end <= span[1]:
      word_answer_end = word_idx
    if word_answer_start and word_answer_end:
      break
  if word_answer_end is None and word_answer_start is not None:
    # TODO(ddohan): Figure out why this is sometimes necessary
    if answer_end > spans[-1][-1]:
      word_answer_end = len(spans) - 1
  if word_answer_end is None or word_answer_start is None:
    raise ValueError
  assert word_answer_end >= word_answer_start
  return word_answer_start, word_answer_end

def encode_question_and_context(question,
                                context,
                                qid=0,
                                embedding_path=None,
                                tokenizer_fn=word_tokenize):
  """Encode given question and context strings to an input dict for qanet."""
  output = dict(id=qid)
  if embedding_path:
    embeddings = get_pretrained_embeddings_cache(embedding_path)

  def add_fields(text, field_name):
    tokens, ids = tokenizer_fn(text)
    output[field_name] = text
    output['%s_tokens' % field_name] = tokens
    output['%s_length' % field_name] = len(tokens)
    output['%s_tokens' % field_name] = tokens
    if ids is not None:
      output['%s_ids' % field_name] = tokens
    if embedding_path:
      emb = [
          embeddings[x] if x in embeddings else embeddings['UNK']
          for x in tokens
      ]
      output['%s_vecs' % field_name] = tf.constant(emb)

  add_fields(question, 'question')
  add_fields(context, 'context')
  output = _tokens_to_bytes(output)
  output = do_renames(output)
  return output


def build_tfrecord_pipeline(filenames):
  """Read TFRecords from disk to create data pipeline."""
  sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.int64, allow_missing=True)
  str_sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.string, allow_missing=True)
  int_feature = tf.FixedLenFeature([], tf.int64)
  str_feature = tf.FixedLenFeature([], tf.string)
  features = {
      'id': str_feature,
      'num_answers': int_feature,
      'answers': str_sequence_feature,
      'answers_start_token': sequence_feature,
      'answers_end_token': sequence_feature,
      'context': str_feature,
      'context_length': int_feature,
      'context_tokens': str_sequence_feature,
      'question': str_feature,
      'question_length': int_feature,
      'question_tokens': str_sequence_feature,
  }

  def _parse(proto):
    return tf.parse_single_example(proto, features=features)

  ds = tf.data.TFRecordDataset(
      filenames,
      # 1 GB
      buffer_size=1024 * 1024 * 1024,
      num_parallel_reads=8)

  ds = ds.map(_parse, num_parallel_calls=16)
  return ds


def build_generator_pipeline(data_path,
                             split,
                             tokenizer_fn=word_tokenize,
                             sort_by_length=False,
                             is_subword=False):
  """Build a data pipeline from raw json SQuAD file."""
  shapes, types = get_shapes_and_types(
      is_tpu=False,
      max_length=None,
      include_bytes=False,
      include_ids=is_subword)

  def generator():
    path = os.path.join(data_path, '%s-v1.1.json' % split)
    return squad_generator(
        path=path,
        tokenizer_fn=tokenizer_fn,
        sort_by_length=sort_by_length,
        is_subword=is_subword)

  ds = tf.data.Dataset.from_generator(
      generator, output_types=types, output_shapes=shapes)
  return ds


FIELD_NAMES = ['context', 'question', 'answers']


def get_shapes_and_types(
    is_tpu=False,
    max_length=None,
    include_bytes=True,
    bytes_per_word=50,
    include_ids=True,
):
  """Build tuple of (shapes, types) dictionaries specifying the dataset."""
  # TODO(ddohan): Explicitly list types & shapes instead of creating in a loop
  types = {}
  shapes = {}
  length = None
  if is_tpu:
    assert max_length
    length = max_length

  for k in FIELD_NAMES:
    if not is_tpu:
      types[k] = tf.string
      types['%s_tokens' % k] = tf.string
      shapes[k] = []
      shapes['%s_tokens' % k] = [length]
    types['%s_length' % k] = tf.int64
    shapes['%s_length' % k] = []

    if include_ids:
      types['%s_ids' % k] = tf.int64
      shapes['%s_ids' % k] = [length]

    if include_bytes:
      types['%s_bytes' % k] = tf.int64
      shapes['%s_bytes' % k] = [length, bytes_per_word]

  for k in ['answers_tokens', 'answers_length', 'answers_bytes', 'answers_ids']:
    if k in types:
      del types[k]
      del shapes[k]

  types['num_answers'] = tf.int64
  types['answers_start_token'] = tf.int64
  types['answers_end_token'] = tf.int64

  shapes['num_answers'] = []
  shapes['answers_start_token'] = []
  shapes['answers_end_token'] = []

  if not is_tpu:
    types['id'] = tf.string
    shapes['id'] = []

  ids = []
  for k in shapes:
    if k.startswith('answer'):
      # TODO(ddohan): Handle multiple answers
      shapes[k] = [1 if is_tpu else None] + shapes[k]
    if k.endswith('_ids'):
      ids.append(k)

  if not include_ids:
    for k in ids:
      del shapes[k]
      del types[k]

  tf.logging.info(sorted(types.keys()))
  return shapes, types


def resample_example(example, max_length=256):
  """Given an example and max length, resample the context to that length.

  Start position randomly chosen from [0, answer_start]. Assumes a single
    answer per context, which is true for the SQuAD training set.

  Args:
    example: A single example containing at least these fields:
      ['answers_start_token', 'answers_end_token', 'context_tokens',
      'context_length']
    max_length: Maximum length. Contexts are resampled to this length.

  Returns:
    Resampled example.
  """

  # TODO(ddohan): Consider randomly cropping to shorter lengths
  # TODO(ddohan): Figure out how to resample the raw text as well. Not necessary
  # for training
  def _resample():
    """Helper method for resampling inside cond."""
    x = example
    ans_start = tf.to_int64(x['answers_start_token'][0])
    ans_end = tf.to_int64(x['answers_end_token'][0])
    min_start = tf.maximum(tf.to_int64(0), ans_end - max_length + 1)
    max_start = ans_start
    min_start = tf.minimum(min_start, max_start)
    start_idx = tf.random_uniform([],
                                  min_start,
                                  max_start + 1, dtype=tf.int64)
    for k in ['answers_start_token', 'answers_end_token']:
      x[k] -= start_idx
    if 'context_ids' in x:
      x['context_ids'] = x['context_ids'][start_idx:start_idx + max_length]
    x['context_tokens'] = x['context_tokens'][start_idx:start_idx + max_length]
    x['context_length'] = tf.to_int64(tf.shape(x['context_tokens'])[0])
    return x

  def identity():
    return example

  return tf.cond(
      tf.greater_equal(
          tf.to_int32(max_length), tf.to_int32(example['context_length'])),
      true_fn=identity,
      false_fn=_resample)


def _tokens_to_bytes_helper(tokens,
                            bytes_per_word=50,
                            bos=_BOS,
                            eos=_EOS,
                            pad=_PAD,
                            method='elmo'):
  """Given a sequence of strings, map to sequence of bytes.

  Args:
    tokens: A tf.string tensor
    bytes_per_word: Size of output
    bos: begin-of-sentence token
    eos: end-of-sentence token
    pad: padding token
    method: 'elmo' or a callable acting as _elmo_token_to_bytes.

  Returns:
    A tensor of shape words.shape + [bytes_per_word] containing byte versions
    of each word.
  """

  def lambda_token_to_bytes(x):
    if method == 'elmo':  # default option
      return _elmo_token_to_bytes(
          x, max_length=bytes_per_word, bos=bos, eos=eos, pad=pad)
    elif callable(method):
      return method(x, max_length=bytes_per_word, bos=bos, eos=eos, pad=pad)
    else:
      raise ValueError('Unknown method %s' % method)

  with tf.device('/cpu:0'):
    tf.assert_rank(tokens, 1)
    shape = tf.shape(tokens)
    tf.logging.info(tokens)
    tokens_flat = tf.reshape(tokens, [-1])
    as_bytes_flat = tf.map_fn(
        fn=lambda_token_to_bytes,
        elems=tokens_flat,
        dtype=tf.int32,
        back_prop=False)
    tf.logging.info(as_bytes_flat)
    as_bytes = tf.reshape(as_bytes_flat, [shape[0], bytes_per_word])
  return as_bytes

def _elmo_token_to_bytes(text, max_length,
                         bos=_BOS, eos=_EOS, pad=_PAD):
  """ELMO-specific way of converting a word into a  byte seq.

  This mimics docqa/elmo/data.py, UnicodeCharsVocabulary.

  Args:
    text: tf.string tensor of shape []
    max_length: Maximum number of bytes per word. Defaults to 50.
    bos: begin-of-sentence token
    eos: end-of-sentence token
    pad: padding token

  Returns:
    A tf.int32 tensor of the byte encoded text.
  """
  byte_ids = tf.to_int32(tf.decode_raw(text, tf.uint8))

  # Special handling for bos and eos
  byte_ids = tf.cond(tf.equal(text, bos),
                     lambda: tf.constant([_BOS_CHAR_ID]), lambda: byte_ids)
  byte_ids = tf.cond(tf.equal(text, eos),
                     lambda: tf.constant([_EOS_CHAR_ID]), lambda: byte_ids)

  byte_ids = byte_ids[:max_length - 2]
  padding = tf.fill([max_length - tf.shape(byte_ids)[0] - 2], _PAD_CHAR_ID)
  byte_ids = tf.concat(
      [[_BOW_CHAR_ID], byte_ids, [_EOW_CHAR_ID], padding], axis=0)
  tf.logging.info(byte_ids)

  byte_ids = tf.reshape(byte_ids, [max_length])
  tf.logging.info(byte_ids.get_shape().as_list())
  return tf.cond(tf.equal(text, pad),
                 lambda: tf.zeros(max_length, dtype=tf.int32),
                 lambda: byte_ids + 1)

def _tokens_to_bytes(x, bytes_per_word=50):
  """Encode tokens to byte values suitable for ELMO."""
  new = {}
  for k in x:
    if k.endswith('_tokens'):
      # ELMO has us add in the BOS/EOS tokens around the actual sentence.
      byte_enc = _tokens_to_bytes_helper(
          tf.concat([tf.constant([_BOS]), x[k],
                     tf.constant([_EOS])], axis=0),
          bytes_per_word=bytes_per_word)
      # Remove `_tokens` and add in `_bytes`
      new['_'.join(k.split('_')[:-1] + ['bytes'])] = byte_enc
  x.update(new)
  return x


def do_renames(example):
  """Rename fields to the expected names inside the QANet models."""
  renames = dict(
      question_bytes='indexed_question_chars',
      question_tokens='question_words',
      context_bytes='indexed_context_chars',
      context_tokens='context_words',
      context_length='context_num_words',
      question_length='question_num_words',
      answers_start_token='word_answer_starts',
      answers_end_token='word_answer_ends',
  )
  for current, old in renames.iteritems():
    if current in example:
      example[old] = example[current]
      del example[current]
  return example


def get_input_fn(split='dev',
                 shuffle=False,
                 num_repeats=1,
                 limit=None,
                 do_embedding=True,
                 cache=True,
                 max_length=None,
                 resample_too_long=True,
                 data_path=None,
                 vocab_path=None,
                 is_tpu=False,
                 use_generator=True,
                 is_training=False,
                 include_bytes=True,
                 legacy_rename=False,
                 shuffle_buffer_size=1024,
                 bytes_per_word=50,
                 sort_by_length=False,
                 tokenizer='word'):
  """Build input function.

  Args:
    split: Split name to use. One of train/dev/test.
    shuffle: Whether to shuffle.
    num_repeats: Number of times to repeat. Infinite if 0.
    limit: If specified, take only the first N examples.
    do_embedding: Whether to do word embeddings lookups
    cache: Whether to cache the data pipeline
    max_length: If specified, specify maximum context length to return.
    resample_too_long: When max_length is specified and this is not None,
      resample the region around the correct answer. If None, these examples
      are filtered out
    data_path: Directory containing {split}-v1.1.json files
    vocab_path: Path to word vector file.
    is_tpu: Whether we are on a TPU.
    use_generator: If true, generate from raw json. Otherwise load tfrecords.
    is_training: Whether we are training.
    include_bytes: Whether to encode tokens in ELMO compatible bytes.
    legacy_rename: Whether to rename to be compatible with old data pipeline.
    shuffle_buffer_size: Size of shuffle buffer.
    bytes_per_word: Use this many bytes to encode each individual token.
    sort_by_length: Whether to sort examples by length.
    tokenizer: Which tokenizer to use. One of `word`, `nltk`

  Returns

  """
  if is_tpu:
    assert max_length

  include_ids = False
  if tokenizer == 'subword':
    do_embedding = False
    include_ids = True

  # Do the GLOVE embedding lookups in the data loader
  if do_embedding:
    # Load and package into the graph directly
    # Vocab is about ~200MB total once filtered down
    embeddings = get_pretrained_embeddings_cache(embeddings_path=vocab_path)

  def _input_fn(params=None):
    """Input function compatible with `Experiment` object.

    Pipeline proceeds as:
    -> generation examples from json or tfrecords
    -> limit to first N examples (if limit is not None)
    -> encode any _tokens fields as bytes
    -> cache the results to avoid recomputing
    -> Lookup tokens in GLOVE embeddings
    -> Shuffle & repeat

    Args:
      params: Params passed to the estimator. Contains 'batch_size'.

    Returns:
      A tuple of feature tensors and target tensors.

    Raises:
      ValueError: If filtering by length is set during eval mode.
    """
    if not is_training:
      assert not is_tpu
    tf.logging.info('Data pipeline given params:\n%s' % params)
    if params:
      if is_training:
        batch_size = params['train_batch_size']
      else:
        batch_size = params['eval_batch_size']

    if use_generator:
      tf.logging.info('Building generator data pipeline.')
      if tokenizer == 'word':
        tf.logging.info('Using word split encoder.')
        tokenizer_fn = word_tokenize
      elif tokenizer == 'nltk':
        tf.logging.info('Using NLTK encoder.')
        tokenizer_fn = build_nltk_tokenizer()
      elif tokenizer == 'subword':
        tokenizer_fn = build_subword_tokenizer(vocab_path=vocab_path)
      else:
        raise ValueError('Unknown tokenizer %s' % tokenizer)
      ds = build_generator_pipeline(
          data_path=data_path,
          split=split,
          tokenizer_fn=tokenizer_fn,
          sort_by_length=sort_by_length,
          is_subword=tokenizer == 'subword')
    else:
      tf.logging.info('Loading TFRecords from %s' % data_path)
      filenames = tf.gfile.Glob(os.path.join(data_path, '%s_*' % split))
      tf.logging.info(filenames)
      ds = build_tfrecord_pipeline(filenames=filenames)

    if max_length:
      if not is_training:
        raise ValueError('Unable to filter or resample examples at eval time.')
      if resample_too_long:

        tf.logging.info('Resampling with max length %s', max_length)
        def _resample(x):
          return resample_example(x, max_length=max_length)

        ds = ds.map(_resample, num_parallel_calls=16)
      else:
        # Filter out examples over our max length to avoid an error downstream.
        tf.logging.info('Filtering out examples over max length %s', max_length)
        def _not_too_long(x):
          return tf.greater_equal(
              tf.to_int32(max_length), tf.to_int32(x['context_length']))

        ds = ds.filter(_not_too_long)

    if limit:
      # Take the first N examples
      ds = ds.take(limit)

    if include_bytes:
      tokens_to_bytes = lambda x: _tokens_to_bytes(x, bytes_per_word)
      ds = ds.map(tokens_to_bytes, num_parallel_calls=16)

    if cache:
      # Cache dataset to avoid hitting the python generator after first epoch
      ds = ds.cache()

    # Subset that we should actually pass back to the caller
    # This is required to filter out tf.string fields which are not TPU
    # compatible
    # Specifically: id, context, question, context_tokens and question_tokens
    # are all string fields that will be removed.
    shapes, _ = get_shapes_and_types(
        is_tpu=is_tpu,
        max_length=max_length,
        include_bytes=include_bytes,
        bytes_per_word=bytes_per_word,
        include_ids=include_ids)

    if do_embedding:
      # Embed tokens with pretrained word vectors

      # Add in shape info before batching
      shapes['context_vecs'] = [max_length if is_tpu else None, 300]
      shapes['question_vecs'] = [max_length if is_tpu else None, 300]

      def lookup(words):
        # Do embedding lookups on a tensor of words
        # We use a py_func so we can check for both upper and lowercase.
        # TODO(ddohan): Revert to embedding_lookup for TPU support

        def embed_words(words):
          def embed_word(word):
            utf_word = word.decode('utf-8')
            for key in [word, word.lower(), utf_word, utf_word.lower(), 'UNK']:
              if key in embeddings:
                return embeddings[key]

          emb = [embed_word(word) for word in words]
          emb = np.array(emb, dtype=np.float32)
          return emb

        embedded = tf.py_func(
            embed_words, inp=[words], Tout=[tf.float32], stateful=False)
        embedded = tf.reshape(embedded, [-1, 300])
        return embedded

      def lookup_fields(d):
        d['context_vecs'] = lookup(d['context_tokens'])
        d['question_vecs'] = lookup(d['question_tokens'])
        return d

      ds = ds.map(lookup_fields, num_parallel_calls=16)

    repeats = num_repeats if num_repeats else None
    if shuffle and repeats != 1:
      tf.logging.info('Shuffle and repeat size: %s' % shuffle_buffer_size)
      ds = ds.apply(
          contrib_data.shuffle_and_repeat(
              buffer_size=shuffle_buffer_size, count=repeats))
    elif repeats != 1:
      tf.logging.info('Repeating')
      ds = ds.repeat(count=repeats)
    elif shuffle:
      tf.logging.info('Shuffle size: %s' % shuffle_buffer_size)
      ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    def filter_fields(example):
      out = {}
      for k in shapes:
        out[k] = example[k]
      return out

    ds = ds.map(filter_fields, num_parallel_calls=16)

    if is_training:
      ds = ds.padded_batch(
          batch_size, padded_shapes=shapes, drop_remainder=True)
    else:
      # Never want to ignore values at eval time
      ds = ds.padded_batch(batch_size, padded_shapes=shapes)
    ds = ds.prefetch(
        tf.data.experimental.AUTOTUNE)  # Buffer a few batches ahead
    if do_embedding:
      iterator = ds.make_initializable_iterator()
      # Must be initialized when the graph is initialized and before the
      # dataset tensors are evaluated.
      # Run `tf.tables_initializer()` before getting first batch
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
    else:
      iterator = ds.make_one_shot_iterator()
    batch = iterator.get_next()

    if legacy_rename:
      batch = do_renames(batch)
    return batch, batch

  return _input_fn
