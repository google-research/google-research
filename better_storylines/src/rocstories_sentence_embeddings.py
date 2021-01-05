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

"""Dataset of embeddings for each sentence in ROC Stories."""

import csv
import enum

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import tensorflow_hub as hub

# pylint: disable=g-import-not-at-top
if tf.__version__.startswith('1'):
  from bert import extract_features
  from bert import modeling
  from bert import tokenization
# pylint: enable=g-import-not-at-top


gfile = tf.io.gfile


_DESCRIPTION = """\
'The ROC Stories dataset and corresponding Story Cloze Test is a commonsense
reasoning framework for evaluating story understanding and story generation.
The train set contains stories that are 5 sentences long. The validation and
test sets contain the first 4 sentences of stories and plus 2 possible ending
sentences for each story. This dataset computes sentence embeddings for each
sentence in the Story Cloze dataset.
"""

# The second citation introduces the source data, while the first
# introduces the specific form (non-anonymized) we use here.
_CITATION = """\
@article{ippolito2020,
  title={Toward Better Storylines with Sentence-Level Language Models},
  author={Ippolito, Daphne and Grangier, David and Eck, Douglas and Callison-Burch, Chris
  journal={Proceedings of the 2020 Conference of the Association for Computational Linguistics},
  year={2020}
}
"""

# BERT PATHS AND OPTIONS
MODEL_NAME = 'cased_L-12_H-768_A-12'
_BERT_CONFIG_FILE = '%s/bert_config.json' % MODEL_NAME
_TOKENIZER_VOCAB_FILE = '%s/vocab.txt' % MODEL_NAME
_INIT_CHECKPOINT = '%s/bert_model.ckpt' % MODEL_NAME

_TOKENIZER_DO_LOWER_CASE = False
_TOKENIZER_MAX_SEQ_LEN = 512
_VOCAB_FREQUENCY_FILE = 'vocab_frequencies'

# UNIVERSAL SENTENCE ENCODER PATHS AND OPTIONS
_TF_HUB_PATH = 'https://tfhub.dev/google/'
# As a note, this is a newer version of the universal sentence embeddings
# than the ones we initially conducted experiments with.
_UNIVERSAL_SENTENCE_ENCODER_PATH = (_TF_HUB_PATH +
                                    'universal-sentence-encoder-large/3')

VALIDATION_2018 = tfds.Split('validation_2018')
VALIDATION_2016 = tfds.Split('validation_2016')
TEST_2016 = tfds.Split('test_2016')
TEST_2018 = tfds.Split('test_2018')


class EmbeddingType(enum.Enum):
  """Indicates what type of BERT-based embedding to use."""
  BERT_REDUCE_MEAN = 'bert_mean_emb'
  BERT_REDUCE_WEIGHTED_MEAN = 'bert_weighted_mean_emb'
  BERT_REDUCE_MIN_MAX = 'bert_min_max_emb'
  BERT_REDUCE_MIN_MAX_MEAN = 'bert_min_max_mean_emb'
  BERT_CLASS_TOKEN = 'bert_class_token'
  UNIVERSAL_SENTENCE = 'universal_sentence_emb'
  MT_SMALL = 'long_tacl_europarl_0602_090043'


def masked_mean(embedding, mask):
  """Take the masked mean of the wordpiece embeddings."""
  mask = np.expand_dims(mask, axis=-1)
  masked_sum = np.sum(embedding * mask, axis=0)
  total = np.sum(mask) + 1e-10
  return masked_sum / total


def masked_min_max(embedding, mask):
  """Take the masked mean of the wordpiece embeddings."""
  mask = np.expand_dims(mask, axis=-1)
  masked_min = np.min(embedding * mask, axis=0)
  masked_max = np.max(embedding * mask, axis=0)
  return np.concatenate([masked_min, masked_max], axis=-1)


def masked_min_max_mean(embedding, mask):
  mean_emb = masked_mean(embedding, mask)
  min_max_emb = masked_min_max(embedding, mask)
  return np.concatenate([mean_emb, min_max_emb], axis=-1)


class ExampleReaderDoFn(beam.DoFn):
  """Yields examples from the Book dataset."""

  def process(self, filepath):
    """Reads in the .csv containing raw ROCStories data."""
    with gfile.GFile(filepath, 'r') as f:
      print('FILEPATH: ', filepath)
      reader = csv.reader(f)
      header = next(reader)
      if header[1] == 'storytitle':
        split = tfds.Split.TRAIN
      elif len(header) == 8:
        split = tfds.Split.VALIDATION
      else:
        split = tfds.Split.TEST

      for line in reader:
        Metrics.counter('ExampleReaderDoFn', 'read_story').inc()
        story_id = line[0]

        if split == tfds.Split.TRAIN:
          story_sentences = line[2:]
          label = None
        elif split == tfds.Split.VALIDATION:
          story_sentences = line[1:7]
          label = int(line[-1]) - 1
        elif split == tfds.Split.TEST:
          story_sentences = line[1:]
          label = None
        Metrics.counter('ExampleReaderDoFn', 'yield_story').inc()
        yield story_id, story_sentences, label


class GenerateUniversalEmbeddings(beam.DoFn):
  """Generate sentence embeddings with Universal Sentence Embeddings."""

  def start_bundle(self):
    tf.compat.v1.disable_eager_execution()
    tf.reset_default_graph()
    self.module = hub.Module(_UNIVERSAL_SENTENCE_ENCODER_PATH)

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    self.session.run(tf.tables_initializer())

    self.messages_placeholder = tf.placeholder(tf.string)
    self.feature_tensor = self.module(self.messages_placeholder)

  def process(self, inputs):
    story_id, story_sentences, label = inputs
    Metrics.counter('GenerateUniversalEmbeddings', 'starting_story').inc()

    embeddings = self.session.run(
        self.feature_tensor,
        feed_dict={self.messages_placeholder: story_sentences})
    Metrics.counter('GenerateUniversalEmbeddings', 'finished_with_story').inc()

    example = {
        'story_id': story_id,
        'embeddings': [embeddings[i, :] for i in range(len(story_sentences))],
        'label': label if label is not None else 2,
    }
    yield story_id, example


class GenerateBERTEmbeddings(beam.DoFn):
  """Generate BERT sentence embeddings."""

  def __init__(self,
               embedding_type,
               vocab_file,
               vocab_frequency_file,
               bert_config_file,
               init_checkpoint,
               max_seq_len,
               do_lower_case):
    if tf.__version__.startswith('2'):
      raise ValueError('Data generation can only be performed with TF1.')

    self._embedding_type = embedding_type
    self._vocab_file = vocab_file
    self._vocab_frequency_file = vocab_frequency_file
    self._bert_config_file = bert_config_file
    self._init_checkpoint = init_checkpoint
    self._max_seq_len = max_seq_len
    self._do_lower_case = do_lower_case
    self._layer_indexes = [-2]
    self._frequencies = None

  def start_bundle(self):
    bert_config = modeling.BertConfig.from_json_file(self._bert_config_file)

    self._tokenizer = tokenization.FullTokenizer(
        vocab_file=self._vocab_file, do_lower_case=self._do_lower_case)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        master=None,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            num_shards=1,
            per_host_input_for_training=is_per_host))

    model_fn = extract_features.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=self._init_checkpoint,
        layer_indexes=self._layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=False)

    self._estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=1)

  def _make_examples(self, texts):
    """Creates BERT examples and input_fn to iterate over them.

    Args:
      texts: List of strings. One example will be created per string.

    Returns:
      Dictionary mapping from unique example ID to example
    """
    print('MAKING EXAMPLES')
    examples = [extract_features.InputExample(i, text, None)
                for i, text in enumerate(texts)]
    features = extract_features.convert_examples_to_features(
        examples, self._max_seq_len, self._tokenizer)
    unique_id_to_feature = {}
    for feature in features:
      unique_id_to_feature[feature.unique_id] = feature
    input_fn = extract_features.input_fn_builder(
        features=features, seq_length=self._max_seq_len)
    return unique_id_to_feature, input_fn

  def _get_or_create_word_frequencies(self):
    """Returns a dictionary mapping from wordpiece ID to its frequency."""
    if self._frequencies is None:
      freq_dict = {}
      with gfile.Open(self._vocab_frequency_file) as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        for line in reader:
          token_id = int(line[0])
          frequency = int(line[-1])
          freq_dict[token_id] = frequency
      total_words = sum(freq_dict.values())
      self._frequencies = [
          freq_dict.get(i, 0) / total_words for i in range(0, 30000)]
    return self._frequencies

  def process(self, inputs):
    story_id, story_sentences, label = inputs

    unique_id_to_feature, input_fn = self._make_examples(story_sentences)

    results = {}
    for result in self._estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result['unique_id'])
      Metrics.counter('GenerateBERTEmbeddings', 'story_%d' % unique_id).inc()

      feature = unique_id_to_feature[unique_id]

      output = result['layer_output_0']

      if self._embedding_type == EmbeddingType.BERT_REDUCE_MEAN:
        sentence_embedding = masked_mean(
            output, feature.input_mask)
      elif self._embedding_type == EmbeddingType.BERT_REDUCE_MIN_MAX:
        sentence_embedding = masked_min_max(
            output, feature.input_mask)
      elif self._embedding_type == EmbeddingType.BERT_REDUCE_MIN_MAX_MEAN:
        sentence_embedding = masked_min_max_mean(
            output, feature.input_mask)
      elif self._embedding_type == EmbeddingType.BERT_REDUCE_WEIGHTED_MEAN:
        frequencies = self._get_or_create_word_frequencies()
        # TODO(dei): Fix that this gives full weight to the CLS and SEP tokens.
        weights = np.take(frequencies, feature.input_ids)
        weights = 0.0001 / (weights + 0.0001)  # a = 0.0001 specified in paper.
        mask = (feature.input_mask * weights).tolist()
        sentence_embedding = masked_mean(output, mask)
      elif self._embedding_type == EmbeddingType.BERT_CLASS_TOKEN:
        sentence_embedding = output[0, :]
      else:
        raise ValueError('Should only be called for creating BERT embeddings.')

      results[unique_id] = sentence_embedding.astype(np.float32)

    Metrics.counter('GenerateBERTEmbeddings', 'finished_with_story').inc()
    example = {
        'story_id': story_id,
        'embeddings': [results[i] for i in range(len(unique_id_to_feature))],
        'label': label if label is not None else 2,
    }
    yield story_id, example


class ROCStoriesEmbeddingConfig(tfds.core.BuilderConfig):
  """BuilderConfig for ROCStories sentence embeddings."""

  def __init__(self, *, embedding_type, output_emb_size, version, **kwargs):
    """BuilderConfig for ROCStories sentence embeddings.

    Args:
      embedding_type: EmbeddingType enum indicating strategy to embed sentence.
      output_emb_size: Dimension of embedding being saved per sentence.
      version: str. "x.y.z"-style dataset version.
      **kwargs: keyword arguments forwarded to super.
    """
    self.embedding_type = embedding_type
    self.output_emb_size = output_emb_size

    version = tfds.core.Version(version)
    super(ROCStoriesEmbeddingConfig, self).__init__(
        name=self.embedding_type.value, version=version, **kwargs)


class ROCStoriesEmbeddings(tfds.core.BeamBasedBuilder):
  """ROCStories datasets for each sentence embedding type."""
  BUILDER_CONFIGS = [
      ROCStoriesEmbeddingConfig(
          embedding_type=EmbeddingType.BERT_REDUCE_MEAN,
          output_emb_size=768,
          version='1.0.1',
          description='Mean of BERT wordpiece embeddings for sentence.',
      ),
      ROCStoriesEmbeddingConfig(
          embedding_type=EmbeddingType.BERT_REDUCE_WEIGHTED_MEAN,
          output_emb_size=768,
          version='1.0.1',
          description='Mean of BERT wordpiece embeddings for sentence, '
                      'weighted by inverse word frequency.',
      ),
      ROCStoriesEmbeddingConfig(
          embedding_type=EmbeddingType.BERT_REDUCE_MIN_MAX,
          output_emb_size=768*2,  # Because of min/max being concatenated.
          version='1.0.1',
          description='Min and max of BERT wordpiece embeddings for sentence.',
      ),
      ROCStoriesEmbeddingConfig(
          embedding_type=EmbeddingType.BERT_REDUCE_MIN_MAX_MEAN,
          output_emb_size=768*3,  # Because of mean/min/max being concatenated.
          version='1.0.1',
          description='Min, max, and mean of BERT wordpiece embeddings.',
      ),
      ROCStoriesEmbeddingConfig(
          embedding_type=EmbeddingType.BERT_CLASS_TOKEN,
          output_emb_size=768,
          version='1.0.1',
          description='Class token returned by BERT.',
      ),
      ROCStoriesEmbeddingConfig(
          embedding_type=EmbeddingType.UNIVERSAL_SENTENCE,
          output_emb_size=512,
          version='1.0.0',
          description='Class token returned by BERT',
      ),
  ]

  def _info(self):
    # Should return a tfds.core.DatasetInfo object

    embedding_feature = tfds.features.Tensor(
        shape=(self.builder_config.output_emb_size,), dtype=tf.float32)

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'embeddings': tfds.features.Sequence(feature=embedding_feature),
            'story_id': tfds.features.Text(),
            'label': tfds.features.ClassLabel(num_classes=3),
        }),
        supervised_keys=None,
        homepage='http://cs.rochester.edu/nlp/rocstories/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    fp = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'
    dl_paths = dl_manager.download({
        'train2017': fp.format('1emH8KL8NVCCumZc2oMu-3YqRWddD3AqZEHvNqMdfgKA'),
        'valid2018': fp.format('1F9vtluzD3kZOn7ULKyMQZfoRnSRzRnnaePyswkRqIdY'),
        'valid2016': fp.format('1FkdPMd7ZEw_Z38AsFSTzgXeiJoLdLyXY_0B_0JIJIbw'),
        'test2016': fp.format('11tfmMQeifqP-Elh74gi2NELp0rx9JMMjnQ_oyGKqCEg'),
        'train2016': fp.format('1bGW8LzTAPelhBvV8FPI6YvNtdLcf-TStowpk12UkExw'),
        'test2018': fp.format('1xLV6YhjKX5HaQ2fnFvJ5HC93oX-iw_Gfm_9ufRbljeQ'),
    })
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,
            gen_kwargs={'filepaths': [
                dl_paths['train2017'], dl_paths['train2016']]},
        ),
        tfds.core.SplitGenerator(
            name=VALIDATION_2018,
            num_shards=1,
            gen_kwargs={'filepaths': [dl_paths['valid2018']]},
        ),
        tfds.core.SplitGenerator(
            name=VALIDATION_2016,
            num_shards=1,
            gen_kwargs={'filepaths': [dl_paths['valid2016']]},
        ),
        tfds.core.SplitGenerator(
            name=TEST_2018,
            num_shards=1,
            gen_kwargs={'filepaths': [dl_paths['test2018']]},
        ),
        tfds.core.SplitGenerator(
            name=TEST_2016,
            num_shards=1,
            gen_kwargs={'filepaths': [dl_paths['test2016']]}
        ),
    ]

  def _build_pcollection(self, pipeline, filepaths):
    """Build PCollection of examples in the raw (text) form."""

    pipeline |= beam.Create(filepaths)
    pipeline |= beam.ParDo(ExampleReaderDoFn())
    pipeline |= beam.Reshuffle()

    if self.builder_config.embedding_type == EmbeddingType.UNIVERSAL_SENTENCE:
      pipeline |= beam.ParDo(GenerateUniversalEmbeddings())
    elif 'bert' in self.builder_config.name:
      pipeline |= beam.ParDo(
          GenerateBERTEmbeddings(
              embedding_type=self.builder_config.embedding_type,
              vocab_file=_TOKENIZER_VOCAB_FILE,
              vocab_frequency_file=_VOCAB_FREQUENCY_FILE,
              bert_config_file=_BERT_CONFIG_FILE,
              init_checkpoint=_INIT_CHECKPOINT,
              max_seq_len=_TOKENIZER_MAX_SEQ_LEN,
              do_lower_case=_TOKENIZER_DO_LOWER_CASE
          )
      )
    else:
      raise ValueError('Unsupported embedding type')
    return pipeline


