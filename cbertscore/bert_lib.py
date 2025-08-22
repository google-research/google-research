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

"""Extract pre-computed feature vectors from BERT."""

import collections
import copy

from bert import modeling
from bert import tokenization

import tensorflow.compat.v1 as tf

gfile = tf.io.gfile
contrib_tpu = tf.compat.v1.estimator.tpu


# A file location of newline-delimited medical words.
MEDICAL_TERMS_TXT = ''


class InputExample(object):
  """Raw Input example of the schema needed by BERT.

  BERT_Score only uses text_a, because it only uses the contextual embeddings.
  text_b is present here in case we use this library for something else besides
  BERT_Score in the future.
  """

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """Features (aka BERT input) for a single sentence.

  This is the sentence fully prepped for input to the model. The raw text has
  been tokenized into word pieces, and for each word piece, we have mapped it
  to an integer id. input_mask records whether each token is part of text_a
  or text_b.
  """

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params['batch_size']

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        'unique_ids':
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        'input_ids':
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        'input_mask':
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        'input_type_ids':
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_index, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  if not isinstance(layer_index, (list, tuple)):
    assert isinstance(layer_index, int)
    layer_index = [layer_index]

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features['unique_ids']
    input_ids = features['input_ids']
    input_mask = features['input_mask']
    input_type_ids = features['input_type_ids']

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError('Only PREDICT modes are supported: %s' % mode)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
      init_string = ''
      if var.name in initialized_variable_names:
        init_string = ', *INIT_FROM_CKPT*'
      tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()

    predictions = {
        'unique_id': unique_ids,
    }

    for l in layer_index:
      predictions[f'layer_output_{l}'] = all_layers[l]

    output_spec = contrib_tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    # Account for [CLS] and [SEP] with '- 2'
    if len(tokens_a) > seq_length - 2:
      tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where 'type_ids' are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the 'sentence vector'. Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append('[CLS]')
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append('[SEP]')
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info('*** Example ***')
      tf.logging.info('unique_id: %s' % example.unique_id)
      tf.logging.info(
          'tokens: %s' %
          ' '.join([tokenization.printable_text(x) for x in tokens]))
      tf.logging.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
      tf.logging.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
      tf.logging.info('input_type_ids: %s' %
                      ' '.join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


class BertModel(object):
  """Contains variables needed to contain a BERT model and get activations."""

  def __init__(self,
               model_dir,
               do_lower_case=True,
               max_seq_length=128,
               layer_index=tuple(range(10)),
               batch_size=32,
               use_tpu=False,
               tpu_master=None,
               num_tpu_cores=None,
               use_one_hot_embeddings=True):
    """All handles needed to evaluate an existing BERT model.

    Args:
      model_dir: str
      do_lower_case: bool
      max_seq_length: int
      layer_index: int
      batch_size: int
      use_tpu: bool
      tpu_master: str
      num_tpu_cores: int
      use_one_hot_embeddings: bool
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    # layer_index = [layer_index]

    # Standard naming of model files and metadata for pre-trained BERT models
    bert_config_file = model_dir + '/bert_config.json'
    vocab_file = model_dir + '/vocab.txt'
    init_checkpoint = model_dir + '/bert_model.ckpt'

    self.max_seq_length = max_seq_length
    self.layer_index = layer_index
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    run_config = contrib_tpu.RunConfig(
        master=tpu_master,
        tpu_config=contrib_tpu.TPUConfig(
            num_shards=num_tpu_cores, per_host_input_for_training=is_per_host))

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        layer_index=layer_index,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    self.estimator = contrib_tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=batch_size,
        train_batch_size=batch_size
    )  # Only needed so no error is thrown when using a TPU

  def get_activations(self, sentences):
    """Gets the activations of particular layers in BERT.

    Args:
      sentences ([str]): List of strings for which you want the activations.

    Returns:
      Tuple with two elements, with a structure as follows:

      tuple[0] (:obj:`np.ndarray`): 2D array of strings where there is a
        first-axis element for each sentence and a second-axis element for
        each wordpiece of the given sentence. If there is more than one sentence
        and they do not contain the same number of word pieces, this will
        instead
        be a 1D `np.ndarray` containing 1D lists.
      tuple[1] (:obj:`np.ndarray`): 3D array of floats where there is a
        first-axis element for each sentence, a second-axis element for
        each wordpiece of the given sentence, and a third-axis element for
        each dimension of the embedding (768 dimensions with the default
        model).
        If there is more than one sentence and they do not contain the same
        number of word pieces, this will instead be a 1D `np.ndarray`
        containing 2D lists.
    """
    assert sentences, 'You must have at least 1 string to get activations!'

    unique_id = 0
    examples = []
    for sentence in sentences:
      # Prediction is run as a single-sentence task (i.e. text_b is None)
      examples.append(
          InputExample(unique_id=unique_id, text_a=sentence, text_b=None))
      unique_id += 1

    features = convert_examples_to_features(
        examples=examples,
        seq_length=self.max_seq_length,
        tokenizer=self.tokenizer)

    unique_id_to_feature = {f.unique_id: f for f in features}

    input_fn = input_fn_builder(
        features=features, seq_length=self.max_seq_length)

    tokens = []
    layers = []
    pred_results = self.estimator.predict(input_fn, yield_single_examples=True)
    for result in pred_results:  # For each example
      example_tokens = []
      example_layers = collections.defaultdict(list)
      layer_nums = [int(x[len('layer_output_'):])
                    for x in result.keys() if 'layer_output_' in x]

      unique_id = int(result['unique_id'])
      feature = unique_id_to_feature[unique_id]
      for (i, token) in enumerate(feature.tokens):  # For each word piece
        example_tokens.append(token)
        for layer_num in layer_nums:
          layer_output = result[f'layer_output_{layer_num}'][i]
          example_layers[layer_num].append(layer_output)

      tokens.append(copy.deepcopy(example_tokens))
      layers.append(copy.deepcopy(example_layers))

    return tokens, layers


def get_medical_terms(medical_terms_txt = MEDICAL_TERMS_TXT):
  """Gets a list of medical terms.

  Args:
    medical_terms_txt: File with newline-delimited words.

  Returns:
    Set of words.
  """

  with gfile.Open(medical_terms_txt, 'r') as f:
    all_cbertscore_medical_words = set(f.read().split('\n'))
  return all_cbertscore_medical_words
