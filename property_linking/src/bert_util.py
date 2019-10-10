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

"""Various bert utils for property linking project.
"""

from bert import tokenization as bert_tokenization

import tensorflow as tf
import tensorflow_hub as hub

# Some functions are copied from bert, but a direct import of
# that file is not possible due to flags and build warnings.


class BertHelper(object):
  """Construct a wrapper around BERT parameters.
  """

  def __init__(self,
               session,
               model_dir,
               max_query_length,
               batch_size,
               module):
    self.tokenizer = create_tokenizer_from_hub_module(model_dir)
    self.max_query_length = max_query_length
    self.bert_model_dir = model_dir
    self.bert_batch_size = batch_size
    self.session = session
    self.module = module


class InputExample(object):
  """A single training/test example for simple sequence classification.

  Copied from run_classifier.py
  """

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


def convert_single_example(ex_id, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`.

  Args:
    ex_id: Index of the example.
    example: InputExample to be converted.
    label_list:
    max_seq_length: Maximum sequence length to truncate or pad to.
    tokenizer: BERT tokenizer.
  Returns:
    features: BERT InputFeatures to be used by BERT module.

  Adapted from run_classifier.py. Assumes there's no token_b
  """

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  # tokens_b = None

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_id < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [bert_tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


class InputFeatures(object):
  """A single set of features of data.

  Copied from run_classifier.py
  """

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`.

  Args:
    examples: List of InputExamples
    label_list: Possible labels
    max_seq_length: Maximum sequence length to truncate or pad to.
    tokenizer: BERT tokenizer.
  Returns:
    features: BERT InputFeatures to be used by BERT module for all examples

  Copied from run_classifier.py
  """

  features = []
  for (ex_id, example) in enumerate(examples):
    if ex_id % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_id, len(examples)))

    feature = convert_single_example(ex_id, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def create_tokenizer_from_hub_module(bert_model_dir):
  """Get the vocab file and casing info from the Hub module.

  Args:
    bert_model_dir: Path to BERT model
  Returns:
    A tokenizer suitable for BERT tokenization (wordpiece).

  Adapted from run_classifier_with_tfhub.py
  """
  with tf.Graph().as_default():
    bert_module = hub.Module(bert_model_dir)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])

  return bert_tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


def get_intermediate_layer(last_layer, total_layers, desired_layer):
  """Get an intermediate layer."""
  intermediate_layer = last_layer.name.split("/")
  intermediate_layer[-1] = intermediate_layer[-1].replace(
      str(total_layers + 1),
      str(desired_layer + 1))
  intermediate_layer_name = "/".join(intermediate_layer)
  layer = tf.get_default_graph().get_tensor_by_name(intermediate_layer_name)
  if desired_layer == 0:
    layer = tf.reshape(layer, tf.shape(last_layer))
  return layer
