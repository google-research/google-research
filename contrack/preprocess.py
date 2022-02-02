# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Convert Context Tracking text data into TF Example protos for model training."""

import os
import re
from typing import List, Text

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from contrack import bert_client as bert_client_module
from contrack import encoding
from contrack import word2vec_client as word2vec_client_module
from contrack import signals as signals_module

# Flags
flags.DEFINE_string('input_file', '/tmp/input.txt', 'input file path')
flags.DEFINE_string('output_dir', '/tmp/output', 'output directory path')
flags.DEFINE_string('wordvec_path', '/tmp/GoogleNews-vectors-negative300.bin',
                    'Path to word2vec embedding file.')
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_string(
    'tokenizer_handle',
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'The TFHub handle of the BERT preprocessing model used '
    'for tokenization.')
flags.DEFINE_string(
    'bert_handle',
    'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'The TFHub handle of the BERT model used for preprocessing.')

FLAGS = flags.FLAGS

# String Constants
ENVIRONMENT = 'env'  # Used for introducing enrefs without message or sender.

GENDER_FLAGS = {'f': 'female', 'm': 'male', 'n': 'neuter', 'u': 'unknown'}


class Message(object):
  """Represents one message in a conversation."""

  def __init__(self, msg_id, sender, words,
               signals, entities,
               enrefs):
    self.msg_id = msg_id
    self.sender = sender
    self.words = list(words)
    self.signals = list(signals)
    self.entities = entities
    self.enrefs = enrefs
    self.wordvecs = None
    self.tokens = None
    self.token_ids = None
    self.token_wordvecs = None
    self.token_signals = None
    self.token_bertvecs = None

  def tokenize(self, tokenizer,
               empty_wordvec):
    """Tokenize message text and re-assign wordvecs and signals ."""
    max_length = FLAGS.max_seq_length - 2  # Account for [CLS] and [SEP].
    utterance = ' '.join(self.words[-max_length:])

    tokens, token_ids = tokenizer.tokenize(utterance)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = [101] + token_ids + [102]

    word_to_token = {0: 1}  # Skip first token which is CLS]
    word_index = 0
    for i in range(1, len(tokens)):  # Also skip [CLS]
      token = tokens[i]
      if token.startswith('['):
        continue
      if token.startswith('##'):
        token = token[2:]
      if (word_index < len(self.words) and
          self.words[word_index].startswith(token)):
        word_to_token[word_index] = i
        word_index += 1
    if word_index != len(self.words):
      logging.info('word_index: %d len(words):%d', word_index, len(self.words))
      logging.info(str(word_to_token))
      logging.fatal('Cannot align words %s with tokens %s', str(self.words),
                    str(tokens))
    word_to_token[len(self.words)] = len(tokens) - 1  # skip [SEP]

    # Convert indices into words to indices into tokens
    enrefs = self.enrefs
    for enref in enrefs:
      (word_start, word_end) = enref.word_span
      enref.token_span = (word_to_token[word_start], word_to_token[word_end])

    # Computer per-token wordvecs and signals
    token_wordvecs = []
    token_signals = []
    prev_wordvec = empty_wordvec
    prev_signals = []
    for word_index, token_index in word_to_token.items():
      while len(token_wordvecs) < token_index:
        token_wordvecs += [prev_wordvec]
        token_signals += [prev_signals]
      if word_index < len(self.wordvecs):
        prev_wordvec = self.wordvecs[word_index]
        prev_signals = self.signals[word_index]
    token_wordvecs += [empty_wordvec]
    token_signals += [[]]

    self.tokens = tokens
    self.token_ids = token_ids
    self.token_wordvecs = token_wordvecs
    self.token_signals = token_signals


class Conversation(object):
  """Represents a conversation to be preprocessed."""

  def __init__(self, conversation_id, scenario_id):
    self.conversation_id = str(conversation_id)
    self.scenario_id = scenario_id
    self.messages = []

  def add_message(self, sender, words, signals,
                  entities,
                  enrefs):
    msg_id = self.conversation_id + ':' + str(len(self.messages))
    msg = Message(msg_id, sender, words, signals, entities, enrefs)
    self.messages.append(msg)

  def add_wordvecs(self,
                   word2vec_client):
    for message in self.messages:
      vecs = word2vec_client.lookup(message.words)
      message.wordvecs = vecs

  def tokenize(self, tokenizer,
               empty_wordvec):
    for message in self.messages:
      message.tokenize(tokenizer, empty_wordvec)


def _build_seq_examples(
    conversation,
    encodings):
  """Builds SequenceExample protos from the conversations."""
  seq_examples = []
  enrefs = []
  participants = conversation.messages[0].tokens[1:-1]
  for example_index, message in enumerate(conversation.messages):
    sender = message.sender
    tokens = message.tokens
    msg_enrefs = message.enrefs
    wordvecs = message.token_wordvecs
    bertvecs = message.token_bertvecs

    seq_example = tf.train.SequenceExample()
    seq_example.context.feature['state_seq_length'].int64_list.value.append(
        len(enrefs))
    seq_example.context.feature['token_seq_length'].int64_list.value.append(
        len(tokens))
    seq_example.context.feature['sender'].bytes_list.value.append(
        message.sender.encode())
    seq_example.context.feature['scenario_id'].bytes_list.value.append(
        conversation.scenario_id.encode())
    for p in participants:
      seq_example.context.feature['participants'].bytes_list.value.append(
          p.encode())

    state_seq = seq_example.feature_lists.feature_list['state_seq']
    token_seq = seq_example.feature_lists.feature_list['token_seq']
    word_seq = seq_example.feature_lists.feature_list['word_seq']
    annotation_seq = seq_example.feature_lists.feature_list['annotation_seq']

    # Add enref sequence
    for enref in enrefs:
      entity_name = enref.entity_name
      enref.enref_context.set_is_sender(entity_name == sender)
      enref.enref_context.set_is_recipient(entity_name != sender and
                                           entity_name in participants)
      enref.enref_context.set_message_offset(
          enref.enref_context.get_message_offset() + 1)

      state_seq.feature.add().float_list.value.extend(np.array(enref.array))

    # Store enref vectors in predictions
    predictions = [encodings.new_prediction_array() for _ in tokens]

    for enref in msg_enrefs:
      start, end = enref.token_span
      enref.wordvec.set(np.mean(wordvecs[start:end], 0))
      enref.bert.set(np.mean(bertvecs[start:end], 0))

      enrefs.append(enref)
      for index in range(start, end):
        prediction_enc = encodings.as_prediction_encoding(predictions[index])
        prediction_enc.enref_meta.replace(enref.enref_meta.slice())
        if enref.enref_meta.is_new() > 0.0 and index != start:
          prediction_enc.enref_meta.set_is_new(False)
          prediction_enc.enref_meta.set_is_new_continued(True)
        prediction_enc.enref_id.replace(enref.enref_id.slice())
        prediction_enc.enref_properties.replace(enref.enref_properties.slice())
        prediction_enc.enref_membership.replace(enref.enref_membership.slice())

    # Add tokens and predictions
    for i, token in enumerate(tokens):
      token_enc = encodings.new_token_encoding(token, message.token_signals[i],
                                               message.token_wordvecs[i],
                                               message.token_bertvecs[i])
      token_seq.feature.add().float_list.value.extend(token_enc.array)
      word_seq.feature.add().bytes_list.value.append(token.encode())

      annotation_seq.feature.add().float_list.value.extend(predictions[i])

    if example_index > 0:
      seq_examples.append(seq_example)

  return seq_examples


def _parse_enrefs(encodings, entities,
                  utterance, sender,
                  declarations):
  """Parses the enref declarations."""
  enrefs = []
  participants = entities[:2]
  for decl in declarations:
    if not decl:
      continue

    is_new = False
    if decl[-1] != ']':
      raise Exception('Missing bracket in enref declaration %s' % decl)
    decl = decl[:-1]
    elements = decl.split(' ')
    if len(elements) != 3:
      raise Exception('Invalid enref declaration %s' % decl)
    entity_name = elements[0]

    domain = 'people'
    if entity_name.startswith('person:') or entity_name.startswith('p:'):
      domain = 'people'
      entity_name = re.sub(r'^.*?:', '', entity_name)
    if entity_name.startswith('location:') or entity_name.startswith('l:'):
      domain = 'locations'
      entity_name = re.sub(r'^.*?:', '', entity_name)

    if entity_name not in entities:
      entities.append(entity_name)
      is_new = True

    span = [int(k.strip()) for k in elements[2].split('-')]
    if len(span) != 2:
      raise Exception('Invalid span in enref declaration %s' % decl)
    span_words = utterance.split(' ')[span[0]:(span[1] + 1)]
    span_text = ' '.join(span_words)

    enref = encodings.new_enref_encoding()
    enref.populate(entity_name, (span[0], span[1] + 1), span_text)

    enref.enref_meta.set_is_enref(True)
    enref.enref_meta.set_is_new(is_new)
    enref.enref_meta.set_is_new_continued(False)
    enref.enref_id.set(entities.index(entity_name))

    enref.enref_properties.set_domain(domain)

    if elements[1].startswith('g'):
      members_decl = re.search(r'\((.*?)\)', elements[1])
      if members_decl is None:
        raise Exception('Cannot parse group declaration: %s' % elements[1])
      members = members_decl.group(1).split(':')
      if members == ['']:
        members = []
      member_ids = [entities.index(m) for m in members]

      enref.enref_properties.set_is_group(True)
      enref.enref_membership.set(member_ids, members)
    else:
      enref.enref_properties.set_is_group(False)
      if domain == 'people':
        gender = GENDER_FLAGS[elements[1][0]]
        enref.enref_properties.set_gender(gender)

    is_sender = entity_name == sender
    is_recipient = not is_sender and entity_name in participants
    enref.enref_context.set_is_sender(is_sender)
    enref.enref_context.set_is_recipient(is_recipient)
    enref.enref_context.set_message_offset(0)

    enref.signals.set([])

    logging.info('enref: %s', str(enref))
    enrefs.append(enref)
  return enrefs


def _add_bert_vecs(conversations,
                   bert_client):
  """Adds BERT embeddings to conversations."""
  msgs = {}
  # BERT embeddings are computed one batch at a time so it's inefficient to
  # add them for each messages individually. Instead we collect all messages in
  # one large dict and then run BERT on that dict and finally copy the
  # embeddings back to the messages.
  for conversation in conversations:
    msgs.update({m.msg_id: m.token_ids for m in conversation.messages})

  embeddings = bert_client.lookup(msgs)

  for conversation in conversations:
    for message in conversation.messages:
      message.token_bertvecs = embeddings[message.msg_id]


def convert(input_path, output_path):
  """Converts a file with conversations into a TF Records file."""
  logging.info('Loading Word2Vec embeddings from %s', FLAGS.wordvec_path)
  wordvec_client = word2vec_client_module.Word2VecClient(FLAGS.wordvec_path)

  logging.info('Loading tokenizer from %s', FLAGS.tokenizer_handle)
  tokenizer = bert_client_module.Tokenizer(FLAGS.tokenizer_handle)

  logging.info('Loading BERT embeddings from %s', FLAGS.bert_handle)
  bert_client = bert_client_module.BertClient(FLAGS.bert_handle)

  encodings = encoding.Encodings()

  logging.info('Converting data from %s', input_path)
  input_file_name = os.path.basename(input_path)
  input_file_name = os.path.splitext(input_file_name)[0]
  entities = []
  conversations = []
  conversation = None
  conversation_id = 0

  scenario_id = None
  with tf.io.gfile.GFile(input_path, 'r') as input_file:
    for line in input_file:
      if not line.strip() and conversation:
        entities = []
        conversations.append(conversation)
        conversation = None
        scenario_id = None
        conversation_id += 1
        continue

      logging.info('read line %s', line)

      # Extract line sections
      sections = line.strip().split('|')
      sender = sections[0].strip()
      utterance = sections[1].strip()
      enrefs_section = sections[2].strip()

      if sender.startswith('conv:'):
        scenario_id = sender[5:]
        sender = ENVIRONMENT
        conversation = Conversation(conversation_id, scenario_id)

      # Parse (enrefs)
      enref_decls = enrefs_section.split('[')
      enrefs = _parse_enrefs(encodings, entities, utterance, sender,
                             enref_decls)

      # Parse words in utterance
      words = utterance.lower().split(' ')
      logging.info(words)

      # Collect signals
      signals = signals_module.collect_signals(words)

      conversation.add_message(sender, words, signals, entities, enrefs)
      conversation.add_wordvecs(wordvec_client)
      conversation.tokenize(tokenizer, wordvec_client.empty_vec)

  _add_bert_vecs(conversations, bert_client)

  # Create output directory
  if not tf.io.gfile.exists(output_path):
    tf.io.gfile.makedirs(output_path)
  output_file_name = (
      os.path.splitext(os.path.basename(input_path))[0] + '.tfrecord')
  data_file_path = os.path.join(output_path, output_file_name)
  logging.info('Writing to %s', data_file_path)

  # Write sequence examples
  with tf.io.TFRecordWriter(data_file_path) as output_file:
    for conversation in conversations:
      seq_examples = _build_seq_examples(conversation, encodings)
      for seq_example in seq_examples:
        output_file.write(seq_example.SerializeToString())


def main(argv):
  del argv
  convert(FLAGS.input_file, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
