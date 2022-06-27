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

"""Converts Contrack text format into jsonfiles format for experiments."""

import json
import os
import re
from typing import List, Text

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from contrack import encoding

flags.DEFINE_string('input_file', '/tmp/input.txt', 'input file path')
flags.DEFINE_string('output_dir', '/tmp/output', 'output directory path')

FLAGS = flags.FLAGS

ENVIRONMENT = 'env'
GENDER_FLAGS = {'f': 'female', 'm': 'male', 'n': 'neuter', 'u': 'unknown'}


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


def convert(input_path, output_path):
  """Converts a file with conversations into a jsonfiles file."""

  encodings = encoding.Encodings()

  logging.info('Converting data from %s', input_path)
  input_file_name = os.path.basename(input_path)
  input_file_name = os.path.splitext(input_file_name)[0]
  entities = []
  conversations = []
  conversation = {}
  conversation_id = 0
  word_count = 0

  scenario_id = None
  with tf.io.gfile.GFile(input_path, 'r') as input_file:
    for line in input_file:
      if not line.strip() and conversation:
        conversation['entities'] = list(entities)
        conversations.append(conversation)
        entities = []
        conversation = {}
        scenario_id = None
        conversation_id += 1
        word_count = 0
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
        conversation = {'conv_id': conversation_id,
                        'scenario_id': scenario_id,
                        'turns': []}

      # Parse (enrefs)
      enref_decls = enrefs_section.split('[')
      enrefs = _parse_enrefs(encodings, entities, utterance, sender,
                             enref_decls)
      for enref in enrefs:
        enref.word_span = (enref.word_span[0] + word_count,
                           enref.word_span[1] - 1 + word_count)

      # Parse words in utterance
      words = utterance.lower().split(' ')
      logging.info(words)

      if sender != ENVIRONMENT:
        turn = {'sender': sender, 'words': words, 'enrefs': enrefs}
        word_count += len(words)
        conversation['turns'].append(turn)

  # Create output directory
  if not tf.io.gfile.exists(output_path):
    tf.io.gfile.makedirs(output_path)
  output_file_name = (
      os.path.splitext(os.path.basename(input_path))[0] + '.jsonlines')
  data_file_path = os.path.join(output_path, output_file_name)
  logging.info('Writing to %s', data_file_path)

  with tf.io.gfile.GFile(data_file_path, 'w') as output_file:
    for conversation in conversations:
      jsonline = {
          'doc_key': 'tc/' + conversation['scenario_id'],
          'sentences': [],
          'speakers': []
      }
      enrefs = []
      for turn in conversation['turns']:
        jsonline['sentences'].append(turn['words'])
        jsonline['speakers'].append([turn['sender']] * len(turn['words']))
        enrefs += turn['enrefs']

      clusters = []
      logging.info(enrefs)
      for e_id, _ in enumerate(conversation['entities']):
        cluster = []
        for e in enrefs:
          if e.enref_id.get() == e_id and e.enref_properties.is_group() <= 0:
            cluster.append(list(e.word_span))
        if cluster:
          clusters.append(cluster)
      jsonline['clusters'] = clusters

      output_file.write(json.dumps(jsonline) + '\n')


def main(argv):
  del argv
  convert(FLAGS.input_file, FLAGS.output_dir)

if __name__ == '__main__':
  app.run(main)
