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

"""Converts character mining json data to Contrack text format."""

import json
import os
import re
from typing import List, Sequence

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

flags.DEFINE_string(
    'input_dir', '',
    'Path to the directory containing the input data json files.')

flags.DEFINE_string(
    'output_dir', '',
    'Path to the directory where the conversations are written.')

FLAGS = flags.FLAGS

KNOWN_CHARACTERS = frozenset([
    'Ross Geller',
    'Rachel Green',
    'Chandler Bing',
    'Monica Geller',
    'Joey Tribbiani',
    'Phoebe Buffay',
    'Emily Waltham',
    'Richard Burke',
    'Carol Willick',
    'Ben Geller',
    'Peter Becker',
    'Judy Geller',
    'Barry Farber',
    'Jack Geller',
    'Kate Miller'
])

ENTITY_GROUPS = {'#GENERAL#': 'general', '#OTHER#': 'other'}

GENDER = {
    'ross': 'm',
    'rachel': 'f',
    'chandler': 'm',
    'monica': 'f',
    'joey': 'm',
    'phoebe': 'f',
    'emily': 'f',
    'richard': 'm',
    'carol': 'f',
    'ben': 'm',
    'peter': 'm',
    'judy': 'f',
    'barry': 'm',
    'jack': 'm',
    'kate': 'f',
    'general': 'u',
    'other': 'u'
}


class Turn(object):
  """Represents a turn in a Contrack conversation."""

  def __init__(self, speaker, msg, enrefs):
    self.speaker = speaker
    self.msg = msg
    self.enrefs = enrefs


class Conversation(object):
  """Represents a Contrack conversation."""

  def __init__(self, conversation_id):
    self.conversation_id = conversation_id
    self.turns = []

  def append(self, turn):
    self.turns.append(turn)


all_speakers = set()


def str_to_id(name):
  if name in KNOWN_CHARACTERS:
    return name.split(' ')[0].lower()
  elif name in ENTITY_GROUPS:
    return ENTITY_GROUPS[name]
  else:
    return 'other'


def build_env_turn(conversation_id):
  """Creates a first turn describing the environment."""
  entities = [n.split(' ')[0].lower() for n in KNOWN_CHARACTERS]
  entities += ENTITY_GROUPS.values()

  enrefs_decls = []
  for i, name in enumerate(entities):
    enrefs_decls.append(f'[{name} {GENDER[name]} {i}-{i}]')
  utterance_str = ' '.join(entities)
  enrefs_str = ''.join(enrefs_decls)

  turn = f'conv:{conversation_id}| {utterance_str}| {enrefs_str}\n'
  return turn


def read_season(num):
  """Reads a Season from file and returns it as a list of conversations."""
  global all_speakers
  filepath = os.path.join(FLAGS.input_dir, f'friends_season_0{num}.json')

  with tf.io.gfile.GFile(filepath, 'r') as json_file:
    season_json = json.load(json_file)

  conversations = []
  for episode_json in season_json['episodes']:
    for scene_json in episode_json['scenes']:
      conversation = Conversation(scene_json['scene_id'])
      participants = []

      for utterance_json in scene_json['utterances']:
        utterance_id = utterance_json['utterance_id']

        # Extract speaker
        speakers = utterance_json['speakers']
        all_speakers.update(speakers)
        if not speakers:
          logging.info('no speaker for turn %s', utterance_id)
          continue
        if len(speakers) > 1:
          logging.info('multiple speakers for turn %s', utterance_id)
        speaker = str_to_id(speakers[0])
        if speaker not in participants:
          participants.append(speaker)

        for sentence_index, tokens in enumerate(utterance_json['tokens']):
          # Extract tokens
          if not tokens or tokens[0] == '_':
            logging.info('empty turn %s', utterance_id)
            continue

          # Extract enrefs
          entities_json = utterance_json['character_entities'][sentence_index]
          enrefs = []
          for entity_json in entities_json:
            if len(entity_json) < 3:
              logging.info('%s: cannot parse entity: %s', utterance_id,
                           entity_json)
              continue
            # logging.info(entity_json)
            start_index = entity_json[0]
            end_index = entity_json[1] - 1
            entities = set([str_to_id(e) for e in entity_json[2:]])
            if not entities:
              logging.info('empty entities list in %s', utterance_id)
              continue
            if len(entities) == 1:
              entity_name = next(iter(entities))
              gender = GENDER[entity_name]
              enref = f'[{entity_name} {gender} {start_index}-{end_index}]'
            else:
              enref_name = tokens[start_index]
              member_decl = ':'.join(entities)
              enref = f'[{enref_name} g({member_decl}) {start_index}-{end_index}]'
            enrefs.append(enref)

          conversation.append(Turn(speaker, ' '.join(tokens), enrefs))
      conversations.append(conversation)
  return conversations


def convert_data():
  """Converts the data from all seasons."""
  for season_num in [1, 2, 3, 4]:
    conversations = read_season(season_num)
    trg_path = os.path.join(FLAGS.output_dir,
                            f'char_ident_trg.txt-0000{season_num - 1}-of-00004')
    tst_path = os.path.join(FLAGS.output_dir,
                            f'char_ident_tst.txt-0000{season_num - 1}-of-00004')

    with tf.io.gfile.GFile(trg_path, 'w') as trg_file:
      with tf.io.gfile.GFile(tst_path, 'w') as tst_file:
        for conversation in conversations:
          c_id = conversation.conversation_id
          episode_nr = int(re.search(r's.._e(..)_c..', c_id).group(1))
          file = trg_file if episode_nr <= 21 else tst_file

          file.write(build_env_turn(conversation.conversation_id))
          for turn in conversation.turns:
            line = turn.speaker + '| '
            line += turn.msg + '| '
            line += ''.join(turn.enrefs)
            line += '\n'
            file.write(line)
          file.write('\n')

  global all_speakers
  for s in all_speakers:
    logging.info(s)


def main(argv):
  del argv
  convert_data()

if __name__ == '__main__':
  app.run(main)
