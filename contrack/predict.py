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

"""Applies a Contrack model on some new data."""

import collections
import json
import logging
import os
from typing import Dict, List, Text, Tuple

from absl import app
from absl import flags
import tensorflow as tf

from contrack import data
from contrack import encoding
from contrack import env
from contrack import model
from contrack import signals

flags.DEFINE_string('model_path', '',
                    'Path to directory where the model is stored.')
flags.DEFINE_bool(
    'eval', True, 'If true, compare with target label containd in the input '
    'data and output accuracy metrics.')
flags.DEFINE_string(
    'input_data_glob', '',
    'A TF glob pattern specifying the location of the evaluation data files.')
flags.DEFINE_string(
    'clusters_file', '',
    'A jsonline file to which the predicted clusters are added.')
flags.DEFINE_bool(
    'teacher_forcing', True,
    'If true, use true instead of predicted labels for repository.')

FLAGS = flags.FLAGS

PRONOUNS = [
    'i', 'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'we',
    'our', 'us', 'they', 'their', 'them', 'there', 'here', 'it'
]

METRICS = [('people', 'new_entity'), ('people', 'entities'),
           ('people', 'properties'), ('people', 'membership'),
           ('locations', 'new_entity'), ('locations', 'entities'),
           ('locations', 'properties'), ('locations', 'membership'),
           ('all', 'new_entity'), ('all', 'entities'),
           ('all', 'properties'), ('all', 'membership')]

EPSILON = 1e-10


def _get_named_slices(y_true, logits,
                      section_name):
  """Returns the slices (given by name) of true and predictied vector."""
  is_entity = y_true.enref_meta.is_enref()
  if section_name == 'new_entity':
    return (y_true.enref_meta.get_is_new_slice(),
            is_entity * logits.enref_meta.get_is_new_slice())
  elif section_name == 'entities':
    return (y_true.enref_id.slice(), is_entity * logits.enref_id.slice())
  elif section_name == 'properties':
    return (y_true.enref_properties.slice(),
            is_entity * logits.enref_properties.slice())
  elif section_name == 'membership':
    is_group = y_true.enref_properties.is_group()
    return (y_true.enref_membership.slice(),
            is_entity * is_group * logits.enref_membership.slice())
  else:
    raise ValueError('Unknown section name %s' % section_name)


def _compute_stats(x, y_pred,
                   environment):
  """Computes statistics about accuracy on enrefs in certain categories."""
  encodings = environment.encodings
  stats = {}
  for m in METRICS:
    stats[f'{m[0]}/{m[1]}/tp'] = 0
    stats[f'{m[0]}/{m[1]}/fp'] = 0
    stats[f'{m[0]}/{m[1]}/fn'] = 0
  stats['people/stats'] = [0, 0, 0]
  stats['locations/stats'] = [0, 0, 0]
  for i in range(0, 30):
    stats[f'by_turn/{i}'] = [0, 0, 0]

  other_entity_tokens = collections.defaultdict(int)

  turn_nr = 0
  prev_scenario_id = ''
  for i in range(0, y_pred.shape[0]):
    if x['scenario_id'][i].decode('utf-8') == prev_scenario_id:
      turn_nr += 1
    else:
      turn_nr = 0
      prev_scenario_id = x['scenario_id'][i].decode('utf-8')

    for j in range(0, x['token_seq_length'][i]):
      true_enc = encodings.as_prediction_encoding(x['annotation_seq'][i, j, :])
      pred_index = x['state_seq_length'][i] + j
      if pred_index >= environment.config.max_seq_len:
        continue
      pred_enc = encodings.as_prediction_encoding(y_pred[i, pred_index, :])

      if true_enc.enref_meta.is_enref() > 0.0:
        word = x['word_seq'][i, j, 0].decode('utf-8')

        if word in signals.FEMALE_NAMES:
          word = 'FFN'
        elif word in signals.MALE_NAMES:
          word = 'MFN'
        elif word not in PRONOUNS:
          other_entity_tokens[word] += 1
          word = 'OTHER'

        if word not in stats:
          stats[word] = [0, 0, 0]

        stats[word][0] += 1
        stats[true_enc.enref_properties.get_domain() + '/stats'][0] += 1
        stats[f'by_turn/{turn_nr}'][0] += 1
        if pred_enc.enref_meta.is_enref() > 0.0:
          stats[word][1] += 1
          stats[true_enc.enref_properties.get_domain() + '/stats'][1] += 1
          stats[f'by_turn/{turn_nr}'][1] += 1
        if pred_enc.enref_id.get() == true_enc.enref_id.get():
          stats[word][2] += 1
          stats[true_enc.enref_properties.get_domain() + '/stats'][2] += 1
          stats[f'by_turn/{turn_nr}'][2] += 1

      for m in METRICS:
        if (m[0] != 'all' and
            m[0] != true_enc.enref_properties.get_domain()):
          continue
        true_y, logits = _get_named_slices(true_enc, pred_enc, m[1])
        pred_y = tf.cast(logits > 0.0, tf.float32)

        stats[f'{m[0]}/{m[1]}/tp'] += tf.reduce_sum(true_y * pred_y).numpy()
        stats[f'{m[0]}/{m[1]}/fp'] += tf.reduce_sum(
            (1.0 - true_y) * pred_y).numpy()
        stats[f'{m[0]}/{m[1]}/fn'] += tf.reduce_sum(
            true_y * (1.0 - pred_y)).numpy()

  for m in METRICS:
    stats[f'{m[0]}/{m[1]}/pr'] = round(
        stats[f'{m[0]}/{m[1]}/tp'] /
        (stats[f'{m[0]}/{m[1]}/tp'] + stats[f'{m[0]}/{m[1]}/fp'] + EPSILON), 3)
    stats[f'{m[0]}/{m[1]}/re'] = round(
        stats[f'{m[0]}/{m[1]}/tp'] /
        (stats[f'{m[0]}/{m[1]}/tp'] + stats[f'{m[0]}/{m[1]}/fn'] + EPSILON), 3)
    stats[f'{m[0]}/{m[1]}/f1'] = round(
        2.0 * (stats[f'{m[0]}/{m[1]}/pr'] * stats[f'{m[0]}/{m[1]}/re']) /
        (stats[f'{m[0]}/{m[1]}/pr'] + stats[f'{m[0]}/{m[1]}/re'] + EPSILON), 3)

  return stats, other_entity_tokens


def find_cluster(stats, m1, m2):
  """Checks if m1 and m2 are in true or predited cluster."""
  in_true_cluster = False
  in_pred_cluster = False
  for cluster in stats['true_clusters'].values():
    if m1 in cluster and m2 in cluster:
      in_true_cluster = True
      break
  for cluster in stats['pred_clusters'].values():
    if m1 in cluster and m2 in cluster:
      in_pred_cluster = True
      break
  return in_true_cluster, in_pred_cluster


def _compute_entity_tracking_stats(x, y_pred,
                                   environment):
  """Computes statistics about accuracy on enrefs in certain categories."""
  encodings = environment.encodings
  el_stats = {}
  for category in ['singular', 'plural', 'both']:
    el_stats.update({
        f'{category}_true': 0,
        f'{category}_pred': 0,
        f'{category}_correct': 0,
    })

  scene_stats = {
      'id': '',
      'm_id': 0,
      'true_clusters': collections.defaultdict(set),
      'pred_clusters': collections.defaultdict(set)
  }

  blanc_stats = [[0, 0], [0, 0]]

  for i in range(0, y_pred.shape[0]):
    for j in range(0, x['token_seq_length'][i]):
      true_enc = encodings.as_prediction_encoding(x['annotation_seq'][i, j, :])
      pred_index = x['state_seq_length'][i] + j
      if pred_index >= environment.config.max_seq_len:
        continue
      pred_enc = encodings.as_prediction_encoding(y_pred[i, pred_index, :])

      # Collect stats for Entity Linking F1 score
      true_entities = []
      if true_enc.enref_meta.is_enref() > 0.0:
        if true_enc.enref_properties.is_group() > 0.0:
          true_entities = true_enc.enref_membership.get_ids()
          el_stats['plural_true'] += len(true_entities)
          el_stats['both_true'] += len(true_entities)
        else:
          true_entities = [true_enc.enref_id.get()]
          el_stats['singular_true'] += 1
          el_stats['both_true'] += 1

      pred_entities = []
      if pred_enc.enref_meta.is_enref() > 0.0:
        if pred_enc.enref_properties.is_group() > 0.0:
          pred_entities = pred_enc.enref_membership.get_ids()
          el_stats['plural_pred'] += len(pred_entities)
          el_stats['both_pred'] += len(pred_entities)
        else:
          pred_entities = [pred_enc.enref_id.get()]
          el_stats['singular_pred'] += 1
          el_stats['both_pred'] += 1

      for entity in true_entities:
        if entity in pred_entities:
          el_stats['both_correct'] += 1
          if true_enc.enref_properties.is_group() > 0.0:
            el_stats['plural_correct'] += 1
          else:
            el_stats['singular_correct'] += 1

      # Collect stats for BLANC
      scene_id = x['scenario_id'][i]
      if not scene_stats['id']:
        scene_stats['id'] = scene_id
      m_id = scene_stats['m_id']
      if scene_id != scene_stats['id']:
        for m1 in range(0, m_id):
          for m2 in range(0, m1):
            in_true_cluster, in_pred_cluster = find_cluster(scene_stats, m1, m2)
            blanc_stats[1 - int(in_true_cluster)][1 - int(in_pred_cluster)] += 1
        scene_stats = {
            'id': scene_id,
            'm_id': 0,
            'true_clusters': collections.defaultdict(set),
            'pred_clusters': collections.defaultdict(set)
        }

      if true_enc.enref_meta.is_enref() > 0.0:
        scene_stats['m_id'] += 1
        if true_enc.enref_properties.is_group() > 0.0:
          for e_id in true_enc.enref_membership.get_ids():
            scene_stats['true_clusters'][e_id].add(m_id)
        else:
          scene_stats['true_clusters'][true_enc.enref_id.get()].add(m_id)

        if pred_enc.enref_meta.is_enref() > 0.0:
          if pred_enc.enref_properties.is_group() > 0.0:
            for e_id in pred_enc.enref_membership.get_ids():
              scene_stats['pred_clusters'][e_id].add(m_id)
          else:
            scene_stats['pred_clusters'][pred_enc.enref_id.get()].add(m_id)

  el_results = {}
  for c in ['singular', 'plural', 'both']:
    el_results.update({
        f'{c}_precision': el_stats[f'{c}_correct'] / el_stats[f'{c}_pred'],
        f'{c}_recall': el_stats[f'{c}_correct'] / el_stats[f'{c}_true'],
    })
    el_results[f'{c}_f1'] = (
        2.0 * (el_results[f'{c}_precision'] * el_results[f'{c}_recall']) /
        (el_results[f'{c}_precision'] + el_results[f'{c}_recall']))

  b = blanc_stats
  logging.info('B: %s', b)
  blanc_results = {
      'Pc': b[0][0] / (b[0][0] + b[1][0]),
      'Rc': b[0][0] / (b[0][0] + b[0][1]),
      'Pn': b[1][1] / (b[1][1] + b[0][1]),
      'Rn': b[1][1] / (b[1][1] + b[1][0]),
  }
  blanc_results['F1c'] = (2.0 * (blanc_results['Pc'] * blanc_results['Rc']) /
                          (blanc_results['Pc'] + blanc_results['Rc']))
  blanc_results['F1n'] = (2.0 * (blanc_results['Pn'] * blanc_results['Rn']) /
                          (blanc_results['Pn'] + blanc_results['Rn']))
  blanc_results.update({
      'P': (blanc_results['Pc'] + blanc_results['Pn']) / 2.0,
      'R': (blanc_results['Rc'] + blanc_results['Rn']) / 2.0,
      'F1': (blanc_results['F1c'] + blanc_results['F1n']) / 2.0
  })

  return el_results, blanc_results


def _save_clusters(x, y_pred,
                   environment, file_name):
  """Saves clusters to jsonlines file."""
  encodings = environment.encodings
  examples = {}
  with tf.io.gfile.GFile(file_name, 'r') as input_file:
    for line in input_file:
      example = json.loads(line)
      examples[example['doc_key']] = example

  prev_id = x['scenario_id'][0].decode('utf-8')
  clusters = {}
  num_tokens = 0
  prev_enref = None
  for i in range(0, y_pred.shape[0]):
    s_id = x['scenario_id'][i].decode('utf-8')
    if s_id != prev_id:
      examples['tc/' + prev_id]['predicted_clusters'] = list(clusters.values())
      prev_id = s_id
      clusters = {}
      num_tokens = 0
      prev_enref = None

    for j in range(0, x['token_seq_length'][i]):
      token = x['word_seq'][i, j, 0].decode('utf-8')
      if token.startswith('['):
        continue

      pred_index = x['state_seq_length'][i] + j
      if pred_index >= environment.config.max_seq_len:
        continue
      pred_enc = encodings.as_prediction_encoding(y_pred[i, pred_index, :])

      pred_id = -1
      if (pred_enc.enref_meta.is_enref() > 0 and
          pred_enc.enref_properties.is_group() <= 0):
        pred_id = pred_enc.enref_id.get()

      if prev_enref and (pred_id == -1 or pred_id != prev_enref[0]):
        cl_id = prev_enref[0]
        if cl_id not in clusters:
          clusters[cl_id] = []
        logging.info('Adding enref %s to cluster %s', prev_enref[2],
                     clusters[cl_id])
        clusters[cl_id].append((prev_enref[1], num_tokens - 1))
        prev_enref = None
      if pred_id >= 0:
        logging.info('Starting new enref: %s (%d)', token, num_tokens)
        prev_enref = (pred_id, num_tokens, token)

      if not token.startswith('##'):
        logging.info('%s: %d', token, num_tokens)
        num_tokens += 1

  examples['tc/' + prev_id]['predicted_clusters'] = list(clusters.values())

  teacher_forcing = 'withtf' if FLAGS.teacher_forcing else 'withouttf'
  output_file_name = (os.path.splitext(file_name)[0] + '_predicted' +
                      teacher_forcing + '.jsonlines')
  logging.info('Writing to %s', output_file_name)
  with tf.io.gfile.GFile(output_file_name, 'w') as output_file:
    for e in examples.values():
      output_file.write(json.dumps(e) + '\n')


def main(argv):
  del argv  # Unused.

  env.Env.init_from_saved_model(FLAGS.model_path)
  environment = env.Env.get()
  if not FLAGS.teacher_forcing:
    environment.config.batch_size = 1
  logging.info('Inference with config:\n%s', environment.config)

  logging.info('Reading data from %s', FLAGS.input_data_glob)
  input_data = data.read_eval_data(FLAGS.input_data_glob, environment.config,
                                   environment.encodings)

  with tf.keras.utils.custom_object_scope(model.get_custom_objects()):
    contrack_model = tf.keras.models.load_model(FLAGS.model_path)

  contrack_model.print_predictions = True
  if not FLAGS.teacher_forcing:
    contrack_model.compile(run_eagerly=True)
    contrack_model.disable_teacher_forcing()

  if FLAGS.eval:
    contrack_model.evaluate(
        input_data, batch_size=environment.config.batch_size)
  else:
    x, y_pred = contrack_model.predict(
        input_data, batch_size=environment.config.batch_size, verbose=1)

    stats, other_entities = _compute_stats(x, y_pred, environment)
    logging.info('Accuracy Stats:')
    for k, v in stats.items():
      logging.info('%s: %s', k, v)
    logging.info('Other entities: ')
    for word, count in sorted(other_entities.items(), key=lambda w: -w[1]):
      if count < 5:
        break
      logging.info('%s: %d', word, count)

    el_stats, blanc_stats = _compute_entity_tracking_stats(x, y_pred,
                                                           environment)
    logging.info('entity linking results: %s', str(el_stats))
    logging.info('BLANC results: %s', str(blanc_stats))

    if FLAGS.clusters_file:
      _save_clusters(x, y_pred, environment, FLAGS.clusters_file)


if __name__ == '__main__':
  app.run(main)
