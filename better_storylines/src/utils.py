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

r"""Utilities for training and evaling next sentence prediction on ROC Stories.
"""

import csv
import os

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

gfile = tf.io.gfile


def train_step(model, optimizer, x, labels, metrics):
  """Performs a single train step on a batch of data."""
  with tf.GradientTape() as tape:
    scores, embedding = model(x)

    main_loss = model.compute_loss(labels, scores)

    if model.small_context_loss_weight > 0.0:
      # For each index in the batch:
      # Get the embeddings for the 4 context sentences. [4, 768]
      # Dot product them with the predicted embedding. [4, 1]
      # Get the score of the ground truth from the total list of scores [1]
      # Concatenate the baove two together [5, 1]
      # Rebatch things to end up with [batch_size, 5]

      def slice_in_context_fn(inputs):
        context_sentences, label_for_ex, scores_for_ex, emb_for_ex = inputs
        context_scores = tf.matmul(
            tf.expand_dims(emb_for_ex, 0), context_sentences, transpose_b=True)
        groundtruth_score = tf.reshape(scores_for_ex[label_for_ex], [1, 1])
        subscores = tf.concat([groundtruth_score, context_scores], -1)
        return tf.squeeze(subscores)

      context_scores = tf.map_fn(
          slice_in_context_fn, [x, labels, scores, embedding], dtype=tf.float32)
      context_labels = tf.zeros_like(labels)

      context_loss = model.compute_loss(context_labels, context_scores)
      context_loss = context_loss * model.small_context_loss_weight
      total_loss = main_loss + context_loss

      metrics['main_loss'](main_loss)
      metrics['small_context_loss'](context_loss)
    else:
      total_loss = main_loss

    metrics['train_loss'](total_loss)
    metrics['train_acc'](labels, scores)

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss, scores


def eval_nolabel_step(model, x, embedding_matrix):
  """Evaluates a single unlabeled example."""
  _, output_embedding = model(x, training=False)
  scores = tf.matmul(output_embedding, embedding_matrix, transpose_b=True)
  predictions = tf.argmax(scores, -1)
  return predictions


def eval_step(model, x, fifth_embedding_1, fifth_embedding_2, label):
  """Evaluates a single example from the validation set."""
  assert x.shape[0] == 1, 'Only supports batch_size=1 for now'

  _, output_embedding = model(x, training=False)
  sim_1 = tf.matmul(output_embedding, fifth_embedding_1, transpose_b=True)
  sim_2 = tf.matmul(output_embedding, fifth_embedding_2, transpose_b=True)

  correct_1 = tf.squeeze(tf.logical_and(
      tf.greater(sim_1, sim_2), tf.equal(label, 0)))
  correct_2 = tf.squeeze(tf.logical_and(
      tf.greater(sim_2, sim_1), tf.equal(label, 1)))
  return tf.logical_or(correct_1, correct_2)


def collect_all_embeddings(tfds_dataset):
  """Returns a giant matrix with all the sentence embeddings."""
  bert_embedding_size = tf.compat.v1.data.get_output_shapes(
      tfds_dataset)['embeddings'][1]
  embedding_matrix = tf.zeros([0, bert_embedding_size])
  all_ids = []

  for idx, example in enumerate(
      tf.compat.v1.data.make_one_shot_iterator(tfds_dataset)):
    if idx % 1000 == 0 and idx > 0:
      logging.info('emb_collect  %d', idx)

    story_id = example['story_id'].numpy()
    num_sentences_per_story = example['embeddings'].shape[0]
    all_ids.extend(zip([story_id] * 5,
                       range(num_sentences_per_story)))

    embeddings = example['embeddings']  # shape is (5 x bert_embedding_size)
    embedding_matrix = tf.concat([embedding_matrix, embeddings], axis=0)

  return embedding_matrix, all_ids


def build_train_style_dataset(tfds_dataset,
                              batch_size,
                              shuffle_input_sentences,
                              return_ids=False,
                              num_examples=-1,
                              is_training=True):
  """Processes a dataset that is in the training data format.

  That is, the TFExamples in the dataset contain embeddings for give sentences,
  without a 6th false Sentence.

  Args:
    tfds_dataset: A TFDS dataset containing embeddings for five sentences.
    batch_size: Batch size for dataset.
    shuffle_input_sentences: If True, shuffle the embeddings for the input
      sentences. This can be seen as a form of data augmentation, but if it does
      well would suggest that the sentence order actually does not matter,
    return_ids: If return_ids, a list of story_ids corresponding to
      each row in the embedding matrix is also returned.
    num_examples: If >0, truncate the dataset
      (including the embedding matrix) to this many examples.
    is_training: If True, then shuffle examples in the dataset. If False then
      don't.
  Returns:
    Dataset containing inputs and targets (as IDs into embedding matrix), and
    embedding matrix containing target embeddings.
  """

  all_x = []
  all_y = []
  all_ids = []  # Contains the story_id for each example.

  bert_embedding_size = tf.compat.v1.data.get_output_shapes(
      tfds_dataset)['embeddings'][1]
  all_target_embeddings = []

  # Read in the data into inputs and targets and create a new dataset from it.
  logging.info('Creating examples.')

  for idx, example in enumerate(
      tf.compat.v1.data.make_one_shot_iterator(tfds_dataset)):
    if idx % 1000 == 0 and idx > 0:
      logging.info('train  %d', idx)
    embeddings = example['embeddings']  # shape is (5 x bert_embedding_size)

    # The input is the concatenated embeddings of the first 4 sentences.
    x = embeddings[0:4, :]
    all_x.append(x)

    # The target is the index of this example's 5th sentence in the embedding
    # matrix.
    all_y.append(idx)

    all_ids.append(example['story_id'])

    # Concatenate all 5th sentence (target) embeddings together.
    target_embedding = tf.reshape(embeddings[4], [1, bert_embedding_size])
    all_target_embeddings.append(target_embedding)

  embedding_matrix = tf.concat(all_target_embeddings, axis=0)
  if return_ids:
    dataset = tf.data.Dataset.from_tensor_slices((all_x, all_y, all_ids))
  else:
    dataset = tf.data.Dataset.from_tensor_slices((all_x, all_y))
  if shuffle_input_sentences:
    # Assumes x is shaped [num_sentences x embedding_size], so random.shuffle
    # shuffles the sentence dimension.
    dataset = dataset.map(lambda x, y: (tf.random.shuffle(x), y))

  if num_examples > 0:
    dataset = dataset.take(num_examples)
    embedding_matrix = embedding_matrix[:num_examples, :]

  if is_training:
    dataset = dataset.shuffle(10000)
  dataset = dataset.batch(batch_size)

  if return_ids:
    return dataset, embedding_matrix, all_ids
  else:
    return dataset, embedding_matrix


def build_validation_dataset(dataset, return_embedding_matrices=False):
  """Processes TFDS dataset into inputs and two candidate targets."""

  all_x = []
  all_5th_embs_1 = []
  all_5th_embs_2 = []
  all_labels = []

  # Read in the data into inputs and targets and create a new dataset from it.
  logging.info('Creating examples.')
  for idx, example in enumerate(
      tf.compat.v1.data.make_one_shot_iterator(dataset)):
    if idx % 500 == 0:
      logging.info('validation  %d', idx)

    embeddings = example['embeddings']  # shape is (5 x bert_embedding_size)

    # The input is the concatenated embeddings of the first 4 sentences.
    x = embeddings[0:4, :]
    all_x.append(x)

    # Concatenate all 5th sentence (target) embeddings together.
    all_5th_embs_1.append(embeddings[4, :])
    all_5th_embs_2.append(embeddings[5, :])

    all_labels.append(example['label'].numpy())

  dataset = tf.data.Dataset.from_tensor_slices(
      (all_x, all_5th_embs_1, all_5th_embs_2, all_labels)).batch(1)

  if return_embedding_matrices:
    return dataset, np.array(all_5th_embs_1), np.array(all_5th_embs_2)
  else:
    return dataset


def build_all_distractor_valid_dataset(dataset):
  """Reads in examples from TFDS valid set with all targets from train set as ditractors."""

  all_x = []
  all_true_5th_embs = []
  all_y = []

  # Read in the data into inputs and targets and create a new dataset from it.
  logging.info('Creating examples.')
  for idx, example in enumerate(
      tf.compat.v1.data.make_one_shot_iterator(dataset)):
    if idx % 500 == 0:
      logging.info('validation  %d', idx)

    embeddings = example['embeddings']  # shape is (5 x bert_embedding_size)

    # The input is the concatenated embeddings of the first 4 sentences.
    x = embeddings[0:4, :]
    all_x.append(x)
    all_y.append(idx)

    label = example['label'].numpy()

    # Concatenate all 5th sentence (target) embeddings together.
    if label == 0:
      all_true_5th_embs.append(embeddings[4, :])
    elif label == 1:
      all_true_5th_embs.append(embeddings[5, :])
    else:
      raise ValueError('Label should be either 0 or 1')

  dataset = tf.data.Dataset.from_tensor_slices(
      (all_x, all_y)).batch(1)

  return dataset, np.array(all_true_5th_embs)


def do_evaluation(model,
                  metrics,
                  datasets,
                  emb_matrices):
  """Runs all of the evaluation loops."""
  for x, labels in datasets['valid_nolabel']:
    predictions = eval_nolabel_step(model, x, emb_matrices['valid_nolabel'])
    metrics['valid_nolabel_acc'](labels, predictions)

  for x, labels in datasets['train_nolabel']:
    predictions = eval_nolabel_step(model, x, emb_matrices['train_nolabel'])
    metrics['train_subset_acc'](labels, predictions)

  for x, emb1, emb2, label in datasets['valid2018']:
    correct = eval_step(model, x, emb1, emb2, label)
    metrics['valid_winter2018_acc'](1, correct)

  for x, emb1, emb2, label in datasets['valid2016']:
    correct = eval_step(model, x, emb1, emb2, label)
    metrics['valid_spring2016_acc'](1, correct)


def pick_best_checkpoint(base_dir, key='valid_spring2016_acc'):
  """Returns path to checkpoint with highest validation accuracy.

  Args:
    base_dir: Base directory of experiment checkpoints.
    key: Name of eval metric to choose best checkpoint based on.

  Returns:
    Path to best checkpoint.
  """

  metrics_file = os.path.join(base_dir, 'eval/all_metrics.csv')
  with gfile.GFile(metrics_file, 'r') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)

    accuracy_index = header.index(key)
    checkpoint_name_index = header.index('checkpoint')

    best_acc = 0.0
    best_checkpoint = None
    for line in csv_reader:
      acc = float(line[accuracy_index])
      if acc > best_acc:
        best_acc = acc
        best_checkpoint = line[checkpoint_name_index]

    logging.warning('Best checkpoint is %s with validation accuracy %f',
                    best_checkpoint, best_acc)

    # This code assumed the checkpoint name contains the epoch and step in the
    # following format.
    best_checkpoint_path = gfile.glob(
        os.path.join(base_dir, best_checkpoint) + '*ckpt*index')
    best_checkpoint_path = best_checkpoint_path[0].replace('.index', '')

    return best_checkpoint_path


def read_all_stories(root_dir):
  """Reads all of the stories into memory."""
  stories = {}
  for fname in gfile.glob(os.path.join(root_dir, '*.csv')):
    with gfile.GFile(fname) as f:
      reader = csv.reader(f)
      header = next(reader)
      if header[1] == 'storytitle':
        split = 'TRAIN'
      elif len(header) == 8:
        split = 'VALIDATION'
      else:
        split = 'TEST'

      for line in reader:
        story_id = line[0]

        if split == 'TRAIN':
          story_sentences = line[2:]
        elif split == 'VALIDATION':
          first_four_story_sentences = line[1:5]
          which_fifth = int(line[-1])
          fifth_sentence = line[5] if which_fifth == 1 else line[6]
          story_sentences = first_four_story_sentences + [fifth_sentence]
        elif split == 'TEST':
          story_sentences = line[1:]

        stories[story_id] = story_sentences
  return stories
