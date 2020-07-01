# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""For a bunch of stories in the validation set, rank possible 5th sentences.

5th sentences being ranked are those from the validation set concatenated with
the train set.
"""

import os

from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
import models
import numpy as np
import rocstories_sentence_embeddings
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
import tensorflow_datasets.public_api as tfds
import utils


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', None, 'Dataset to read from.')
flags.DEFINE_string('save_dir', '/tmp/model', 'Where to save model.')

flags.DEFINE_string('base_dir', '/tmp/model',
                    'Base directory containing checkpoints and .gin config.')

flags.DEFINE_string('data_dir', 'tfds_datasets/',
                    'Where to look for TFDS datasets.')

flags.DEFINE_string('rocstories_root_dir',
                    'ROCStories_csvs/',
                    'Base directory where ROC Stories *.csv files are stored.')
tf.enable_v2_behavior()


def build_all_distractor_valid_dataset(dataset, batch_size):
  """Read training data from TFDS dataset into inputs and targets embeddings."""

  all_x = []
  all_true_5th_embs = []
  all_y = []
  all_story_ids = []

  # Read in the data into inputs and targets and create a new dataset from it.
  logging.info('Creating examples.')
  for idx, example in enumerate(
      tf.compat.v1.data.make_one_shot_iterator(dataset)):
    if idx % 500 == 0:
      logging.info('validation  %d', idx)

    embeddings = example['embeddings']  # shape is (5 x bert_embedding_size)
    label = example['label'].numpy()

    # The input is the concatenated embeddings of the first 4 sentences.
    x = embeddings[0:4, :]
    all_x.append(x)
    all_y.append(idx)
    all_story_ids.append(example['story_id'])

    # Concatenate all 5th sentence (target) embeddings together.
    if label == 0:
      all_true_5th_embs.append(embeddings[4, :])
    elif label == 1:
      all_true_5th_embs.append(embeddings[5, :])
    else:
      raise ValueError('Label should be either 0 or 1')

  dataset = tf.data.Dataset.from_tensor_slices(
      (all_x, all_y, all_story_ids)).batch(batch_size)

  return dataset, np.array(all_true_5th_embs), all_story_ids


@gin.configurable('dataset')
def prepare_dataset(dataset_name=gin.REQUIRED,
                    shuffle_input_sentences=False,
                    num_eval_examples=2000,
                    batch_size=32):
  """Create batched, properly-formatted datasets from the TFDS datasets.

  Args:
    dataset_name: Name of TFDS dataset.
    shuffle_input_sentences: Not used during evaluation, but arg still needed
      for gin compatibility.
    num_eval_examples: Number of examples to use during evaluation. For the
      nolabel evaluation, this is also the number of distractors we choose
      between.
    batch_size: Batch size.

  Returns:
    The validation dataset, the story identifiers for each story in the
      embedding matrix, and the embedding matrix.
  """

  del num_eval_examples
  del shuffle_input_sentences

  splits_to_load = [tfds.Split.TRAIN,
                    rocstories_sentence_embeddings.VALIDATION_2018,]
  tfds_train, tfds_valid = tfds.load(
      dataset_name,
      data_dir=FLAGS.data_dir,
      split=splits_to_load)

  _, train_embs, train_story_ids = utils.build_train_style_dataset(
      tfds_train, batch_size, shuffle_input_sentences=False, return_ids=True,
      is_training=False)
  out = build_all_distractor_valid_dataset(tfds_valid, batch_size=batch_size)
  valid_dataset, valid_embs, valid_story_ids = out

  all_story_ids = valid_story_ids + train_story_ids
  all_emb_matrix = tf.concat([valid_embs, train_embs], axis=0)

  return valid_dataset, all_story_ids, all_emb_matrix


def reranking_eval(base_dir):
  """Outputs top-ranked ending sentences for stories in validation set."""
  stories_text = utils.read_all_stories(FLAGS.rocstories_root_dir)

  valid_dataset, all_story_ids, all_emb_matrix = prepare_dataset()

  num_input_sentences = tf.compat.v1.data.get_output_shapes(
      valid_dataset)[0][1]
  model = models.build_model(
      num_input_sentences=num_input_sentences, embedding_matrix=all_emb_matrix)

  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_path = utils.pick_best_checkpoint(
      base_dir, 'valid_nolabel_acc')

  logging.info('LOADING FROM CHECKPOINT: %s', checkpoint_path)

  result = checkpoint.restore(checkpoint_path).expect_partial()
  result.assert_nontrivial_match()

  for x, labels, story_ids in valid_dataset:
    _, output_embedding = model(x, training=False)

    scores = tf.matmul(output_embedding, all_emb_matrix, transpose_b=True)
    sorted_indices = tf.argsort(scores, axis=-1, direction='DESCENDING')

    batch_size = x.shape[0]
    for batch_index in range(batch_size):
      story_id = story_ids[batch_index].numpy()
      story_id = story_id.decode('utf-8')
      story_text = stories_text[story_id]

      logging.info('Groundtruth story start:')
      for i in range(4):
        logging.info('  %d:\t%s', i+1, story_text[i])

      logging.info('Groundtruth 5th sentence: %s', story_text[4])

      logging.info('Guessed endings: ')
      for story_index in sorted_indices[batch_index, :10].numpy():
        chosen_story_id = all_story_ids[story_index].numpy().decode('utf-8')
        story_text = stories_text[chosen_story_id]
        score = scores[batch_index, story_index].numpy()
        logging.info('(%f)  %s', score, story_text[-1])

      # Note: the indexes in labels are only valid for the validation set
      # because it happens ot be first in the embedding matric.
      gt_score = scores[batch_index, labels[batch_index]].numpy()
      label = labels[batch_index].numpy()
      gt_rank = sorted_indices[batch_index].numpy().tolist().index(label)
      logging.info('Rank of GT: %d', gt_rank)
      logging.info('Score for GT: %f', gt_score)

      logging.info('')


def main(argv):
  del argv
  base_dir = FLAGS.base_dir

  # Load gin.config settings stored in model directory. It might take some time
  # for the train script to start up and actually write out a gin config file.
  # Wait 10 minutes (periodically checking for file existence) before giving up.
  gin_config_path = os.path.join(base_dir, 'config.gin')
  if not gfile.exists(gin_config_path):
    raise ValueError('Could not find config.gin in "%s"' % base_dir)

  gin.parse_config_file(gin_config_path, skip_unknown=True)
  gin.finalize()

  reranking_eval(base_dir)


if __name__ == '__main__':
  app.run(main)

