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

"""Computes model accuracy with all available 5th sentences as distractors.
"""

import os

from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
import models
import rocstories_sentence_embeddings  # pylint: disable=unused-import
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
import tensorflow_datasets.public_api as tfds
import utils


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', None, 'Dataset to read from.')

flags.DEFINE_string('base_dir', '/tmp/model',
                    'Base directory containing checkpoints and .gin config.')

flags.DEFINE_string('data_dir', 'tfds_datasets/',
                    'Where to look for TFDS datasets.')

tf.enable_v2_behavior()


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

  splits_to_load = ['train',
                    rocstories_sentence_embeddings.VALIDATION_2018,]
  tfds_train, tfds_valid2018 = tfds.load(
      dataset_name,
      data_dir=FLAGS.data_dir,
      split=splits_to_load)

  _, train_embs = utils.build_train_style_dataset(
      tfds_train, batch_size=batch_size, shuffle_input_sentences=False)
  valid_dataset, valid_embs = utils.build_all_distractor_valid_dataset(
      tfds_valid2018)

  # Put the validation set embeddings first in the joint embedding matrix so
  # that the labels for validation examples are still correct.
  all_emb_matrix = tf.concat([valid_embs, train_embs], axis=0)

  return valid_dataset, all_emb_matrix


def all_distractors_eval(base_dir):
  """Computes model accuracy with all possible last sentences as distractors."""
  valid_dataset, all_emb_matrix = prepare_dataset()

  num_input_sentences = tf.compat.v1.data.get_output_shapes(
      valid_dataset)[0][1]
  model = models.build_model(
      num_input_sentences=num_input_sentences, embedding_matrix=all_emb_matrix)

  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_path = utils.pick_best_checkpoint(base_dir, 'valid_nolabel_acc')

  logging.info('LOADING FROM CHECKPOINT: %s', checkpoint_path)

  result = checkpoint.restore(checkpoint_path).expect_partial()
  result.assert_nontrivial_match()

  ranks = []
  label_in_top10 = []

  for x, labels in valid_dataset:
    _, output_embedding = model(x, training=False)

    scores = tf.matmul(
        output_embedding, all_emb_matrix, transpose_b=True)
    sorted_indices = tf.argsort(scores, axis=-1, direction='DESCENDING')

    batch_size = x.shape[0]
    for batch_index in range(batch_size):
      # Note: the indexes in labels are only valid for the validation set
      # because it happens ot be first in the embedding matrix.
      label = labels[batch_index].numpy()
      gt_rank = sorted_indices[batch_index].numpy().tolist().index(label)

      top10predicted = sorted_indices[batch_index, :10]
      label_in_top10.append(1 if label in top10predicted else 0)

      top_predicted = sorted_indices[batch_index, 0]
      label_in_top10.append(1 if label == top_predicted else 0)
      ranks.append(gt_rank)


def main(argv):
  del argv

  # Load gin.config settings stored in model directory. It might take some time
  # for the train script to start up and actually write out a gin config file.
  # Wait 10 minutes (periodically checking for file existence) before giving up.
  gin_config_path = os.path.join(FLAGS.base_dir, 'config.gin')
  if not gfile.exists(gin_config_path):
    raise ValueError('Could not find config.gin in "%s"' % FLAGS.base_dir)

  gin.parse_config_file(gin_config_path, skip_unknown=True)
  gin.finalize()
  all_distractors_eval(FLAGS.base_dir)


if __name__ == '__main__':
  app.run(main)

