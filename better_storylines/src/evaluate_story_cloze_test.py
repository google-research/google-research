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

"""Output the overall test accuracy on the 2016 test set.
"""

import os

from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
import models
import rocstories_sentence_embeddings
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import utils

gfile = tf.io.gfile

FLAGS = flags.FLAGS
flags.DEFINE_string('base_dir', '/tmp/model',
                    'Base directory containing checkpoints and .gin config.')
flags.DEFINE_string('data_dir', 'tfds_datasets',
                    'Where to look for TFDS datasets.')

flags.DEFINE_multi_string('gin_bindings', [], 'Not used.')

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
    A dictionary mapping from the dataset split to a Dataset object.
  """

  del batch_size
  del num_eval_examples
  del shuffle_input_sentences

  dataset = tfds.load(
      dataset_name,
      data_dir=FLAGS.data_dir,
      split=rocstories_sentence_embeddings.TEST_2016,
      download=False)
  dataset = utils.build_validation_dataset(dataset)
  return dataset


def eval_single_checkpoint(model, dataset):
  """Runs quantitative evaluation on a single checkpoint."""
  test_2016_accuracy = tf.keras.metrics.Accuracy(name='test_spring2016_acc')

  for x, fifth_embedding_1, fifth_embedding_2, label in dataset:
    correct = utils.eval_step(
        model, x, fifth_embedding_1, fifth_embedding_2, label)
    test_2016_accuracy(1, correct)

  logging.warning('Test accuracy: %f', test_2016_accuracy.result())
  return test_2016_accuracy.result().numpy().tolist()


def run_eval(base_dir):
  """Writes model's predictions in proper format to [base_dir]/answer.txt."""
  best_checkpoint_name = utils.pick_best_checkpoint(base_dir)

  dataset = prepare_dataset()
  checkpoint_path = os.path.join(base_dir, best_checkpoint_name)

  embedding_dim = tf.compat.v1.data.get_output_shapes(dataset)[0][-1]
  num_input_sentences = tf.compat.v1.data.get_output_shapes(dataset)[0][1]
  model = models.build_model(
      num_input_sentences=num_input_sentences, embedding_dim=embedding_dim)

  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_path).expect_partial()

  logging.info('Evaluating with checkpoint: "%s"', checkpoint_path)
  test_accuracy = eval_single_checkpoint(model, dataset)

  with gfile.GFile(os.path.join(base_dir, 'test_spring2016_acc.txt'), 'w') as f:
    f.write(str(test_accuracy))


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

  run_eval(base_dir)


if __name__ == '__main__':
  app.run(main)
