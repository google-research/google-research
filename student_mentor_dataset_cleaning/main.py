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

"""Main program to run student-mentor training."""

from absl import app
from absl import flags
import tensorflow as tf

from student_mentor_dataset_cleaning.training.loss.triplet_loss import TripletLoss
from student_mentor_dataset_cleaning.training.trainers import trainer
from student_mentor_dataset_cleaning.training.trainers import trainer_triplet

FLAGS = flags.FLAGS

flags.DEFINE_integer('mini_batch_size', 32, 'Mini-batch size')
flags.DEFINE_integer('max_iteration_count', 20, 'Maximum iteration count')
flags.DEFINE_integer('student_epoch_count', 30,
                     'Maximum number of training epochs for the student.')
flags.DEFINE_integer('mentor_epoch_count', 30,
                     'Maximum number of training epochs for the mentor.')
flags.DEFINE_string(
    'mode', 'softmax',
    'Training mode to use. Options are "softmax" (default) and "triplet". '
    'Softmax mode will always use the MNIST dataset and ignore "csv_path".'
)
flags.DEFINE_string('save_dir', '', 'Path to model save dir.')
flags.DEFINE_string('tensorboard_log_dir', '', 'Path to tensorboard log dir.')
flags.DEFINE_string('train_dataset_dir', '',
                    'Path to the training dataset dir.')
flags.DEFINE_string('csv_path', '', 'Path to the training dataset dataframe.')
flags.DEFINE_string('student_initial_model', '',
                    'Path to the student model initialization.')
flags.DEFINE_integer(
    'delg_embedding_layer_dim', 2048,
    'Size of the FC whitening layer (embedding layer). Used only if'
    'delg_global_features:True.')


def verify_arguments():
  """Verifies the validity of the command-line arguments."""

  assert FLAGS.mini_batch_size > 0, '`mini_batch_size` must be positive.'
  assert FLAGS.max_iteration_count > 0, ('`max_iteration_count` must be '
                                         'positive.')
  assert FLAGS.student_epoch_count > 0, ('`student_epoch_count` must be '
                                         'positive.')
  assert FLAGS.mentor_epoch_count > 0, '`mentor_epoch_count` must be positive.'
  assert FLAGS.mode in ['softmax', 'triplet'], ('`mode` must be either '
                                                '`softmax` or `triplet`')


def run_softmax():
  """Runs the program in softmax mode using the MNIST dataset."""

  tf.compat.v1.enable_eager_execution()

  student = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28), name='student_flatten'),
      tf.keras.layers.Dense(128, activation='relu', name='student_hidden0'),
      tf.keras.layers.Dense(10, name='student_output')
  ])
  student_optimizer = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False,
      name='StudentAdam')
  student.compile(
      optimizer=student_optimizer,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, name='student_categorical_crossentropy_loss'),
      metrics=[
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=1, name='student_top_1_categorical_accuracy'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=2, name='student_top_2_categorical_accuracy'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=3, name='student_top_3_categorical_accuracy'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=4, name='student_top_4_categorical_accuracy'),
          tf.keras.metrics.SparseCategoricalCrossentropy(
              name='student_categorical_crossentropy', from_logits=True)
      ])

  mentor = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(101770,), name='mentor_flatten'),
      tf.keras.layers.Dense(50, activation='relu', name='mentor_hidden0'),
      tf.keras.layers.Dense(1, activation='sigmoid', name='mentor_output')
  ])
  mentor_optimizer = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False,
      name='StudentAdam')
  mentor.compile(
      optimizer=mentor_optimizer,
      loss=tf.keras.losses.BinaryCrossentropy(
          name='mentor_binary_crossentropy_loss'),
      metrics=[
          tf.keras.metrics.BinaryAccuracy(
              name='mentor_binary_accuracy', threshold=0.5),
          tf.keras.metrics.BinaryCrossentropy(
              name='mentor_binary_crossentropy'),
          tf.keras.metrics.FalseNegatives(name='mentor_false_negatives'),
          tf.keras.metrics.FalsePositives(name='mentor_false_positives'),
          tf.keras.metrics.TrueNegatives(name='mentor_true_negatives'),
          tf.keras.metrics.TruePositives(name='mentor_true_positives')
      ])

  student, mentor = trainer.train(student, mentor, FLAGS.mini_batch_size,
                                  FLAGS.max_iteration_count,
                                  FLAGS.student_epoch_count,
                                  FLAGS.mentor_epoch_count, FLAGS.save_dir,
                                  FLAGS.tensorboard_log_dir)


def run_triplet():
  """Runs the program in triplet mode on the provided CSV dataset."""

  tf.compat.v1.enable_eager_execution()

  student = tf.keras.applications.ResNet152V2(
      include_top=False,
      weights='imagenet',
      input_shape=[321, 321, 3],
      pooling='avg')

  student_optimizer = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False,
      name='StudentAdam')
  student.compile(
      optimizer=student_optimizer,
      loss=TripletLoss(
          embedding_size=FLAGS.delg_embedding_layer_dim, train_ratio=1.0))

  mentor = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(104000,), name='mentor_flatten'),
      tf.keras.layers.Dense(50, activation='relu', name='mentor_hidden0'),
      tf.keras.layers.Dense(1, activation='sigmoid', name='mentor_output')
  ])
  mentor_optimizer = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False,
      name='StudentAdam')
  mentor.compile(
      optimizer=mentor_optimizer,
      loss=tf.keras.losses.BinaryCrossentropy(
          name='mentor_binary_crossentropy_loss'),
      metrics=[
          tf.keras.metrics.BinaryAccuracy(
              name='mentor_binary_accuracy', threshold=0.5),
          tf.keras.metrics.BinaryCrossentropy(
              name='mentor_binary_crossentropy'),
          tf.keras.metrics.FalseNegatives(name='mentor_false_negatives'),
          tf.keras.metrics.FalsePositives(name='mentor_false_positives'),
          tf.keras.metrics.TrueNegatives(name='mentor_true_negatives'),
          tf.keras.metrics.TruePositives(name='mentor_true_positives')
      ])

  student, mentor = trainer_triplet.train(
      student, mentor, FLAGS.mini_batch_size, FLAGS.max_iteration_count,
      FLAGS.student_epoch_count, FLAGS.mentor_epoch_count,
      FLAGS.train_dataset_dir, FLAGS.csv_path, FLAGS.save_dir,
      FLAGS.tensorboard_log_dir)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  verify_arguments()

  if FLAGS.mode == 'softmax':
    run_softmax()
  elif FLAGS.mode == 'triplet':
    run_triplet()


if __name__ == '__main__':
  app.run(main)
