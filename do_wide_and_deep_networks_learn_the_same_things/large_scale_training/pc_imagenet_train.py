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

"""Code to train ImageNet ResNet models on Patch Camelyon dataset."""
import functools
import os

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from do_wide_and_deep_networks_learn_the_same_things.large_scale_training import alt_resnet
from do_wide_and_deep_networks_learn_the_same_things.large_scale_training import train_lib
from tensorflow_models.official.legacy.image_classification import augment
from tensorflow_models.official.legacy.image_classification import preprocessing
from tensorflow_models.official.legacy.image_classification.efficientnet import efficientnet_model

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
# Define training hyperparameters.
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train for.')
flags.DEFINE_integer('epochs_between_evals', 1,
                     'Number of epochs between evals and checkpoints.')
flags.DEFINE_float('weight_decay', 5e-5, 'L2 regularization.')
# Define model & data hyperparameters.
flags.DEFINE_float('width_multiplier', 1.0, 'Width multiplier.')
flags.DEFINE_float(
    'depth_multiplier', 1.0,
    'Depth multiplier. Although EfficientNet uses fractional '
    'values here, it might be best to avoid them, since they increase '
    'the number of repeats of different blocks by different amounts.')
flags.DEFINE_float('subsample', 1.0, 'Amount of dataset to use.')
flags.DEFINE_float('dropout_rate', 0.0, 'Amount of dropout.')
flags.DEFINE_integer(
    'copy', 0,
    'If the same model configuration has been run before, train another copy '
    'with a different random initialization.')
flags.DEFINE_string('base_dir', None, 'Where the trained model will be saved.')
flags.DEFINE_string('model', 'resnet', 'Model to train.')
flags.DEFINE_bool('use_autoaugment', False, 'Whether to use AutoAugment.')
flags.DEFINE_string('pretrain_dir', None,
                    'Where the pretrained model will be loaded.')

RESNET50_SPEC = [
    ('bottleneck', 64, 3),
    ('bottleneck', 128, 4),
    ('bottleneck', 256, 6),
    ('bottleneck', 512, 3),
]


def preprocess_data(data, is_training):
  """Patch Camelyon data preprocessing"""
  image = tf.image.convert_image_dtype(data['image'], tf.float32)
  if is_training:
    image = tf.image.random_crop(image, [64, 64, 3])
    image = tf.image.random_flip_up_down(image)
  else:
    image = tf.image.resize_with_crop_or_pad(image, 64, 64)  # central crop
  return {'input_1': image, 'label': data['label']}


def load_test_data(batch_size, shuffle=True):
  """Load Patch Camelyon test data"""
  builder = tfds.builder('patch_camelyon')
  test_dataset = builder.as_dataset(tfds.Split.TEST, decoders={})
  test_dataset_size = builder.info.splits[tfds.Split.TEST].num_examples
  test_dataset = test_dataset.map(
      functools.partial(preprocess_data, is_training=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
  if shuffle:
    test_dataset = test_dataset.shuffle(test_dataset_size)
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return test_dataset


def main(argv):
  del argv

  builder = tfds.builder('patch_camelyon')
  decoders = {}

  read_config = tfds.ReadConfig(
      interleave_cycle_length=96, interleave_block_length=2)

  train_dataset_size = builder.info.splits[tfds.Split.TRAIN].num_examples
  train_split = tfds.Split.TRAIN
  if FLAGS.subsample:
    train_dataset_size = int(round(train_dataset_size * FLAGS.subsample))
    train_split = tfds.core.ReadInstruction(
        train_split, to=FLAGS.subsample * 100, unit='%')
  train_dataset = builder.as_dataset(
      train_split,
      decoders=decoders,
      shuffle_files=False,
      read_config=read_config).cache()
  train_dataset = train_dataset.shuffle(train_dataset_size).repeat()
  train_dataset = train_dataset.map(
      functools.partial(preprocess_data, is_training=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  test_dataset = builder.as_dataset(tfds.Split.VALIDATION, decoders=decoders)
  test_dataset = test_dataset.map(
      functools.partial(preprocess_data, is_training=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
  test_dataset = test_dataset.batch(FLAGS.batch_size)
  test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  test_dataset_size = builder.info.splits[tfds.Split.VALIDATION].num_examples

  steps_per_epoch = train_dataset_size // FLAGS.batch_size
  steps_between_evals = int(FLAGS.epochs_between_evals * steps_per_epoch)
  train_steps = FLAGS.epochs * steps_per_epoch
  eval_steps = ((test_dataset_size - 1) // FLAGS.batch_size) + 1

  model_dir_name = (
      '%s-depth-%s-width-%s-bs-%d-lr-%f-reg-%f-dropout-%f-aa-%s' % \
      (FLAGS.model, FLAGS.depth_multiplier, FLAGS.width_multiplier,
       FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay,
       FLAGS.dropout_rate, FLAGS.use_autoaugment))
  if FLAGS.copy > 0:
    model_dir_name += '-copy-%d' % FLAGS.copy
  experiment_dir = os.path.join(FLAGS.base_dir, model_dir_name)

  def model_optimizer_fn(pretrain_dir):
    schedule = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                                 train_steps)
    if FLAGS.model == 'efficientnet':
      config = efficientnet_model.ModelConfig.from_args(
          width_coefficient=FLAGS.width_multiplier,
          depth_coefficient=FLAGS.depth_multiplier,
          resolution=64,
          weight_decay=FLAGS.weight_decay,
          dropout_rate=FLAGS.dropout_rate)
      model = efficientnet_model.EfficientNet(config)
    elif FLAGS.model == 'resnet':
      model = alt_resnet.Resnet(
          block_fn=alt_resnet.BottleneckBlock,
          layers=[3, 4, int(round(FLAGS.depth_multiplier * 6)), 3],
          width_multipliers=[1, 1, 1, FLAGS.width_multiplier, 1],
          num_classes=2,
          kernel_regularizer=tf.keras.regularizers.l2(FLAGS.weight_decay))
    elif FLAGS.model == 'resnet_scale_all':
      model = alt_resnet.Resnet(
          block_fn=alt_resnet.BottleneckBlock,
          layers=[int(round(FLAGS.depth_multiplier * x)) for x in [3, 4, 6, 3]],
          width_multipliers=[1] + [FLAGS.width_multiplier] * 4,
          num_classes=2,
          kernel_regularizer=tf.keras.regularizers.l2(FLAGS.weight_decay))
    else:
      raise ValueError('Unknown model {}'.format(FLAGS.model))
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)

    if pretrain_dir is not None:
      model = alt_resnet.Resnet(
          block_fn=alt_resnet.BottleneckBlock,
          layers=[int(round(FLAGS.depth_multiplier * x)) for x in [3, 4, 6, 3]],
          width_multipliers=[1] + [FLAGS.width_multiplier] * 4,
          num_classes=2,
          kernel_regularizer=tf.keras.regularizers.l2(FLAGS.weight_decay),
          skip_dense_layer=True)
      checkpoint = tf.train.Checkpoint(model=model)
      checkpoint.restore('/readahead/1G' + FLAGS.pretrain_dir)
      model.add_dense_layer()

    return model, optimizer

  train_lib.train(
      model_optimizer_fn=functools.partial(
          model_optimizer_fn, pretrain_dir=FLAGS.pretrain_dir),
      train_steps=train_steps,
      eval_steps=eval_steps,
      steps_between_evals=steps_between_evals,
      train_dataset=train_dataset,
      test_dataset=test_dataset,
      experiment_dir=experiment_dir)


if __name__ == '__main__':
  app.run(main)
