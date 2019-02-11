# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

r"""Training script for implementing ROAR benchmark.

This script trains a ResNet 50 model on either:
1) a raw unmodified dataset,
2) a modified dataset where a fraction of the most important pixels
have been replaced with the mean
3) a randomly modified dataset where a random fraction of
pixels has been replaced with the mean.

Use FLAGS.transformation to alternate between 1-3. For 2), use
FLAGS.saliency_method to specify the method to estimate which pixels are
important and FLAGS.threshold to indicate the fraction of the image that
will be replaced.

To run this script:

python -m interpretability_benchmark.data_input_test

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf  # tf

from interpretability_benchmark import data_input
from interpretability_benchmark.utils import resnet_model

# model params
flags.DEFINE_integer(
    'steps_per_checkpoint', 500,
    'Controls how often checkpoints are generated. More steps per '
    'checkpoint = higher utilization of TPU and generally higher '
    'steps/sec')
flags.DEFINE_string('data_format', 'channels_last',
                    'A flag to override the data format used in the model.')
flags.DEFINE_integer('steps_per_eval', 1251,
                     'Controls how often evaluation is performed.')
flags.DEFINE_integer('num_cores', 8, 'Number of cores.')
flags.DEFINE_string('output_dir', '',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_string('mode', 'train',
                    'One of {"train_and_eval", "train", "eval"}.')
flags.DEFINE_enum('dataset_name', 'birdsnap',
                  ('food_101', 'imagenet', 'birdsnap'),
                  'What dataset is the model trained on.')
flags.DEFINE_string('base_dir', '',
                    'The location of the tfrecords used for training.')
flags.DEFINE_string('master', 'local',
                    'Name of the TensorFlow master to use.')

# parameters for ROAR benchmark
flags.DEFINE_enum(
    'transformation', 'raw_image',
    ('raw_image', 'random_baseline', 'modified_image'),
    'String to indicate how the raw image should be modified.'
    'modified_image= pixels removed according to FLAG.saliency_method'
    'raw_image=resnet model is trained on the unmodified'
    ' image. random_baseline=training resnet on a randomly'
    'modified image.')
flags.DEFINE_enum(
    'saliency_method', 'ig_smooth_2',
    ('gradient_image', 'gradient_smooth', 'gradient_smooth_2',
     'gradient_vargrad', 'ig_image', 'ig_smooth', 'ig_smooth_2', 'IG_vargrad',
     'gb_image', 'gb_smooth', 'GB_vargrad', 'gb_smooth_2', 'sobel'),
    'Estimator is used to estimate the most/least important pixels in'
    'image.')
flags.DEFINE_bool(
    'keep_information', False,
    'whether to remove (False) or preserve (True) the fraction of pixels'
    'estimated to be most important.')
flags.DEFINE_bool(
    'squared_value', True,
    'whether to compute ranking based upon squared value or'
    'the raw pixel ranking')
flags.DEFINE_float(
    'threshold', 80.,
    'Fraction of all input features that are modified according'
    'to the estimator.')


FLAGS = flags.FLAGS

saliency_dict = {
    'ig_smooth': 'IG_SG',
    'ig_image': 'IG',
    'ig_smooth_2': 'IG_SG_2',
    'IG_vargrad': 'IG_V',
    'gradient_image': 'SH',
    'gradient_smooth': 'SH_SG',
    'gradient_smooth_2': 'SH_SG_2',
    'gradient_vargrad': 'SH_V',
    'gb_image': 'GB',
    'gb_smooth': 'GB_SG',
    'gb_smooth_2': 'GB_SG_2',
    'GB_vargrad': 'GB_V',
    'sobel': 'SOBEL'
}

imagenet_params = {
    'train_batch_size': 4096,
    'num_train_images': 1281167,
    'num_eval_images': 50000,
    'num_label_classes': 1000,
    'num_train_steps': 32000,
    'base_learning_rate': 0.1,
    'weight_decay': 1e-4,
    'eval_batch_size': 1024,
    'mean_rgb': [0.485 * 255, 0.456 * 255, 0.406 * 255],
    'stddev_rgb': [0.229 * 255, 0.224 * 255, 0.225 * 255]
}

food_101_params = {
    'train_batch_size': 256,
    'num_train_images': 75750,
    'num_eval_images': 25250,
    'num_label_classes': 101,
    'num_train_steps': 20000,
    'base_learning_rate': 0.7,
    'weight_decay': 0.0001,
    'eval_batch_size': 256,
    'mean_rgb': [0.561, 0.440, 0.312],
    'stddev_rgb': [0.252, 0.256, 0.259]
}

birdsnap_params = {
    'train_batch_size': 256,
    'num_train_images': 47386,
    'num_eval_images': 2443,
    'num_label_classes': 500,
    'num_train_steps': 20000,
    'base_learning_rate': 1.0,
    'weight_decay': 0.0001,
    'eval_batch_size': 224,
    'mean_rgb': [0.491, 0.506, 0.451],
    'stddev_rgb': [0.229, 0.226, 0.267]
}

MOMENTUM = 0.9
# Learning rate schedule
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

MEAN_RGB = [0.4855, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def compute_lr(current_epoch, initial_learning_rate, train_batch_size):
  """Computes learning rate schedule."""
  scaled_lr = initial_learning_rate * (train_batch_size / 256.0)

  decay_rate = (
      scaled_lr * LR_SCHEDULE[0][0] * current_epoch / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """Setup of training and eval for modified dataset using a ResNet-50.

  Args:
    features: A float32 batch of images.
    labels: A int32 batch of labels.
    mode: Specifies whether training or evaluation.
    params: Dictionary of parameters passed to the model.

  Returns:
    Model estimator w specifications.
  """

  if isinstance(features, dict):
    features = features['feature']

  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  train_batch_size = params['train_batch_size']

  steps_per_epoch = params['num_train_images'] / train_batch_size
  initial_learning_rate = params['base_learning_rate']
  num_label_classes = params['num_label_classes']

  network = resnet_model.resnet_50(
      num_classes=num_label_classes,
      data_format=FLAGS.data_format)

  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  output_dir = params['output_dir']
  weight_decay = params['weight_decay']

  one_hot_labels = tf.one_hot(labels, num_label_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=one_hot_labels, label_smoothing=0.1)

  loss = cross_entropy + weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
  ])
  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:

    global_step = tf.train.get_global_step()

    steps_per_epoch = params['num_train_images'] / train_batch_size
    current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
    learning_rate = compute_lr(current_epoch, initial_learning_rate,
                               train_batch_size)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=MOMENTUM, use_nesterov=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.name_scope('train'):
      train_op = optimizer.minimize(loss, global_step)

    summary_writer = tf.contrib.summary.create_file_writer(output_dir)

    with summary_writer.as_default():
      with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', loss, step=global_step)
        tf.contrib.summary.scalar(
            'learning_rate', learning_rate, step=global_step)
        tf.contrib.summary.scalar(
            'current_epoch', current_epoch, step=global_step)
        tf.contrib.summary.scalar(
            'steps_per_epoch', steps_per_epoch, step=global_step)
        tf.contrib.summary.scalar(
            'weight_decay', weight_decay, step=global_step)
        tf.contrib.summary.all_summary_ops()
  else:
    train_op = None

  eval_metrics = {}
  if mode == tf.estimator.ModeKeys.EVAL:
    train_op = None
    predictions = tf.argmax(logits, axis=1)
    eval_metrics['top_1_accuracy'] = tf.metrics.accuracy(labels, predictions)
    in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
    eval_metrics['top_5_accuracy'] = tf.metrics.mean(in_top_5)

  return tf.estimator.EstimatorSpec(
      training_hooks=host_call,
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics)


def main(argv):
  del argv  # Unused.

  if FLAGS.squared_value:
    is_squared = 'squared'
  else:
    is_squared = 'not_squared'
  if FLAGS.keep_information:
    info_keep = 'keep'
  else:
    info_keep = 'remove'

  if FLAGS.dataset_name == 'food_101':
    params = food_101_params
  elif FLAGS.dataset_name == 'imagenet':
    params = imagenet_params
  elif FLAGS.dataset_name == 'birdsnap':
    params = birdsnap_params
  else:
    raise ValueError('Dataset type is not known %s' % (FLAGS.dataset))

  model_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name,
                           FLAGS.transformation, str(FLAGS.threshold),
                           str(params['base_learning_rate']),
                           str(params['weight_decay']), is_squared, info_keep)

  if FLAGS.transformation in ['modified_image', 'raw_saliency_map']:
    model_dir = os.path.join(model_dir, FLAGS.saliency_method)

  if FLAGS.mode == 'eval':
    split = 'validation'
  else:
    split = 'training'

  sal_method = saliency_dict[FLAGS.saliency_method]
  data_directory = os.path.join(FLAGS.base_dir, FLAGS.dataset_name,
                                '2018-12-10', 'resnet_50', sal_method,
                                split + '*')
  mean_stats = [0.485, 0.456, 0.406],
  std_stats = [0.229, 0.224, 0.225],

  dataset_ = data_input.DataIterator(
      mode=FLAGS.mode,
      data_directory=data_directory,
      saliency_method=FLAGS.saliency_method,
      transformation=FLAGS.transformation,
      threshold=FLAGS.threshold,
      keep_information=FLAGS.keep_information,
      use_squared_value=FLAGS.squared_value,
      mean_stats=mean_stats,
      std_stats=std_stats,
      num_cores=FLAGS.num_cores)

  params['output_dir'] = model_dir
  if FLAGS.mode == 'train':
    params['batch_size'] = params['train_batch_size']
  else:
    params['batch_size'] = params['eval_batch_size']

  num_train_steps = params['num_train_steps']


  eval_steps = params['num_eval_images'] // params['batch_size']
  run_config = tf.estimator.RunConfig(
      model_dir=model_dir, save_checkpoints_steps=FLAGS.steps_per_checkpoint)

  classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn,
      model_dir=model_dir,
      params=params,
      config=run_config)

  if FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.contrib.training.checkpoints_iterator(model_dir):
      tf.logging.info('Starting to evaluate.')
      try:
        classifier.evaluate(
            input_fn=dataset_.input_fn, steps=eval_steps, checkpoint_path=ckpt)
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= num_train_steps:
          print('Evaluation finished after training step %d' % current_step)
          break

      except tf.errors.NotFoundError:
        logging.info('Checkpoint was not found, skipping checkpoint.')

  else:
    if FLAGS.mode == 'train':
      print('start training...')
      classifier.train(input_fn=dataset_.input_fn, max_steps=num_train_steps)
      print('finished training.')


if __name__ == '__main__':
  app.run(main)
