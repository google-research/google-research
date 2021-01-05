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

# Lint as: python3
r"""Training script to sparsify a ResNet-50.

"""
import os
from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from pruning_identified_exemplars.utils import model_utils

# model params
flags.DEFINE_integer(
    'steps_per_checkpoint', 500,
    'Controls how often checkpoints are generated. More steps per '
    'checkpoint = higher utilization of TPU and generally higher '
    'steps/sec')
flags.DEFINE_float('label_smoothing', 0.1,
                   'Relax confidence in the labels by (1-label_smoothing).')
flags.DEFINE_integer('steps_per_eval', 1251,
                     'Controls how often evaluation is performed.')
flags.DEFINE_integer('num_cores', 8, 'Number of cores.')
flags.DEFINE_string('output_dir', '',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_string('mode', 'train',
                    'One of {"train_and_eval", "train", "eval"}.')
flags.DEFINE_string('train_dir', '',
                    'The location of the tfrecords used for training.')
flags.DEFINE_string('eval_dir', '',
                    'The location of the tfrecords used for eval.')
flags.DEFINE_string('master', 'local', 'Name of the TensorFlow master to use.')

# pruning flags
flags.DEFINE_string('pruning_hparams', '',
                    'Comma separated list of pruning-related hyperparameters')
flags.DEFINE_float('end_sparsity', 0.1,
                   'Target sparsity desired by end of training.')
flags.DEFINE_integer('sparsity_begin_step', 5000, 'Step to begin pruning at.')
flags.DEFINE_integer('sparsity_end_step', 8000, 'Step to end pruning at.')
flags.DEFINE_integer('pruning_frequency', 500, 'Step interval between pruning.')
flags.DEFINE_enum(
    'pruning_method', 'baseline',
    ('threshold', 'random_independent', 'random_cumulative', 'baseline'),
    'Method used for pruning'
    'Specify as baseline if no pruning is used.')
flags.DEFINE_bool('log_class_level_summaries', True,
                  'Boolean for whether to log class level precision/accuracy.')
flags.DEFINE_float('expansion_factor', 6.,
                   'how much to expand filters before depthwise conv')
flags.DEFINE_float(
    'training_steps_multiplier', 1.0,
    'Training schedule is shortened or extended with the '
    'multiplier, if it is not 1.')
flags.DEFINE_integer('block_width', 1, 'width of block')
flags.DEFINE_integer('block_height', 1, 'height of block')

# set this flag to true to do a test run of this code with synthetic data
flags.DEFINE_bool('test_small_sample', True,
                  'Boolean for whether to test internally.')

FLAGS = flags.FLAGS

imagenet_params = {
    'sloppy_shuffle': True,
    'num_cores': 8,
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


def main(argv):
  del argv  # Unused.

  initial_sparsity = 0.0
  pruning_hparams_string = ('begin_pruning_step={0},'
                            'sparsity_function_begin_step={0},'
                            'end_pruning_step={1},'
                            'sparsity_function_end_step={1},'
                            'target_sparsity={2},'
                            'initial_sparsity={3},'
                            'pruning_frequency={4},'
                            'threshold_decay=0,'
                            'block_width={5},'
                            'block_height={6}'.format(
                                FLAGS.sparsity_begin_step,
                                FLAGS.sparsity_end_step, FLAGS.end_sparsity,
                                initial_sparsity, FLAGS.pruning_frequency,
                                FLAGS.block_width, FLAGS.block_height))

  params = imagenet_params

  if FLAGS.test_small_sample:
    output_dir = '/tmp/imagenet_train_eval/'
  else:
    # configures train directories based upon hyperparameters.
    if FLAGS.pruning_method:
      folder_stub = os.path.join(FLAGS.pruning_method, str(FLAGS.end_sparsity),
                                 str(FLAGS.sparsity_begin_step),
                                 str(FLAGS.sparsity_end_step))
    else:
      folder_stub = os.path.join('baseline', str(0.0), str(0.0), str(0.0),
                                 str(0.0), str(FLAGS.resnet_depth))
      output_dir = os.path.join(FLAGS.output_dir, folder_stub)

  update_params = {
      'lr_schedule': [(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)],
      'momentum': 0.9,
      'data_format': 'channels_last',
      'output_dir': output_dir,
      'label_smoothing': FLAGS.label_smoothing,
  }
  params.update(update_params)

  if FLAGS.pruning_method != 'baseline':
    params['pruning_method'] = FLAGS.pruning_method
  else:
    params['pruning_method'] = None

  params['mode'] = FLAGS.mode
  if FLAGS.mode == 'train':
    params['batch_size'] = params['train_batch_size']
    params['task'] = 'imagenet_training'
    params['data_dir'] = FLAGS.train_dir
  else:
    params['batch_size'] = params['eval_batch_size']
    params['task'] = 'imagenet_eval'
    params['data_dir'] = FLAGS.eval_dir

  if FLAGS.test_small_sample:
    update_params = {
        'batch_size': 2,
        'num_train_steps': 10,
        'num_images': 2,
        'num_train_images': 10,
        'num_eval_images': 10,
    }
    params['test_small_sample'] = True
    params.update(update_params)
  else:
    params['test_small_sample'] = False

  if FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in tf2.train.checkpoints_iterator(params['output_dir']):
      tf.logging.info('Starting to evaluate.')
      try:
        _ = model_utils.initiate_task_helper(
            ckpt_directory=ckpt, model_params=params, pruning_params=None)
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= params['num_train_steps']:
          tf.logging.info('Evaluation finished')
          break
      except tf.errors.NotFoundError:
        tf.logging.info('Checkpoint was not found, skipping checkpoint.')

  else:
    if FLAGS.mode == 'train':
      tf.logging.info('start training...')
      model_utils.initiate_task_helper(
          ckpt_directory=None,
          model_params=params,
          pruning_params=pruning_hparams_string)
      tf.logging.info('finished training.')


if __name__ == '__main__':
  app.run(main)
