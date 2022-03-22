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

# Lint as: python3
r"""This script trains a ResNet model that implements regularizers.


"""
import os
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from optimizing_interpretability import regularizers as reg
from optimizing_interpretability.imagenet import data_helper
from optimizing_interpretability.imagenet import resnet_model
from optimizing_interpretability.imagenet import utils

flags.DEFINE_integer('num_workers', 1, 'Number of training workers.')
flags.DEFINE_float(
    'momentum',
    default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))
flags.DEFINE_integer('ps_task', 0,
                     'Task id of the replica running the training.')
flags.DEFINE_float(
    'weight_decay',
    default=1e-4,
    help=('Weight decay coefficient for l2 regularization.'))
flags.DEFINE_string('master', '', 'Master job.')
flags.DEFINE_string('tpu_job_name', None, 'For complicated TensorFlowFlock')
flags.DEFINE_integer(
    'steps_per_checkpoint',
    default=1000,
    help=('Controls how often checkpoints are generated. More steps per '
          'checkpoint = higher utilization of TPU and generally higher '
          'steps/sec'))
flags.DEFINE_integer(
    'keep_checkpoint_max',
    default=5,
    help=('The number of checkpoints to save over the course of training'))
flags.DEFINE_string(
    'data_format',
    default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))
flags.DEFINE_bool(
    'transpose_input',
    default=False,
    help='Use TPU double transpose optimization')
flags.DEFINE_float('label_smoothing', 0.1,
                   'Relax confidence in the labels by (1-label_smoothing).')
flags.DEFINE_integer(
    'steps_per_eval',
    default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))
flags.DEFINE_bool(
    'use_tpu',
    default=False,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))
flags.DEFINE_integer(
    'iterations_per_loop',
    default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))
flags.DEFINE_integer(
    'num_parallel_calls',
    default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))
flags.DEFINE_integer(
    'num_cores',
    default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))
flags.DEFINE_string('output_dir', '/tmp/',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_float(
    'base_learning_rate',
    default=0.2,
    help=('Base learning rate when train batch size is 256.'))
flags.DEFINE_enum('dataset', 'imagenet', ('imagenet'),
                  'Name of dataset to train on.')
flags.DEFINE_bool('test_workflow', True,
                  'Whether or not to apply sparsification to the first layer')
flags.DEFINE_enum(
    'mode',
    default='train',
    enum_values=('train_and_eval', 'eval', 'train'),
    help='The Estimator mode.')
# specify the regularizer type to penalize model explanations.
flags.DEFINE_enum(
    'regularizer',
    default='spectreg',
    enum_values=('datagrad', 'spectreg', 'l2', 'tv', 'mse', 'grad_diff',
                 'sobel_edges', 'psnr', 'psnr_hvs', 'ssim', 'ssim_unfiltered',
                 'cor', 'tv_abs', 'tv_abs_unscaled'),
    help='Gradient regularizer.')
flags.DEFINE_float('reg_scale', 0.5,
                   'Scale factor applied to gradient regularization.')
flags.DEFINE_bool('regularize_gradients', True,
                  'Whether to regularize input gradients.')
flags.DEFINE_bool('use_checkpoint', False,
                  'Whether to run eval from a stored checkpoint.')
# add the checkpoint you want to finetune and regularize from here.
flags.DEFINE_string('ckpt_directory', '', 'The ckpt.')
flags.DEFINE_integer('finetune_steps', 500, 'Step interval between pruning.')
flags.DEFINE_enum('noise', 'r_uniform', ('r_normal', 'r_uniform'),
                  'Noise distribution added to input')
flags.DEFINE_float('multiple_image_std', 4.2, 'jitter multiplier.')
flags.DEFINE_integer('num_images', 2, 'Number of images in batch.')

FLAGS = flags.FLAGS

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# add the path to the training and eval directory below.
# when test_small_sample is set to True, the workflow will be tested
# using fake images generated in data_helper.py
imagenet_params = {
    'train_directory': '',
    'eval_directory': '',
    'train_batch_size': 512,
    'eval_batch_size': 512,
    'num_train_images': 1281167,
    'num_eval_images': 50000,
    'num_label_classes': 1000,
    'train_steps': 32000,
    'base_learning_rate': 0.1,
    'finetuning_steps': 1000,
    'weight_decay': 1e-4,
    'mean_rgb': [0.485 * 255, 0.456 * 255, 0.406 * 255],
    'stddev_rgb': [0.229 * 255, 0.224 * 255, 0.225 * 255],
    'resnet_depth': 50,
    'size': 224
}


def compute_lr(current_epoch, initial_learning_rate, train_batch_size,
               lr_schedule):
  """Computes learning rate schedule."""
  scaled_lr = initial_learning_rate * (train_batch_size / 256.0)

  decay_rate = (
      scaled_lr * lr_schedule[0][0] * current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)
  return decay_rate


def create_eval_metrics(labels, logits):
  """Creates the evaluation metrics for the model."""

  eval_metrics = {}
  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  labels = tf.cast(labels, tf.int64)

  eval_metrics['eval_accuracy'] = tf.metrics.accuracy(
      labels=labels, predictions=predictions)

  in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
  eval_metrics['top_5_eval_accuracy'] = tf.metrics.mean(in_top_5)

  return eval_metrics


def train_function(loss, params, global_step):
  """Training script for resnet model.

  Args:
   loss: tensor float32 of the cross entropy + regularization losses.
   params: dictionary of params for training
   global_step: int representing the global step.

  Returns:
    host_call: summary tensors to be computed at each training step.
    train_op: the optimization term.
  """

  num_train_images = params['num_train_images']

  fraction_images = (num_train_images / params['num_train_images'])
  train_batch_size = int(params['train_batch_size'] * fraction_images)

  steps_per_epoch = params['num_train_images'] / train_batch_size
  current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)

  params['steps_per_epoch'] = steps_per_epoch

  if params['use_checkpoint']:
    learning_rate = tf.constant(0.0002, dtype=tf.float32)
  else:
    learning_rate = utils.learning_rate_schedule(params, current_epoch,
                                                 train_batch_size,
                                                 num_train_images)

  optimizer = tf.compat.v1.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=params['momentum'],
      use_nesterov=True)

  # UPDATE_OPS needs to be added as a dependency due to batch norm
  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops), tf.name_scope('train'):
    train_op = optimizer.minimize(loss, global_step)

  if params['num_workers'] > 0:
    optimizer = tf.compat.v1.train.SyncReplicasOptimizer(
        optimizer,
        replicas_to_aggregate=params['num_workers'],
        total_num_replicas=params['num_workers'])
    optimizer.make_session_run_hook(True)

  with tf.summary.create_file_writer(params['output_dir']).as_default():
    with tf.summary.record_if(True):
      tf.summary.scalar('loss', loss, step=global_step)
      tf.summary.scalar('learning_rate', learning_rate, step=global_step)
      tf.summary.scalar('current_epoch', current_epoch, step=global_step)
      tf.summary.scalar('steps_per_epoch', steps_per_epoch, step=global_step)
      tf.summary.scalar(
          'weight_decay', params['weight_decay'], step=global_step)

    tf.compat.v1.summary.all_v2_summary_ops()

  return train_op


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet-50.

  Args:
    features: A dictionary with different features
    labels: A int32 batch of labels.
    mode: Specifies whether training or evaluation.
    params: Dictionary of parameters passed to the model.

  Returns:
    A TPUEstimatorSpec for the model
  """

  images = features['images_batch']
  labels = tf.reshape(features['labels_batch'], [-1])
  if params['dataset'] == 'imagenet':
    images -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype)
    images /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype)

  def build_network():
    network = resnet_model.resnet_v1_(
        resnet_depth=params['resnet_depth'],
        num_classes=params['num_label_classes'],
        data_format=FLAGS.data_format)
    return network(
        inputs=images, is_training=(mode == tf_estimator.ModeKeys.TRAIN))

  logits = build_network()

  output_dir = params['output_dir']  # pylint: disable=unused-variable
  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, params['num_label_classes'])

  cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing)

  reg_loss = 0.0
  if mode == tf_estimator.ModeKeys.TRAIN:
    if params['regularize_gradients']:
      ## if regularize_aux evaluate perceptual quality at earlier layer
      one_hot_labels = tf.one_hot(labels, params['num_label_classes'])
      reg_loss = reg.compute_reg_loss(params['regularizer'], logits, images,
                                      one_hot_labels)
      reg_loss *= params['reg_scale']

  # Add weight decay to the loss for non-batch-normalization variables.
  # Add the regularizer to optimize for gradient heatmap with higher
  # perceptual quality.
  loss = cross_entropy + reg_loss + FLAGS.weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.compat.v1.trainable_variables()
      if 'batch_normalization' not in v.name
  ])
  global_step = tf.compat.v1.train.get_global_step()
  if mode == tf_estimator.ModeKeys.TRAIN:
    train_op = train_function(loss, params, global_step)
    tf.summary.scalar('reg_loss', reg_loss, step=global_step)
    tf.summary.scalar('cross_entropy', cross_entropy, step=global_step)
  else:
    train_op = None

  eval_metrics = None
  if mode == tf_estimator.ModeKeys.EVAL:
    eval_metrics = (create_eval_metrics, [labels, logits])

  return tf_estimator.EstimatorSpec(
      mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics)


def main(argv):
  del argv  # Unused.

  params = imagenet_params
  params['dataset'] = 'imagenet'

  if FLAGS.use_checkpoint:
    params['train_steps'] = FLAGS.finetune_steps

  # tests workflow with limited number of images
  params['test_small_sample'] = False
  if FLAGS.test_workflow:
    params['train_batch_size'] = 2
    params['eval_batch_size'] = 2
    params['batch_size'] = 2
    params['num_train_images'] = 10
    params['num_eval_images'] = 10
    params['num_val_images'] = 10
    params['train_steps'] = 4
    params['test_small_sample'] = True

  # we pass the updated eval and train string to the params dictionary.
  params['use_tpu'] = FLAGS.use_tpu
  params['num_cores'] = FLAGS.num_cores
  params['sloppy_shuffle'] = True
  params['momentum'] = FLAGS.momentum
  params['mode'] = FLAGS.mode
  params['num_train_images'] = params['num_train_images']
  if FLAGS.mode == 'eval':
    params['batch_size'] = params['eval_batch_size']
  if FLAGS.mode == 'train':
    params['batch_size'] = params['train_batch_size']
  params['base_learning_rate'] = params['base_learning_rate']
  params['num_workers'] = FLAGS.num_workers
  params['regularizer'] = FLAGS.regularizer
  params['regularize_gradients'] = FLAGS.regularize_gradients
  params['reg_scale'] = FLAGS.reg_scale
  params['use_checkpoint'] = FLAGS.use_checkpoint
  params['visualize_image'] = False

  # for the constrained explanations, we will want to use a clean checkpoint
  # loaded and constrained for a few steps.
  if FLAGS.use_checkpoint:
    reg_scale = FLAGS.reg_scale

    # create a new save pathway to log the hyperparameters used during
    # regularization stage.
    if FLAGS.regularize_gradients:
      params['output_dir'] = os.path.join(FLAGS.output_dir, 'regularizer',
                                          FLAGS.regularizer, FLAGS.noise,
                                          str(reg_scale),
                                          str(FLAGS.multiple_image_std),
                                          str(FLAGS.finetune_steps))
    else:
      params['output_dir'] = os.path.join(FLAGS.output_dir, 'regularizer',
                                          'baseline', str(0.0))
    if not tf.io.gfile.isdir(params['output_dir']):
      tf.io.gfile.makedirs(params['output_dir'])
    warm_start_from = tf.train.latest_checkpoint(FLAGS.ckpt_directory)
  else:
    warm_start_from = None
    params['output_dir'] = FLAGS.output_dir

  if FLAGS.mode == 'train':
    params['batch_size'] = params['train_batch_size']
    params['data_dir'] = params['train_directory']
  else:
    params['batch_size'] = params['eval_batch_size']
    params['data_dir'] = params['eval_directory']

  run_config = tf_estimator.RunConfig(
      save_summary_steps=300,
      save_checkpoints_steps=1000,
      log_step_count_steps=100)

  classifier = tf_estimator.Estimator(
      model_fn=resnet_model_fn,
      config=run_config,
      params=params,
      warm_start_from=warm_start_from)

  eval_steps = params['num_eval_images'] // params['eval_batch_size']

  if FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(params['output_dir']):
      tf.logging.info('Starting to evaluate.')
      try:
        classifier.evaluate(
            input_fn=data_helper.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= params['train_steps']:
          tf.logging.info('Evaluation finished')
          break
      except tf.errors.NotFoundError:
        tf.logging.info('Checkpoint was not found, skipping checkpoint.')

  else:
    if FLAGS.mode == 'train':
      print('start training...')
      classifier.train(
          input_fn=data_helper.input_fn, max_steps=params['train_steps'])


if __name__ == '__main__':
  app.run(main)
