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

"""This script trains a ResNet model that implements various pruning methods.

Code partially branched out from
third_party/cloud_tpu/models/resnet/resnet_main.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


from state_of_sparsity.sparse_rn50 import imagenet_input
from state_of_sparsity.sparse_rn50 import resnet_model
from state_of_sparsity.sparse_rn50 import utils
from tensorflow.contrib import estimator as contrib_estimator
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.training.python.training import evaluation

flags.DEFINE_string(
    'precision',
    default='float32',
    help=('Precision to use; one of: {bfloat16, float32}'))
flags.DEFINE_integer('num_workers', 1, 'Number of training workers.')
flags.DEFINE_float(
    'base_learning_rate',
    default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum',
    default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))
flags.DEFINE_integer('ps_task', 0,
                     'Task id of the replica running the training.')
flags.DEFINE_float(
    'weight_decay',
    default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))
flags.DEFINE_string('master', '', 'Master job.')
flags.DEFINE_string('tpu_job_name', None, 'For complicated TensorFlowFlock')
flags.DEFINE_integer(
    'steps_per_checkpoint',
    default=1000,
    help=('Controls how often checkpoints are generated. More steps per '
          'checkpoint = higher utilization of TPU and generally higher '
          'steps/sec'))
flags.DEFINE_integer(
    'keep_checkpoint_max', default=0, help=('Number of checkpoints to hold.'))
flags.DEFINE_string(
    'data_directory',
    None,
    'The location of the sstable used for training.')
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
flags.DEFINE_integer(
    'resnet_depth',
    default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))
flags.DEFINE_float('label_smoothing', 0.1,
                   'Relax confidence in the labels by (1-label_smoothing).')
flags.DEFINE_integer(
    'train_steps',
    default=2,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))
flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')
flags.DEFINE_integer(
    'eval_batch_size', default=1000, help='Batch size for evaluation.')
flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')
flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')
flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')
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
flags.DEFINE_string('output_dir', '/tmp/imagenet/',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_integer(
    'checkpoint_step',
    128000,
    'Checkpoint step to evaluate for mode=\'eval_once\'')
flags.DEFINE_string(
    'mode',
    default='train',
    help='One of {"eval_once", "train_and_eval", "train", "eval"}.')
flags.DEFINE_integer('export_model_freq', 2502,
                     'The rate at which estimator exports the model.')

# pruning flags
flags.DEFINE_float('end_sparsity', 0.9,
                   'Target sparsity desired by end of training.')
flags.DEFINE_integer('sparsity_begin_step', 5000, 'Step to begin pruning at.')
flags.DEFINE_integer('sparsity_end_step', 8000, 'Step to end pruning at.')
flags.DEFINE_integer('pruning_frequency', 2000,
                     'Step interval between pruning.')
flags.DEFINE_enum(
    'pruning_method', 'baseline',
    ('baseline', 'threshold', 'variational_dropout', 'l0_regularization'),
    'Method used for pruning. baseline means no pruning is used.')
flags.DEFINE_enum(
    'init_method', 'baseline', ('baseline', 'sparse'),
    'Method for initialization.  If sparse and pruning_method=scratch, then'
    ' use initializers that take into account starting sparsity.')
flags.DEFINE_float('reg_scalar', 0., 'Weight placed on variational dropout'
                   'regularizer.')
flags.DEFINE_float('clip_log_alpha', 8.0, 'Threshold for clipping log alpha.')
flags.DEFINE_float('log_alpha_threshold', 3.0,
                   'Threshold for thresholding log alpha during eval.')
flags.DEFINE_bool(
    'is_warm_up',
    default=True,
    help=('Boolean for whether to scale weight of regularizer.'))
flags.DEFINE_float(
    'width', -1., 'Multiplier for the number of channels in each layer.')
# first and last layer are somewhat special.  First layer has almost no
# parameters, but 3% of the total flops.  Last layer has only .05% of the total
# flops but 10% of the total parameters.  Depending on whether the goal is max
# compression or max acceleration, pruning goals will be different.
flags.DEFINE_bool('prune_first_layer', True,
                  'Whether or not to apply sparsification to the first layer')
flags.DEFINE_bool('prune_last_layer', True,
                  'Whether or not to apply sparsification to the last layer')
flags.DEFINE_float(
    'first_layer_sparsity', -1.,
    'Sparsity to use for the first layer.  Overrides default of end_sparsity.')
flags.DEFINE_float(
    'last_layer_sparsity', -1.,
    'Sparsity to use for the last layer. Overrides default of end_sparsity.')
flags.DEFINE_string(
    'load_mask_dir', '',
    'Directory of a trained model from which to load only the mask')
flags.DEFINE_string(
    'initial_value_checkpoint', '',
    'Directory of a model from which to load only the parameters')

FLAGS = flags.FLAGS

# Learning rate schedule (multiplier, epoch to start) tuples
LR_SCHEDULE = [(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)]

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# TODO(shooker): verify hyperparameter defaults once code is stabilized.


def lr_schedule(current_epoch):
  """Computes learning rate schedule."""
  scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

  decay_rate = (
      scaled_lr * LR_SCHEDULE[0][0] * current_epoch / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)
  return decay_rate


def train_function(pruning_method, loss, output_dir, use_tpu):
  """Training script for resnet model.

  Args:
   pruning_method: string indicating pruning method used to compress model.
   loss: tensor float32 of the cross entropy + regularization losses.
   output_dir: string tensor indicating the directory to save summaries.
   use_tpu: boolean indicating whether to run script on a tpu.

  Returns:
    host_call: summary tensors to be computed at each training step.
    train_op: the optimization term.
  """

  global_step = tf.train.get_global_step()

  steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
  current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
  learning_rate = lr_schedule(current_epoch)
  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=True)

  if use_tpu:
    # use CrossShardOptimizer when using TPU.
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

  # UPDATE_OPS needs to be added as a dependency due to batch norm
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops), tf.name_scope('train'):
    train_op = optimizer.minimize(loss, global_step)

  if not use_tpu:
    if FLAGS.num_workers > 0:
      optimizer = tf.train.SyncReplicasOptimizer(
          optimizer,
          replicas_to_aggregate=FLAGS.num_workers,
          total_num_replicas=FLAGS.num_workers)
      optimizer.make_session_run_hook(True)

  metrics = {
      'global_step': tf.train.get_or_create_global_step(),
      'loss': loss,
      'learning_rate': learning_rate,
      'current_epoch': current_epoch
  }

  if pruning_method == 'threshold':
    # construct the necessary hparams string from the FLAGS
    hparams_string = ('begin_pruning_step={0},'
                      'sparsity_function_begin_step={0},'
                      'end_pruning_step={1},'
                      'sparsity_function_end_step={1},'
                      'target_sparsity={2},'
                      'pruning_frequency={3},'
                      'threshold_decay=0,'
                      'use_tpu={4}'.format(
                          FLAGS.sparsity_begin_step,
                          FLAGS.sparsity_end_step,
                          FLAGS.end_sparsity,
                          FLAGS.pruning_frequency,
                          FLAGS.use_tpu,
                      ))

    # Parse pruning hyperparameters
    pruning_hparams = pruning.get_pruning_hparams().parse(hparams_string)

    # The first layer has so few parameters, we don't need to prune it, and
    # pruning it a higher sparsity levels has very negative effects.
    if FLAGS.prune_first_layer and FLAGS.first_layer_sparsity >= 0.:
      pruning_hparams.set_hparam(
          'weight_sparsity_map',
          ['resnet_model/initial_conv:%f' % FLAGS.first_layer_sparsity])
    if FLAGS.prune_last_layer and FLAGS.last_layer_sparsity >= 0:
      pruning_hparams.set_hparam(
          'weight_sparsity_map',
          ['resnet_model/final_dense:%f' % FLAGS.last_layer_sparsity])

    # Create a pruning object using the pruning hyperparameters
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

    # We override the train op to also update the mask.
    with tf.control_dependencies([train_op]):
      train_op = pruning_obj.conditional_mask_update_op()

    masks = pruning.get_masks()
    metrics.update(utils.mask_summaries(masks))
  elif pruning_method == 'scratch':
    masks = pruning.get_masks()
    # make sure the masks have the sparsity we expect and that it doesn't change
    metrics.update(utils.mask_summaries(masks))
  elif pruning_method == 'variational_dropout':
    masks = utils.add_vd_pruning_summaries(threshold=FLAGS.log_alpha_threshold)
    metrics.update(masks)
  elif pruning_method == 'l0_regularization':
    summaries = utils.add_l0_summaries()
    metrics.update(summaries)
  elif pruning_method == 'baseline':
    pass
  else:
    raise ValueError('Unsupported pruning method', FLAGS.pruning_method)

  host_call = (functools.partial(utils.host_call_fn, output_dir),
               utils.format_tensors(metrics))

  return host_call, train_op


def resnet_model_fn_w_pruning(features, labels, mode, params):
  """The model_fn for ResNet-50 with pruning.

  Args:
    features: A float32 batch of images.
    labels: A int32 batch of labels.
    mode: Specifies whether training or evaluation.
    params: Dictionary of parameters passed to the model.

  Returns:
    A TPUEstimatorSpec for the model
  """

  width = 1. if FLAGS.width <= 0 else FLAGS.width
  if isinstance(features, dict):
    features = features['feature']

  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input  # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  if FLAGS.transpose_input and mode != tf_estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  pruning_method = params['pruning_method']
  use_tpu = params['use_tpu']
  log_alpha_threshold = params['log_alpha_threshold']

  def build_network():
    """Construct the network in the graph."""
    model_pruning_method = pruning_method
    if pruning_method == 'scratch':
      model_pruning_method = 'threshold'

    network = resnet_model.resnet_v1_(
        resnet_depth=FLAGS.resnet_depth,
        num_classes=FLAGS.num_label_classes,
        # we need to construct the model with the pruning masks, but they won't
        # be updated if we're doing scratch training
        pruning_method=model_pruning_method,
        init_method=FLAGS.init_method,
        width=width,
        prune_first_layer=FLAGS.prune_first_layer,
        prune_last_layer=FLAGS.prune_last_layer,
        data_format=FLAGS.data_format,
        end_sparsity=FLAGS.end_sparsity,
        clip_log_alpha=FLAGS.clip_log_alpha,
        log_alpha_threshold=log_alpha_threshold,
        weight_decay=FLAGS.weight_decay)
    return network(
        inputs=features, is_training=(mode == tf_estimator.ModeKeys.TRAIN))

  if FLAGS.precision == 'bfloat16':
    with contrib_tpu.bfloat16_scope():
      logits = build_network()
    logits = tf.cast(logits, tf.float32)
  elif FLAGS.precision == 'float32':
    logits = build_network()

  if mode == tf_estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf_estimator.export.PredictOutput(predictions)
        })

  output_dir = params['output_dir']  # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)

  # make sure we reuse the same label smoothing parameter is we're doing
  # scratch / lottery ticket experiments.
  label_smoothing = FLAGS.label_smoothing
  if FLAGS.pruning_method == 'scratch':
    label_smoothing = float(FLAGS.load_mask_dir.split('/')[15])
  loss = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=label_smoothing)
  # Add regularization loss term
  loss += tf.losses.get_regularization_loss()

  if pruning_method == 'variational_dropout':
    reg_loss = utils.variational_dropout_dkl_loss(
        reg_scalar=FLAGS.reg_scalar,
        start_reg_ramp_up=FLAGS.sparsity_begin_step,
        end_reg_ramp_up=FLAGS.sparsity_end_step,
        warm_up=FLAGS.is_warm_up,
        use_tpu=use_tpu)
    loss += reg_loss
    tf.losses.add_loss(reg_loss, loss_collection=tf.GraphKeys.LOSSES)
  elif pruning_method == 'l0_regularization':
    reg_loss = utils.l0_regularization_loss(
        reg_scalar=FLAGS.reg_scalar,
        start_reg_ramp_up=FLAGS.sparsity_begin_step,
        end_reg_ramp_up=FLAGS.sparsity_end_step,
        warm_up=FLAGS.is_warm_up,
        use_tpu=use_tpu)
    loss += reg_loss
    tf.losses.add_loss(reg_loss, loss_collection=tf.GraphKeys.LOSSES)

  host_call = None
  if mode == tf_estimator.ModeKeys.TRAIN:
    host_call, train_op = train_function(pruning_method, loss, output_dir,
                                         use_tpu)

  else:
    train_op = None

  eval_metrics = None
  if mode == tf_estimator.ModeKeys.EVAL:

    def metric_fn(labels, logits):
      """Calculate eval metrics."""
      logging.info('In metric function')
      eval_metrics = {}
      predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      eval_metrics['top_5_eval_accuracy'] = tf.metrics.mean(in_top_5)
      eval_metrics['eval_accuracy'] = tf.metrics.accuracy(
          labels=labels, predictions=predictions)

      return eval_metrics

    def vd_metric_fn(labels, logits, global_sparsity):
      eval_metrics = metric_fn(labels, logits)
      eval_metrics['global_sparsity'] = tf.metrics.mean(global_sparsity)
      return eval_metrics

    tensors = [labels, logits]
    metric_function = metric_fn

    if FLAGS.pruning_method == 'variational_dropout':
      batch_size = labels.shape[0]
      ones = tf.ones([batch_size, 1])
      mask_metrics = utils.add_vd_pruning_summaries(
          threshold=FLAGS.log_alpha_threshold)
      tensors.append(mask_metrics['global_sparsity'] * ones)
      metric_function = vd_metric_fn

    eval_metrics = (metric_function, tensors)

  # define a custom scaffold function to enable initializing the mask from an
  # already trained checkpoint.
  def initialize_mask_from_ckpt(ckpt_path):
    """Load mask from an existing checkpoint."""
    model_dir = FLAGS.output_dir
    already_has_ckpt = model_dir and tf.train.latest_checkpoint(
        model_dir) is not None
    if already_has_ckpt:
      tf.logging.info(
          'Training already started on this model, not loading masks from'
          'previously trained model')
      return

    reader = tf.train.NewCheckpointReader(ckpt_path)
    mask_names = reader.get_variable_to_shape_map().keys()
    mask_names = [x for x in mask_names if x.endswith('mask')]

    variable_map = {}
    for var in tf.global_variables():
      var_name = var.name.split(':')[0]
      if var_name in mask_names:
        tf.logging.info('Loading mask variable from checkpoint: %s', var_name)
        variable_map[var_name] = var
      elif 'mask' in var_name:
        tf.logging.info('Cannot find mask variable in checkpoint, skipping: %s',
                        var_name)
    tf.train.init_from_checkpoint(ckpt_path, variable_map)

  def initialize_parameters_from_ckpt(ckpt_path):
    """Load parameters from an existing checkpoint."""
    model_dir = FLAGS.output_dir
    already_has_ckpt = model_dir and tf.train.latest_checkpoint(
        model_dir) is not None
    if already_has_ckpt:
      tf.logging.info(
          'Training already started on this model, not loading masks from'
          'previously trained model')
      return

    reader = tf.train.NewCheckpointReader(ckpt_path)
    param_names = reader.get_variable_to_shape_map().keys()
    param_names = [x for x in param_names if not x.endswith('mask')]

    variable_map = {}
    for var in tf.global_variables():
      var_name = var.name.split(':')[0]
      if var_name in param_names:
        tf.logging.info('Loading parameter variable from checkpoint: %s',
                        var_name)
        variable_map[var_name] = var
      elif 'mask' not in var_name:
        tf.logging.info(
            'Cannot find parameter variable in checkpoint, skipping: %s',
            var_name)
    tf.train.init_from_checkpoint(ckpt_path, variable_map)

  if FLAGS.pruning_method == 'scratch':
    if FLAGS.load_mask_dir:

      def scaffold_fn():
        initialize_mask_from_ckpt(FLAGS.load_mask_dir)
        if FLAGS.initial_value_checkpoint:
          initialize_parameters_from_ckpt(FLAGS.initial_value_checkpoint)
        return tf.train.Scaffold()
    else:
      raise ValueError('Must supply a mask directory to use scratch method')
  else:
    scaffold_fn = None

  return contrib_tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


class ExportModelHook(tf.train.SessionRunHook):
  """Train hooks called after each session run for exporting the model."""

  def __init__(self, classifier, export_dir):
    self.classifier = classifier
    self.global_step = None
    self.export_dir = export_dir
    self.last_export = 0
    self.supervised_input_receiver_fn = (
        contrib_estimator.build_raw_supervised_input_receiver_fn(
            {
                'feature':
                    tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            }, tf.placeholder(dtype=tf.int32, shape=[None])))

  def begin(self):
    self.global_step = tf.train.get_or_create_global_step()

  def after_run(self, run_context, run_values):
    # export saved model
    global_step = run_context.session.run(self.global_step)

    if global_step - self.last_export >= FLAGS.export_model_freq:
      tf.logging.info(
          'Export model for prediction (step={}) ...'.format(global_step))

      self.last_export = global_step
      contrib_estimator.export_all_saved_models(
          self.classifier, os.path.join(self.export_dir, str(global_step)), {
              tf_estimator.ModeKeys.EVAL:
                  self.supervised_input_receiver_fn,
              tf_estimator.ModeKeys.PREDICT:
                  imagenet_input.image_serving_input_fn
          })


def main(_):

  if FLAGS.pruning_method in ['threshold']:
    folder_stub = os.path.join(FLAGS.pruning_method, str(FLAGS.end_sparsity),
                               str(FLAGS.sparsity_begin_step),
                               str(FLAGS.sparsity_end_step),
                               str(FLAGS.pruning_frequency),
                               str(FLAGS.label_smoothing))
  elif FLAGS.pruning_method == 'variational_dropout':
    folder_stub = os.path.join(FLAGS.pruning_method,
                               str(FLAGS.sparsity_begin_step),
                               str(FLAGS.sparsity_end_step),
                               str(FLAGS.reg_scalar),
                               str(FLAGS.label_smoothing))
  elif FLAGS.pruning_method == 'l0_regularization':
    folder_stub = os.path.join(FLAGS.pruning_method,
                               str(FLAGS.sparsity_begin_step),
                               str(FLAGS.sparsity_end_step),
                               str(FLAGS.reg_scalar),
                               str(FLAGS.label_smoothing))
  elif FLAGS.pruning_method == 'baseline':
    folder_stub = os.path.join(FLAGS.pruning_method, str(0.0), str(0.0),
                               str(0.0), str(0.0))
  elif FLAGS.pruning_method == 'scratch':
    run_info = FLAGS.load_mask_dir.split('/')
    run_type = run_info[10]
    run_sparsity = run_info[11]
    run_begin = run_info[12]
    run_end = run_info[13]
    run_freq = run_info[14]
    run_label_smoothing = run_info[15]
    folder_stub = os.path.join(FLAGS.pruning_method, run_type, run_sparsity,
                               run_begin, run_end, run_freq,
                               run_label_smoothing, FLAGS.init_method)
  else:
    raise ValueError('Pruning method is not known %s' % (FLAGS.pruning_method))

  output_dir = os.path.join(FLAGS.output_dir, folder_stub)

  export_dir = os.path.join(output_dir, 'export_dir')

  # we pass the updated eval and train string to the params dictionary.
  params = {}
  params['output_dir'] = output_dir
  params['pruning_method'] = FLAGS.pruning_method
  params['use_tpu'] = FLAGS.use_tpu
  params['log_alpha_threshold'] = FLAGS.log_alpha_threshold

  imagenet_train, imagenet_eval = [
      imagenet_input.ImageNetInput(  # pylint: disable=g-complex-comprehension
          is_training=is_training,
          data_dir=FLAGS.data_directory,
          transpose_input=False,
          num_parallel_calls=FLAGS.num_parallel_calls,
          use_bfloat16=False) for is_training in [True, False]
  ]

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=output_dir,
      save_checkpoints_steps=FLAGS.steps_per_checkpoint,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          tpu_job_name=FLAGS.tpu_job_name))

  classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn_w_pruning,
      params=params,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  cpu_classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn_w_pruning,
      params=params,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      export_to_tpu=False,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.num_eval_images % FLAGS.eval_batch_size != 0:
    raise ValueError(
        'eval_batch_size (%d) must evenly divide num_eval_images(%d)!' %
        (FLAGS.eval_batch_size, FLAGS.num_eval_images))

  eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size

  if FLAGS.mode == 'eval_once':
    ckpt = FLAGS.output_dir + 'model.ckpt-{}'.format(FLAGS.checkpoint_step)
    classifier.evaluate(
        input_fn=imagenet_eval.input_fn,
        steps=eval_steps,
        checkpoint_path=ckpt,
        name='{0}'.format(int(FLAGS.log_alpha_threshold * 10)))
  elif FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(output_dir):
      print('Starting to evaluate.')
      try:
        classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt,
            name='{0}'.format(int(FLAGS.log_alpha_threshold * 10)))
        # Terminate eval job when final checkpoint is reached
        global_step = int(os.path.basename(ckpt).split('-')[1])
        if global_step >= FLAGS.train_steps:
          print('Evaluation finished after training step %d' % global_step)
          break

      except tf.errors.NotFoundError:
        logging('Checkpoint no longer exists,skipping checkpoint.')

  else:
    global_step = tf_estimator._load_global_step_from_checkpoint_dir(output_dir)  # pylint: disable=protected-access,line-too-long
    # Session run hooks to export model for prediction
    export_hook = ExportModelHook(cpu_classifier, export_dir)
    hooks = [export_hook]

    if FLAGS.mode == 'train':
      print('start training...')
      classifier.train(
          input_fn=imagenet_train.input_fn,
          hooks=hooks,
          max_steps=FLAGS.train_steps)
    else:
      assert FLAGS.mode == 'train_and_eval'
      print('start training and eval...')
      while global_step < FLAGS.train_steps:
        next_checkpoint = min(global_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        global_step = next_checkpoint
        logging('Completed training up to step :', global_step)
        classifier.evaluate(input_fn=imagenet_eval.input_fn, steps=eval_steps)


if __name__ == '__main__':
  app.run(main)
