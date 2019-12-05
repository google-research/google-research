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

"""Finetuning the pre-trained model on the target set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from inputs import data_input
import model
from models import resnet_params
import tensorflow as tf
from tensorflow.python.estimator import estimator  # pylint: disable=g-direct-tensorflow-import
from utils import model_utils
from tensorflow.contrib import learn as contrib_learn
from tensorflow.contrib import training as contrib_training

flags.DEFINE_integer('pre_train_steps', 100, help=('pretrain steps'))
flags.DEFINE_integer('finetune_steps', 100, help=('pretrain steps'))
flags.DEFINE_integer('ctrl_steps', 100, help=('pretrain steps'))

flags.DEFINE_string(
    'param_file',
    None,
    help=(
        'Base set of model parameters to use with this model. To see '
        'documentation on the parameters, see the docstring in resnet_params.'))
flags.DEFINE_multi_string(
    'param_overrides',
    None,
    help=('Model parameter overrides for this model. For example, if '
          'experimenting with larger numbers of train_steps, a possible value '
          'is --param_overrides=train_steps=28152. If you have a collection of '
          'parameters that make sense to use together repeatedly, consider '
          'extending resnet_params.param_sets_table.'))
flags.DEFINE_string(
    'data_dir',
    '',
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))
flags.DEFINE_string(
    'model_dir',
    None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))
flags.DEFINE_string(
    'mode',
    'train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')
flags.DEFINE_integer(
    'log_step_count_steps', 64, 'The number of steps at '
    'which the global step information is logged.')
flags.DEFINE_string('model_name', 'resnet',
                    'Serving model name used for the model server.')
flags.DEFINE_multi_integer(
    'inference_batch_sizes', [8],
    'Known inference batch sizes used to warm up for each core.')
flags.DEFINE_integer('use_cosine_lr', 0, '')
flags.DEFINE_integer('start_finetune_step', 0, '')
flags.DEFINE_string(
    'master', '',
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')
flags.DEFINE_integer('train_steps', 100, 'Number of steps for training.')
flags.DEFINE_float('target_base_learning_rate', 100,
                   'Target base learning rate.')
flags.DEFINE_string('model_type', 'resnet', 'Model type.')
flags.DEFINE_string('optimizer', 'momentum', 'The optimizer to use.')
flags.DEFINE_integer('target_batch_size', 1024,
                     'The batch size for the target dataset.')
flags.DEFINE_bool('moving_average', True, 'Whether to do moving average.')
flags.DEFINE_float('weight_decay', 0.00004, 'The value for weight decay.')
flags.DEFINE_string('dataset_split', 'train',
                    'Dataset split used at this stage.')
flags.DEFINE_string('target_dataset', '', 'Name of the target dataset.')

FLAGS = flags.FLAGS

MOVING_AVERAGE_DECAY = 0.9
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.


def rampcosine():
  """Cosine decay."""
  cur_finetune_step = tf.train.get_global_step()
  current_step = cur_finetune_step - FLAGS.start_finetune_step
  scaled_lr = FLAGS.target_base_learning_rate * (
      FLAGS.target_batch_size / 256.0)
  total_train_steps = FLAGS.train_steps
  target_learning_rate = scaled_lr

  return model_utils.multi_stage_lr(
      total_train_steps,
      target_learning_rate,
      current_step,
  )


def get_model_fn(run_config):
  """Returns the model definition."""
  bird_num_classes = data_input.num_classes_map[FLAGS.target_dataset]

  def resnet_model_fn(features, labels, mode, params):
    """Returns the model function."""
    global_step = tf.train.get_global_step()

    feature = features['feature']
    labels = labels['label']
    one_hot_labels = model_utils.get_label(
        labels, params, bird_num_classes, batch_size=params['batch_size'])

    def get_logits():
      """Return the logits."""
      end_points, aux_logits = None, None
      if FLAGS.model_type == 'resnet':
        avg_pool = model.resnet_v1_model(feature, labels, mode, params)
      else:
        assert False
      name = 'final_dense_dst'
      with tf.variable_scope('target_CLS'):
        logits = tf.layers.dense(
            inputs=avg_pool,
            units=bird_num_classes,
            kernel_initializer=tf.random_normal_initializer(stddev=.01),
            name=name)
        if end_points is not None:
          aux_pool = end_points['AuxLogits_Pool']
          aux_logits = tf.layers.dense(
              inputs=aux_pool,
              units=bird_num_classes,
              kernel_initializer=tf.random_normal_initializer(stddev=.001),
              name='Aux{}'.format(name))
      return logits, aux_logits, end_points

    logits, _, _ = get_logits()
    logits = tf.cast(logits, tf.float32)

    if FLAGS.model_type == 'resnet':
      dst_loss = tf.losses.softmax_cross_entropy(
          logits=logits,
          weights=1.,
          onehot_labels=one_hot_labels,
          label_smoothing=params['label_smoothing'])
      dst_l2_loss = FLAGS.weight_decay * tf.add_n([
          tf.nn.l2_loss(v)
          for v in tf.trainable_variables()
          if 'batch_normalization' not in v.name
      ])
      loss = dst_loss + dst_l2_loss

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      cur_finetune_step = tf.train.get_global_step()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        if FLAGS.model_type == 'resnet':
          finetune_learning_rate = rampcosine()
        else:
          finetune_learning_rate = rampcosine()
        if FLAGS.optimizer == 'momentum':
          optimizer = tf.train.MomentumOptimizer(
              learning_rate=finetune_learning_rate,
              momentum=params['momentum'],
              use_nesterov=True)
        elif FLAGS.optimizer == 'RMS':
          optimizer = tf.train.RMSPropOptimizer(
              finetune_learning_rate,
              RMSPROP_DECAY,
              momentum=RMSPROP_MOMENTUM,
              epsilon=RMSPROP_EPSILON)
        elif FLAGS.optimizer == 'adam':
          optimizer = tf.train.AdamOptimizer(finetune_learning_rate)

        optimizer = tf.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=FLAGS.sync_replicas,
            total_num_replicas=run_config.num_worker_replicas)
        train_op = contrib_training.create_train_op(loss, optimizer)
        with tf.variable_scope('finetune'):
          train_op = optimizer.minimize(loss, cur_finetune_step)
        if FLAGS.moving_average:
          ema = tf.train.ExponentialMovingAverage(
              decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
          variables_to_average = (
              tf.trainable_variables() + tf.moving_average_variables())
          with tf.control_dependencies([train_op]):
            with tf.name_scope('moving_average'):
              train_op = ema.apply(variables_to_average)
    else:
      train_op = None

    batch_size = params['batch_size']  # pylint: disable=unused-variable
    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = model_utils.metric_fn(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.control_dependencies([train_op]):
        tf.summary.scalar('classifier/finetune_loss', loss)
        tf.summary.scalar('classifier/finetune_lr', finetune_learning_rate)
    else:
      train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
    )

  return resnet_model_fn


def main(unused_argv):
  params = resnet_params.from_file(FLAGS.param_file)
  params = resnet_params.override(params, FLAGS.param_overrides)

  params['batch_size'] = FLAGS.target_batch_size

  resnet_params.log_hparams_to_model_dir(params, FLAGS.model_dir)
  print('Model params: {}'.format(params))

  if params['use_async_checkpointing']:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = FLAGS.pre_train_steps + FLAGS.finetune_steps + FLAGS.ctrl_steps
    save_checkpoints_steps = max(1000, params['iterations_per_loop'])
  run_config_args = {
      'model_dir': FLAGS.model_dir,
      'save_checkpoints_steps': save_checkpoints_steps,
      'log_step_count_steps': FLAGS.log_step_count_steps,
      'keep_checkpoint_max': 100,
  }

  run_config_args['master'] = FLAGS.master
  config = contrib_learn.RunConfig(**run_config_args)

  resnet_classifier = tf.estimator.Estimator(
      get_model_fn(config), config=config)

  use_bfloat16 = params['precision'] == 'bfloat16'

  def _merge_datasets(train_batch):
    feature, label = train_batch
    features = {
        'feature': feature,
    }
    labels = {
        'label': label,
    }
    return (features, labels)

  def make_input_dataset(params):
    """Returns input dataset."""
    finetune_dataset = data_input.ImageNetInput(
        dataset_name=FLAGS.target_dataset,
        num_classes=data_input.num_classes_map[FLAGS.target_dataset],
        task_id=1,
        is_training=True,
        data_dir=FLAGS.data_dir,
        dataset_split=FLAGS.dataset_split,
        transpose_input=params['transpose_input'],
        cache=False,
        image_size=params['image_size'],
        num_parallel_calls=params['num_parallel_calls'],
        use_bfloat16=use_bfloat16)
    finetune_data = finetune_dataset.input_fn(params)
    dataset = tf.data.Dataset.zip((finetune_data,))
    dataset = dataset.map(_merge_datasets)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

  # pylint: disable=protected-access
  current_step = estimator._load_global_step_from_checkpoint_dir(
      FLAGS.model_dir)

  train_steps = FLAGS.train_steps
  while current_step < train_steps:
    next_checkpoint = train_steps
    resnet_classifier.train(
        input_fn=make_input_dataset, max_steps=next_checkpoint)
    current_step = next_checkpoint


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
