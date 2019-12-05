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

"""The runners."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np

import tensorflow as tf
from capsule_em import model as f_model
from capsule_em.mnist \
  import mnist_record
from capsule_em.norb \
  import norb_record
from tensorflow.contrib import tfprof as contrib_tfprof
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_prime_capsules', 32,
                            'Number of first layer capsules.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_integer('routing_iteration', 3,
                            'Number of iterations for softmax routing')
tf.app.flags.DEFINE_float(
    'routing_rate', 1,
    'ratio for combining routing logits and routing feedback')
tf.app.flags.DEFINE_float('decay_rate', 0.96, 'ratio for learning rate decay')
tf.app.flags.DEFINE_integer('decay_steps', 20000,
                            'number of steps for learning rate decay')
tf.app.flags.DEFINE_bool('normalize_kernels', False,
                         'Normalize the capsule weight kernels')
tf.app.flags.DEFINE_integer('num_second_atoms', 16,
                            'number of capsule atoms for the second layer')
tf.app.flags.DEFINE_integer('num_primary_atoms', 16,
                            'number of capsule atoms for the first layer')
tf.app.flags.DEFINE_integer('num_start_conv', 32,
                            'number of channels for the start layer')
tf.app.flags.DEFINE_integer('kernel_size', 5,
                            'kernel size for the start layer.')
tf.app.flags.DEFINE_integer(
    'routing_iteration_prime', 1,
    'number of routing iterations for primary capsules.')
tf.app.flags.DEFINE_integer('max_steps', 2000000,
                            'Number of steps to run trainer.')
tf.app.flags.DEFINE_string('data_dir', '/datasets/mnist/',
                           'Directory for storing input data')
tf.app.flags.DEFINE_string('summary_dir',
                           '/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                           'Summaries log directory')
tf.app.flags.DEFINE_bool('train', True, 'train or test.')
tf.app.flags.DEFINE_integer(
    'checkpoint_steps', 1500,
    'number of steps before saving a training checkpoint.')
tf.app.flags.DEFINE_bool('verbose_image', False, 'whether to show images.')
tf.app.flags.DEFINE_bool('multi', True,
                         'whether to use multiple digit dataset.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'whether to evaluate once on the ckpnt file.')
tf.app.flags.DEFINE_integer('eval_size', 24300,
                            'number of examples to evaluate.')
tf.app.flags.DEFINE_string(
    'ckpnt',
    '/tmp/tensorflow/mnist/logs/mnist_with_summaries/train/model.ckpnt',
    'The checkpoint to load and evaluate once.')
tf.app.flags.DEFINE_integer('keep_ckpt', 5, 'number of examples to evaluate.')
tf.app.flags.DEFINE_bool(
    'clip_lr', False, 'whether to clip learning rate to not go bellow 1e-5.')
tf.app.flags.DEFINE_integer('stride_1', 2,
                            'stride for the first convolutinal layer.')
tf.app.flags.DEFINE_integer('kernel_2', 9,
                            'kernel size for the secon convolutinal layer.')
tf.app.flags.DEFINE_integer('stride_2', 2,
                            'stride for the second convolutinal layer.')
tf.app.flags.DEFINE_string('padding', 'VALID',
                           'the padding method for conv layers.')
tf.app.flags.DEFINE_integer('extra_caps', 2, 'number of extra conv capsules.')
tf.app.flags.DEFINE_string('caps_dims', '32,32',
                           'output dim for extra conv capsules.')
tf.app.flags.DEFINE_string('caps_strides', '2,1',
                           'stride for extra conv capsules.')
tf.app.flags.DEFINE_string('caps_kernels', '3,3',
                           'kernel size for extra conv capsuls.')
tf.app.flags.DEFINE_integer('extra_conv', 0, 'number of extra conv layers.')

tf.app.flags.DEFINE_string('conv_dims', '', 'output dim for extra conv layers.')
tf.app.flags.DEFINE_string('conv_strides', '', 'stride for extra conv layers.')
tf.app.flags.DEFINE_string('conv_kernels', '',
                           'kernel size for extra conv layers.')
tf.app.flags.DEFINE_bool('leaky', False, 'Use leaky routing.')
tf.app.flags.DEFINE_bool('staircase', False, 'Use staircase decay.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train.')
tf.app.flags.DEFINE_bool('adam', True, 'Use Adam optimizer.')
tf.app.flags.DEFINE_bool('pooling', False, 'Pooling after convolution.')
tf.app.flags.DEFINE_bool('use_caps', True, 'Use capsule layers.')
tf.app.flags.DEFINE_integer(
    'extra_fc', 512, 'number of units in the extra fc layer in no caps mode.')
tf.app.flags.DEFINE_bool('dropout', False, 'Dropout before last layer.')
tf.app.flags.DEFINE_bool('tweak', False, 'During eval recons from tweaked rep.')
tf.app.flags.DEFINE_bool('softmax', False, 'softmax loss in no caps.')
tf.app.flags.DEFINE_bool('c_dropout', False, 'dropout after conv capsules.')
tf.app.flags.DEFINE_bool(
    'distort', True,
    'distort mnist images by cropping to 24 * 24 and rotating by 15 degrees.')
tf.app.flags.DEFINE_bool('restart', False, 'Clean train checkpoints.')
tf.app.flags.DEFINE_bool('use_em', True,
                         'If set use em capsules with em routing.')
tf.app.flags.DEFINE_float('final_beta', 0.01, 'Temperature at the sigmoid.')
tf.app.flags.DEFINE_bool('eval_ensemble', False, 'eval over aggregated logits.')
tf.app.flags.DEFINE_string('part1', 'ok', 'ok')
tf.app.flags.DEFINE_string('part2', 'ok', 'ok')
tf.app.flags.DEFINE_bool('debug', False, 'If set use tfdbg wrapper.')
tf.app.flags.DEFINE_bool('reduce_mean', False,
                         'If set normalize mean of each image.')
tf.app.flags.DEFINE_float('loss_rate', 1.0,
                          'classification to regularization rate.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size.')
tf.app.flags.DEFINE_integer('norb_pixel', 48, 'Batch size.')
tf.app.flags.DEFINE_bool('patching', True, 'If set use patching for eval.')

tf.app.flags.DEFINE_string('data_set', 'norb', 'the data set to use.')
tf.app.flags.DEFINE_string('cifar_data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('norb_data_dir', '/tmp/smallNORB/',
                           """Path to the norb data directory.""")
tf.app.flags.DEFINE_string('affnist_data_dir', '/tmp/affnist_data',
                           """Path to the affnist data directory.""")


num_classes = {
    'mnist': 10,
    'cifar10': 10,
    'mnist_multi': 10,
    'svhn': 10,
    'affnist': 10,
    'expanded_mnist': 10,
    'norb': 5,
}


def get_features(train, total_batch):
  """Return batched inputs."""
  print(FLAGS.data_set)
  batch_size = total_batch // max(1, FLAGS.num_gpus)
  split = 'train' if train else 'test'
  features = []
  for i in xrange(FLAGS.num_gpus):
    with tf.device('/cpu:0'):
      with tf.name_scope('input_tower_%d' % (i)):
        if FLAGS.data_set == 'norb':
          features += [
              norb_record.inputs(
                  train_dir=FLAGS.norb_data_dir,
                  batch_size=batch_size,
                  split=split,
                  multi=FLAGS.multi,
                  image_pixel=FLAGS.norb_pixel,
                  distort=FLAGS.distort,
                  patching=FLAGS.patching,
              )
          ]
        elif FLAGS.data_set == 'affnist':
          features += [
              mnist_record.inputs(
                  train_dir=FLAGS.affnist_data_dir,
                  batch_size=batch_size,
                  split=split,
                  multi=FLAGS.multi,
                  shift=0,
                  height=40,
                  train_file='test.tfrecords')
          ]
        elif FLAGS.data_set == 'expanded_mnist':
          features += [
              mnist_record.inputs(
                  train_dir=FLAGS.data_dir,
                  batch_size=batch_size,
                  split=split,
                  multi=FLAGS.multi,
                  height=40,
                  train_file='train_6shifted_6padded_mnist.tfrecords',
                  shift=6)
          ]
        else:
          if train and not FLAGS.distort:
            shift = 2
          else:
            shift = 0
          features += [
              mnist_record.inputs(
                  train_dir=FLAGS.data_dir,
                  batch_size=batch_size,
                  split=split,
                  multi=FLAGS.multi,
                  shift=shift,
                  distort=FLAGS.distort)
          ]
  print(features)
  return features


def run_training():
  """Train."""
  with tf.Graph().as_default():
    # Input images and labels.
    features = get_features(True, FLAGS.batch_size)
    model = f_model.multi_gpu_model
    print('so far so good!')
    result = model(features)
    param_stats = contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer
        .TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
    merged = result['summary']
    train_step = result['train']
    # test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    if FLAGS.debug:
      sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='curses')
      sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver(max_to_keep=FLAGS.keep_ckpt)
    if tf.gfile.Exists(FLAGS.summary_dir + '/train'):
      ckpt = tf.train.get_checkpoint_state(FLAGS.summary_dir + '/train/')
      print(ckpt)
      if (not FLAGS.restart) and ckpt and ckpt.model_checkpoint_path:
        print('hesllo')
        saver.restore(sess, ckpt.model_checkpoint_path)
        prev_step = int(
            ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      else:
        print('what??')
        tf.gfile.DeleteRecursively(FLAGS.summary_dir + '/train')
        tf.gfile.MakeDirs(FLAGS.summary_dir + '/train')
        prev_step = 0
    else:
      tf.gfile.MakeDirs(FLAGS.summary_dir + '/train')
      prev_step = 0
    train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train',
                                         sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      for i in range(prev_step, FLAGS.max_steps):
        step += 1
        summary, _ = sess.run([merged, train_step])
        train_writer.add_summary(summary, i)
        if (i + 1) % FLAGS.checkpoint_steps == 0:
          saver.save(
              sess,
              os.path.join(FLAGS.summary_dir + '/train', 'model.ckpt'),
              global_step=i + 1)
    except tf.errors.OutOfRangeError:
      print('Done training for %d steps.' % step)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    train_writer.close()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def run_eval():
  """Evaluate on test or validation."""
  with tf.Graph().as_default():
    # Input images and labels.
    features = get_features(False, 5)
    model = f_model.multi_gpu_model
    result = model(features)
    merged = result['summary']
    correct_prediction_sum = result['correct']
    almost_correct_sum = result['almost']
    saver = tf.train.Saver()
    test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
    seen_step = -1
    time.sleep(3 * 60)
    paused = 0
    while paused < 360:
      ckpt = tf.train.get_checkpoint_state(FLAGS.summary_dir + '/train/')
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoin
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        time.sleep(2 * 60)
        paused += 2
        continue
      while seen_step == int(global_step):
        time.sleep(2 * 60)
        ckpt = tf.train.get_checkpoint_state(FLAGS.summary_dir + '/train/')
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        paused += 2
        if paused > 360:
          test_writer.close()
          return
      paused = 0

      seen_step = int(global_step)
      print(seen_step)
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      saver.restore(sess, ckpt.model_checkpoint_path)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        total_tp = 0
        total_almost = 0
        for i in range(FLAGS.eval_size // 5):
          summary_j, tp, almost = sess.run(
              [merged, correct_prediction_sum, almost_correct_sum])
          total_tp += tp
          total_almost += almost

        total_false = FLAGS.eval_size - total_tp
        total_almost_false = FLAGS.eval_size - total_almost
        summary_tp = tf.Summary.FromString(summary_j)
        summary_tp.value.add(tag='correct_prediction', simple_value=total_tp)
        summary_tp.value.add(tag='wrong_prediction', simple_value=total_false)
        summary_tp.value.add(
            tag='almost_wrong_prediction', simple_value=total_almost_false)
        test_writer.add_summary(summary_tp, global_step)
        print('write done')
      except tf.errors.OutOfRangeError:
        print('Done eval for %d steps.' % i)
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()
      # Wait for threads to finish.
      coord.join(threads)
      sess.close()
    test_writer.close()


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


def eval_ensemble(ckpnts):
  """Evaluate on an ensemble of checkpoints."""
  with tf.Graph().as_default():
    first_features = get_features(False, 100)[0]
    h = first_features['height']
    d = first_features['depth']
    features = {
        'images': tf.placeholder(tf.float32, shape=(100, d, h, h)),
        'labels': tf.placeholder(tf.float32, shape=(100, 10)),
        'recons_image': tf.placeholder(tf.float32, shape=(100, d, h, h)),
        'recons_label': tf.placeholder(tf.int32, shape=(100)),
        'height': first_features['height'],
        'depth': first_features['depth']
    }

    model = f_model.multi_gpu_model
    result = model([features])
    logits = result['logits']
    config = tf.ConfigProto(allow_soft_placement=True)
    # saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpnt))
    batch_logits = np.zeros((FLAGS.eval_size // 100, 100, 10), dtype=np.float32)
    batch_recons_label = np.zeros((FLAGS.eval_size // 100, 100),
                                  dtype=np.float32)
    batch_labels = np.zeros((FLAGS.eval_size // 100, 100, 10), dtype=np.float32)
    batch_images = np.zeros((FLAGS.eval_size // 100, 100, d, h, h),
                            dtype=np.float32)
    batch_recons_image = np.zeros((FLAGS.eval_size // 100, 100, d, h, h),
                                  dtype=np.float32)
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      for i in range(FLAGS.eval_size // 100):
        (batch_recons_label[i, Ellipsis], batch_labels[i, Ellipsis], batch_images[i, Ellipsis],
         batch_recons_image[i, Ellipsis]) = sess.run([
             first_features['recons_label'], first_features['labels'],
             first_features['images'], first_features['recons_image']
         ])
      for ckpnt in ckpnts:
        saver.restore(sess, ckpnt)
        for i in range(FLAGS.eval_size // 100):
          logits_i = sess.run(
              logits,
              feed_dict={
                  features['recons_label']: batch_recons_label[i, Ellipsis],
                  features['labels']: batch_labels[i, Ellipsis],
                  features['images']: batch_images[i, Ellipsis],
                  features['recons_image']: batch_recons_image[i, Ellipsis]
              })
          # batch_logits[i, ...] += softmax(logits_i)
          batch_logits[i, Ellipsis] += logits_i
    except tf.errors.OutOfRangeError:
      print('Done eval for %d steps.' % i)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    batch_pred = np.argmax(batch_logits, axis=2)
    total_wrong = np.sum(np.not_equal(batch_pred, batch_recons_label))
    print(total_wrong)


def eval_once(ckpnt):
  """Evaluate on one checkpoint once."""
  ptches = np.zeros((14, 14, 32, 32))
  for i in range(14):
    for j in range(14):
      ind_x = i * 2
      ind_y = j * 2
      for k in range(5):
        for h in range(5):
          ptches[i, j, ind_x + k, ind_y + h] = 1
  ptches = np.reshape(ptches, (14 * 14, 32, 32))

  with tf.Graph().as_default():
    features = get_features(False, 1)[0]
    if FLAGS.patching:
      features['images'] = features['cc_images']
      features['recons_label'] = features['cc_recons_label']
      features['labels'] = features['cc_labels']
    model = f_model.multi_gpu_model
    result = model([features])
    # merged = result['summary']
    correct_prediction_sum = result['correct']
    # almost_correct_sum = result['almost']
    # mid_act = result['mid_act']
    logits = result['logits']

    saver = tf.train.Saver()
    test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test_once')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    # saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpnt))
    saver.restore(sess, ckpnt)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    try:
      total_tp = 0
      for i in range(FLAGS.eval_size):
        #, g_ac, ac
        lb, tp, lg = sess.run([
            features['recons_label'],
            correct_prediction_sum,
            logits,
        ])
        if FLAGS.patching:
          batched_lg = np.sum(lg / np.sum(lg, axis=1, keepdims=True), axis=0)
          batch_pred = np.argmax(batched_lg)
          tp = np.equal(batch_pred, lb[0])

        total_tp += tp
      total_false = FLAGS.eval_size - total_tp
      print('false:{}, true:{}'.format(total_false, total_tp))
      # summary_tp = tf.Summary.FromString(summary_j)
      # summary_tp.value.add(tag='correct_prediction', simple_value=total_tp)
      # summary_tp.value.add(tag='wrong_prediction', simple_value=total_false)
      # summary_tp.value.add(
      #     tag='almost_wrong_prediction', simple_value=total_almost_false)
      # test_writer.add_summary(summary_tp, i + 1)
    except tf.errors.OutOfRangeError:
      print('Done eval for %d steps.' % i)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    test_writer.close()


def main(_):
  if FLAGS.eval_ensemble:
    if tf.gfile.Exists(FLAGS.summary_dir + '/test_ensemble'):
      tf.gfile.DeleteRecursively(FLAGS.summary_dir + '/test_ensemble')
    tf.gfile.MakeDirs(FLAGS.summary_dir + '/test_ensemble')
    ensem = []
    for i in range(1, 12):
      f_name = '/tmp/cifar10/{}{}{}-600000'.format(FLAGS.part1, i, FLAGS.part2)
      if tf.train.checkpoint_exists(f_name):
        ensem += [f_name]

    print(len(ensem))
    eval_ensemble(ensem)
  elif FLAGS.eval_once:
    if tf.gfile.Exists(FLAGS.summary_dir + '/test_once'):
      tf.gfile.DeleteRecursively(FLAGS.summary_dir + '/test_once')
    tf.gfile.MakeDirs(FLAGS.summary_dir + '/test_once')
    eval_once(FLAGS.ckpnt)
  elif FLAGS.train:
    run_training()
  else:
    if tf.gfile.Exists(FLAGS.summary_dir + '/test_once'):
      tf.gfile.DeleteRecursively(FLAGS.summary_dir + '/test_once')
    tf.gfile.MakeDirs(FLAGS.summary_dir + '/test_once')
    if tf.gfile.Exists(FLAGS.summary_dir + '/test'):
      tf.gfile.DeleteRecursively(FLAGS.summary_dir + '/test')
    tf.gfile.MakeDirs(FLAGS.summary_dir + '/test')
    run_eval()


if __name__ == '__main__':
  tf.app.run()
