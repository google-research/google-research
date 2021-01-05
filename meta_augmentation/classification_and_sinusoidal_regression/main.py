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

"""Entry point for the training and testing jobs."""
# pylint: disable=invalid-name
import csv
import pickle
import random

from data_generator import DataGenerator
from maml import MAML
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid',
                    'sinusoid or omniglot or miniimagenet or dclaw')
flags.DEFINE_string('expt_number', '0', '1 or 2 etc')
flags.DEFINE_string(
    'expt_name', 'intershuffle',
    'non_exclusive or intrashuffle or intershuffle or sin_noise')
flags.DEFINE_string(
    'dclaw_pn', '1',
    '1 or 2 or 3; dataset permutation number for dclaw. Does differnt train/val/test splits'
)
flags.DEFINE_integer(
    'num_classes', 5,
    'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0,
                     'number of pre-training iterations.')
flags.DEFINE_integer(
    'metatrain_iterations', 15000,
    'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25,
                     'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer(
    'update_batch_size', 5,
    'number of examples used for inner gradient update (K for K-shot learning).'
)
flags.DEFINE_float(
    'update_lr', 1e-3,
    'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1,
                     'number of inner gradient updates during training.')
flags.DEFINE_integer(
    'sine_seed', '1',
    'seed for the random operations inside sine generator; sinuosidal regression expt'
)

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer(
    'num_filters', 64,
    'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool(
    'conv', True,
    'whether or not to use a convolutional network, only applicable in some cases'
)
flags.DEFINE_bool(
    'max_pool', False,
    'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool(
    'stop_grad', False,
    'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True,
                  'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data',
                    'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True,
                  'resume training if there is a model available')
flags.DEFINE_bool('rand_init', False,
                  'initialise with the random network for testing')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1,
                     'iteration to load model (-1 for latest model)')
flags.DEFINE_bool(
    'test_set', False,
    'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer(
    'train_update_batch_size', -1,
    'number of examples used for gradient update during training (use if you want to test with a different number).'
)
flags.DEFINE_float(
    'train_update_lr', -1,
    'value of inner gradient step step during training. (use if you want to test with a different value)'
)  # 0.1 for omniglot

flags.DEFINE_bool('label_smooth', False,
                  'Whether to add label noise at training time')
flags.DEFINE_float(
    'max_smooth', 0.3,
    'How much noise to add at most. CURRENTLY IGNORED, was part of old experiment'
)


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
  """Trains constructed model."""
  SUMMARY_INTERVAL = 100
  SAVE_INTERVAL = 1000
  if FLAGS.datasource == 'sinusoid':
    PRINT_INTERVAL = 1000
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
  else:
    PRINT_INTERVAL = 200
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

  if FLAGS.log:
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string,
                                         sess.graph)
    valid_writer = tf.summary.FileWriter(
        FLAGS.logdir + '/' + exp_string + '/valid', sess.graph)
  print('Done initializing, starting training.')
  prelosses, postlosses = [], []

  validation_pre_perf, validation_post_perf = [], []

  num_classes = data_generator.num_classes  # for classification, 1 otherwise

  if FLAGS.datasource == 'sinusoid':
    best_val_loss = 1000
  for itr in range(resume_itr,
                   FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
    feed_dict = {}
    if 'generate' in dir(data_generator):
      batch_x, batch_y, amp, phase = data_generator.generate()

      if FLAGS.baseline == 'oracle':
        batch_x = np.concatenate(
            [batch_x,
             np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
        for i in range(FLAGS.meta_batch_size):
          batch_x[i, :, 1] = amp[i]
          batch_x[i, :, 2] = phase[i]

      inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
      labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
      inputb = batch_x[:, num_classes *
                       FLAGS.update_batch_size:, :]  # b used for testing
      labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
      feed_dict = {
          model.inputa: inputa,
          model.inputb: inputb,
          model.labela: labela,
          model.labelb: labelb
      }

    if itr < FLAGS.pretrain_iterations:
      input_tensors = [model.pretrain_op]
    else:
      input_tensors = [model.metatrain_op]

    if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
      input_tensors.extend([
          model.summ_op, model.total_loss1,
          model.total_losses2[FLAGS.num_updates - 1]
      ])
      if model.classification:
        input_tensors.extend([
            model.total_accuracy1,
            model.total_accuracies2[FLAGS.num_updates - 1]
        ])

    result = sess.run(input_tensors, feed_dict)

    if itr % SUMMARY_INTERVAL == 0:
      prelosses.append(result[-2])
      if FLAGS.log:
        train_writer.add_summary(result[1], itr)
      postlosses.append(result[-1])

    if (itr != 0) and itr % PRINT_INTERVAL == 0:
      if itr < FLAGS.pretrain_iterations:
        print_str = 'Pretrain Iteration ' + str(itr)
      else:
        print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
      print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(
          np.mean(postlosses))
      print(print_str)
      prelosses, postlosses = [], []

    if (itr != 0) and itr % SAVE_INTERVAL == 0:
      # To enable early-stopping, we disabling saving here and only save when
      # we hit a new best validation accuracy.
      # saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
      pass

    if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
      validation_pre_batches = []
      validation_post_batches = []
      # This takes too long to use large number of batches every eval interval.
      NUM_EVAL_BATCHES = 10
      if 'generate' not in dir(data_generator):
        feed_dict = {}
        if model.classification:
          input_tensors = [
              model.metaval_total_accuracy1,
              model.metaval_total_accuracies2[FLAGS.num_updates - 1],
              model.summ_op
          ]
        else:
          input_tensors = [
              model.metaval_total_loss1,
              model.metaval_total_losses2[FLAGS.num_updates - 1], model.summ_op
          ]
      else:
        batch_x, batch_y, amp, phase = data_generator.generate(train=False)
        inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
        inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
        labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
        labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
        feed_dict = {
            model.inputa: inputa,
            model.inputb: inputb,
            model.labela: labela,
            model.labelb: labelb,
            model.meta_lr: 0.0
        }
        if model.classification:
          input_tensors = [
              model.total_accuracy1,
              model.total_accuracies2[FLAGS.num_updates - 1]
          ]
        else:
          input_tensors = [
              model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]
          ]
      if FLAGS.datasource == 'sinusoid':
        result = sess.run(input_tensors, feed_dict)
        print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))
        if result[1] < best_val_loss:
          saver.save(sess,
                     FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
          best_val_loss = result[1]
          print('saving model')
      else:
        for eval_itr in range(NUM_EVAL_BATCHES):
          if eval_itr % 100 == 0:
            print('Evaluation iter %d' % eval_itr)
          result = sess.run(input_tensors, feed_dict)
          validation_pre_batches.append(result[0])
          validation_post_batches.append(result[1])

        validation_pre_batches = np.array(validation_pre_batches)
        validation_post_batches = np.array(validation_post_batches)
        pre_perf_mean = np.mean(validation_pre_batches, 0)
        post_perf_mean = np.mean(validation_post_batches, 0)
        pre_perf_std = np.std(validation_pre_batches, 0)
        post_perf_std = np.std(validation_post_batches, 0)

        print('Validation results step %d means: ' % itr + str(pre_perf_mean) +
              ', ' + str(post_perf_mean))
        print('Validation results step %d stds: ' % itr + str(pre_perf_std) +
              ', ' + str(post_perf_std))
        if FLAGS.log:
          # Manually create summary object for the mean value.
          summary = tf.Summary(value=[
              tf.Summary.Value(
                  tag='metaval_preupdate_average', simple_value=pre_perf_mean),
          ])
          valid_writer.add_summary(summary, itr)
          summary = tf.Summary(value=[
              tf.Summary.Value(
                  tag='metaval_postupdate_average',
                  simple_value=post_perf_mean),
          ])
          valid_writer.add_summary(summary, itr)

        validation_pre_perf.append(pre_perf_mean)
        validation_post_perf.append(post_perf_mean)
        if model.classification:
          best = max(validation_post_perf)
        else:
          best = min(validation_post_perf)
        if best == validation_post_perf[-1]:
          saver.save(
              sess,
              FLAGS.logdir + '/' + exp_string + '/model-early-stop' + str(itr))


# calculated for omniglot
NUM_TEST_POINTS = 600


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
  """Tests current model, saving results to CSV and pickle files."""
  del test_num_updates  # Unused, num updates based off constructed model graph.
  del saver  # Unused
  num_classes = data_generator.num_classes  # for classification, 1 otherwise

  np.random.seed(1)
  random.seed(1)

  metaval_accuracies = []

  for _ in range(NUM_TEST_POINTS):
    if 'generate' not in dir(data_generator):
      feed_dict = {}
      feed_dict = {model.meta_lr: 0.0}
    else:
      batch_x, batch_y, amp, phase = data_generator.generate(train=False)

      if FLAGS.baseline == 'oracle':  # NOTE - this flag is specific to sinusoid
        batch_x = np.concatenate(
            [batch_x,
             np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
        batch_x[0, :, 1] = amp[0]
        batch_x[0, :, 2] = phase[0]

      inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
      inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
      labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
      labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

      feed_dict = {
          model.inputa: inputa,
          model.inputb: inputb,
          model.labela: labela,
          model.labelb: labelb,
          model.meta_lr: 0.0
      }

    if model.classification:
      result = sess.run([model.metaval_total_accuracy1] +
                        model.metaval_total_accuracies2, feed_dict)
    else:  # this is for sinusoid
      result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)
    metaval_accuracies.append(result)

  metaval_accuracies = np.array(metaval_accuracies)
  means = np.mean(metaval_accuracies, 0)
  stds = np.std(metaval_accuracies, 0)
  ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

  print('Mean validation accuracy/loss, stddev, and confidence intervals')
  print((means, stds, ci95))

  out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
      FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
  out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
      FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
  with open(out_pkl, 'wb') as f:
    pickle.dump({'mses': metaval_accuracies}, f)
  with open(out_filename, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['update' + str(i) for i in range(len(means))])
    writer.writerow(means)
    writer.writerow(stds)
    writer.writerow(ci95)


def main():
  if FLAGS.datasource == 'sinusoid':
    if FLAGS.train:
      test_num_updates = 1
    else:
      test_num_updates = 10
  else:
    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'dclaw':
      if FLAGS.train:
        test_num_updates = 1  # eval on at least one update during training
      else:
        test_num_updates = 10
    else:
      test_num_updates = 10

  if not FLAGS.train:
    orig_meta_batch_size = FLAGS.meta_batch_size
    # always use meta batch size of 1 when testing.
    FLAGS.meta_batch_size = 1

  if FLAGS.datasource == 'sinusoid':
    data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                   FLAGS.meta_batch_size)
  else:
    if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
      assert FLAGS.meta_batch_size == 1
      assert FLAGS.update_batch_size == 1
      data_generator = DataGenerator(
          1, FLAGS.meta_batch_size)  # only use one datapoint,
    else:
      if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'dclaw':
        if FLAGS.train:
          data_generator = DataGenerator(
              FLAGS.update_batch_size + 15, FLAGS.meta_batch_size
          )  # only use one datapoint for testing to save memory
        else:
          data_generator = DataGenerator(
              FLAGS.update_batch_size * 2, FLAGS.meta_batch_size
          )  # only use one datapoint for testing to save memory
      else:
        data_generator = DataGenerator(
            FLAGS.update_batch_size * 2, FLAGS.meta_batch_size
        )  # only use one datapoint for testing to save memory

  dim_output = data_generator.dim_output
  if FLAGS.baseline == 'oracle':
    assert FLAGS.datasource == 'sinusoid'
    dim_input = 3
    FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
    FLAGS.metatrain_iterations = 0
  else:
    dim_input = data_generator.dim_input

  if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'dclaw':
    tf_data_load = True
    num_classes = data_generator.num_classes

    if FLAGS.train:  # only construct training model if needed
      random.seed(5)
      image_tensor, label_tensor = data_generator.make_data_tensor()
      inputa = tf.slice(image_tensor, [0, 0, 0],
                        [-1, num_classes * FLAGS.update_batch_size, -1])
      inputb = tf.slice(image_tensor,
                        [0, num_classes * FLAGS.update_batch_size, 0],
                        [-1, -1, -1])
      labela = tf.slice(label_tensor, [0, 0, 0],
                        [-1, num_classes * FLAGS.update_batch_size, -1])
      labelb = tf.slice(label_tensor,
                        [0, num_classes * FLAGS.update_batch_size, 0],
                        [-1, -1, -1])
      input_tensors = {
          'inputa': inputa,
          'inputb': inputb,
          'labela': labela,
          'labelb': labelb
      }

    random.seed(6)
    image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
    inputa = tf.slice(image_tensor, [0, 0, 0],
                      [-1, num_classes * FLAGS.update_batch_size, -1])
    inputb = tf.slice(image_tensor,
                      [0, num_classes * FLAGS.update_batch_size, 0],
                      [-1, -1, -1])
    labela = tf.slice(label_tensor, [0, 0, 0],
                      [-1, num_classes * FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor,
                      [0, num_classes * FLAGS.update_batch_size, 0],
                      [-1, -1, -1])
    metaval_input_tensors = {
        'inputa': inputa,
        'inputb': inputb,
        'labela': labela,
        'labelb': labelb
    }
  else:
    tf_data_load = False
    input_tensors = None

  model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
  if FLAGS.train or not tf_data_load:
    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
  if tf_data_load:
    model.construct_model(
        input_tensors=metaval_input_tensors, prefix='metaval_')
  model.summ_op = tf.summary.merge_all()

  saver = tf.train.Saver(
      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

  sess = tf.InteractiveSession()

  if not FLAGS.train:
    # change to original meta batch size when loading model.
    FLAGS.meta_batch_size = orig_meta_batch_size

  if FLAGS.train_update_batch_size == -1:
    FLAGS.train_update_batch_size = FLAGS.update_batch_size
  if FLAGS.train_update_lr == -1:
    FLAGS.train_update_lr = FLAGS.update_lr

  exp_string = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(
      FLAGS.meta_batch_size) + '.ubs_' + str(
          FLAGS.train_update_batch_size) + '.numstep' + str(
              FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

  if FLAGS.num_filters != 64:
    exp_string += 'hidden' + str(FLAGS.num_filters)
  if FLAGS.max_pool:
    exp_string += 'maxpool'
  if FLAGS.stop_grad:
    exp_string += 'stopgrad'
  if FLAGS.baseline:
    exp_string += FLAGS.baseline
  if FLAGS.norm == 'batch_norm':
    exp_string += 'batchnorm'
  elif FLAGS.norm == 'layer_norm':
    exp_string += 'layernorm'
  elif FLAGS.norm == 'None':
    exp_string += 'nonorm'
  else:
    print('Norm setting not recognized.')

  resume_itr = 0
  model_file = None

  tf.global_variables_initializer().run()
  tf.train.start_queue_runners()

  if not FLAGS.rand_init:
    if FLAGS.datasource == 'sinusoid':
      if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
          model_file = model_file[:model_file.index('model')] + 'model' + str(
              FLAGS.test_iter)
        if model_file:
          ind1 = model_file.index('model')
          resume_itr = int(model_file[ind1 + 5:])
          print('Restoring model weights from ' + model_file)
          saver.restore(sess, model_file)
    else:
      if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        search = 'model-early-stop'
        if FLAGS.test_iter > 0:
          model_file = model_file[:model_file.index(search)] + search + str(
              FLAGS.test_iter)
        if model_file:
          search = 'model-early-stop'
          ind1 = model_file.index(search)
          resume_itr = int(model_file[ind1 + len(search):])
          print('Restoring model weights from ' + model_file)
          saver.restore(sess, model_file)

  if FLAGS.train:
    train(model, saver, sess, exp_string, data_generator, resume_itr)
  else:
    test(model, saver, sess, exp_string, data_generator, test_num_updates)


if __name__ == '__main__':
  main()
