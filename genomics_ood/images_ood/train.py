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

r"""Training an pixel_cnn model.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

from absl import app
from absl import flags
import tensorflow as tf

from genomics_ood.images_ood import pixel_cnn
from genomics_ood.images_ood import utils

tf.compat.v1.disable_v2_behavior()

flags.DEFINE_string('data_dir', '/tmp/image_data',
                    'Directory to data np arrays.')
flags.DEFINE_string('out_dir', '/tmp/pixelcnn',
                    'Directory to write results and logs.')
flags.DEFINE_boolean('save_im', False, 'If True, save image to npy.')
flags.DEFINE_string('exp', 'fashion', 'cifar or fashion')

# general hyper-parameters
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('total_steps', 10, 'Max steps for training')
flags.DEFINE_integer('random_seed', 1234, 'Fixed random seed to use.')
flags.DEFINE_integer('eval_every', 10, 'Interval to evaluate model.')

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.999995, 'LR decay every step.')

flags.DEFINE_integer('num_logistic_mix', 1,
                     'Number of components in decoder mixture distribution.')
flags.DEFINE_integer('num_hierarchies', 1, 'Number of hierarchies in '
                     'Pixel CNN.')
flags.DEFINE_integer(
    'num_resnet', 5, 'Number of convoluational layers '
    'before depth changes in Pixel CNN.')
flags.DEFINE_integer('num_filters', 32, 'Number of pixel cnn filters')
flags.DEFINE_float('dropout_p', 0.0, 'Dropout probability.')
flags.DEFINE_float('reg_weight', 0.0, 'L2 regularization weight.')
flags.DEFINE_float('mutation_rate', 0.0, 'Mutation rate.')
flags.DEFINE_boolean('use_weight_norm', False,
                     'If True, use weight normalization.')
flags.DEFINE_boolean('data_init', False,
                     ('If True, use data-dependent initialization',
                      ' (has no effect if use_weight_norm is False'))

flags.DEFINE_float('momentum', 0.95, 'Momentum parameter (beta1) for Adam'
                   'optimizer.')
flags.DEFINE_float('momentum2', 0.9995, 'Second momentum parameter (beta2) for'
                   'Adam optimizer.')
flags.DEFINE_boolean('rescale_pixel_value', False,
                     'If True, rescale pixel values into [-1,1].')

FLAGS = flags.FLAGS


def main(unused_argv):

  out_dir = FLAGS.out_dir

  exp_dir = 'exp%s' % FLAGS.exp
  model_dir = 'rescale%s' % FLAGS.rescale_pixel_value
  param_dir = 'reg%.2f_mr%.2f' % (FLAGS.reg_weight, FLAGS.mutation_rate)
  job_dir = os.path.join(out_dir, exp_dir, model_dir, param_dir)
  print('job_dir={}'.format(job_dir))

  job_model_dir = os.path.join(job_dir, 'model')
  job_log_dir = os.path.join(job_dir, 'log')

  for sub_dir in out_dir, job_dir, job_model_dir, job_log_dir:
    tf.compat.v1.gfile.MakeDirs(sub_dir)

  params = {
      'job_model_dir': job_model_dir,
      'job_log_dir': job_log_dir,
      'job_dir': job_dir,
      'dropout_p': FLAGS.dropout_p,
      'reg_weight': FLAGS.reg_weight,
      'num_resnet': FLAGS.num_resnet,
      'num_hierarchies': FLAGS.num_hierarchies,
      'num_filters': FLAGS.num_filters,
      'num_logistic_mix': FLAGS.num_logistic_mix,
      'use_weight_norm': FLAGS.use_weight_norm,
      'data_init': FLAGS.data_init,
      'mutation_rate': FLAGS.mutation_rate,
      'batch_size': FLAGS.batch_size,
      'learning_rate': FLAGS.learning_rate,
      'learning_rate_decay': FLAGS.learning_rate_decay,
      'momentum': FLAGS.momentum,
      'momentum2': FLAGS.momentum2,
      'eval_every': FLAGS.eval_every,
      'save_im': FLAGS.save_im,
      'n_dim': 28 if FLAGS.exp == 'fashion' else 32,
      'n_channel': 1 if FLAGS.exp == 'fashion' else 3,
      'exp': FLAGS.exp,
      'rescale_pixel_value': FLAGS.rescale_pixel_value,
  }

  # Print and write parameter settings
  with tf.io.gfile.GFile(
      os.path.join(params['job_model_dir'], 'params.json'), mode='w') as f:
    f.write(json.dumps(params, sort_keys=True))

  # Fix the random seed - easier to debug separate runs
  tf.compat.v1.set_random_seed(FLAGS.random_seed)

  tf.keras.backend.clear_session()
  sess = tf.compat.v1.Session()
  tf.compat.v1.keras.backend.set_session(sess)

  # Load the datasets
  if FLAGS.exp == 'fashion':
    datasets = utils.load_fmnist_datasets(FLAGS.data_dir)
  else:
    datasets = utils.load_cifar_datasets(FLAGS.data_dir)

  # pylint: disable=g-long-lambda
  tr_in_ds = datasets['tr_in'].map(lambda x: utils.image_preprocess_add_noise(
      x, params['mutation_rate'])).batch(
          params['batch_size']).repeat().shuffle(1000).make_one_shot_iterator()
  tr_in_im = tr_in_ds.get_next()

  # repeat valid dataset because it will be used for training
  val_in_ds = datasets['val_in'].map(utils.image_preprocess).batch(
      params['batch_size']).repeat().make_one_shot_iterator()
  val_in_im = val_in_ds.get_next()

  # Define a Pixel CNN network
  input_shape = (params['n_dim'], params['n_dim'], params['n_channel'])
  dist = pixel_cnn.PixelCNN(
      image_shape=input_shape,
      dropout_p=params['dropout_p'],
      reg_weight=params['reg_weight'],
      num_resnet=params['num_resnet'],
      num_hierarchies=params['num_hierarchies'],
      num_filters=params['num_filters'],
      num_logistic_mix=params['num_logistic_mix'],
      use_weight_norm=params['use_weight_norm'],
      rescale_pixel_value=params['rescale_pixel_value'],
  )

  # Define the training loss and optimizer
  log_prob_i = dist.log_prob(tr_in_im['image'], return_per_pixel=False)
  loss = -tf.reduce_mean(log_prob_i)

  log_prob_i_val_in = dist.log_prob(val_in_im['image'])
  loss_val_in = -tf.reduce_mean(log_prob_i_val_in)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  learning_rate = tf.compat.v1.train.exponential_decay(
      params['learning_rate'], global_step, 1, params['learning_rate_decay'])
  opt = tf.compat.v1.train.AdamOptimizer(
      learning_rate=learning_rate,
      beta1=params['momentum'],
      beta2=params['momentum2'])
  tr_op = opt.minimize(loss, global_step=global_step)

  init_op = tf.compat.v1.global_variables_initializer()
  sess.run(init_op)

  # write tensorflow summaries
  saver = tf.compat.v1.train.Saver(max_to_keep=50000)
  merged_tr = tf.compat.v1.summary.merge([
      tf.compat.v1.summary.scalar('loss', loss),
      tf.compat.v1.summary.scalar('train/learning_rate', learning_rate)
  ])
  merged_val_in = tf.compat.v1.summary.merge(
      [tf.compat.v1.summary.scalar('loss', loss_val_in)])
  tr_writer = tf.compat.v1.summary.FileWriter(job_log_dir + '/tr_in',
                                              sess.graph)
  val_in_writer = tf.compat.v1.summary.FileWriter(job_log_dir + '/val_in',
                                                  sess.graph)

  # If previous ckpt exists, load ckpt
  ckpt_file = tf.compat.v2.train.latest_checkpoint(job_model_dir)
  if ckpt_file:
    prev_step = int(
        os.path.basename(ckpt_file).split('model_step')[1].split('.ckpt')[0])
    tf.compat.v1.logging.info(
        'previous ckpt exist, prev_step={}'.format(prev_step))
    saver.restore(sess, ckpt_file)
  else:
    prev_step = 0

  # Train the model
  with sess.as_default():  # this is a must otherwise localhost error
    for step in range(prev_step, FLAGS.total_steps + 1, 1):
      _, loss_tr_np, summary = sess.run([tr_op, loss, merged_tr])
      if step % params['eval_every'] == 0:
        ckpt_name = 'model_step%d.ckpt' % step
        ckpt_path = os.path.join(job_model_dir, ckpt_name)
        while not tf.compat.v1.gfile.Exists(ckpt_path + '.index'):
          _ = saver.save(sess, ckpt_path, write_meta_graph=False)
          time.sleep(10)
        tr_writer.add_summary(summary, step)

        # Evaluate loss on the valid_in
        loss_val_in_np, summary_val_in = sess.run([loss_val_in, merged_val_in])
        val_in_writer.add_summary(summary_val_in, step)

        print('step=%d, tr_in_loss=%.4f, val_in_loss=%.4f' %
              (step, loss_tr_np, loss_val_in_np))

        tr_writer.flush()
        val_in_writer.flush()

  tr_writer.close()
  val_in_writer.close()


if __name__ == '__main__':
  app.run(main)
