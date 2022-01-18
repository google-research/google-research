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

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v1 as tf
import sonnet as snt
import numpy as np
import os
import random
import collections
import sys
from tqdm import tqdm
from scipy.special import softmax
import matplotlib.pyplot as plt

from ebp.common.data_utils.curve_reader import get_reader
from ebp.common.cmd_args import cmd_args
from ebp.common.f_family import ScoreFunc, VAE
from ebp.common.generator import HyperGen
from ebp.common.plot_utils.plot_2d import plot_samples


def get_gen_loss(args, xfake, ll_fake, score_func, z_outer):
  opt_gen = tf.train.AdamOptimizer(
      learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)

  f_sampled_x = score_func(xfake, z_outer)
  loss = -tf.reduce_mean(f_sampled_x) + args.ent_lam * tf.reduce_mean(ll_fake)

  gvs = opt_gen.compute_gradients(
      loss, var_list=tf.trainable_variables(scope='generator'))
  gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val)
         for grad, val in gvs
         if grad is not None]
  train_gen = opt_gen.apply_gradients(gvs)
  return loss, train_gen


def get_disc_loss(args, x, x_fake, score_func, z_outer, neg_kl_outer):
  opt_disc = tf.train.AdamOptimizer(
      learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)

  fx = score_func(x, z_outer)
  f_fake_x = score_func(x_fake, z_outer)
  f_loss = tf.reduce_mean(-fx) + tf.reduce_mean(f_fake_x)

  loss = f_loss + tf.reduce_mean(-neg_kl_outer)
  if args.gp_lambda > 0:  # add gradient penalty
    alpha = tf.random.uniform(shape=(tf.shape(x)[0], 1, 1))
    x_hat = alpha * x + (1 - alpha) * x_fake
    d_hat = score_func(x_hat, tf.stop_gradient(z_outer))
    ddx = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=[1, 2]))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * args.gp_lambda
    loss = loss + ddx
  gvs = opt_disc.compute_gradients(
      loss, var_list=tf.trainable_variables(scope='score_func'))
  gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val)
         for grad, val in gvs
         if grad is not None]
  train_disc = opt_disc.apply_gradients(gvs)
  return f_loss, train_disc


if __name__ == '__main__':
  random.seed(cmd_args.seed)
  np.random.seed(cmd_args.seed)
  tf.set_random_seed(cmd_args.seed)

  # Train dataset
  dataset_train = get_reader(cmd_args, testing=False)
  data_train = dataset_train.generate_curves()

  # plot dataset
  dataset_plot = get_reader(cmd_args, testing=False, bsize=1)
  data_plot = dataset_plot.generate_curves()
  data_test = data_plot

  with tf.variable_scope('score_func'):
    x_true = tf.concat([data_train.query[1], data_train.target_y], axis=-1)

    posterior = VAE(sigma_eps=cmd_args.sigma_eps)

    z_outer, mu_outer, sigma_outer, neg_kl_outer = posterior(x_true)
    x_test = tf.concat([data_test.query[1], data_test.target_y], axis=-1)
    _, mu_test, _, _ = posterior(x_test)
    score_func = ScoreFunc(embed_dim=32)

  with tf.variable_scope('generator'):
    cond_gen = HyperGen(dim=1, condx_dim=1, condz_dim=32, num_layers=10)

    x_fake, ll_fake = cond_gen(data_train.query[1], z_outer)

    test_fake_gen = cond_gen(data_test.query[1], mu_test)[0]

  disc_loss, train_disc = get_disc_loss(cmd_args, x_true, x_fake, score_func,
                                        z_outer, neg_kl_outer)
  gen_loss, train_gen = get_gen_loss(cmd_args, x_fake, ll_fake, score_func,
                                     z_outer)

  # for plot
  ph_x = tf.placeholder(tf.float32, shape=(None, 1))
  ph_y = tf.placeholder(tf.float32, shape=(None, 1))
  ph_x_plot_cond = tf.placeholder(tf.float32, shape=(1, None, 2))
  z_plot, _, _, _ = posterior(ph_x_plot_cond)
  z_plot = tf.tile(z_plot, [tf.shape(ph_x)[0], 1])
  x_plot = tf.concat([ph_x, ph_y], axis=-1)
  x_plot = tf.expand_dims(x_plot, 1)
  score_plot = score_func(x_plot, z_plot)
  x_plot_cond = tf.concat([data_plot.query[1], data_plot.target_y], axis=-1)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  saver = tf.train.Saver()
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    model_dir = os.path.join(cmd_args.save_dir, 'model')
    if cmd_args.epoch_load >= 0:
      model_path = os.path.join(model_dir,
                                'model-%d.ckpt' % cmd_args.epoch_load)
      saver.restore(sess, model_path)
      points_fake, target_y, whole_query = sess.run(
          [test_fake_gen, data_test.target_y, data_test.query])
      (context_x, context_y), target_x = whole_query
      out_file = os.path.join(cmd_args.save_dir,
                              'fig-%d.pdf' % cmd_args.epoch_load)
      points_fake = np.reshape(points_fake[0], [-1, 2])
      plot_samples(points_fake, out_file)
      w = 50
      size = 2
      x = np.linspace(-size, size, w)
      y = np.linspace(-size, size, w)
      xx, yy = np.meshgrid(x, y)
      xx = np.reshape(xx, [-1, 1])
      yy = np.reshape(yy, [-1, 1])
      avg_score = 0
      samples = 100
      plot_scale = 10
      np_plot_cond = sess.run(x_plot_cond)
      out_file = os.path.join(cmd_args.save_dir,
                              'observe-%d.pdf' % cmd_args.epoch_load)
      plot_samples(np_plot_cond[0], out_file)
      for i in tqdm(range(samples)):
        cur_score = sess.run(
            score_plot,
            feed_dict={
                ph_x: xx,
                ph_y: yy,
                ph_x_plot_cond: np_plot_cond
            })
        a = cur_score.reshape((w, w))
        a = softmax(a * plot_scale, axis=0)
        a = np.flip(a, axis=1)
        avg_score += a
      a = avg_score / samples
      plt.imshow(a)
      plt.axis('equal')
      plt.axis('off')
      out_file = os.path.join(cmd_args.save_dir,
                              'heat-%d.pdf' % cmd_args.epoch_load)
      plt.savefig(out_file, bbox_inches='tight')
      plt.close()

    if cmd_args.epoch_load >= 0:
      sys.exit()
    x, y = sess.run([data_test.query[1], data_test.target_y])
    x = x[0].flatten()
    y = y[0].flatten()
    points = np.stack((x, y)).transpose()
    plot_samples(points, os.path.join(cmd_args.save_dir, 'sample.pdf'))
    for epoch in range(cmd_args.num_epochs):
      pbar = tqdm(range(cmd_args.iters_per_eval), unit='batch')

      points_fake, target_y, whole_query = sess.run(
          [test_fake_gen, data_test.target_y, data_test.query])
      (context_x, context_y), target_x = whole_query
      out_file = os.path.join(cmd_args.save_dir, 'plot-%d.pdf' % epoch)
      points_fake = np.reshape(points_fake[0], [-1, 2])
      plot_samples(points_fake, out_file)
      for pos in pbar:
        # optimize discriminator
        for i in range(1):
          _, np_disc_loss = sess.run([train_disc, disc_loss])
        # optimize generator
        for i in range(3):
          _, np_gen_loss = sess.run([train_gen, gen_loss])
        pbar.set_description('disc_loss: %.4f, gen_loss: %.4f' %
                             (np_disc_loss, np_gen_loss))
      if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
      saver.save(sess, os.path.join(model_dir, 'model-%d.ckpt' % epoch))
