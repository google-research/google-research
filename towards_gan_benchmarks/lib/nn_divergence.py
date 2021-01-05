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

"""Code for computing neural net divergences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import time
import numpy as np
import tensorflow.compat.v1 as tf
import towards_gan_benchmarks.lib.flags
import towards_gan_benchmarks.lib.gan
import towards_gan_benchmarks.lib.logger
from tensorflow.contrib import slim as contrib_slim

lib = towards_gan_benchmarks.lib


def set_flags(flags):
  """Populate flags object with defaults."""
  flags.set_if_empty('batch_size', 256)
  flags.set_if_empty('ema', 0.999)
  flags.set_if_empty('final_eval_iters', 10 * 1000)
  flags.set_if_empty('gan', lib.flags.Flags())
  flags.set_if_empty('iters', 100 * 1000)
  flags.set_if_empty('log', False)
  flags.set_if_empty('output_dir', None)
  flags.set_if_empty('fix_tf_seed', False)
  flags.gan.set_if_empty('algorithm', 'wgan-gp')
  flags.gan.set_if_empty('initializer_d', 'he')
  flags.gan.set_if_empty('lr_d', 2e-4)
  flags.gan.set_if_empty('lr_decay', 'linear')
  flags.gan.set_if_empty('nonlinearity_d', 'swish')
  flags.gan.set_if_empty('norm_d', False)
  flags.gan.set_if_empty('l2_reg_d', 0.)
  flags.gan.set_if_empty('wgangp_minimax', True)
  lib.gan.set_flags(flags.gan)


def _make_runner(flags):
  """Return a function which, when called, computes an NN divergence."""
  with tf.Graph().as_default() as graph:
    step_ = tf.placeholder(tf.int32, None)
    real_inputs = tf.placeholder(tf.int32, [None, 32, 32, 3])
    fake_inputs = tf.placeholder(tf.int32, [None, 32, 32, 3])

    def scale_and_dequantize(inputs):
      inputs = tf.cast(inputs - 128, tf.float32) / 128.
      return inputs + tf.random_uniform(tf.shape(inputs), 0, 1. / 128)

    reals = scale_and_dequantize(real_inputs)
    fakes = scale_and_dequantize(fake_inputs)
    # initialize discriminator weights
    lib.gan.discriminator(reals, flags.gan, scope='discriminator')
    discriminator = functools.partial(
        lib.gan.discriminator,
        flags=flags.gan,
        scope='discriminator',
        reuse=True)

    def generator(_):
      return fakes

    disc_params = contrib_slim.get_model_variables('discriminator')
    _, disc_cost, divergence = lib.gan.losses(generator, discriminator, reals,
                                              None, disc_params, flags.gan)
    disc_train_op = lib.gan.disc_train_op(disc_cost, disc_params, step_,
                                          flags.iters, flags.gan)

    # EMA
    ema = tf.train.ExponentialMovingAverage(decay=flags.ema)
    maintain_averages_op = ema.apply(disc_params)
    with tf.control_dependencies([disc_train_op]):
      disc_train_op = tf.group(maintain_averages_op)
    # Vars/ops to swap in and out the EMA vars
    emas_ = [ema.average(x) for x in disc_params]
    temps_ = [tf.Variable(x, trainable=False) for x in disc_params]
    vars_to_temps_ = tf.group(
        *[tf.assign(a, b) for a, b in zip(temps_, disc_params)])
    emas_to_vars_ = tf.group(
        *[tf.assign(a, b) for a, b in zip(disc_params, emas_)])
    temps_to_vars_ = tf.group(
        *[tf.assign(a, b) for a, b in zip(disc_params, temps_)])

    init_op = tf.variables_initializer(contrib_slim.get_variables())

    def run_gan_divergence(real_gen, fake_gen):
      """Train a discriminator and measures a GAN divergence."""
      with tf.Session(graph=graph) as session:
        if flags.fix_tf_seed:
          tf.set_random_seed(1)
        print('Training GAN divergence model')
        if flags.log:
          logger = lib.logger.Logger(flags.output_dir)
        session.run(init_op)
        # Train loop
        step = 0
        for step in range(flags.iters):
          start_time = time.time()
          real_inputs_ = next(real_gen)
          fake_inputs_ = next(fake_gen)
          disc_cost_, divergence_, _ = session.run(
              [disc_cost, divergence, disc_train_op],
              feed_dict={
                  step_: step,
                  real_inputs: real_inputs_,
                  fake_inputs: fake_inputs_
              })
          if flags.log:
            logger.add_scalar('time', time.time() - start_time, step)
            logger.add_scalar('train_cost', disc_cost_, step)
            logger.add_scalar('train_divergence', divergence_, step)
            logger.print(step)
        # Final evaluation
        session.run(vars_to_temps_)
        session.run(emas_to_vars_)
        eval_costs, eval_divs = [], []
        for _ in range(flags.final_eval_iters):
          real_inputs_ = next(real_gen)
          fake_inputs_ = next(fake_gen)
          cost_, div_ = session.run([disc_cost, divergence],
                                    feed_dict={
                                        real_inputs: real_inputs_,
                                        fake_inputs: fake_inputs_
                                    })
          eval_costs.append(cost_)
          eval_divs.append(div_)
        session.run(temps_to_vars_)
        # Log data
        if flags.log:
          logger.add_scalar('final_ema_cost', np.mean(eval_costs), step)
          logger.add_scalar('final_ema_divergence', np.mean(eval_divs), step)
          logger.flush()

        print('final_ema_divergence {}'.format(np.mean(eval_divs)))
        return np.mean(eval_divs)

    return run_gan_divergence


def run(flags, real_gen, fake_gen):
  return _make_runner(flags)(real_gen, fake_gen)
