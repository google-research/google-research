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

"""Very minimal GAN library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim


def set_flags(flags):
  """Populate flags object with defaults."""
  flags.set_if_empty('acts_loss', 0.)
  flags.set_if_empty('algorithm', 'vanilla')
  flags.set_if_empty('architecture', 'dcgan')
  flags.set_if_empty('dim', 64)
  flags.set_if_empty('dim_z', 128)
  flags.set_if_empty('extra_depth', 0)
  flags.set_if_empty('initializer_d', 'xavier')
  flags.set_if_empty('lr_decay', 'none')
  flags.set_if_empty('nonlinearity', 'default')
  flags.set_if_empty('norm', True)
  flags.set_if_empty('l2_reg_d', 1e-3)
  flags.set_if_empty('weight_clip_d', -1)
  flags.set_if_empty('weight_decay_g', None)
  flags.set_if_empty('weight_decay_d', None)
  flags.set_if_empty('z_seed', None)
  if flags.algorithm in [
      'wgan-gp', 'wgan-lp', 'wgan-v3', 'wgan-gp-quadratic', 'r1', 'r1-ns'
  ]:
    flags.set_if_empty('lr', 1e-4)
    flags.set_if_empty('beta1', 0.)
    flags.set_if_empty('beta2', 0.9)
    flags.set_if_empty('disc_iters', 5)
    flags.set_if_empty('wgangp_lambda', 10)
    flags.set_if_empty('wgangp_minimax', False)
    flags.set_if_empty('wgangp_compressive_loss', False)
  elif flags.algorithm in ['vanilla', 'vanilla_minimax']:
    flags.set_if_empty('lr', 2e-4)
    flags.set_if_empty('beta1', 0.5)
    flags.set_if_empty('beta2', 0.999)
    flags.set_if_empty('disc_iters', 1)
  else:
    raise Exception('invalid gan flags.algorithm')
  flags.set_if_empty('dim_g', flags.dim)
  flags.set_if_empty('dim_d', flags.dim)
  flags.set_if_empty('extra_depth_g', flags.extra_depth)
  flags.set_if_empty('extra_depth_d', flags.extra_depth)
  flags.set_if_empty('downsample_conv_filt_size', 5)
  flags.set_if_empty('extra_conv_filt_size', 3)
  flags.set_if_empty('extra_top_conv', False)
  flags.set_if_empty('lr_d', flags.lr)
  flags.set_if_empty('lr_g', flags.lr)
  flags.set_if_empty('nonlinearity_g', flags.nonlinearity)
  flags.set_if_empty('nonlinearity_d', flags.nonlinearity)
  flags.set_if_empty('norm_g', flags.norm)
  flags.set_if_empty('norm_d', flags.norm)


def random_latents(batch_size, flags, antithetic_sampling=False):
  if antithetic_sampling:
    half = tf.random_normal([batch_size // 2, flags.dim_z], seed=flags.z_seed)
    return tf.concat([half, -half], axis=0)
  else:
    return tf.random_normal([batch_size, flags.dim_z], seed=flags.z_seed)


def _leaky_relu(x):
  return tf.maximum(0.2 * x, x)


def _swish(x):
  return x * tf.nn.sigmoid(x)


def _softplus(x):
  return tf.nn.softplus(x)


def _elu_softplus(x):
  """softplus that looks roughly like elu but is smooth."""
  return (tf.nn.softplus((2 * x) + 2) / 2) - 1


def nonlinearity_fn(flag, is_discriminator):
  """Returns the appropriate nonlinearity function based on flags."""
  if flag == 'default':
    if is_discriminator:
      return _leaky_relu
    else:
      return tf.nn.relu
  elif flag == 'leaky_relu':
    return _leaky_relu
  elif flag == 'relu':
    return tf.nn.relu
  elif flag == 'elu':
    return tf.nn.elu
  elif flag == 'swish':
    return _swish
  elif flag == 'softplus':
    return _softplus
  elif flag == 'elu_softplus':
    return _elu_softplus
  elif flag == 'exp':
    return tf.exp
  elif flag == 'tanh':
    return tf.tanh
  elif flag == 'sigmoid':
    return tf.nn.sigmoid
  else:
    raise Exception('invalid nonlinearity {}'.format(flag))


def generator(z, flags, scope=None, reuse=None):
  if flags.architecture == 'dcgan':
    return dcgan_generator(z, flags, scope, reuse)


def discriminator(x, flags, scope=None, reuse=None, return_acts=False):
  if flags.architecture == 'dcgan':
    return dcgan_discriminator(x, flags, scope, reuse, return_acts=return_acts)


def dcgan_generator(z, flags, scope=None, reuse=None):
  """DCGAN-style generator network."""
  nonlinearity = nonlinearity_fn(flags.nonlinearity_g, False)
  ds_fs = flags.downsample_conv_filt_size
  x_fs = flags.extra_conv_filt_size

  if not flags.norm_g:
    normalizer = None
  else:
    normalizer = contrib_slim.batch_norm

  with tf.variable_scope(scope, reuse=reuse):
    out = contrib_slim.fully_connected(
        z,
        4 * 4 * (4 * flags.dim_g),
        scope='fc',
        normalizer_fn=normalizer,
        activation_fn=nonlinearity)
    out = tf.reshape(out, [-1, 4, 4, 4 * flags.dim_g])

    if flags.extra_top_conv:
      out = contrib_slim.conv2d(
          out,
          4 * flags.dim_d,
          x_fs,
          scope='extratopconv',
          activation_fn=nonlinearity,
          normalizer_fn=normalizer)

    out = contrib_slim.conv2d_transpose(
        out,
        2 * flags.dim_g,
        ds_fs,
        scope='conv1',
        stride=2,
        normalizer_fn=normalizer,
        activation_fn=nonlinearity)

    for i in range(flags.extra_depth_g):
      out = contrib_slim.conv2d(
          out,
          2 * flags.dim_g,
          x_fs,
          scope='extraconv1.{}'.format(i),
          normalizer_fn=normalizer,
          activation_fn=nonlinearity)

    out = contrib_slim.conv2d_transpose(
        out,
        flags.dim_g,
        ds_fs,
        scope='conv2',
        stride=2,
        normalizer_fn=normalizer,
        activation_fn=nonlinearity)

    for i in range(flags.extra_depth_g):
      out = contrib_slim.conv2d(
          out,
          flags.dim_g,
          x_fs,
          scope='extraconv2.{}'.format(i),
          normalizer_fn=normalizer,
          activation_fn=nonlinearity)

    out = contrib_slim.conv2d_transpose(
        out, 3, ds_fs, scope='conv3', stride=2, activation_fn=tf.tanh)

    return out


def dcgan_discriminator(x, flags, scope=None, reuse=None, return_acts=False):
  """DCGAN-style discriminator network."""
  nonlinearity = nonlinearity_fn(flags.nonlinearity_d, True)
  ds_fs = flags.downsample_conv_filt_size
  x_fs = flags.extra_conv_filt_size

  acts = []
  with tf.variable_scope(scope, reuse=reuse):
    if not flags.norm_d:
      normalizer = None
    elif flags.algorithm == 'vanilla':
      normalizer = contrib_slim.batch_norm
    else:
      normalizer = contrib_slim.layer_norm

    if flags.initializer_d == 'xavier':
      initializer = contrib_layers.xavier_initializer()
    elif flags.initializer_d == 'orth_gain2':
      initializer = tf.orthogonal_initializer(gain=2.)
    elif flags.initializer_d == 'he':
      initializer = contrib_layers.variance_scaling_initializer()
    elif flags.initializer_d == 'he_uniform':
      initializer = contrib_layers.variance_scaling_initializer(uniform=True)

    out = contrib_slim.conv2d(
        x,
        flags.dim_d,
        ds_fs,
        scope='conv1',
        stride=2,
        activation_fn=nonlinearity,
        weights_initializer=initializer)
    acts.append(out)

    for i in range(flags.extra_depth_d):
      out = contrib_slim.conv2d(
          out,
          flags.dim_d,
          x_fs,
          scope='extraconv1.{}'.format(i),
          activation_fn=nonlinearity,
          normalizer_fn=normalizer,
          weights_initializer=initializer)
      acts.append(out)

    out = contrib_slim.conv2d(
        out,
        2 * flags.dim_d,
        ds_fs,
        scope='conv2',
        stride=2,
        activation_fn=nonlinearity,
        normalizer_fn=normalizer,
        weights_initializer=initializer)
    acts.append(out)

    for i in range(flags.extra_depth_d):
      out = contrib_slim.conv2d(
          out,
          2 * flags.dim_d,
          x_fs,
          scope='extraconv2.{}'.format(i),
          activation_fn=nonlinearity,
          normalizer_fn=normalizer,
          weights_initializer=initializer)
      acts.append(out)

    out = contrib_slim.conv2d(
        out,
        4 * flags.dim_d,
        ds_fs,
        scope='conv3',
        stride=2,
        activation_fn=nonlinearity,
        normalizer_fn=normalizer,
        weights_initializer=initializer)
    acts.append(out)

    if flags.extra_top_conv:
      out = contrib_slim.conv2d(
          out,
          4 * flags.dim_d,
          x_fs,
          scope='extratopconv',
          activation_fn=nonlinearity,
          normalizer_fn=normalizer,
          weights_initializer=initializer)
      acts.append(out)

    out = tf.reshape(out, [-1, 4 * 4 * (4 * flags.dim_d)])
    out = contrib_slim.fully_connected(out, 1, scope='fc', activation_fn=None)
    acts.append(out)

    if return_acts:
      return out, acts
    else:
      return out


def losses(generator_fn, discriminator_fn, real_data, z,
           disc_params, flags):
  """Returns loss variables for the generator and discriminator."""
  fake_data = generator_fn(z)

  if flags.acts_loss > 0.:
    disc_real, disc_real_acts = discriminator_fn(real_data, return_acts=True)
    disc_fake, disc_fake_acts = discriminator_fn(fake_data, return_acts=True)
  else:
    disc_real = discriminator_fn(real_data)
    disc_fake = discriminator_fn(fake_data)

  acts_l2_loss = 0.
  acts_count = 1.
  if flags.acts_loss > 0.:
    all_disc_acts = disc_real_acts + disc_fake_acts
    for act in all_disc_acts:
      acts_l2_loss += tf.nn.l2_loss(act)
      acts_count += tf.reduce_sum(tf.ones_like(act))

  l2_reg_d_cost = 0.
  if flags.l2_reg_d > 0:
    for p in disc_params:
      if 'weights' in p.name:
        l2_reg_d_cost += tf.nn.l2_loss(p)
    l2_reg_d_cost *= flags.l2_reg_d

    def cn(x):
      """compressive nonlinearity."""
      return tf.asinh(4. * x) / 4.

  if flags.algorithm == 'vanilla':
    gen_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.ones_like(disc_fake)))
    disc_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones_like(disc_real)))
    divergence = gen_cost
    disc_cost += l2_reg_d_cost
    disc_cost += flags.acts_loss * (acts_l2_loss / (1e-2 + acts_count))

  elif flags.algorithm == 'vanilla_minimax':
    disc_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones_like(disc_real)))
    gen_cost = -disc_cost
    divergence = ((-disc_cost) + tf.log(4.)) / 2.
    disc_cost += l2_reg_d_cost
    disc_cost += flags.acts_loss * (acts_l2_loss / (1e-2 + acts_count))

  elif flags.algorithm == 'wgan-gp':
    input_ndim = len(real_data.get_shape())
    if flags.wgangp_compressive_loss:
      disc_fake = cn(disc_fake)
      disc_real = cn(disc_real)
    wgan_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[tf.shape(real_data)[0]] + [1 for i in range(input_ndim - 1)],
        minval=0.,
        maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    if flags.acts_loss > 0.:
      disc_interps, disc_interp_acts = discriminator_fn(
          interpolates, return_acts=True)
    else:
      disc_interps = discriminator_fn(interpolates)
    gradients = tf.gradients(disc_interps, [interpolates])[0]
    slopes = tf.sqrt(1e-8 + tf.reduce_sum(
        tf.square(gradients),
        reduction_indices=[i for i in range(1, input_ndim)]))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    disc_cost = wgan_disc_cost + (flags.wgangp_lambda * gradient_penalty)
    disc_cost += l2_reg_d_cost

    if flags.acts_loss > 0.:
      for act in disc_interp_acts:
        acts_l2_loss += flags.acts_loss * tf.nn.l2_loss(act)
        acts_count += tf.reduce_sum(tf.ones_like(act))
    disc_cost += flags.acts_loss * (acts_l2_loss / (1e-2 + acts_count))

    if flags.wgangp_minimax:
      gen_cost = -disc_cost
      divergence = -disc_cost
    else:
      gen_cost = -tf.reduce_mean(disc_fake)
      divergence = -wgan_disc_cost

  elif flags.algorithm == 'r1':
    disc_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones_like(disc_real)))
    gen_cost = -disc_cost
    divergence = ((-disc_cost) + tf.log(4.)) / 2.

    input_ndim = len(real_data.get_shape())
    gradients = tf.gradients(tf.nn.sigmoid(disc_real), [real_data])[0]
    slopes = tf.sqrt(1e-8 + tf.reduce_sum(
        tf.square(gradients),
        reduction_indices=[i for i in range(1, input_ndim)]))
    gradient_penalty = 0.5 * tf.reduce_mean(slopes**2)

    disc_cost += flags.wgangp_lambda * gradient_penalty
    disc_cost += l2_reg_d_cost
    disc_cost += flags.acts_loss * (acts_l2_loss / (1e-2 + acts_count))

  elif flags.algorithm == 'r1-ns':
    disc_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones_like(disc_real)))
    divergence = ((-disc_cost) + tf.log(4.)) / 2.
    gen_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.ones_like(disc_fake)))

    input_ndim = len(real_data.get_shape())
    gradients = tf.gradients(tf.nn.sigmoid(disc_real), [real_data])[0]
    slopes = tf.sqrt(1e-8 + tf.reduce_sum(
        tf.square(gradients),
        reduction_indices=[i for i in range(1, input_ndim)]))
    gradient_penalty = 0.5 * tf.reduce_mean(slopes**2)

    disc_cost += flags.wgangp_lambda * gradient_penalty
    disc_cost += l2_reg_d_cost
    disc_cost += flags.acts_loss * (acts_l2_loss / (1e-2 + acts_count))

  return gen_cost, disc_cost, divergence


def gen_train_op(cost, params, step, iters, flags):
  """Build the generator train op."""
  if flags.lr_decay == 'linear':
    step_lr = (1. - (tf.cast(step, tf.float32) / iters))
  elif flags.lr_decay == 'quadratic':
    step_lr = ((1. - (tf.cast(step, tf.float32) / iters))**2)
  elif flags.lr_decay == 'none':
    step_lr = 1.
  train_op = tf.train.AdamOptimizer(step_lr * flags.lr_g, flags.beta1,
                                    flags.beta2).minimize(
                                        cost,
                                        var_list=params,
                                        colocate_gradients_with_ops=True)

  if flags.weight_decay_g is not None:
    decay = (step_lr * flags.weight_decay_g)
    with tf.control_dependencies([train_op]):
      weights = [p for p in params if 'weights' in p.name]
      decayed = [w - (decay * w) for w in weights]
      decay_op = tf.group(*[tf.assign(w, d) for w, d in zip(weights, decayed)])
    train_op = decay_op

  return train_op


def disc_train_op(cost, params, step, iters, flags):
  """Build the discriminator train op."""
  if flags.lr_decay == 'linear':
    step_lr = (1. - (tf.cast(step, tf.float32) / iters))
  elif flags.lr_decay == 'quadratic':
    step_lr = ((1. - (tf.cast(step, tf.float32) / iters))**2)
  elif flags.lr_decay == 'drop_after_90k':
    step_lr = tf.cond(step > 90000, lambda: 0.1, lambda: 1.0)
  elif flags.lr_decay == 'none':
    step_lr = 1.
  train_op = tf.train.AdamOptimizer(step_lr * flags.lr_d, flags.beta1,
                                    flags.beta2).minimize(
                                        cost,
                                        var_list=params,
                                        colocate_gradients_with_ops=True)

  if flags.weight_decay_d is not None:
    decay = (step_lr * flags.weight_decay_d)
    with tf.control_dependencies([train_op]):
      weights = [p for p in params if 'weights' in p.name]
      decayed = [w - (decay * w) for w in weights]
      decay_op = tf.group(*[tf.assign(w, d) for w, d in zip(weights, decayed)])
    train_op = decay_op

  if flags.weight_clip_d >= 0:
    # Clip *all* the params, like the original WGAN implementation
    clip = flags.weight_clip_d
    with tf.control_dependencies([train_op]):
      clipped = [tf.clip_by_value(p, -clip, clip) for p in params]
      clip_op = tf.group(*[tf.assign(p, c) for c, p in zip(clipped, params)])
    train_op = clip_op

  return train_op
