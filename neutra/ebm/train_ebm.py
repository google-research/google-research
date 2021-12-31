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

# python3
"""EBM definition and training logic."""
import math
import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

# pylint: disable=g-import-not-at-top
USE_LOCAL_FUN_MC = True

if USE_LOCAL_FUN_MC:
  from fun_mc import using_tensorflow as fun_mc  # pylint: disable=reimported

from neutra.ebm import ebm_util
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS
tfd = tfp.distributions
tfb = tfp.bijectors

flags.DEFINE_string('logdir', None, 'Directory to place all the artifacts in.')

flags.DEFINE_integer('seed', 1, 'PRNG seed.')

flags.DEFINE_enum('dataset', 'celeba', ['celeba', 'mnist'],
                  'Dataset to train on.')

flags.DEFINE_integer('batch_size', 64, 'Training minibatch size.')
flags.DEFINE_integer('train_steps', int(1e5),
                     'Total number of training iterations')
flags.DEFINE_integer('save_steps', 1000, 'How often to save checkpoints.')

flags.DEFINE_boolean('use_mcmc', False,
                     'Whether to use MCMC when training EBM.')
flags.DEFINE_integer('mcmc_num_steps', 10, 'How many MCMC steps to take.')
flags.DEFINE_float('mcmc_step_size', 0.1, 'MCMC initial step size.')
flags.DEFINE_integer('mcmc_leapfrog_steps', 3, 'MCMC number of leapfrog steps.')
flags.DEFINE_boolean('mcmc_adapt_step_size', True,
                     'Whether to adapt the step size.')
flags.DEFINE_float('mcmc_momentum_stddev', 0.1, 'Initial momentum stddev.')

flags.DEFINE_enum('p_loss', 'neutra_iid', ['neutra_hmc', 'neutra_iid'],
                  'What loss to use to train the EBM.')
flags.DEFINE_float('p_learning_rate', 1e-4, 'EBM learning rate.')
flags.DEFINE_float('p_adam_beta_1', 0.5, 'EBM ADAM beta_1.')
flags.DEFINE_float('p_prior_weight', 0.01,
                   'Strength of the EBM quadratic prior.')
flags.DEFINE_float('p_temperature', 1.0, 'EBM temperature.')
flags.DEFINE_string('p_activation', 'relu', 'EBM activation function.')
flags.DEFINE_float('p_center_regularizer', 0.0,
                   'Regularizer to center the EBM energy around 0.')

flags.DEFINE_string('q_type', 'mean_field_gaussian', 'What Q to use.')
flags.DEFINE_enum('q_loss', 'reverse_kl_mle',
                  ['reverse_kl', 'forward_kl', 'reverse_kl_mle', 'mle'],
                  'What loss to use to train the flow model.')
flags.DEFINE_float('q_learning_rate', 1e-4, 'Q learning rate.')
flags.DEFINE_float('q_temperature', 1.0, 'Q temperature when sampling.')
flags.DEFINE_integer(
    'q_sub_steps', 1, 'Number of Q updates per training step. '
    'Only applicable to reverse_kl_mle loss.')
flags.DEFINE_float('q_entropy_weight', 1.0,
                   'Entropy coefficient for reverse KL losses.')
flags.DEFINE_float('q_mle_coefficient', 1.0,
                   'MLE coefficient for the reverse_kl_mle loss.')
flags.DEFINE_float('q_rkl_weight', 0.01,
                   'Q reverse KL coefficient for reverse_kl_mle loss')

flags.DEFINE_integer('plot_steps', 500, 'How often to generate plots.')


N_CH = 3  # Number of channels
N_WH = 32  # Width of the images.

tfk = tf.keras
tfkl = tfk.layers


class EbmConv(tf.keras.Model):
  """A convolutional EBM."""

  def __init__(self, activation=tf.nn.relu, anchor_size=32):
    """Initializes the network.

    Args:
      activation: What activation to use for the conv layers.
      anchor_size: A base size to use for the filter channels, setting the
        overall depth of the network.
    """
    super(EbmConv, self).__init__()
    sn = ebm_util.SpectralNormalization
    self.net = tf.keras.Sequential([
        sn(
            tfkl.Conv2D(
                filters=anchor_size,
                kernel_size=3,
                strides=(1, 1),
                padding='SAME',
                activation=activation,
                input_shape=[N_WH, N_WH, N_CH])),
        sn(
            tfkl.Conv2D(
                filters=anchor_size * 2,
                kernel_size=4,
                strides=(2, 2),
                padding='SAME',
                activation=activation)),
        sn(
            tfkl.Conv2D(
                filters=anchor_size * 4,
                kernel_size=4,
                strides=(2, 2),
                padding='SAME',
                activation=activation)),
        sn(
            tfkl.Conv2D(
                filters=anchor_size * 4,
                kernel_size=4,
                strides=(2, 2),
                padding='SAME',
                activation=activation)),
        sn(tfkl.Conv2D(filters=1, kernel_size=4, strides=(1, 1))),
    ])

  def call(self, x):
    x = tf.reshape(x, shape=[-1, N_WH, N_WH, N_CH])
    prior = tf.reduce_sum((x**2), axis=[1, 2, 3])
    energy = tf.squeeze(self.net(x))
    return FLAGS.p_prior_weight * prior + energy / FLAGS.p_temperature


def make_u():
  """Create an energy function."""
  activation = {'relu': tf.nn.relu, 'lipswish': ebm_util.lipswish}
  u = EbmConv(activation=activation[FLAGS.p_activation])
  return u


class MeanFieldGaussianQ(tf.Module):
  """A mean-field Gaussian Q."""

  def __init__(self):
    super(MeanFieldGaussianQ, self).__init__()
    image_shape = [N_WH, N_WH, N_CH]
    ndims = np.prod(image_shape)
    zeros = tf.zeros(ndims)
    ones = tf.ones(ndims)
    self._base = tfd.Independent(tfd.Normal(zeros, ones), 1)
    b = tfb.ScaleMatvecDiag(
        scale_diag=tfp.util.TransformedVariable(ones, tfb.Softplus()))
    b = tfb.Shift(shift=tf.Variable(zeros))(b)
    b = tfb.Reshape(image_shape)(b)
    self._bijector = b

  def forward(self, x):
    """Encodes a datapoint into the latent vector."""
    return (self._bijector.inverse(x),
            self._bijector.inverse_log_det_jacobian(x, event_ndims=3))

  def reverse(self, z):
    """Decodes a latent vector into the data space."""
    return (self._bijector.forward(z),
            self._bijector.forward_log_det_jacobian(z, event_ndims=1))

  def log_prob(self, x):
    return self._bijector(self._base).log_prob(x)

  def sample_with_log_prob(self, n, temp=1.0):
    # TODO(siege): How to incorporate temperature here?
    z = self._base.sample(n)
    x = self._bijector.forward(z)
    fldj = self._bijector.forward_log_det_jacobian(z, event_ndims=1)
    return z, x, fldj + self._base.log_prob(z)


@tf.function(autograph=False)
def train_q_fwd_kl(q, x, opt_q):
  """KL[P || Q].

  Args:
    q: `ModelQ`.
    x: A batch of positive examples.
    opt_q: A `tf.optimizer.Optimizer`.

  Returns:
    loss: The mean loss across the batch.
  """
  with tf.GradientTape() as tape:
    tape.watch(q.trainable_variables)
    loss = -tf.reduce_mean(q.log_prob(x))

  variables = tape.watched_variables()
  grads = tape.gradient(loss, variables)
  grads_and_vars = list(zip(grads, variables))
  opt_q.apply_gradients(grads_and_vars)
  return loss


def q_rev_kl(q, u):
  """KL[Q || U].

  Args:
    q: `ModelQ`.
    u: A callable representing the energy function.

  Returns:
    loss: The mean loss across the batch.
    entropy: Entropy estimate of the `q` model.
    new_e_q: Mean energy of the negative samples sampled from `q`.
  """
  _, x, log_p = q.sample_with_log_prob(
      FLAGS.batch_size, temp=FLAGS.q_temperature)

  # TODO(nijkamp): Is this normalization correct?
  n_pixel = N_WH * N_WH * N_CH
  entropy = tf.reduce_mean(-log_p / (math.log(2) * n_pixel))
  neg_e_q = tf.reduce_mean(u(x))

  return neg_e_q - FLAGS.q_entropy_weight * entropy, entropy, neg_e_q


@tf.function(autograph=False)
def train_q_rev_kl(q, u, opt_q):
  """KL[Q || U].

  Args:
    q: `ModelQ`.
    u: A callable representing the energy function.
    opt_q: A `tf.optimizer.Optimizer`.

  Returns:
    loss: The mean loss across the batch.
    entropy: Entropy estimate of the `q` model.
  """
  with tf.GradientTape() as tape:
    tape.watch(q.trainable_variables)
    loss, entropy, _ = q_rev_kl(q, u)

  variables = q.trainable_variables
  grads = tape.gradient(loss, variables)
  grads_and_vars = list(zip(grads, variables))
  opt_q.apply_gradients(grads_and_vars)
  return loss, entropy


@tf.function(autograph=False)
def train_q_rev_kl_mle(q, u, x_pos, alpha, opt_q):
  """alpha KL[Q || U] + beta KL[data || Q].

  Args:
    q: `ModelQ`.
    u: A callable representing the energy function.
    x_pos: A batch of positive examples.
    alpha: Factor for the RKL term.
    opt_q: A `tf.optimizer.Optimizer`.

  Returns:
    loss: The mean overall loss across the batch.
    entropy: Entropy estimate of the `q` model.
    new_e_q: Mean energy of the negative samples sampled from `q`.
    mle_loss: Mean ML loss across the batch.
    grads_ebm_norm: Global norm of gradients for the RKL term.
    grads_mle_norm: Global norm of gradients for the MLE term.
  """
  with tf.GradientTape() as tape1:
    tape1.watch(q.trainable_variables)
    ebm_loss, entropy, neg_e_q = q_rev_kl(q, u)
    ebm_loss = alpha * ebm_loss

  with tf.GradientTape() as tape2:
    tape2.watch(q.trainable_variables)
    mle_loss = FLAGS.q_mle_coefficient * tf.reduce_mean(
        -q.log_prob(x_pos) /
        (np.log(2.) * int(x_pos.get_shape()[1]) * int(x_pos.get_shape()[2]) *
         int(x_pos.get_shape()[3])))  # bits per subpixel

  loss = ebm_loss + mle_loss

  variables = q.trainable_variables

  grads_ebm = tape1.gradient(ebm_loss, variables)
  grads_mle = tape2.gradient(mle_loss, variables)

  grads_ebm_norm = tf.norm(
      tf.concat([tf.reshape(t, [-1]) for t in grads_ebm], axis=0))
  grads_mle_norm = tf.norm(
      tf.concat([tf.reshape(t, [-1]) for t in grads_mle], axis=0))

  grads_and_vars = list(zip(grads_ebm, variables))
  opt_q.apply_gradients(grads_and_vars)

  grads_and_vars = list(zip(grads_mle, variables))
  opt_q.apply_gradients(grads_and_vars)

  return loss, entropy, neg_e_q, mle_loss, grads_ebm_norm, grads_mle_norm


@tf.function(autograph=False)
def train_q_mle(q, x_pos, opt_q):
  """KL[data || Q].

  Args:
    q: `ModelQ`.
    x_pos: A batch of positive examples.
    opt_q: A `tf.optimizer.Optimizer`.

  Returns:
    loss: The mean overall loss across the batch.
  """
  with tf.GradientTape() as tape:
    tape.watch(q.trainable_variables)

    mle_loss = tf.reduce_mean(
        -q.log_prob(x_pos) /
        (np.log(2.) * int(x_pos.get_shape()[1]) * int(x_pos.get_shape()[2]) *
         int(x_pos.get_shape()[3])))  # bits per subpixel

  # variables = tape.watched_variables()
  # assert len(variables) == len(q.trainable_variables)
  variables = q.trainable_variables
  grads = tape.gradient(mle_loss, variables)
  grads_and_vars = list(zip(grads, variables))
  opt_q.apply_gradients(grads_and_vars)
  return mle_loss


@tf.function(autograph=False)
def train_p(q, u, x_pos, step_size, opt_p):
  """Train P using the standard CD objective.

  Args:
    q: `ModelQ`.
    u: A callable representing the energy function.
    x_pos: A batch of positive examples.
    step_size: Step size to use for HMC.
    opt_p: A `tf.optimizer.Optimizer`.

  Returns:
    x_neg_q: Negative samples sampled from `q`.
    x_neg_p: Negative samples used to train `p`, possibly generated via HMC.
    p_accept: Acceptance rate of HMC.
    step_size: The new step size, possibly adapted to adjust the acceptance
      rate.
    pos_e: Mean energy of the positive samples across the batch.
    pos_e: Mean energy of the positive samples across the batch, after the
      parameter update.
    neg_e_q: Mean energy of `x_neg_q` across the batch.
    neg_e_p: Mean energy of `x_neg_p` across the batch.
    neg_e_p_updated: Mean energy of `x_neg_p` across the batch, after the
      parameter update.
  """

  def create_momentum_sample_fn(state):
    sample_fn = lambda seed: tf.random.normal(  # pylint: disable=g-long-lambda
        tf.shape(state),
        stddev=FLAGS.mcmc_momentum_stddev)
    return sample_fn

  _, x_neg_q, _ = q.sample_with_log_prob(
      FLAGS.batch_size, temp=FLAGS.q_temperature)
  neg_e_q = tf.reduce_mean(u(x_neg_q))

  def p_log_prob(x):
    return -u(x)

  if FLAGS.use_mcmc:

    def log_prob_non_transformed(x):
      p_log_p = p_log_prob(x)

      return p_log_p, (x,)

    # TODO(siege): Why aren't we actually using NeuTra?
    # def log_prob_transformed(z):
    #   x, logdet = q.reverse(z)
    #   p_log_p = p_log_prob(x)

    #   return p_log_p + logdet, (x,)

    def kernel(hmc_state, step_size, step):
      """HMC kernel."""
      hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
          hmc_state,
          step_size=step_size,
          num_integrator_steps=FLAGS.mcmc_leapfrog_steps,
          momentum_sample_fn=create_momentum_sample_fn(hmc_state.state),
          target_log_prob_fn=log_prob_non_transformed)

      mean_p_accept = tf.reduce_mean(
          tf.exp(tf.minimum(0., hmc_extra.log_accept_ratio)))

      if FLAGS.mcmc_adapt_step_size:
        step_size = fun_mc.sign_adaptation(
            step_size, output=mean_p_accept, set_point=0.9)

      return (hmc_state, step_size, step + 1), hmc_extra

    hmc_state, is_accepted = fun_mc.trace(
        state=(fun_mc.hamiltonian_monte_carlo_init(x_neg_q,
                                                   log_prob_non_transformed),
               step_size, 0),
        fn=kernel,
        num_steps=FLAGS.mcmc_num_steps,
        trace_fn=lambda _, hmc_extra: hmc_extra.is_accepted)

    x_neg_p = hmc_state[0].state_extra[0]
    step_size = hmc_state[1]

    p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
  else:
    x_neg_p = x_neg_q
    p_accept = 0.0
    step_size = 0.0

  with tf.GradientTape() as tape:
    tape.watch(u.trainable_variables)
    pos_e = tf.reduce_mean(u(x_pos))
    neg_e_p = tf.reduce_mean(u(x_neg_p))
    loss = pos_e - neg_e_p + tf.square(pos_e) * FLAGS.p_center_regularizer

  variables = u.trainable_variables
  grads = tape.gradient(loss, variables)
  grads_and_vars = list(zip(grads, variables))
  opt_p.apply_gradients(grads_and_vars)

  pos_e_updated = tf.reduce_mean(u(x_pos))
  neg_e_p_updated = tf.reduce_mean(u(x_neg_p))

  return (x_neg_q, x_neg_p, p_accept, step_size, pos_e, pos_e_updated, neg_e_q,
          neg_e_p, neg_e_p_updated)


@tf.function(autograph=False)
def train_p_mh(q, u, x_pos, step_size, opt_p):
  """Train P using a CD objective with negatives generated via IID MH step.

  Args:
    q: `ModelQ`.
    u: A callable representing the energy function.
    x_pos: A batch of positive examples.
    step_size: Step size to use for HMC.
    opt_p: A `tf.optimizer.Optimizer`.

  Returns:
    x_neg_q: Negative samples sampled from `q`.
    x_neg_p: Negative samples used to train `p`, possibly generated via HMC.
    p_accept: Acceptance rate of HMC.
    step_size: The new step size, possibly adapted to adjust the acceptance
      rate.
    pos_e: Mean energy of the positive samples across the batch.
    pos_e: Mean energy of the positive samples across the batch, after the
      parameter update.
    neg_e_q: Mean energy of `x_neg_q` across the batch.
    neg_e_p: Mean energy of `x_neg_p` across the batch.
    neg_e_p_updated: Mean energy of `x_neg_p` across the batch, after the
      parameter update.
  """

  # (1) nt-iid q

  n = x_pos.shape[0]

  def p_log_prob(x):
    return -u(x)

  _, x1, g_x_1 = q.sample_with_log_prob(n=n, temp=1.0)
  _, x2, g_x_2 = q.sample_with_log_prob(n=n, temp=1.0)
  p_x_1 = p_log_prob(x1)
  p_x_2 = p_log_prob(x2)

  log_accept_ratio = p_x_2 - p_x_1 + g_x_1 - g_x_2
  log_accept_ratio_min = tf.math.minimum(
      tf.zeros_like(log_accept_ratio), log_accept_ratio)
  log_uniform = tf.math.log(
      tf.random.uniform(
          shape=tf.shape(log_accept_ratio), dtype=log_accept_ratio.dtype))

  is_accepted = log_uniform < log_accept_ratio_min

  def _expand_is_accepted_like(x):
    """Helper to expand `is_accepted` like the shape of some input arg."""
    with tf.name_scope('expand_is_accepted_like'):
      if x.shape is not None and is_accepted.shape is not None:
        expand_shape = list(is_accepted.shape) + [1] * (
            len(x.shape) - len(is_accepted.shape))
      else:
        expand_shape = tf.concat([
            tf.shape(is_accepted),
            tf.ones([tf.rank(x) - tf.rank(is_accepted)], dtype=tf.int32),
        ],
                                 axis=0)
      return tf.reshape(is_accepted, expand_shape)

  x = tf.where(_expand_is_accepted_like(x1), x2, x1)

  # (2) update p

  neg_e_q = tf.reduce_mean(u(x1))

  x_neg_q = x1
  x_neg_p = x
  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
  step_size = 0.0

  with tf.GradientTape() as tape:
    tape.watch(u.trainable_variables)
    pos_e = tf.reduce_mean(u(x_pos))
    neg_e_p = tf.reduce_mean(u(x_neg_p))
    loss = pos_e - neg_e_p + tf.square(pos_e) * FLAGS.p_center_regularizer

  # variables = tape.watched_variables()
  variables = u.trainable_variables
  grads = tape.gradient(loss, variables)
  grads_and_vars = list(zip(grads, variables))
  opt_p.apply_gradients(grads_and_vars)

  pos_e_updated = tf.reduce_mean(u(x_pos))
  neg_e_p_updated = tf.reduce_mean(u(x_neg_p))

  return (x_neg_q, x_neg_p, p_accept, step_size, pos_e, pos_e_updated, neg_e_q,
          neg_e_p, neg_e_p_updated)


def main(unused_args):
  del unused_args

  #
  # General setup.
  #

  ebm_util.init_tf2()

  ebm_util.set_seed(FLAGS.seed)

  output_dir = FLAGS.logdir
  checkpoint_dir = os.path.join(output_dir, 'checkpoint')
  samples_dir = os.path.join(output_dir, 'samples')

  tf.io.gfile.makedirs(samples_dir)
  tf.io.gfile.makedirs(checkpoint_dir)

  log_f = tf.io.gfile.GFile(os.path.join(output_dir, 'log.out'), mode='w')
  logger = ebm_util.setup_logging('main', log_f, console=False)
  logger.info({k: v._value for (k, v) in FLAGS._flags().items()})  # pylint: disable=protected-access

  #
  # Data
  #

  if FLAGS.dataset == 'mnist':
    x_train = ebm_util.mnist_dataset(N_CH)
  elif FLAGS.dataset == 'celeba':
    x_train = ebm_util.celeba_dataset()
  else:
    raise ValueError(f'Unknown dataset. {FLAGS.dataset}')
  train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(
      FLAGS.batch_size)

  #
  # Models
  #

  if FLAGS.q_type == 'mean_field_gaussian':
    q = MeanFieldGaussianQ()
  u = make_u()

  #
  # Optimizers
  #

  def lr_p(step):
    lr = FLAGS.p_learning_rate * (1. - (step / (1.5 * FLAGS.train_steps)))
    return lr

  def lr_q(step):
    lr = FLAGS.q_learning_rate * (1. - (step / (1.5 * FLAGS.train_steps)))
    return lr

  opt_q = tf.optimizers.Adam(learning_rate=ebm_util.LambdaLr(lr_q))
  opt_p = tf.optimizers.Adam(
      learning_rate=ebm_util.LambdaLr(lr_p), beta_1=FLAGS.p_adam_beta_1)

  #
  # Checkpointing
  #

  global_step_var = tf.Variable(0, trainable=False)
  checkpoint = tf.train.Checkpoint(
      opt_p=opt_p, opt_q=opt_q, u=u, q=q, global_step_var=global_step_var)

  checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
  if tf.io.gfile.exists(checkpoint_path + '.index'):
    print(f'Restoring from {checkpoint_path}')
    checkpoint.restore(checkpoint_path)

  #
  # Stats initialization
  #

  stat_i = []
  stat_keys = [
      'E_pos',  # Mean energy of the positive samples.
      'E_neg_q',  # Mean energy of the negative samples (pre-HMC).
      'E_neg_p',  # Mean energy of the negative samples (post-HMC).
      'H',  # Entropy of Q (if known).
      'pd_pos',  # Pairse differences of the positive samples.
      'pd_neg_q',  # Pairwise differences of the negative samples (pre-HMC).
      'pd_neg_p',  # Pairwise differences of the negative samples (post-HMC).
      'hmc_disp',  # L2 distance between initial and final entropyMC samples.
      'hmc_p_accept',  # entropyMC P(accept).
      'hmc_step_size',  # entropyMC step size.
      'x_neg_p_min',  # Minimum value of the negative samples (post-HMC).
      'x_neg_p_max',  # Maximum value of the negative samples (post-HMC).
      'time',  # Time taken to do the training step.
  ]
  stat = {k: [] for k in stat_keys}

  def array_to_str(a, fmt='{:>8.4f}'):
    return ' '.join([fmt.format(v) for v in a])

  def stats_callback(step, entropy, pd_neg_q):
    del step, entropy, pd_neg_q


  step_size = FLAGS.mcmc_step_size

  train_ds_iter = iter(train_ds)
  x_pos_1 = ebm_util.data_preprocess(next(train_ds_iter))
  x_pos_2 = ebm_util.data_preprocess(next(train_ds_iter))

  global_step = global_step_var.numpy()

  while global_step < (FLAGS.train_steps + 1):
    for x_pos in train_ds:

      # Drop partial batches.
      if x_pos.shape[0] != FLAGS.batch_size:
        continue

      #
      # Update
      #

      start_time = time.time()

      x_pos = ebm_util.data_preprocess(x_pos)
      x_pos = ebm_util.data_discrete_noise(x_pos)

      if FLAGS.p_loss == 'neutra_hmc':
        (x_neg_q, x_neg_p, p_accept, step_size, pos_e, pos_e_updated, neg_e_q,
         neg_e_p, neg_e_p_updated) = train_p(q, u, x_pos, step_size, opt_p)
      elif FLAGS.p_loss == 'neutra_iid':
        (x_neg_q, x_neg_p, p_accept, step_size, pos_e, pos_e_updated, neg_e_q,
         neg_e_p, neg_e_p_updated) = train_p_mh(q, u, x_pos, step_size, opt_p)
      else:
        raise ValueError(f'Unknown P loss {FLAGS.p_loss}')

      if FLAGS.q_loss == 'forward_kl':
        train_q_fwd_kl(q, x_neg_p, opt_q)
        entropy = 0.0
        mle_loss = 0.0
      elif FLAGS.q_loss == 'reverse_kl':
        for _ in range(10):
          _, entropy = train_q_rev_kl(q, u, opt_q)
        mle_loss = 0.0
      elif FLAGS.q_loss == 'reverse_kl_mle':
        for _ in range(FLAGS.q_sub_steps):
          alpha = FLAGS.q_rkl_weight
          (_, entropy, _, mle_loss, norm_grads_ebm,
           norm_grads_mle) = train_q_rev_kl_mle(q, u, x_pos,
                                                tf.convert_to_tensor(alpha),
                                                opt_q)

      elif FLAGS.q_loss == 'mle':
        mle_loss = train_q_mle(q, x_pos, opt_q)
        entropy = 0.0
      else:
        raise ValueError(f'Unknown Q loss {FLAGS.q_loss}')

      end_time = time.time()

      #
      # Stats
      #

      hmc_disp = tf.reduce_mean(
          tf.norm(
              tf.reshape(x_neg_q, [64, -1]) - tf.reshape(x_neg_p, [64, -1]),
              axis=1))

      if global_step % FLAGS.plot_steps == 0:

        # Positives + negatives.
        ebm_util.plot(
            tf.reshape(
                ebm_util.data_postprocess(x_neg_q),
                [FLAGS.batch_size, N_WH, N_WH, N_CH]),
            os.path.join(samples_dir, f'x_neg_q_{global_step}.png'))
        ebm_util.plot(
            tf.reshape(
                ebm_util.data_postprocess(x_neg_p),
                [FLAGS.batch_size, N_WH, N_WH, N_CH]),
            os.path.join(samples_dir, f'x_neg_p_{global_step}.png'))
        ebm_util.plot(
            tf.reshape(
                ebm_util.data_postprocess(x_pos),
                [FLAGS.batch_size, N_WH, N_WH, N_CH]),
            os.path.join(samples_dir, f'x_pos_{global_step}.png'))

        # Samples for various temperatures.
        for t in [0.1, 0.5, 1.0, 2.0, 4.0]:
          _, x_neg_q_t, _ = q.sample_with_log_prob(FLAGS.batch_size, temp=t)
          ebm_util.plot(
              tf.reshape(
                  ebm_util.data_postprocess(x_neg_q_t),
                  [FLAGS.batch_size, N_WH, N_WH, N_CH]),
              os.path.join(samples_dir, f'x_neg_t_{t}_{global_step}.png'))

        stats_callback(global_step, entropy,
                       ebm_util.nearby_difference(x_neg_q))

        stat_i.append(global_step)
        stat['E_pos'].append(pos_e_updated)
        stat['E_neg_q'].append(neg_e_q)
        stat['E_neg_p'].append(neg_e_p)
        stat['H'].append(entropy)
        stat['pd_neg_q'].append(ebm_util.nearby_difference(x_neg_q))
        stat['pd_neg_p'].append(ebm_util.nearby_difference(x_neg_p))
        stat['pd_pos'].append(ebm_util.nearby_difference(x_pos))
        stat['hmc_disp'].append(hmc_disp)
        stat['hmc_p_accept'].append(p_accept)
        stat['hmc_step_size'].append(step_size)
        stat['x_neg_p_min'].append(tf.reduce_min(x_neg_p))
        stat['x_neg_p_max'].append(tf.reduce_max(x_neg_p))
        stat['time'].append(end_time - start_time)

        ebm_util.plot_stat(stat_keys, stat, stat_i, output_dir)

        # Doing a linear interpolation in the latent space.
        z_pos_1 = q.forward(x_pos_1)[0]
        z_pos_2 = q.forward(x_pos_2)[0]

        x_alphas = []
        n_steps = 10
        for j in range(0, n_steps + 1):
          alpha = (j / n_steps)
          z_alpha = (1. - alpha) * z_pos_1 + (alpha) * z_pos_2
          x_alpha = q.reverse(z_alpha)[0]
          x_alphas.append(x_alpha)

        ebm_util.plot_n_by_m(
            ebm_util.data_postprocess(
                tf.reshape(
                    tf.stack(x_alphas, axis=1),
                    [(n_steps + 1) * FLAGS.batch_size, N_WH, N_WH, N_CH])),
            os.path.join(samples_dir, f'x_alpha_{global_step}.png'),
            FLAGS.batch_size, n_steps + 1)

        # Doing random perturbations in the latent space.
        for eps in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 2e0, 2.5e0, 3e0]:
          z_pos_2_eps = z_pos_2 + eps * tf.random.normal(z_pos_2.shape)
          x_alpha = q.reverse(z_pos_2_eps)[0]
          ebm_util.plot(
              tf.reshape(
                  ebm_util.data_postprocess(x_alpha),
                  [FLAGS.batch_size, N_WH, N_WH, N_CH]),
              os.path.join(samples_dir, f'x_alpha_eps_{eps}_{global_step}.png'))

        # Checking the log-probabilites of positive and negative examples under
        # Q.
        z_neg_test, x_neg_test, _ = q.sample_with_log_prob(
            FLAGS.batch_size, temp=FLAGS.q_temperature)
        z_pos_test = q.forward(x_pos)[0]

        z_neg_test_pd = ebm_util.nearby_difference(z_neg_test)
        z_pos_test_pd = ebm_util.nearby_difference(z_pos_test)

        z_norms_neg = tf.reduce_mean(tf.norm(z_neg_test, axis=1))
        z_norms_pos = tf.reduce_mean(tf.norm(z_pos_test, axis=1))

        log_prob_neg = tf.reduce_mean(q.log_prob(x_neg_test))
        log_prob_pos = tf.reduce_mean(q.log_prob(x_pos))

        logger.info('  '.join([
            f'i={global_step:6d}',
            # Pre-update, post-update
            (f'E_pos=[{pos_e:10.4f} {pos_e_updated:10.4f} ' +
             f'{pos_e_updated - pos_e:10.4f}]'),
            # Pre-update pre-HMC, pre-update post-HMC, post-update post-HMC
            (f'E_neg=[{neg_e_q:10.4f} {neg_e_p:10.4f} ' +
             f'{neg_e_p_updated:10.4f} {neg_e_p_updated - neg_e_p:10.4f}]'),
            f'mle={tf.reduce_mean(mle_loss):8.4f}',
            f'H={entropy:8.4f}',
            f'norm_grads_ebm={norm_grads_ebm:8.4f}',
            f'norm_grads_mle={norm_grads_mle:8.4f}',
            f'pd(x_pos)={ebm_util.nearby_difference(x_pos):8.4f}',
            f'pd(x_neg_q)={ebm_util.nearby_difference(x_neg_q):8.4f}',
            f'pd(x_neg_p)={ebm_util.nearby_difference(x_neg_p):8.4f}',
            f'hmc_disp={hmc_disp:8.4f}',
            f'p(accept)={p_accept:8.4f}',
            f'step_size={step_size:8.4f}',
            # Min, max.
            (f'x_neg_q=[{tf.reduce_min(x_neg_q):8.4f} ' +
             f'{tf.reduce_max(x_neg_q):8.4f}]'),
            (f'x_neg_p=[{tf.reduce_min(x_neg_p):8.4f} ' +
             f'{tf.reduce_max(x_neg_p):8.4f}]'),
            f'z_neg_norm={array_to_str(z_norms_neg)}',
            f'z_pos_norm={array_to_str(z_norms_pos)}',
            f'z_neg_test_pd={z_neg_test_pd:>8.2f}',
            f'z_pos_test_pd={z_pos_test_pd:>8.2f}',
            f'log_prob_neg={log_prob_neg:12.2f}',
            f'log_prob_pos={log_prob_pos:12.2f}',
        ]))

      if global_step % FLAGS.save_steps == 0:

        global_step_var.assign(global_step)
        checkpoint.write(os.path.join(checkpoint_dir, 'checkpoint'))

      global_step += 1


if __name__ == '__main__':
  flags.mark_flag_as_required('log_dir')
  app.run(main)
