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
"""Binary for the DisARM experiments on VAE with multiple stochastic layers."""

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_probability as tfp
from disarm import dataset
from disarm import networks

tfd = tfp.distributions

layers = tf.keras.layers


flags.DEFINE_enum('dataset', 'dynamic_mnist',
                  ['static_mnist', 'dynamic_mnist',
                   'fashion_mnist', 'omniglot'],
                  'Dataset to use.')
flags.DEFINE_float('genmo_lr', 1e-4,
                   'Learning rate for decoder, Generation network.')
flags.DEFINE_float('infnet_lr', 1e-4,
                   'Learning rate for encoder, Inference network.')
flags.DEFINE_float('prior_lr', 1e-2,
                   'Learning rate for prior variables.')
flags.DEFINE_integer('batch_size', 50, 'Training batch size.')
flags.DEFINE_integer('num_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_enum('grad_type', 'disarm',
                  ['arm', 'disarm', 'reinforce_loo', 'relax'],
                  'Choice of gradient estimator.')
flags.DEFINE_string('logdir', '/tmp/disarm',
                    'Directory for storing logs.')
flags.DEFINE_bool('verbose', False,
                  'Whether to turn on training result logging.')
flags.DEFINE_float('temperature', None,
                   'Temperature for RELAX estimator.')
flags.DEFINE_float('scaling_factor', None,
                   'Scaling factor for RELAX estimator.')
flags.DEFINE_bool('bias_check', False,
                  'Carry out bias check for RELAX and baseline')
flags.DEFINE_bool('demean_input', False,
                  'Demean for encoder and decoder inputs.')
flags.DEFINE_bool('initialize_with_bias', False,
                  'Initialize the final layer bias of decoder '
                  'with dataset mean.')
flags.DEFINE_integer('seed', 1, 'Global random seed.')
flags.DEFINE_bool('estimate_grad_basket', False,
                  'Estimate gradients for multiple estimators.')
flags.DEFINE_integer('num_eval_samples', 100,
                     'Number of samples for evaluation')
flags.DEFINE_integer('num_train_samples', 1,
                     'Number of samples for evaluation')
flags.DEFINE_bool('shared_randomness', False,
                  'Whether to use shared randomness in multilayer setting.')
flags.DEFINE_bool('debug', False, 'Turn on debugging mode.')
FLAGS = flags.FLAGS


def process_batch_input(input_batch):
  input_batch = tf.reshape(input_batch, [tf.shape(input_batch)[0], -1])
  input_batch = tf.cast(input_batch, tf.float32)
  return input_batch


def initialize_grad_variables(target_variable_list):
  return [tf.Variable(tf.zeros(shape=i.shape)) for i in target_variable_list]


def estimate_gradients(input_batch, bvae_model, gradient_type):
  """Estimate gradient for inference and generation networks."""
  if gradient_type == 'relax':
    with tf.GradientTape(persistent=True) as tape:
      genmo_loss, reparam_loss, learning_signal, log_q = (
          bvae_model.get_multilayer_relax_loss(
              input_batch,
              temperature=FLAGS.temperature,
              scaling_factor=FLAGS.scaling_factor))

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_grads = []
    for i in range(len(bvae_model.encoder_list)):
      log_q_i = log_q[i]
      learning_signal_i = learning_signal[i]
      reparam_loss_i = reparam_loss[i]

      infnet_vars_i = bvae_model.encoder_list[i].trainable_variables
      infnet_grads_1_i = tape.gradient(log_q_i, infnet_vars_i,
                                       output_gradients=learning_signal_i)
      infnet_grads_2_i = tape.gradient(reparam_loss_i, infnet_vars_i)
      # infnet_grads_1/2 are list of tf.Tensors.
      infnet_grads_i = [infnet_grads_1_i[j] + infnet_grads_2_i[j]
                        for j in range(len(infnet_vars_i))]
      infnet_grads.extend(infnet_grads_i)

  else:
    with tf.GradientTape(persistent=True) as tape:
      elbo, sample_list, infnet_logits, _ = bvae_model(input_batch)
      genmo_loss = -1. * tf.reduce_mean(elbo)

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    batched_infnet_grad = []
    uniform_sample_list = bvae_model.get_multilayer_uniform_sample(
        batch_shape=input_batch.shape[:1])
    for l in range(bvae_model.num_layers):
      grad_layer_l = bvae_model.get_multilayer_grad_estimation(
          sample_list, uniform_sample_list,
          grad_type=gradient_type, start_layer=l)
      batched_infnet_grad.append(grad_layer_l)
    infnet_grad_multiplier = -1. * tf.concat(batched_infnet_grad, axis=-1)
    infnet_grads = tape.gradient(
        infnet_logits,
        bvae_model.encoder_vars,
        output_gradients=infnet_grad_multiplier)

  del tape
  return (genmo_grads, prior_grads, infnet_grads, genmo_loss)


@tf.function
def train_one_step(
    train_batch_i,
    bvae_model,
    genmo_optimizer,
    infnet_optimizer,
    prior_optimizer,
    theta_optimizer,
    encoder_grad_variable,
    encoder_grad_sq_variable,
    grad_variable_dict,
    grad_sq_variable_dict):
  """Train Discrete VAE for 1 step."""
  metrics = {}
  input_batch = process_batch_input(train_batch_i)

  if FLAGS.grad_type == 'relax':
    with tf.GradientTape(persistent=True) as theta_tape:
      (genmo_grads, prior_grads, infnet_grads, genmo_loss) = estimate_gradients(
          input_batch, bvae_model, FLAGS.grad_type)

      # Update generative model
      genmo_vars = bvae_model.decoder_vars
      genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

      prior_vars = bvae_model.prior_vars
      prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

      infnet_vars = bvae_model.encoder_vars
      infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

      infnet_grads_sq = [tf.square(grad_i) for grad_i in infnet_grads]
      theta_vars = []
      if bvae_model.control_nn:
        theta_vars.extend(bvae_model.control_nn.trainable_variables)
      if FLAGS.temperature is None:
        theta_vars.append(bvae_model.log_temperature_variable)
      if FLAGS.scaling_factor is None:
        theta_vars.append(bvae_model.scaling_variable)
      theta_grads = theta_tape.gradient(infnet_grads_sq, theta_vars)
      theta_optimizer.apply_gradients(zip(theta_grads, theta_vars))
    del theta_tape

    metrics['learning_signal'] = bvae_model.mean_learning_signal

  else:
    (genmo_grads, prior_grads, infnet_grads, genmo_loss) = estimate_gradients(
        input_batch, bvae_model, FLAGS.grad_type)

    genmo_vars = bvae_model.decoder_vars
    genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

    prior_vars = bvae_model.prior_vars
    prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

    infnet_vars = bvae_model.encoder_vars
    infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

  batch_size_sq = tf.cast(FLAGS.batch_size * FLAGS.batch_size, tf.float32)
  encoder_grad_var = bvae_model.compute_grad_variance(
      encoder_grad_variable, encoder_grad_sq_variable,
      infnet_grads) / batch_size_sq

  if grad_variable_dict is not None:
    variance_dict = dict()
    for k in grad_variable_dict.keys():
      encoder_grads = estimate_gradients(
          input_batch, bvae_model, gradient_type=k)[2]
      variance_dict['var/' + k] = bvae_model.compute_grad_variance(
          grad_variable_dict[k], grad_sq_variable_dict[k],
          encoder_grads) / batch_size_sq
  else:
    variance_dict = None

  return (encoder_grad_var, variance_dict, genmo_loss, metrics)


@tf.function
def evaluate(model, tf_dataset, max_step=1000, num_eval_samples=None):
  """Evaluate the model."""
  if num_eval_samples:
    num_samples = num_eval_samples
  elif FLAGS.num_eval_samples:
    num_samples = FLAGS.num_eval_samples
  else:
    num_samples = 1
  tf.print('Evaluate with samples: ', num_samples)
  loss = 0.
  n = 0.
  for batch in tf_dataset.map(process_batch_input):
    if n >= max_step:  # used for train_ds, which is a `repeat` dataset.
      break
    if num_samples > 1:
      batch_size = tf.shape(batch)[0]
      input_batch = tf.tile(batch, [num_samples, 1])
      elbo = tf.reshape(model(input_batch)[0], [num_samples, batch_size])
      objectives = (tf.reduce_logsumexp(elbo, axis=0, keepdims=False) -
                    tf.math.log(tf.cast(tf.shape(elbo)[0], tf.float32)))
    else:
      objectives = model(batch)[0]
    loss -= tf.reduce_mean(objectives)
    n += 1.
  return loss / n


def run_bias_check(model, batch, target_type, baseline_type):
  """Run bias check."""
  tf.print(f'Running a bias check comparing {target_type} and {baseline_type}.')
  mu = 0.
  s = 0.
  for step in range(1, int(1e6) + 1):
    diff = run_bias_check_step(
        batch,
        model,
        target_type=target_type,
        baseline_type=baseline_type)
    prev_mu = mu
    mu = mu + (diff - mu) / step
    s = s + (diff - mu) * (diff - prev_mu)

    if step % 1000 == 0:
      sigma = tf.math.sqrt(s / step)
      z_score = mu / (sigma / tf.math.sqrt(float(step)))
      tf.print(step, 'z_score: ', z_score, 'sigma: ', sigma)


@tf.function
def run_bias_check_step(
    train_batch_i,
    bvae_model,
    target_type='disarm',
    baseline_type='reinforce_loo'):
  """Run bias check for 1 batch."""
  input_batch = process_batch_input(train_batch_i)

  infnet_grads = estimate_gradients(
      input_batch, bvae_model, target_type)[2]
  baseline_infnet_grads = estimate_gradients(
      input_batch, bvae_model, baseline_type)[2]

  diff = tf.concat([tf.reshape(x - y, [-1])
                    for x, y in zip(infnet_grads, baseline_infnet_grads)],
                   axis=0)
  return tf.reduce_mean(diff)


def main(_):
  tf.random.set_seed(FLAGS.seed)

  logdir = FLAGS.logdir

  os.makedirs(logdir, exist_ok=True)

  genmo_lr = tf.constant(FLAGS.genmo_lr)
  infnet_lr = tf.constant(FLAGS.infnet_lr)
  prior_lr = tf.constant(FLAGS.prior_lr)

  genmo_optimizer = tf.keras.optimizers.Adam(learning_rate=genmo_lr)
  infnet_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr)
  prior_optimizer = tf.keras.optimizers.SGD(learning_rate=prior_lr)
  theta_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr,
                                             beta_1=0.999)

  batch_size = FLAGS.batch_size

  if FLAGS.dataset == 'static_mnist':
    train_ds, valid_ds, test_ds = dataset.get_static_mnist_batch(batch_size)
    train_size = 50000
  elif FLAGS.dataset == 'dynamic_mnist':
    train_ds, valid_ds, test_ds = dataset.get_dynamic_mnist_batch(batch_size)
    train_size = 50000
  elif FLAGS.dataset == 'fashion_mnist':
    train_ds, valid_ds, test_ds = dataset.get_dynamic_mnist_batch(
        batch_size, fashion_mnist=True)
    train_size = 50000
  elif FLAGS.dataset == 'omniglot':
    train_ds, valid_ds, test_ds = dataset.get_omniglot_batch(batch_size)
    train_size = 23000

  num_steps_per_epoch = int(train_size / batch_size)
  train_ds_mean = dataset.get_mean_from_iterator(
      train_ds, dataset_size=train_size, batch_size=batch_size)

  if FLAGS.initialize_with_bias:
    bias_value = -tf.math.log(
        1./tf.clip_by_value(train_ds_mean, 0.001, 0.999) - 1.)
    bias_initializer = tf.keras.initializers.Constant(bias_value)
  else:
    bias_initializer = 'zeros'

  encoder = [
      networks.BinaryNetwork(
          [200],
          ['linear'],
          mean_xs=train_ds_mean,
          demean_input=FLAGS.demean_input,
          name='bvae_encoder1'),
      networks.BinaryNetwork(
          [200],
          ['linear'],
          name='bvae_encoder2'),
      networks.BinaryNetwork(
          [200],
          ['linear'],
          name='bvae_encoder3'),
      networks.BinaryNetwork(
          [200],
          ['linear'],
          name='bvae_encoder4')]
  decoder = [
      networks.BinaryNetwork(
          [200],
          ['linear'],
          name='bvae_decoder1'),
      networks.BinaryNetwork(
          [200],
          ['linear'],
          name='bvae_decoder2'),
      networks.BinaryNetwork(
          [200],
          ['linear'],
          name='bvae_decoder3'),
      networks.BinaryNetwork(
          [784],
          ['linear'],
          demean_input=FLAGS.demean_input,
          final_layer_bias_initializer=bias_initializer,
          name='bvae_decoder4')]

  prior_logit = tf.Variable(tf.zeros([200], tf.float32))

  if FLAGS.grad_type == 'relax':
    control_network = []
    for _ in range(len(encoder)):
      control_network_i = tf.keras.Sequential()
      control_network_i.add(
          layers.Dense(137, activation=layers.LeakyReLU(alpha=0.3)))
      control_network_i.add(
          layers.Dense(1))
      control_network.append(control_network_i)
  else:
    control_network = None

  bvae_model = networks.DiscreteVAE(
      encoder,
      decoder,
      prior_logit,
      grad_type=FLAGS.grad_type,
      control_nn=control_network,
      shared_randomness=FLAGS.shared_randomness)

  bvae_model.build(input_shape=(None, 784))

  tensorboard_file_writer = tf.summary.create_file_writer(logdir)

  # In order to use `tf.train.ExponentialMovingAverage`, one has to
  # use `tf.Variable`.
  encoder_grad_variable = initialize_grad_variables(bvae_model.encoder_vars)
  encoder_grad_sq_variable = initialize_grad_variables(bvae_model.encoder_vars)

  if FLAGS.estimate_grad_basket:
    if FLAGS.grad_type == 'reinforce_loo':
      grad_basket = ['arm', 'disarm', 'reinforce_loo', 'relax']
    else:
      raise NotImplementedError

    grad_variable_dict = {
        k: initialize_grad_variables(bvae_model.encoder_vars)
        for k in grad_basket}
    grad_sq_variable_dict = {
        k: initialize_grad_variables(bvae_model.encoder_vars)
        for k in grad_basket}
    ckpt = tf.train.Checkpoint(
        genmo_optimizer=genmo_optimizer,
        infnet_optimizer=infnet_optimizer,
        theta_optimizer=theta_optimizer,
        encoder_grad_variable=encoder_grad_variable,
        encoder_grad_sq_variable=encoder_grad_sq_variable,
        grad_variable_dict=grad_variable_dict,
        grad_sq_variable_dict=grad_sq_variable_dict,
        bvae_model=bvae_model)

  else:
    grad_variable_dict = None
    grad_sq_variable_dict = None

    ckpt = tf.train.Checkpoint(
        genmo_optimizer=genmo_optimizer,
        infnet_optimizer=infnet_optimizer,
        theta_optimizer=theta_optimizer,
        encoder_grad_variable=encoder_grad_variable,
        encoder_grad_sq_variable=encoder_grad_sq_variable,
        bvae_model=bvae_model)

  ckpt_manager = tf.train.CheckpointManager(
      ckpt, logdir, max_to_keep=5)

  if not FLAGS.debug and ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    logging.info(
        'Last checkpoint was restored: %s.', ckpt_manager.latest_checkpoint)
  else:
    tf.print('No checkpoint to load.')
    logging.info('No checkpoint to load.')

  start_step = infnet_optimizer.iterations.numpy()
  logging.info('Training start from step: %s', start_step)

  train_iter = train_ds.__iter__()
  for step_i in range(start_step, FLAGS.num_steps):
    (encoder_grad_var, variance_dict, genmo_loss, metrics) = train_one_step(
        train_iter.next(),
        bvae_model,
        genmo_optimizer,
        infnet_optimizer,
        prior_optimizer,
        theta_optimizer,
        encoder_grad_variable,
        encoder_grad_sq_variable,
        grad_variable_dict,
        grad_sq_variable_dict)
    train_loss = tf.reduce_mean(genmo_loss)

    # Summarize
    if step_i % 1000 == 0:
      metrics.update({
          'train_objective': train_loss,
          'eval_metric/train': evaluate(
              bvae_model, train_ds,
              max_step=num_steps_per_epoch,
              num_eval_samples=FLAGS.num_train_samples),
          'eval_metric/valid': evaluate(
              bvae_model, valid_ds,
              num_eval_samples=FLAGS.num_eval_samples),
          'eval_metric/test': evaluate(
              bvae_model, test_ds,
              num_eval_samples=FLAGS.num_eval_samples),
          'var/grad': encoder_grad_var
      })
      if FLAGS.grad_type == 'relax':
        if FLAGS.temperature is None:
          metrics['relax/temperature'] = tf.math.exp(
              bvae_model.log_temperature_variable)
        if FLAGS.scaling_factor is None:
          metrics['relax/scaling'] = bvae_model.scaling_variable
      tf.print(step_i, metrics)

      with tensorboard_file_writer.as_default():
        for k, v in metrics.items():
          tf.summary.scalar(k, v, step=step_i)
        if variance_dict is not None:
          tf.print(variance_dict)
          for k, v in variance_dict.items():
            tf.summary.scalar(k, v, step=step_i)

    # Checkpoint
    if step_i % 10000 == 0:
      ckpt_save_path = ckpt_manager.save()
      logging.info('Saving checkpoint for step %d at %s.',
                   step_i, ckpt_save_path)

  if FLAGS.bias_check:
    if FLAGS.grad_type == 'reinforce_loo':
      baseline_type = 'disarm'
    else:
      baseline_type = 'reinforce_loo'
    run_bias_check(bvae_model,
                   train_iter.next(),
                   FLAGS.grad_type,
                   baseline_type)


if __name__ == '__main__':
  app.run(main)
