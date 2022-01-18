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
"""Python binary for running the coupled estimator experiments."""

import os

from absl import app
from absl import flags
from absl import logging
import dataset
import networks as categorical_networks
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

layers = tf.keras.layers

flags.DEFINE_enum('dataset', 'static_mnist',
                  ['static_mnist', 'dynamic_mnist',
                   'fashion_mnist', 'omniglot',
                   'binarized_mnist', 'celeba'],
                  'Dataset to use.')
flags.DEFINE_float('genmo_lr', 1e-4,
                   'Learning rate for decoder, Generation network.')
flags.DEFINE_float('infnet_lr', 1e-4,
                   'Learning rate for encoder, Inference network.')
flags.DEFINE_float('prior_lr', 1e-2,
                   'Learning rate for prior variables.')
flags.DEFINE_integer('batch_size', 200, 'Training batch size.')
flags.DEFINE_integer('num_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_string('encoder_type', 'nonlinear',
                    'Choice supported: nonlinear')
flags.DEFINE_enum('grad_type', 'reinforce_loo',
                  ['reinforce_loo', 'arsm', 'ars', 'arsp', 'arsmp', 'disarm'],
                  'Gradient estimator type.')
flags.DEFINE_string('logdir', '/tmp/logdir',
                    'Directory for storing logs.')
flags.DEFINE_bool('verbose', False,
                  'Whether to turn on training result logging.')
flags.DEFINE_integer('repeat_idx', 0,
                     'Dummy flag to label the experiments in repeats.')
flags.DEFINE_bool('eager', False, 'Enable eager execution.')
flags.DEFINE_bool('bias_check', False,
                  'Carry out bias check for RELAX and baseline')
flags.DEFINE_bool('demean_input', False,
                  'Demean for encoder and decoder inputs.')
flags.DEFINE_bool('initialize_with_bias', False,
                  'Initialize the final layer bias of decoder '
                  'with dataset mean.')
flags.DEFINE_integer('seed', 1, 'Global random seed.')
flags.DEFINE_list('estimate_grad_basket', [],
                  'List of gradient estimators to compute in addition '
                  'for variance.')
flags.DEFINE_integer('num_eval_samples', 100,
                     'Number of samples for evaluation, default to None, '
                     'when the num_samples will be used.')
flags.DEFINE_integer('num_train_samples', 1,
                     'Number of samples for evaluation, default to None, '
                     'when the num_samples will be used.')
flags.DEFINE_integer('num_categories', 10,
                     'Number of categories for categorical variables.')
flags.DEFINE_integer('num_variables', 20,
                     'Number of hidden categorical varibles.')
flags.DEFINE_integer('num_samples', None,
                     'Number of samples for REINFORCE Baseline.'
                     'Default to None, when the num_categories will be used.')
flags.DEFINE_bool('stick_breaking', False,
                  'Use stick breaking augmentation for categorical variables.')
flags.DEFINE_bool('tree_structure', False,
                  'Use tree structure stick breaking.')
flags.DEFINE_bool('importance_weight', False,
                  'Use importance weight stick breaking.')
flags.DEFINE_bool('one_hot', False,
                  'Use one-hot categorical representation.')
flags.DEFINE_bool('debug', False, 'Turn on debugging mode.')
flags.DEFINE_string('logits_order', None,
                    'The order to sort the logits: [None, abs, ascending, '
                    'descending].')
flags.DEFINE_float('weight_scale', None,
                   'Scale of initializer.')
FLAGS = flags.FLAGS


def process_batch_input(input_batch):
  if FLAGS.dataset == 'celeba':
    return input_batch
  else:
    input_batch = tf.reshape(input_batch, [tf.shape(input_batch)[0], -1])
    input_batch = tf.cast(input_batch, tf.float32)
    return input_batch


def initialize_grad_variables(target_variable_list):
  return [tf.Variable(tf.zeros(shape=i.shape)) for i in target_variable_list]


def estimate_gradients(input_batch, bvae_model, gradient_type,
                       stick_breaking=False,
                       tree_structure=False,
                       importance_weight=False,
                       logits_sorting_order=None,
                       num_samples=None):
  """Estimate gradient for inference and generation networks."""
  if num_samples is None:
    num_samples = FLAGS.num_samples
  if gradient_type == 'reinforce_loo' and  stick_breaking:
    with tf.GradientTape(persistent=True) as tape:
      elbo, _, encoder_logits, _ = bvae_model(
          input_batch, num_samples=1,
          stick_breaking=True,
          tree_structure=tree_structure)
      genmo_loss = -1. * tf.reduce_mean(elbo)

      learning_signal, encoder_llk = bvae_model.get_layer_grad_estimation(
          input_batch,
          grad_type=gradient_type,
          num_samples=num_samples,
          stick_breaking=True,
          tree_structure=tree_structure,
          logits_sorting_order=logits_sorting_order)

      infnet_objective = tf.reduce_sum(
          tf.reduce_mean(tf.stop_gradient(-1. * learning_signal) * encoder_llk,
                         axis=0),  # reduce num_samples
          axis=0)  # reduce batch dims

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_grads = tape.gradient(
        infnet_objective,
        bvae_model.encoder_vars)

  elif gradient_type == 'disarm' and  importance_weight:
    with tf.GradientTape(persistent=True) as tape:
      elbo, _, encoder_logits, _ = bvae_model(
          input_batch, num_samples=1,
          stick_breaking=True,
          tree_structure=tree_structure)
      genmo_loss = -1. * tf.reduce_mean(elbo)

      learning_signal, encoder_llk_diff = bvae_model.get_layer_grad_estimation(
          input_batch,
          grad_type=gradient_type,
          num_samples=1,
          stick_breaking=True,
          tree_structure=False,
          logits_sorting_order=None,
          importance_weighting=True)

      infnet_objective = tf.reduce_sum(
          tf.reduce_mean(
              tf.stop_gradient(-1. * learning_signal) * encoder_llk_diff,
              axis=0),  # reduce num_samples
          axis=0)  # reduce batch dims

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_grads = tape.gradient(
        infnet_objective,
        bvae_model.encoder_vars)

  elif gradient_type == 'disarm' and  stick_breaking:
    with tf.GradientTape(persistent=True) as tape:
      elbo = bvae_model(input_batch, num_samples=1,
                        stick_breaking=True,
                        tree_structure=tree_structure,
                        logits_sorting_order=logits_sorting_order)[0]
      genmo_loss = -1. * tf.reduce_mean(elbo)

      learning_signal, encoder_logits = bvae_model.get_layer_grad_estimation(
          input_batch,
          grad_type=gradient_type,
          num_samples=1,  # num_samples,
          stick_breaking=True,
          tree_structure=tree_structure,
          logits_sorting_order=logits_sorting_order)

      infnet_objective = tf.reduce_sum(
          tf.reduce_mean(
              tf.stop_gradient(-1. * learning_signal) * encoder_logits,
              axis=0),  # reduce num_samples
          axis=0)  # reduce batch dims

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_grads = tape.gradient(
        infnet_objective,
        bvae_model.encoder_vars)

  elif gradient_type in ['reinforce_loo', 'arsm', 'ars', 'arsp', 'arsmp']:
    with tf.GradientTape(persistent=True) as tape:
      elbo, _, encoder_logits, _ = bvae_model(input_batch, stick_breaking=False)
      genmo_loss = -1. * tf.reduce_mean(elbo)

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_grad_multiplier = -1. * bvae_model.get_layer_grad_estimation(
        input_batch,
        grad_type=gradient_type,
        num_samples=num_samples)
    infnet_grads = tape.gradient(
        encoder_logits,
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
    grad_variable_dict,
    grad_sq_variable_dict):
  """Train Discrete VAE for 1 step."""
  metrics = {}
  input_batch = process_batch_input(train_batch_i)

  (genmo_grads, prior_grads, infnet_grads, genmo_loss) = estimate_gradients(
      input_batch, bvae_model, FLAGS.grad_type,
      stick_breaking=FLAGS.stick_breaking,
      tree_structure=FLAGS.tree_structure,
      importance_weight=FLAGS.importance_weight,
      logits_sorting_order=FLAGS.logits_order)

  genmo_vars = bvae_model.decoder_vars
  genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

  prior_vars = bvae_model.prior_vars
  prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

  infnet_vars = bvae_model.encoder_vars
  infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

  batch_size_sq = tf.cast(FLAGS.batch_size * FLAGS.batch_size, tf.float32)
  encoder_grad_var = bvae_model.compute_grad_variance(
      grad_variable_dict[FLAGS.grad_type],
      grad_sq_variable_dict[FLAGS.grad_type],
      infnet_grads) / batch_size_sq

  variance_dict = {}
  if (FLAGS.grad_type == 'reinforce_loo') and FLAGS.estimate_grad_basket:
    for method in FLAGS.estimate_grad_basket:
      if method == FLAGS.grad_type:
        continue  # Already computed
      if ('disarm' in method) and ('tree' not in method):
        main_method, logits_order = method.split('-')
        logits_order = None if logits_order == 'null' else logits_order
      (_, _, infnet_grads, _) = estimate_gradients(
          input_batch, bvae_model, main_method,
          stick_breaking=True,
          tree_structure=False,
          logits_sorting_order=logits_order)
      variance_dict[method] = bvae_model.compute_grad_variance(
          grad_variable_dict[method],
          grad_sq_variable_dict[method],
          infnet_grads) / batch_size_sq

  return (encoder_grad_var, variance_dict, genmo_loss, metrics)


# @tf.function
def evaluate(model, tf_dataset, max_step=1000, num_eval_samples=None,
             stick_breaking=False,
             tree_structure=False):
  """Evaluate the model."""
  if num_eval_samples:
    num_samples = num_eval_samples
  elif FLAGS.num_eval_samples:
    num_samples = FLAGS.num_eval_samples
  else:
    num_samples = FLAGS.num_samples
  # tf.print('Evaluate with samples: %d.', num_samples)
  loss = 0.
  n = 0.
  for batch in tf_dataset.map(process_batch_input):
    if n >= max_step:  # used for train_ds, which is a `repeat` dataset.
      break
    if num_samples > 1:
      batch_size = tf.shape(batch)[0]
      input_batch = tf.tile(
          batch, [num_samples] + [1] * (len(batch.shape)-1))
      elbo = tf.reshape(model(input_batch,
                              stick_breaking=stick_breaking,
                              tree_structure=tree_structure)[0],
                        [num_samples, batch_size])
      objectives = (tf.reduce_logsumexp(elbo, axis=0, keepdims=False) -
                    tf.math.log(tf.cast(tf.shape(elbo)[0], tf.float32)))
    else:
      objectives = model(batch,
                         stick_breaking=stick_breaking,
                         tree_structure=tree_structure)[0]
    loss -= tf.reduce_mean(objectives)
    n += 1.
  return loss / n


# @tf.function
def maxprob_histogram(
    model, tf_dataset, max_step=1000,
    stick_breaking=False,
    tree_structure=False):
  """Evaluate the model."""
  results = []
  n = 0
  for batch in tf_dataset.map(process_batch_input):
    if n >= max_step:  # used for train_ds, which is a `repeat` dataset.
      break
    encoder_logits = model(
        batch,
        stick_breaking=stick_breaking,
        tree_structure=tree_structure)[2]
    max_prob = tf.reshape(
        tf.math.reduce_max(
            tf.nn.softmax(encoder_logits, axis=-1),
            axis=-1),
        [-1])
    results.append(max_prob)
    n += 1
  return tf.concat(results, axis=-1)


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

    if step % 100 == 0:
      sigma = tf.math.sqrt(s / step)
      z_score = mu / (sigma / tf.math.sqrt(float(step)))
      tf.print(step, 'z_score: ', z_score, 'sigma: ', sigma)


@tf.function
def run_bias_check_step(
    train_batch_i,
    bvae_model,
    target_type='local-armpp',
    baseline_type='vimco'):
  """Run bias check for 1 batch."""
  input_batch = process_batch_input(train_batch_i)

  if target_type == 'disarm':
    infnet_grads = estimate_gradients(
        input_batch, bvae_model, 'disarm',
        stick_breaking=True,
        tree_structure=FLAGS.tree_structure,
        importance_weight=FLAGS.importance_weight,
        num_samples=1)[2]
    baseline_infnet_grads = estimate_gradients(
        input_batch, bvae_model, 'reinforce_loo',
        stick_breaking=False,
        tree_structure=False,
        num_samples=2)[2]
  else:
    infnet_grads = estimate_gradients(
        input_batch, bvae_model, target_type,
        stick_breaking=FLAGS.stick_breaking)[2]
    baseline_infnet_grads = estimate_gradients(
        input_batch, bvae_model, baseline_type,
        stick_breaking=False)[2]

  diff = tf.concat([tf.reshape(x - y, [-1])
                    for x, y in zip(infnet_grads, baseline_infnet_grads)],
                   axis=0)
  return tf.reduce_mean(diff)


def main(_):

  tf.random.set_seed(FLAGS.seed)

  logdir = FLAGS.logdir

  if not os.path.exists(logdir):
    os.makedirs(logdir)

  if FLAGS.eager:
    tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  genmo_lr = tf.constant(FLAGS.genmo_lr)
  infnet_lr = tf.constant(FLAGS.infnet_lr)
  prior_lr = tf.constant(FLAGS.prior_lr)

  genmo_optimizer = tf.keras.optimizers.Adam(learning_rate=genmo_lr)
  infnet_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr)
  prior_optimizer = tf.keras.optimizers.SGD(learning_rate=prior_lr)
  theta_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr,
                                             beta_1=0.999)

  batch_size = FLAGS.batch_size

  if FLAGS.dataset == 'celeba':
    train_ds, valid_ds, test_ds, train_ds_mean, train_size = (
        dataset.get_celeba_batch(batch_size))
    num_steps_per_epoch = int(train_size / batch_size)

    encoder = categorical_networks.CnnEncoderNetwork(
        hidden_size=FLAGS.num_variables,
        num_categories=FLAGS.num_categories,
        train_mean=train_ds_mean)
    decoder = categorical_networks.CnnDecoderNetwork(
        train_mean=train_ds_mean)

  else:
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
    elif FLAGS.dataset == 'binarized_mnist':
      train_ds, valid_ds, test_ds = dataset.get_binarized_mnist_batch(
          batch_size)
      train_size = 50000

    num_steps_per_epoch = int(train_size / batch_size)
    train_ds_mean = dataset.get_mean_from_iterator(
        train_ds, dataset_size=train_size, batch_size=batch_size)

    if FLAGS.initialize_with_bias:
      bias_value = -tf.math.log(
          1./tf.clip_by_value(train_ds_mean, 0.001, 0.999) - 1.)
      bias_initializer = tf.keras.initializers.Constant(bias_value)
    else:
      bias_initializer = 'zeros'

    if FLAGS.encoder_type == 'nonlinear':
      encoder_hidden_sizes = [512, 256, FLAGS.num_variables]
      encoder_activations = [layers.LeakyReLU(0.2),
                             layers.LeakyReLU(0.2),
                             None]
      decoder_hidden_sizes = [256, 512, 784]
      decoder_activations = [layers.LeakyReLU(0.2),
                             layers.LeakyReLU(0.2),
                             None]
    elif FLAGS.encoder_type == 'linear':
      encoder_hidden_sizes = [FLAGS.num_variables]
      encoder_activations = [None]
      decoder_hidden_sizes = [784]
      decoder_activations = [None]
    else:
      raise NotImplementedError

    if FLAGS.weight_scale is not None:
      kernel_initializer = tf.keras.initializers.VarianceScaling(
          scale=FLAGS.weight_scale, seed=FLAGS.seed)
    else:
      kernel_initializer = 'glorot_uniform'

    encoder = categorical_networks.CategoricalNetwork(
        encoder_hidden_sizes,
        encoder_activations,
        num_categories=FLAGS.num_categories,
        mean_xs=train_ds_mean,
        demean_input=FLAGS.demean_input,
        name='bvae_encoder',
        kernel_initializer=kernel_initializer)
    decoder = categorical_networks.BinaryNetwork(
        decoder_hidden_sizes,
        decoder_activations,
        demean_input=FLAGS.demean_input,
        final_layer_bias_initializer=bias_initializer,
        name='bvae_decoder',
        kernel_initializer=kernel_initializer)

  prior_logit = tf.Variable(
      tf.zeros([FLAGS.num_variables, FLAGS.num_categories], tf.float32))

  if FLAGS.grad_type == 'relax':
    control_network = tf.keras.Sequential()
    control_network.add(
        layers.Dense(137, activation=layers.LeakyReLU(alpha=0.3)))
    control_network.add(
        layers.Dense(1))
  else:
    control_network = None

  bvae_model = categorical_networks.CategoricalVAE(
      encoder,
      decoder,
      prior_logit,
      FLAGS.num_categories,
      one_hot_sample=FLAGS.one_hot,
      grad_type=FLAGS.grad_type)

  if FLAGS.dataset == 'celeba':
    bvae_model.build(input_shape=(FLAGS.batch_size, 64, 64, 3))
  else:
    bvae_model.build(input_shape=(FLAGS.batch_size, 784))

  tensorboard_file_writer = tf.summary.create_file_writer(logdir)

  # In order to use `tf.train.ExponentialMovingAverage`, one has to
  # use `tf.Variable`.
  grad_variable_dict = {}
  grad_sq_variable_dict = {}
  for method in set([FLAGS.grad_type] + FLAGS.estimate_grad_basket):
    grad_variable_dict[method] = initialize_grad_variables(
        bvae_model.encoder_vars)
    grad_sq_variable_dict[method] = initialize_grad_variables(
        bvae_model.encoder_vars)

  ckpt = tf.train.Checkpoint(
      genmo_optimizer=genmo_optimizer,
      infnet_optimizer=infnet_optimizer,
      theta_optimizer=theta_optimizer,
      grad_variable_dict=grad_variable_dict,
      grad_sq_variable_dict=grad_sq_variable_dict,
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
              num_eval_samples=FLAGS.num_train_samples,
              stick_breaking=FLAGS.stick_breaking,
              tree_structure=FLAGS.tree_structure),
          'eval_metric/valid': evaluate(
              bvae_model, valid_ds,
              num_eval_samples=FLAGS.num_eval_samples,
              stick_breaking=FLAGS.stick_breaking,
              tree_structure=FLAGS.tree_structure),
          'eval_metric/test': evaluate(
              bvae_model, test_ds,
              num_eval_samples=FLAGS.num_eval_samples,
              stick_breaking=FLAGS.stick_breaking,
              tree_structure=FLAGS.tree_structure),
          f'var/{FLAGS.grad_type}': encoder_grad_var,
      })

      if FLAGS.grad_type == 'relax':
        if FLAGS.temperature is None:
          metrics['relax/temperature'] = tf.math.exp(
              bvae_model.log_temperature_variable)
        if FLAGS.scaling_factor is None:
          metrics['relax/scaling'] = bvae_model.scaling_variable
      tf.print(step_i, metrics)

      max_prob = maxprob_histogram(
          bvae_model, train_ds,
          stick_breaking=FLAGS.stick_breaking,
          tree_structure=FLAGS.tree_structure)

      with tensorboard_file_writer.as_default():
        for k, v in metrics.items():
          tf.summary.scalar(k, v, step=step_i)
        tf.summary.histogram('max_prob', max_prob, step=step_i)
        if variance_dict:  # if variance_dict == {}
          tf.print(variance_dict)
          for k, v in variance_dict.items():
            tf.summary.scalar(f'var/{k}_minus_{FLAGS.grad_type}',
                              v - encoder_grad_var, step=step_i)
            tf.summary.scalar(f'var/{k}', v, step=step_i)

    # Checkpoint
    if step_i % 10000 == 0:
      ckpt_save_path = ckpt_manager.save()
      logging.info('Saving checkpoint for step %d at %s.',
                   step_i, ckpt_save_path)

  if FLAGS.bias_check:
    if FLAGS.grad_type == 'reinforce_loo':
      baseline_type = 'ars'
    else:
      baseline_type = 'reinforce_loo'
    run_bias_check(bvae_model,
                   train_iter.next(),
                   FLAGS.grad_type,
                   baseline_type)


if __name__ == '__main__':
  app.run(main)
