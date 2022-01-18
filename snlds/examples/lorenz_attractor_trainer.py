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

r"""Example SNLDS script.

The script can be launch locally with:
python -m snlds/examples/lorenz_attractor_trainer \
 --num_steps=100000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from snlds import model_cavi_snlds
from snlds import utils
from snlds.examples import config_utils
from snlds.examples import datasets
from snlds.examples import tensorboard_utils


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Mini-batch size for training.")
flags.DEFINE_integer(
    "hidden_dim",
    default=8,
    help="Number of continuous latent variables, z[t].")
flags.DEFINE_string(
    "logdir",  # `log_dir` is already defined by absl
    default="/tmp/snlds/logs/{timestamp}",
    help="Directory in which to write TensorBoard logs.")
flags.DEFINE_integer(
    "log_steps",
    default=2000,
    help="Fequency, in steps, of TensorBoard logging.")
flags.DEFINE_string(
    "model_dir",
    default="/tmp/snlds/models/{timestamp}",
    help="Directory in which to save model checkpoints.")
flags.DEFINE_integer(
    "num_categories",
    default=3,
    help="Number of categories/regimes, s[t].")
flags.DEFINE_integer(
    "num_steps",
    default=100000,
    help="Number of steps to train the model.")
flags.DEFINE_integer(
    "seed",
    default=131,
    help="Random seed for tensorflow and various generators in the program.")
flags.DEFINE_boolean(
    "use_diff",
    default=False,
    help="Use position difference as input of model.")
flags.DEFINE_integer(
    "rnndim",
    default=4,
    help="RNN hidden state size.")
flags.DEFINE_string(
    "rnntype",
    default="simplernn",
    help="Transition RNN type.")
flags.DEFINE_integer(
    "num_samples",
    default=1,
    help="Number of trajectories to sample from posterior.")
flags.DEFINE_enum(
    "objective",
    default="elbo",
    enum_values=["elbo", "iwae"],
    help="Type of lower bound to use for the objective.")

# Cross Entropy Regularization Config
flags.DEFINE_boolean(
    "cross_entropy_annealing",
    default=True,
    help="Whether to use cross entropy annealing for each time-step.")
flags.DEFINE_float(
    "xent_init",  # "cross_entropy_initial_coef",
    default=0.,
    help="Initial value of multiplier for cross entropy penalty.")
flags.DEFINE_float(
    "xent_rate",  # "cross entropy decay rate"
    default=0.99,
    help="Entropy exponential decay rate for cross entropy penalty.")
flags.DEFINE_float(
    "xent_steps",  # "cross entropy decay steps"
    default=50,
    help="Steps for update cross entropy penalty.")
flags.DEFINE_integer(
    "xent_kickin_steps",
    default=0,
    help="Kickin point for cross entropy coefficient decay.")

# Learning Rate Schedule Config
flags.DEFINE_boolean(
    "flat_learning_rate",
    default=False,
    help="Whether to use flat learning rate instead of learning rate decay")
flags.DEFINE_float(
    "learning_rate",
    default=1.e-4,
    help="Learning rate during training.")
flags.DEFINE_boolean(
    "use_inverse_annealing_lr",
    default=False,
    help="Use inverse annealing for Learning Rate decay.")


# Temperature Annealing Config
flags.DEFINE_boolean(
    "temperature_annealing",
    default=True,
    help="Whether to use temperature annealing for discrete state transitions.")
flags.DEFINE_float(
    "t_init",  # "initial_temperature",
    default=0.,
    help="Initial value of temperature annealing schedule.")
flags.DEFINE_float(
    "t_min",
    default=1.,
    help="Minimal temperature after decay.")
flags.DEFINE_float(
    "annealing_rate",  # "temperature decay rates"
    default=0.99,
    help="Temperature exponential decay rate.")
flags.DEFINE_integer(
    "annealing_steps",
    default=50,
    help="Temperature exponential decay step.")
flags.DEFINE_integer(
    "annealing_kickin_steps",
    default=0,
    help="Starting point of temperature annealing.")


@tf.function
def eval_step(test_batch, snlds_model, num_samples, temperature):
  """Runs evaluation of model on the test set and returns evaluation metrics.

  Args:
    test_batch: a batch of the test data.
    snlds_model: tf.keras.Model, SNLDS model to be evaluated.
    num_samples: int, number of samples per trajectories to use at eval time.
    temperature: float, annealing temperature to use on the model.

  Returns:
    Dictionary of metrics, str -> list[tf.Tensor],
      aggregates the result dictionaries returned by the model.
  """
  test_values = collections.defaultdict(list)
  for _ in range(10):
    result_dict = snlds_model(
        test_batch, temperature, num_samples=num_samples)
    for k, v in result_dict.items():
      test_values[k].append(v)

  return test_values


@tf.function
def train_step(train_batch, snlds_model, optimizer, num_samples,
               objective, learning_rate, temperature, cross_entropy_coef):
  """Runs one training step and returns metrics evaluated on the train set.

  Args:
    train_batch:  a batch of the training set.
    snlds_model: tf.keras.Model, model to be trained.
    optimizer: tf.keras.optimizers.Optimizer,
      optimizer to use for back-propagation.
    num_samples: int, number of samples per trajectories to use at train time.
    objective: str, which objective to use ("elbo" or "iwae").
    learning_rate: float, learning rate to use for back-propagation.
    temperature: float, annealing temperature to use on the model.
    cross_entropy_coef: float, weight of the cross-entropy loss.

  Returns:
    Dictionary of metrics, str -> tf.Tensor
      log_likelihood: value of the estimated train loglikelihood.
      cross_entropy: value of the estimated cross-entropy loss.
      objective: value of the estimate objective to minimize.
  """
  with tf.GradientTape() as tape:
    train_result = snlds_model(
        train_batch, temperature, num_samples=num_samples)
    train_loglikelihood = train_result[objective]
    train_xent = train_result["cross_entropy"]
    # TODO(zhedong): populate snlds_model output with
    # train_result["elbo_training_objective"],
    # train_result["iwae_training_objective"], and
    # train_result["entropy_regularizer"] to improve usability.
    train_objective = -1. * (
        train_loglikelihood + cross_entropy_coef * train_xent)

    grads = tape.gradient(train_objective, snlds_model.trainable_variables)
    clipped_grads = [tf.clip_by_value(grad, -5.0, 5.0) for grad in grads]
    optimizer.learning_rate = learning_rate
    optimizer.apply_gradients(
        list(zip(clipped_grads, snlds_model.trainable_variables)))
    train_loglikelihood = tf.reduce_mean(train_loglikelihood)
    train_xent = tf.reduce_mean(train_xent)

    train_result["objective"] = train_objective
    return train_result


def main(argv):
  del argv  # unused

  tf.random.set_seed(FLAGS.seed)

  timestamp = datetime.datetime.strftime(datetime.datetime.today(),
                                         "%y%m%d_%H%M%S")
  logdir = FLAGS.logdir.format(
      timestamp=timestamp)
  model_dir = FLAGS.model_dir.format(
      timestamp=timestamp)
  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

  ##############################################
  # populate the flags
  ##############################################
  train_data_config = config_utils.get_data_config(
      batch_size=FLAGS.batch_size)
  test_data_config = config_utils.get_data_config(
      batch_size=1)

  # regularization and annealing config
  cross_entropy_config = config_utils.get_cross_entropy_config(
      decay_rate=FLAGS.xent_rate,
      decay_steps=FLAGS.xent_steps,
      initial_value=FLAGS.xent_init,
      kickin_steps=FLAGS.xent_kickin_steps,
      use_entropy_annealing=FLAGS.cross_entropy_annealing)

  learning_rate_config = config_utils.get_learning_rate_config(
      flat_learning_rate=FLAGS.flat_learning_rate,
      inverse_annealing_lr=FLAGS.use_inverse_annealing_lr,
      decay_steps=FLAGS.num_steps,
      learning_rate=FLAGS.learning_rate,
      warmup_steps=1000)

  temperature_config = config_utils.get_temperature_config(
      decay_rate=FLAGS.annealing_rate,
      decay_steps=FLAGS.annealing_steps,
      initial_temperature=FLAGS.t_init,
      minimal_temperature=FLAGS.t_min,
      kickin_steps=FLAGS.annealing_kickin_steps,
      use_temperature_annealing=FLAGS.temperature_annealing)

  # Build Dataset and Model
  train_ds = datasets.create_lorenz_attractor_by_generator(
      batch_size=train_data_config.batch_size,
      random_seed=FLAGS.seed)
  test_ds = datasets.create_lorenz_attractor_by_generator(
      batch_size=test_data_config.batch_size,
      random_seed=FLAGS.seed)

  # configuring emission distribution p(x[t] | z[t])
  config_emission = config_utils.get_distribution_config(
      triangular_cov=False,
      trainable_cov=False)

  emission_network = utils.build_dense_network(
      [8, 32, 3],
      ["relu", "relu", None])

  # configuring q(z[t]|h[t]=f_RNN(h[t-1], z[t-1], h[t]^b))
  config_inference = config_utils.get_distribution_config(triangular_cov=True)
  # the `network_posterior_rnn` is a RNN cell,
  # `h[t]=f_RNN(h[t-1], z[t-1], input[t])`,
  # which recursively takes previous step RNN states `h`, previous step
  # sampled dynamical state `z[t-1]`, and conditioned input `input[t]`.
  posterior_rnn = utils.build_rnn_cell(
      rnn_type=FLAGS.rnntype, rnn_hidden_dim=FLAGS.rnndim)

  # the `posterior_mlp` is a dense network emitting mean tensor for
  # the distribution of hidden states, p(z[t] | h[t])
  posterior_mlp = utils.build_dense_network(
      [32, FLAGS.hidden_dim],
      ["relu", None])

  # configuring p(z[0])
  config_z_initial = config_utils.get_distribution_config(triangular_cov=True)

  # configuring p(z[t] | z[t-1], s[t])
  config_z_transition = config_utils.get_distribution_config(
      triangular_cov=True,
      trainable_cov=True,
      sigma_scale=0.1,
      raw_sigma_bias=1.e-5,
      sigma_min=1.e-5)

  z_transition_networks = [
      utils.build_dense_network(
          [256, FLAGS.hidden_dim], ["relu", None])
      for _ in range(FLAGS.num_categories)]

  # `network_s_transition` is a network returning the transition probability
  # `log p(s[t] |s[t-1], x[t-1])`
  num_categ_squared = FLAGS.num_categories * FLAGS.num_categories
  network_s_transition = utils.build_dense_network(
      [4 * num_categ_squared, num_categ_squared],
      ["relu", None])

  snlds_model = model_cavi_snlds.create_model(
      num_categ=FLAGS.num_categories,
      hidden_dim=FLAGS.hidden_dim,
      observation_dim=3,  # Lorenz attractor has input o[t] = [x, y, z].
      config_emission=config_emission,
      config_inference=config_inference,
      config_z_initial=config_z_initial,
      config_z_transition=config_z_transition,
      network_emission=emission_network,
      network_input_embedding=lambda x: x,
      network_posterior_mlp=posterior_mlp,
      network_posterior_rnn=posterior_rnn,
      network_s_transition=network_s_transition,
      networks_z_transition=z_transition_networks,
      name="snlds")

  snlds_model.build(input_shape=(FLAGS.batch_size, 200, 3))

  # learning rate decay
  def _get_learning_rate(global_step):
    """Construct Learning Rate Schedule."""
    if learning_rate_config.flat_learning_rate:
      lr_schedule = learning_rate_config.learning_rate
    elif learning_rate_config.inverse_annealing_lr:
      lr_schedule = utils.inverse_annealing_learning_rate(
          global_step,
          target_lr=learning_rate_config.learning_rate)
    else:
      lr_schedule = utils.learning_rate_schedule(global_step,
                                                 learning_rate_config)
    return lr_schedule

  # Learning rate for optimizer will be applied on the fly in the training loop.
  optimizer = tf.keras.optimizers.Adam()

  # temperature annealing
  def _get_temperature(step):
    """Construct Temperature Annealing Schedule."""
    if temperature_config.use_temperature_annealing:
      temperature_schedule = utils.schedule_exponential_decay(
          step, temperature_config,
          temperature_config.minimal_temperature)
    else:
      temperature_schedule = temperature_config.initial_temperature
    return temperature_schedule

  # cross entropy penalty decay
  def _get_cross_entropy_coef(step):
    """Construct Cross Entropy Coefficient Schedule."""
    if cross_entropy_config.use_entropy_annealing:
      cross_entropy_schedule = utils.schedule_exponential_decay(
          step, cross_entropy_config)
    else:
      cross_entropy_schedule = 0.
    return cross_entropy_schedule

  tensorboard_file_writer = tf.summary.create_file_writer(
      logdir, flush_millis=100)

  ckpt = tf.train.Checkpoint(
      optimizer=optimizer, model=snlds_model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=5)

  latest_checkpoint = tf.train.latest_checkpoint(model_dir)
  if latest_checkpoint:
    logging.info("Loading checkpoint from %s.", latest_checkpoint)
    ckpt.restore(latest_checkpoint)
  else:
    logging.info("Start training from scratch.")

  train_iter = train_ds.as_numpy_iterator()
  test_iter = test_ds.as_numpy_iterator()

  while optimizer.iterations < FLAGS.num_steps:
    learning_rate = _get_learning_rate(optimizer.iterations)
    temperature = _get_temperature(optimizer.iterations)
    cross_entropy_coef = _get_cross_entropy_coef(optimizer.iterations)
    train_metrics = train_step(train_iter.next(), snlds_model, optimizer,
                               FLAGS.num_samples, FLAGS.objective,
                               learning_rate, temperature, cross_entropy_coef)
    if (optimizer.iterations.numpy() % FLAGS.log_steps) == 0:
      step = optimizer.iterations.numpy()
      test_metrics = eval_step(test_iter.next(), snlds_model, FLAGS.num_samples,
                               temperature)
      test_log_likelihood = tf.reduce_mean(test_metrics[FLAGS.objective])
      train_objective = tf.reduce_mean(train_metrics["objective"])
      logging.info("log step: %d, train loss %f, test loss %f.",
                   step, train_objective,
                   test_log_likelihood)
      summary_items = {
          "params/learning_rate": learning_rate,
          "params/temperature": temperature,
          "params/cross_entropy_coef": cross_entropy_coef,
          "elbo/training": tf.reduce_mean(train_metrics[FLAGS.objective]),
          "elbo/testing": test_log_likelihood,
          "xent/training": tf.reduce_mean(train_metrics["cross_entropy"]),
          "xent/testing": tf.reduce_mean(test_metrics["cross_entropy"])
      }
      with tensorboard_file_writer.as_default():
        for k, v in summary_items.items():
          tf.summary.scalar(k, v, step=step)

        original_inputs = train_metrics["inputs"][0]
        reconstructed_inputs = train_metrics["reconstructions"][0]
        most_likely_states = tf.math.argmax(
            train_metrics["posterior_llk"],
            axis=-1,
            output_type=tf.int32)[0]
        hidden_states = train_metrics["sampled_z"][0]
        discrete_states_lk = tf.exp(train_metrics["posterior_llk"][0])

        # Show lorenz attractor reconstruction side-by-side with original.

        matplotlib_fig = tensorboard_utils.show_lorenz_attractor_3d(
            fig_size=(10, 5),
            inputs=original_inputs,
            reconstructed_inputs=reconstructed_inputs,
            fig_title="input_reconstruction")
        fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
        tf.summary.image("Reconstruction", fig_numpy_array, step=step)

        # Show discrete state segmentation on input data along each dimension.
        matplotlib_fig = tensorboard_utils.show_lorenz_segmentation(
            fig_size=(10, 6),
            inputs=original_inputs,
            segmentation=most_likely_states)
        fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
        tf.summary.image("Segmentation", fig_numpy_array, step=step)

        # Show z[t] and segmentation.
        matplotlib_fig = tensorboard_utils.show_hidden_states(
            fig_size=(12, 3),
            zt=hidden_states,
            segmentation=most_likely_states)
        fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
        tf.summary.image("Hidden_State_zt", fig_numpy_array, step=step)

        # Show s[t] posterior likelihood.
        matplotlib_fig = tensorboard_utils.show_discrete_states(
            fig_size=(12, 3),
            discrete_states_lk=discrete_states_lk,
            segmentation=most_likely_states)
        fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
        tf.summary.image("Discrete_State_st", fig_numpy_array, step=step)

      ckpt_save_path = ckpt_manager.save()
      logging.info("Saving checkpoint for step %d at %s.",
                   step, ckpt_save_path)


if __name__ == "__main__":
  app.run(main)
