# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Training/eval loop for DReGs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf
import tensorflow_probability as tfp

from dreg_estimators import model
from dreg_estimators import utils

tfd = tfp.distributions
flags = tf.flags

flags.DEFINE_enum("estimator", "iwae", [
    "iwae", "rws", "stl", "dreg", "dreg-cv", "rws-dreg", "rws-dreg-norm",
    "dreg-norm", "jk", "jk-dreg", "dreg-alpha"
], "Estimator type to use.")
flags.DEFINE_float("alpha", 0.9, "Weighting for DReG(alpha)")
flags.DEFINE_integer("batch_size", 32, "The batch size.")
flags.DEFINE_integer("num_samples", 64, "The numer of samples to use.")
flags.DEFINE_integer("latent_dim", 50, "The dimension of the VAE latent space.")
flags.DEFINE_float("learning_rate", 3e-4, "The learning rate for ADAM.")
flags.DEFINE_integer("max_steps", int(5e6), "The number of steps to train for.")
flags.DEFINE_integer("summarize_every", 100,
                     "Number of steps between summaries.")
flags.DEFINE_string("logdir", "/tmp/dreg",
                    "The directory to put summaries and checkpoints.")
flags.DEFINE_bool("bias_check", False,
                  "Run a bias check instead of training the model.")
flags.DEFINE_string(
    "initial_checkpoint_dir", None,
    ("Initial checkpoint directory to start from. This also disables model ",
     "training. Only the inference network is trained.")
)
flags.DEFINE_integer(
    "run", 0,
    ("A number to distinguish which run this is. This allows us to run ",
     "multiple trials with the same params.")
)
flags.DEFINE_string(
    "var_calc", None,
    ("Comma separated list of estimators to calculate the variance of on this",
     "trajectory")
)
flags.DEFINE_enum("dataset", "mnist", [
    "mnist",
    "struct_mnist",
    "omniglot",
], "Dataset to use.")
flags.DEFINE_bool("image_summary", False, "Create visualizations")

FLAGS = flags.FLAGS


def create_logging_hook(metrics):

  def summary_formatter(d):
    return ", ".join(
        ["%s: %g" % (key, float(value)) for key, value in sorted(d.items())])

  logging_hook = tf.train.LoggingTensorHook(
      metrics, formatter=summary_formatter, every_n_iter=FLAGS.summarize_every)
  return logging_hook


def main(unused_argv):
  proposal_hidden_dims = [200, 200]
  likelihood_hidden_dims = [200, 200]

  with tf.Graph().as_default():
    alpha = tf.Variable(0.0, name="alpha_cv")
    beta = tf.Variable(0.0, name="beta_cv")
    gamma = tf.Variable(0.0, name="gamma_cv")
    delta = tf.Variable(0.0, name="delta_cv")
    tf.add_to_collection("CONTROL_VARIATES", alpha)
    tf.add_to_collection("CONTROL_VARIATES", beta)
    tf.add_to_collection("CONTROL_VARIATES", gamma)
    tf.add_to_collection("CONTROL_VARIATES", delta)

    if FLAGS.dataset in ["mnist", "struct_mnist"]:
      train_xs, valid_xs, test_xs = utils.load_mnist()
    elif FLAGS.dataset == "omniglot":
      train_xs, valid_xs, test_xs = utils.load_omniglot()

    # Compute bias initializer on the training set
    mean_xs = np.mean(train_xs, axis=0)
    clipped_mean_xs = np.clip(mean_xs, 1e-3, 1 - 1e-3)
    bias_init = np.log(clipped_mean_xs / (1 - clipped_mean_xs))

    # Placeholder for input mnist digits.
    observations_ph = tf.placeholder("float32", [None, 784])

    if FLAGS.dataset == "struct_mnist":
      context_mean_xs = np.split(mean_xs, 2, 0)[0]
      prior = model.ConditionalNormal(
          FLAGS.latent_dim,
          likelihood_hidden_dims,
          mean_center=context_mean_xs,
          hidden_activation_fn=tf.nn.tanh)
      proposal = model.ConditionalNormal(
          FLAGS.latent_dim,
          proposal_hidden_dims,
          mean_center=mean_xs,
          hidden_activation_fn=tf.nn.tanh)
      likelihood = model.ConditionalBernoulli(
          784 // 2,
          likelihood_hidden_dims,
          bias_init=np.split(bias_init, 2, 0)[1],
          hidden_activation_fn=tf.nn.tanh)
      observations, contexts = tf.split(
          observations_ph, num_or_size_splits=2, axis=1)
      # pylint: disable=g-long-lambda
      get_model_params = (lambda: likelihood.get_variables() +
                          prior.get_variables())  # pytype: disable=attribute-error
      # pylint: enable=g-long-lambda
    else:
      # prior is Normal(0, 1)
      prior_loc = tf.zeros([FLAGS.latent_dim], dtype=tf.float32)
      prior_scale = tf.ones([FLAGS.latent_dim], dtype=tf.float32)
      prior = lambda _: tfd.Normal(loc=prior_loc, scale=prior_scale)
      proposal = model.ConditionalNormal(
          FLAGS.latent_dim,
          proposal_hidden_dims,
          mean_center=mean_xs,
          hidden_activation_fn=tf.nn.tanh)
      likelihood = model.ConditionalBernoulli(
          784,
          likelihood_hidden_dims,
          bias_init=bias_init,
          hidden_activation_fn=tf.nn.tanh)
      observations, contexts = observations_ph, None
      get_model_params = likelihood.get_variables

    # Compute the lower bound and the loss
    estimators = model.iwae(
        prior,
        likelihood,
        proposal,
        observations,
        FLAGS.num_samples, [alpha, beta, gamma, delta],
        contexts=contexts)

    log_p_hat, neg_model_loss, neg_inference_loss = estimators[FLAGS.estimator]
    model_loss = -tf.reduce_mean(neg_model_loss)
    inference_loss = -tf.reduce_mean(neg_inference_loss)
    log_p_hat_mean = tf.reduce_mean(log_p_hat)

    model_params = get_model_params()
    inference_network_params = proposal.get_variables()

    # Compute and apply the gradients, summarizing the gradient variance.
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    model_grads = opt.compute_gradients(model_loss, var_list=model_params)
    inference_grads = opt.compute_gradients(
        inference_loss, var_list=inference_network_params)

    # If we're using control variates, add gradients for these too.
    if "cv" in FLAGS.estimator:
      vectorized_inference_grads = tf.concat(
          [tf.reshape(g, [-1]) for g, _ in inference_grads if g is not None],
          axis=0)
      cv_grads = opt.compute_gradients(
          tf.reduce_mean(tf.square(vectorized_inference_grads)),
          var_list=tf.get_collection("CONTROL_VARIATES"))
      tf.summary.scalar("alpha", alpha)
      tf.summary.scalar("beta", beta)
      tf.summary.scalar("gamma", gamma)
      tf.summary.scalar("delta", delta)
    else:
      cv_grads = []

    if FLAGS.initial_checkpoint_dir:
      # Just train the inference network from the initial checkpoint
      train_op = opt.apply_gradients(
          inference_grads + cv_grads, global_step=global_step)
      model_grad_variance = tf.constant(0.)
      inference_grad_variance = tf.constant(0.)
      inference_grad_snr_sq = tf.constant(0.)
    else:
      grads = model_grads + inference_grads + cv_grads
      model_ema_op, model_grad_variance, _ = (
          utils.summarize_grads(model_grads))
      inference_ema_op, inference_grad_variance, inference_grad_snr_sq = (
          utils.summarize_grads(inference_grads))

      ema_ops = [model_ema_op, inference_ema_op]
      if FLAGS.var_calc is not None:
        var_calc = FLAGS.var_calc.split(",")
        for estimator in var_calc:
          var_calc_inference_grads = opt.compute_gradients(
              -tf.reduce_mean(estimators[estimator][-1]),
              var_list=inference_network_params)
          (var_calc_inference_ema_op, var_calc_inference_grad_variance,
           var_calc_inference_grad_snr_sq
          ) = utils.summarize_grads(var_calc_inference_grads)
          ema_ops.append(var_calc_inference_ema_op)

          # Add summaries
          tf.summary.scalar("inference_grad_variance/%s" % estimator,
                            var_calc_inference_grad_variance)
          tf.summary.scalar("inference_grad_snr_sq/%s" % estimator,
                            var_calc_inference_grad_snr_sq)

      with tf.control_dependencies(ema_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

      tf.summary.scalar("model_grad_variance", model_grad_variance)
      tf.summary.scalar("inference_grad_variance/%s" % FLAGS.estimator,
                        inference_grad_variance)
      tf.summary.scalar("inference_grad_snr_sq/%s" % FLAGS.estimator,
                        inference_grad_snr_sq)
    tf.summary.scalar("log_p_hat/train", log_p_hat_mean)

    # Define an op to compute the paired t statistic (for bias checking)
    iwae_inference_grads = opt.compute_gradients(
        -log_p_hat_mean, var_list=inference_network_params)
    n = 0.
    t_stat = 0.
    for (g_a, _), (g_b, _) in zip(inference_grads, iwae_inference_grads):
      n += tf.to_float(tf.size(g_a))
      t_stat += tf.reduce_sum(g_a - g_b)
    t_stat /= n

    exp_name = "%s.lr-%g.n_samples-%d.alpha-%g.dataset-%s.run-%d" % (
        FLAGS.estimator, FLAGS.learning_rate, FLAGS.num_samples, FLAGS.alpha,
        FLAGS.dataset, FLAGS.run)
    checkpoint_dir = os.path.join(FLAGS.logdir, exp_name)
    if FLAGS.initial_checkpoint_dir and not tf.gfile.Exists(checkpoint_dir):
      tf.gfile.MakeDirs(checkpoint_dir)
      f = "checkpoint"
      tf.gfile.Copy(
          os.path.join(FLAGS.initial_checkpoint_dir, f),
          os.path.join(checkpoint_dir, f))

    with tf.train.MonitoredTrainingSession(
        is_chief=True,
        hooks=[
            create_logging_hook({
                "Step": global_step,
                "log_p_hat": log_p_hat_mean,
                "model_grad_variance": model_grad_variance,
                "infer_grad_varaince": inference_grad_variance,
                "infer_grad_snr_sq": inference_grad_snr_sq,
                "alpha": alpha,
                "beta": beta,
            })
        ],
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_secs=120,
        save_summaries_steps=FLAGS.summarize_every,
        log_step_count_steps=FLAGS.summarize_every * 10) as sess:
      writer = tf.summary.FileWriter(checkpoint_dir)
      t_stats = []
      cur_step = -1
      indices = list(range(train_xs.shape[0]))
      n_epoch = 0

      def run_eval(cur_step, split="valid", eval_batch_size=256):
        """Run evaluation on a datasplit."""
        if split == "valid":
          eval_dataset = valid_xs
        elif split == "test":
          eval_dataset = test_xs

        log_p_hat_vals = []
        for i in range(0, eval_dataset.shape[0], eval_batch_size):
          batch_xs = utils.binarize_batch_xs(
              eval_dataset[i:(i + eval_batch_size)])
          log_p_hat_vals.append(
              sess.run(log_p_hat_mean, feed_dict={observations_ph: batch_xs}))

        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="log_p_hat/%s" % split,
                simple_value=np.mean(log_p_hat_vals))
        ])
        writer.add_summary(summary, cur_step)
        print("log_p_hat/%s: %g" % (split, np.mean(log_p_hat_vals)))

      while cur_step < FLAGS.max_steps and not sess.should_stop():
        n_epoch += 1
        random.shuffle(indices)

        for i in range(0, train_xs.shape[0], FLAGS.batch_size):
          if sess.should_stop() or cur_step > FLAGS.max_steps:
            break

          # Get a batch, then dynamically binarize
          ns = indices[i:i + FLAGS.batch_size]
          batch_xs = utils.binarize_batch_xs(train_xs[ns])

          if FLAGS.bias_check and cur_step > 1000:
            t_stat_val, = sess.run([t_stat],
                                   feed_dict={observations_ph: batch_xs})
            t_stats.append(t_stat_val)
            aggregate_t_stat = (
                np.mean(t_stats) /
                (np.std(t_stats, ddof=1) / np.sqrt(len(t_stats))))
            print(
                len(t_stats), np.mean(t_stats), np.std(t_stats, ddof=1),
                aggregate_t_stat)
          else:
            _, cur_step = sess.run([train_op, global_step],
                                   feed_dict={observations_ph: batch_xs})

        if n_epoch % 10 == 0:
          # Run a validation step and test step
          run_eval(cur_step, "valid")
          run_eval(cur_step, "test")


if __name__ == "__main__":
  tf.app.run(main)
