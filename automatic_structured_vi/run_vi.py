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

"""Run surrogate posterior benchmarks."""
import json
import os
import pathlib
import time

from absl import app
from absl import flags
from inference_gym import using_tensorflow as inference_gym
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp


from automatic_structured_vi import make_surrogate_posteriors


gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_enum('model_name', 'brownian_motion', [
    'brownian_motion', 'stochastic_volatility', 'radon', 'eight_schools',
    'lorenz_bridge', 'lorenz_bridge_global', 'brownian_motion_global'
], 'Inference Gym model')
flags.DEFINE_enum('posterior_type', 'asvi', [
    'asvi', 'large_iaf', 'small_iaf', 'maf', 'mean_field', 'mvn',
    'autoregressive'
], 'Type of surrogate posterior to use.')
flags.DEFINE_integer('num_steps', 100000, 'Number of optimization steps')
flags.DEFINE_float('learning_rate', 1e-2, 'Optimizer learning rate')
flags.DEFINE_float('prior_weight', 0.5, 'Initialization value of prior_weight.')
flags.DEFINE_integer('ensemble_num', 0, 'Ensemble member ID.')
flags.DEFINE_string('output_dir',
                    os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'run_vi/'),
                    'Directory to store output files.')


def main(_):
  model_name = FLAGS.model_name
  num_steps = FLAGS.num_steps
  learning_rate = FLAGS.learning_rate
  posterior_type = FLAGS.posterior_type
  prior_weight = FLAGS.prior_weight
  xid = FLAGS.xm_xid if hasattr(FLAGS, 'xm_xid') else -1


  output_dir = '{}_xid{}/{}'.format(FLAGS.output_dir, xid,
                                    wid) if xid > -1 else FLAGS.output_dir
  output_dir = pathlib.Path(output_dir)
  gfile.makedirs(output_dir)

  if model_name == 'brownian_motion':
    model = inference_gym.targets.BrownianMotionMissingMiddleObservations()
  elif model_name == 'stochastic_volatility':
    model = inference_gym.targets.StochasticVolatilitySP500Small()
  elif model_name == 'eight_schools':
    model = inference_gym.targets.EightSchools()
  elif model_name == 'lorenz_bridge':
    model = inference_gym.targets.ConvectionLorenzBridge()
  elif model_name == 'lorenz_bridge_global':
    model = inference_gym.targets.ConvectionLorenzBridgeUnknownScales()
  elif model_name == 'brownian_motion_global':
    model = inference_gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations(
    )
  elif model_name == 'radon':
    model = inference_gym.targets.RadonContextualEffectsHalfNormalMinnesota(
        dtype=tf.float32)

  else:
    raise NotImplementedError(
        '"{}" is not a valid value for `model_name`'.format(model_name))

  prior = model.prior_distribution()
  if isinstance(prior.event_shape, dict):
    target_log_prob = lambda **values: model.log_likelihood(  # pylint: disable=g-long-lambda
        values) + prior.log_prob(values)
  else:
    target_log_prob = lambda *values: model.log_likelihood(  # pylint: disable=g-long-lambda
        values) + prior.log_prob(values)

  opt = tf.optimizers.Adam(learning_rate)

  if posterior_type == 'asvi':
    surrogate_dist = tfp.experimental.vi.build_asvi_surrogate_posterior(
        prior, initial_prior_weight=prior_weight)
  elif posterior_type == 'mean_field':
    surrogate_dist = tfp.experimental.vi.build_asvi_surrogate_posterior(
        prior, mean_field=True)
  elif posterior_type == 'large_iaf':
    surrogate_dist = make_surrogate_posteriors.make_flow_posterior(
        prior, num_hidden_units=512, invert=True)
  elif posterior_type == 'small_iaf':
    surrogate_dist = make_surrogate_posteriors.make_flow_posterior(
        prior, num_hidden_units=8, invert=True)
  elif posterior_type == 'maf':
    surrogate_dist = make_surrogate_posteriors.make_flow_posterior(
        prior, num_hidden_units=512, invert=False)
  elif posterior_type == 'mvn':
    surrogate_dist = make_surrogate_posteriors.make_mvn_posterior(prior)
  elif posterior_type == 'autoregressive':
    surrogate_dist = make_surrogate_posteriors.build_autoregressive_surrogate_posterior(
        prior, make_surrogate_posteriors.make_conditional_linear_gaussian)

  @tf.function(experimental_compile=False)
  def fit_vi():
    return tfp.vi.fit_surrogate_posterior(
        target_log_prob,
        surrogate_dist,
        optimizer=opt,
        num_steps=num_steps)

  start = time.time()
  losses = fit_vi()
  trace_run_time = time.time() - start

  # Actual Run Time
  start = time.time()
  fit_vi()
  run_time = time.time() - start

  losses = losses.numpy()
  posterior_samples = surrogate_dist.sample(100)

  samples = surrogate_dist.sample(1000)

  if isinstance(prior.event_shape, dict):
    final_elbo = tf.reduce_mean(
        target_log_prob(**samples)
        - surrogate_dist.log_prob(samples)).numpy().tolist()
  else:
    final_elbo = tf.reduce_mean(
        target_log_prob(*samples)
        - surrogate_dist.log_prob(samples)).numpy().tolist()

  json_output = {
      'losses': losses.tolist(),
      'trace_time': trace_run_time - run_time,
      'run_time': run_time,
      'num_steps': num_steps,
      'final_elbo': final_elbo,
      'learning_rate': learning_rate,
      'ensemble_num': FLAGS.ensemble_num,
      'xm_xid': str(xid)
  }


  fig, ax = plt.subplots()
  ax.plot(losses)
  ax.set_xlabel('Iterations')
  ax.set_ylabel('Loss')
  with tf.io.gfile.GFile(
      os.path.join(output_dir, 'loss_plot.png'), 'w') as fp:
    fig.savefig(fp)

  with tf.io.gfile.GFile(
      os.path.join(output_dir, 'results.json'), 'w') as out_file:
    json.dump(json_output, out_file)

  with tf.io.gfile.GFile(
      os.path.join(output_dir, 'samples.json'), 'w') as out_file:
    json.dump(tf.nest.map_structure(
        lambda x: x.numpy().tolist(), posterior_samples), out_file)

if __name__ == '__main__':
  app.run(main)
