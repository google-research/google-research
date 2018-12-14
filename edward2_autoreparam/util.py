# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tools."""
# pylint: disable=missing-docstring,g-doc-args,g-doc-return-or-yield
# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary
# pylint: disable=invalid-name,broad-except

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

from tensorflow_probability.python.edward2.generated_random_variables import Normal
from tensorflow_probability.python.edward2.interceptor import tape
from tensorflow_probability.python.edward2.program_transformations import make_log_joint_fn

from edward2_autoreparam.tfp import program_transformations

# pylint: disable=g-import-not-at-top
try:
  import __builtin__
except ImportError:
  # Python 3
  import builtins as __builtin__
# pylint: enable=g-import-not-at-top


__all__ = [
    'condition_number_cp',
    'condition_number_ncp',
    'compute_V_cp',
    'compute_V_ncp',
    'mean_field_variational_inference',
    'approximate_mcmc_step_size',
]


def compute_V_cp(q, v):
  r = (v * q + q + 1.)
  return np.array([[1. + v, 1.], [1., q*v + 1.]]) / r


def compute_V_ncp(q, v):
  r = 1 / (v * q + q + 1)
  return r * np.array([[q + 1, -np.sqrt(v)*q], [-np.sqrt(v)*q, v*q + 1]])


def condition_number_cp(q, v):
  sqrt_det = 2 * np.sqrt((v*q + 1) * (v*q + 1)  -
                         v * (v*q + q + 1) * (v*q + 1) / (v + 1))
  lambda1 = 2*(v*q + 1) - sqrt_det
  lambda2 = 2*(v*q + 1) + sqrt_det
  return lambda2 / lambda1


def condition_number_ncp(q, v):
  sqrt_det = 2 * np.sqrt((v*q + 1) * (v*q + 1)  -
                         (v*q + q + 1) * (v*q + 1) / (q + 1))
  lambda1 = 2*(v*q + 1) - sqrt_det
  lambda2 = 2*(v*q + 1) + sqrt_det
  return lambda2 / lambda1


def mean_field_variational_inference(model, *args, **kwargs):
  num_optimization_steps = kwargs.get('num_optimization_steps', 2000)
  del kwargs['num_optimization_steps']

  (variational_model,
   variational_parameters) = program_transformations.make_variational_model(
       model, *args, **kwargs)

  log_joint = make_log_joint_fn(model)
  def target(**parameters):
    full_kwargs = dict(parameters, **kwargs)
    return log_joint(*args, **full_kwargs)

  log_joint_q = make_log_joint_fn(variational_model)
  def target_q(**parameters):
    return log_joint_q(*args, **parameters)

  elbo_sum = 0.
  for _ in range(16):
    with tape() as variational_tape:
      _ = variational_model(*args)

    params = variational_tape
    elbo_sum = elbo_sum + target(**params) - target_q(**params)

  elbo = elbo_sum / 16.

  best_elbo = None

  learning_rate_ph = tf.placeholder(shape=[], dtype=tf.float32)
  learning_rate = tf.Variable(learning_rate_ph, trainable=False)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train = optimizer.minimize(-elbo)
  init = tf.global_variables_initializer()

  start_time = time.time()
  for learning_rate_val in [0.01, 0.1, 0.01, 0.1, 0.01, 0.1]:
    feed_dict = {learning_rate_ph: learning_rate_val}
    with tf.Session() as sess:
      sess.run(init, feed_dict=feed_dict)

      this_timeline = []
      print('VI with {} optimization steps'.format(num_optimization_steps))
      for _ in range(num_optimization_steps):
        _, e = sess.run([train, elbo], feed_dict=feed_dict)
        this_timeline.append(e)

      this_elbo = np.mean(this_timeline[-100:])
      if best_elbo is None or best_elbo < this_elbo:
        timeline = this_timeline
        best_elbo = this_elbo

        vals = sess.run(list(variational_parameters.values()),
                        feed_dict=feed_dict)
        learned_variational_params = collections.OrderedDict(
            zip(variational_parameters.keys(), vals))

      vi_time = time.time() - start_time

  results = collections.OrderedDict()
  results['vp'] = learned_variational_params
  print('ELBO: {}'.format(best_elbo))

  return results, best_elbo, timeline, vi_time


def _marshal(*rvs):
  """Args: a list of ed.RandomVariables each with vector or scalar event shape
  (which must be staticly known), and all having the same batch shape.

  Returns: a Tensor from concatenating their values along a single vector
  dimension.
  """
  vector_rvs = []
  for rv in rvs:
    v = rv.value
    if v.shape.ndims == 0:
      vector_rvs.append([v])
    else:
      vector_rvs.append(v)
  print(vector_rvs)
  return tf.concat(vector_rvs, axis=-1)


def _to_vector_shape(tensor_shape):
  if tensor_shape.ndims > 1:
    raise Exception('cannot convert {} to vector shape!'.format(tensor_shape))
  elif tensor_shape.ndims == 0:
    return tf.TensorShape([1])
  return tensor_shape


def _tensorshape_size(tensor_shape):
  if tensor_shape.ndims > 1:
    raise Exception(
        'shapes of ndims >1 are bad! (saw: {})!'.format(tensor_shape))
  elif tensor_shape.ndims == 0:
    return 1
  return tensor_shape[0].value


def get_iaf_elbo(target, num_mc_samples, param_shapes):
  shape_sizes = [_tensorshape_size(pshape) for pshape in param_shapes.values()]
  overall_shape = [sum(shape_sizes)]

  def unmarshal(variational_sample):
    results = []
    n_dimensions_used = 0
    for (n_to_add, result_shape) in zip(shape_sizes, param_shapes.values()):
      result = variational_sample[
          Ellipsis, n_dimensions_used:n_dimensions_used + n_to_add]
      results.append(tf.reshape(result, result_shape))
      n_dimensions_used += n_to_add
    return tuple(results)

  variational_dist = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Invert(
          tfb.MaskedAutoregressiveFlow(
              shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                  hidden_layers=[256, 256]))),
      event_shape=overall_shape,
      name='q_iaf')

  variational_samples = variational_dist.sample(num_mc_samples)
  target_q_sum = tf.reduce_sum(variational_dist.log_prob(variational_samples))
  target_sum = 0.
  for s in range(num_mc_samples):
    params = unmarshal(variational_samples[s, Ellipsis])
    target_sum = target_sum + target(*params)

  energy = target_sum / float(num_mc_samples)
  entropy = -target_q_sum / float(num_mc_samples)
  elbo = energy + entropy

  tf.summary.scalar('energy', energy)
  tf.summary.scalar('entropy', entropy)
  tf.summary.scalar('elbo', elbo)

  return elbo


def get_mean_field_elbo(model, target, num_mc_samples, model_args, vi_kwargs):

  variational_model, variational_parameters = make_variational_model_special(
      model, *model_args, **vi_kwargs)

  log_joint_q = make_log_joint_fn(variational_model)
  def target_q(**parameters):
    return log_joint_q(*model_args, **parameters)

  target_sum = 0.
  target_q__sum = 0.
  for _ in range(num_mc_samples):
    with tape() as variational_tape:
      _ = variational_model(*model_args)

    params = variational_tape.values()
    target_sum = target_sum + target(*params)
    target_q__sum = target_q__sum + target_q(**variational_tape)

  energy = target_sum / float(num_mc_samples)
  entropy = -target_q__sum / float(num_mc_samples)
  elbo = energy + entropy

  tf.summary.scalar('energy', energy)
  tf.summary.scalar('entropy', entropy)
  tf.summary.scalar('elbo', elbo)

  return elbo, variational_parameters


def get_approximate_step_size(variational_parameters, num_leapfrog_steps):
  return [
      variational_parameters[key] / num_leapfrog_steps**2
      for key in variational_parameters.keys()
      if key.endswith('_scale')
  ]


# FIXME: need to make this nicer than with all these weird kwargs
def approximate_mcmc_step_size(model, *args, **kwargs):

  with tf.variable_scope('approx_step_size_{}'.format(model.__name__)):
    if 'diagnostics' in kwargs.keys():
      diagnostics = kwargs.pop('diagnostics')
    else:
      diagnostics = False

    if 'num_leapfrog_steps' in kwargs.keys():
      num_leapfrog_steps = kwargs.pop('num_leapfrog_steps')
    else:
      num_leapfrog_steps = 4

    results, final_elbo_val, _, vi_time = mean_field_variational_inference(
        model, *args, **kwargs)
    stepsize = [(np.array(np.array(results['vp'][key], dtype=np.float32)) /
                 (float(num_leapfrog_steps)**2))
                for key in results['vp'].keys()
                if key.endswith('_scale')]

    if diagnostics:
      print('Estimated goodness of {}: {}'.format(model.__name__,
                                                  final_elbo_val))
      print('Estimated stepsize of {}: {}'.format(model.__name__, stepsize))

  return stepsize, final_elbo_val, vi_time


def stddvs_to_mcmc_step_sizes(results, num_leapfrog_steps):
  stepsize = [(np.sqrt(2 * np.mean(results[key])) / float(num_leapfrog_steps))
              for key in results.keys()
              if key.endswith('_scale')]

  return stepsize


def estimate_true_mean(sample_groups, esss):

  true_mean = [0 for group in range(len(sample_groups))]

  r = float(sum(esss))

  for group in range(len(sample_groups)):

    samples = sample_groups[group]
    mean = [np.mean(v) for v in samples]

    true_mean[group] = [(true_mean[group] + esss[group] * var_mean / r)
                        for var_mean in mean]

  return true_mean


def make_variational_model_special(model, *args, **kwargs):

  variational_parameters = collections.OrderedDict()
  param_params = kwargs['parameterisation']

  def get_or_init(name, a, b, shape=None):

    loc_name = model.__name__ + '_' + 'q_' + name + '_loc'
    scale_name = model.__name__ + '_' + 'q_' + name + '_scale'

    if loc_name in variational_parameters.keys() and \
        scale_name in variational_parameters.keys():
      return (variational_parameters[loc_name],
              variational_parameters[scale_name])
    else:
      # shape must not be None
      pre_loc = tf.get_variable(
          name=loc_name, initializer=1e-10 * tf.ones(shape, dtype=tf.float32))
      pre_scale = tf.nn.softplus(tf.get_variable(
          name=scale_name, initializer=-4*tf.ones(shape, dtype=tf.float32)))
      variational_parameters[loc_name] = (a + 0.1) * pre_loc
      variational_parameters[scale_name] = pre_scale**(b + 0.1)

      return (variational_parameters[loc_name],
              variational_parameters[scale_name])

  def mean_field(rv_constructor, *rv_args, **rv_kwargs):

    name = rv_kwargs['name']
    if name not in kwargs.keys():
      rv = rv_constructor(*rv_args, **rv_kwargs)

      try:
        a, b = param_params[name[:-5] + 'a'], param_params[name[:-5] + 'b']
      except Exception as err:
        print(
            'couldn\'t get centering params for variable {}: {}'.format(
                name, err))
        a, b = 1., 1.
      loc, scale = get_or_init(name, a=a, b=b, shape=rv.shape)

      # NB: name must be the same as original variable,
      # in order to be able to do black-box VI (setting
      # parameters to variational values obtained via trace).
      return Normal(loc=loc, scale=scale, name=name)
    else:
      rv_kwargs['value'] = kwargs[name]
      return rv_constructor(*rv_args, **rv_kwargs)

  def variational_model(*args):
    with ed.interception(mean_field):
      return model(*args)

  _ = variational_model(*args)

  return variational_model, variational_parameters


def print(*args):  # pylint: disable=redefined-builtin
  __builtin__.print(*args)
  logging.info(' '.join(args))
