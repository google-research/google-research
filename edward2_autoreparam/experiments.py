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

"""Implementations of the core algorithms."""
# pylint: disable=missing-docstring,g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation,g-no-space-after-docstring-summary

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

from tensorflow_probability.python import mcmc
from edward2_autoreparam import util
from edward2_autoreparam.tfp import interleaved
from edward2_autoreparam.tfp import program_transformations as ed_transforms


def noop(x):
  return x


def transform_mcmc_states(states, transform_fn):
  """Apply a joint transformation to each of a set of MCMC samples.

  Args:
    states: list of `Tensors`, such as returned from `tfp.mcmc.sample_chain`,
      where the `i`th element has shape `concat([[num_results], rv_shapes[i]])`.
    transform_fn: callable that takes as argument a single state of the chain,
      i.e., a list of `Tensors` where the `i`th element has shape `rv_shapes[i]`
      representing a single rv value, and returns a transformed state, i.e., a
      list of `Tensors` where the `i`th element has shape
      `transformed_rv_shapes[i]`.

  Returns:
    transformed_states: list of `Tensors` representing samples from a
      transformed model, where the `i`th element has shape
      `concat([[num_results], transformed_rv_shapes[i]])`.
  """

  num_samples = states[0].shape[0].value
  transformed_states = zip(*[
      transform_fn([rv_states[sample_idx, Ellipsis]
                    for rv_states in states])
      for sample_idx in range(num_samples)
  ])
  return [
      tf.stack(transformed_rv_states)
      for transformed_rv_states in transformed_states
  ]


def _run_hmc(target, param_shapes, transform=noop, step_size_init=0.1,
             num_samples=2000, burnin=1000, num_adaptation_steps=500,
             num_leapfrog_steps=4):

  g = tf.Graph()
  with g.as_default():

    step_size = [tf.get_variable(
        name='step_size'+str(i),
        initializer=np.array(step_size_init[i], dtype=np.float32),
        use_resource=True,  # For TFE compatibility.
        trainable=False) for i in range(len(step_size_init))]

    kernel = mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size_update_fn=mcmc.make_simple_step_size_update_policy(
            num_adaptation_steps=num_adaptation_steps, target_rate=0.85))

    states, kernel_results = mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=burnin,
        current_state=[
            tf.zeros(param_shapes[param]) for param in param_shapes.keys()
        ],
        kernel=kernel,
        num_steps_between_results=1)

    tr_states = transform_mcmc_states(states, transform)
    ess_op = tfp.mcmc.effective_sample_size(tr_states)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      start_time = time.time()
      (orig_samples, samples,
       is_accepted, ess,
       final_step_size,
       log_accept_ratio) = sess.run([states, tr_states,
                                     kernel_results.is_accepted,
                                     ess_op,
                                     kernel_results.extra.step_size_assign,
                                     kernel_results.log_accept_ratio])
      sampling_time = time.time() - start_time

    results = collections.OrderedDict()
    results['samples'] = collections.OrderedDict()
    i = 0
    for param in param_shapes.keys():
      results['samples'][param] = samples[i]
      i = i + 1

    results['orig_samples'] = orig_samples

    results['is_accepted'] = is_accepted
    results['acceptance_rate'] = np.sum(is_accepted) * 100. / float(num_samples)
    results['ess'] = ess
    results['step_size'] = [s[0] for s in final_step_size]
    results['sampling_time'] = sampling_time
    results['log_accept_ratio'] = log_accept_ratio

    return results


def _run_hmc_interleaved(target_cp, target_ncp, param_shapes,
                         step_size_cp=0.1, step_size_ncp=0.1,
                         to_centered=noop,
                         to_noncentered=noop,
                         num_samples=2000,
                         burnin=1000,
                         num_leapfrog_steps=4):

  g = tf.Graph()
  with g.as_default():

    inner_kernel_cp = mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_cp,
        step_size=step_size_cp,
        num_leapfrog_steps=num_leapfrog_steps)

    inner_kernel_ncp = mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_ncp,
        step_size=step_size_ncp,
        num_leapfrog_steps=num_leapfrog_steps)

    kernel = interleaved.Interleaved(inner_kernel_cp,
                                     inner_kernel_ncp,
                                     to_centered,
                                     to_noncentered)

    states, kernel_results = mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=burnin,
        current_state=[
            tf.zeros(param_shapes[param]) for param in param_shapes.keys()
        ],
        kernel=kernel)

    ess_op = tfp.mcmc.effective_sample_size(states)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      init.run()
      start_time = time.time()
      samples, is_accepted, ess = sess.run(
          [states, kernel_results.is_accepted, ess_op])
      sampling_time = time.time() - start_time

    results = collections.OrderedDict()
    results['samples'] = collections.OrderedDict()
    i = 0
    for param in param_shapes.keys():
      results['samples'][param] = samples[i]
      i = i + 1

    results['is_accepted'] = is_accepted
    results['acceptance_rate'] = 'NA'
    results['ess'] = ess
    results['sampling_time'] = sampling_time
    results['step_size'] = 'NA'

    return results


def run_centered_hmc(model_config,
                     num_samples=2000,
                     burnin=1000,
                     num_leapfrog_steps=4,
                     num_adaptation_steps=500,
                     num_optimization_steps=2000):
  """Runs HMC on the provided (centred) model."""

  tf.reset_default_graph()

  log_joint_centered = ed.make_log_joint_fn(model_config.model)

  with ed.tape() as model_tape:
    _ = model_config.model(*model_config.model_args)

  param_shapes = collections.OrderedDict()
  target_cp_kwargs = {}
  for param in model_tape.keys():
    if param not in model_config.observed_data.keys():
      param_shapes[param] = model_tape[param].shape
    else:
      target_cp_kwargs[param] = model_config.observed_data[param]

  def target_cp(*param_args):
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_cp_kwargs[param] = param_args[i]
        i = i + 1

    return log_joint_centered(*model_config.model_args, **target_cp_kwargs)

  stepsize_kwargs = {'num_leapfrog_steps': num_leapfrog_steps}
  stepsize_kwargs = {'num_optimization_steps': num_optimization_steps}
  for key in model_config.observed_data:
    stepsize_kwargs[key] = model_config.observed_data[key]
  (step_size_init_cp,
   stepsize_elbo_cp, vi_time) = util.approximate_mcmc_step_size(
       model_config.model, *model_config.model_args, **stepsize_kwargs)

  results = _run_hmc(target_cp, param_shapes,
                     step_size_init=step_size_init_cp,
                     num_samples=num_samples,
                     burnin=burnin,
                     num_adaptation_steps=num_adaptation_steps,
                     num_leapfrog_steps=num_leapfrog_steps)

  results['elbo'] = stepsize_elbo_cp
  results['vi_time'] = vi_time
  return results


def run_noncentered_hmc(model_config,
                        num_samples=2000,
                        burnin=1000,
                        num_leapfrog_steps=4,
                        num_adaptation_steps=500,
                        num_optimization_steps=2000):
  """Given a (centred) model, this function transforms it to a fully non-centred
  one, and runs HMC on the reparametrised model.
  """

  tf.reset_default_graph()

  return run_parametrised_hmc(
      model_config=model_config,
      interceptor=ed_transforms.ncp,
      num_samples=num_samples,
      burnin=burnin,
      num_leapfrog_steps=num_leapfrog_steps,
      num_adaptation_steps=num_adaptation_steps,
      num_optimization_steps=num_optimization_steps)


def run_parametrised_hmc(model_config,
                         interceptor,
                         num_samples=2000,
                         burnin=1000,
                         num_leapfrog_steps=4,
                         num_adaptation_steps=500,
                         num_optimization_steps=2000):
  """Given a (centred) model, this function transforms it based on the provided
  interceptor, and runs HMC on the reparameterised model.
  """

  def model_ncp(*params):
    with ed.interception(interceptor):
      return model_config.model(*params)

  log_joint_noncentered = ed.make_log_joint_fn(model_ncp)

  with ed.tape() as model_tape:
    _ = model_ncp(*model_config.model_args)

  param_shapes = collections.OrderedDict()
  target_ncp_kwargs = {}
  for param in model_tape.keys():
    if param not in model_config.observed_data.keys():
      param_shapes[param] = model_tape[param].shape
    else:
      target_ncp_kwargs[param] = model_config.observed_data[param]

  def target_ncp(*param_args):
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_ncp_kwargs[param] = param_args[i]
        i = i + 1

    return log_joint_noncentered(*model_config.model_args, **target_ncp_kwargs)

  stepsize_kwargs = {'num_leapfrog_steps': num_leapfrog_steps}
  stepsize_kwargs = {'num_optimization_steps': num_optimization_steps}
  for key in model_config.observed_data:
    stepsize_kwargs[key] = model_config.observed_data[key]
  (step_size_init_ncp,
   stepsize_elbo_ncp,
   vi_time) = util.approximate_mcmc_step_size(
       model_ncp, *model_config.model_args, **stepsize_kwargs)

  results = _run_hmc(target_ncp, param_shapes,
                     step_size_init=step_size_init_ncp,
                     transform=model_config.to_centered,
                     num_samples=num_samples,
                     burnin=burnin,
                     num_adaptation_steps=num_adaptation_steps,
                     num_leapfrog_steps=num_leapfrog_steps)

  results['elbo'] = stepsize_elbo_ncp
  results['vi_time'] = vi_time
  return results


def run_interleaved_hmc(model_config,
                        num_samples=2000, step_size_cp=0.1, step_size_ncp=0.1,
                        burnin=1000, num_leapfrog_steps=4):
  """Given a (centred) model, this function transforms it to a fully
  non-centred one, and uses both models to run interleaved HMC.
  """

  tf.reset_default_graph()

  log_joint_centered = ed.make_log_joint_fn(model_config.model)

  with ed.tape() as model_tape_cp:
    _ = model_config.model(*model_config.model_args)

  param_shapes = collections.OrderedDict()
  target_cp_kwargs = {}
  for param in model_tape_cp.keys():
    if param not in model_config.observed_data.keys():
      param_shapes[param] = model_tape_cp[param].shape
    else:
      target_cp_kwargs[param] = model_config.observed_data[param]

  def target_cp(*param_args):
    i = 0
    for param in model_tape_cp.keys():
      if param not in model_config.observed_data.keys():
        target_cp_kwargs[param] = param_args[i]
        i = i + 1

    return log_joint_centered(*model_config.model_args, **target_cp_kwargs)

  def model_noncentered(*params):
    with ed.interception(ed_transforms.ncp):
      return model_config.model(*params)

  log_joint_noncentered = ed.make_log_joint_fn(model_noncentered)

  with ed.tape() as model_tape_ncp:
    _ = model_noncentered(*model_config.model_args)

  param_shapes = collections.OrderedDict()
  target_ncp_kwargs = {}
  for param in model_tape_ncp.keys():
    if param not in model_config.observed_data.keys():
      param_shapes[param] = model_tape_ncp[param].shape
    else:
      target_ncp_kwargs[param] = model_config.observed_data[param]

  def target_ncp(*param_args):
    i = 0
    for param in model_tape_ncp.keys():
      if param not in model_config.observed_data.keys():
        target_ncp_kwargs[param] = param_args[i]
        i = i + 1

    return log_joint_noncentered(*model_config.model_args, **target_ncp_kwargs)

  return _run_hmc_interleaved(target_cp, target_ncp, param_shapes,
                              to_centered=model_config.to_centered,
                              to_noncentered=model_config.to_noncentered,
                              num_samples=num_samples,
                              step_size_cp=step_size_cp,
                              step_size_ncp=step_size_ncp,
                              burnin=burnin,
                              num_leapfrog_steps=num_leapfrog_steps)


def gen_id():
  return np.random.randint(10000, 99999)


def run_vip_hmc_continuous(model_config,
                           num_samples=2000,
                           burnin=1000,
                           use_iaf_posterior=False,
                           num_leapfrog_steps=4,
                           num_adaptation_steps=500,
                           num_optimization_steps=2000,
                           num_mc_samples=32,
                           tau=1.,
                           do_sample=True,
                           description='',
                           experiments_dir=''):

  tf.reset_default_graph()

  if use_iaf_posterior:
    # IAF posterior doesn't give us stddevs for step sizes for HMC (we could
    # extract them by sampling but I haven't implemented that), and we mostly
    # care about it for ELBOs anyway.
    do_sample = False

  init_val_loc = tf.placeholder('float', shape=())
  init_val_scale = tf.placeholder('float', shape=())

  (learnable_parameters,
   learnable_parametrisation, _) = ed_transforms.make_learnable_parametrisation(
       init_val_loc=init_val_loc, init_val_scale=init_val_scale, tau=tau)

  def model_vip(*params):
    with ed.interception(learnable_parametrisation):
      return model_config.model(*params)

  log_joint_vip = ed.make_log_joint_fn(model_vip)

  with ed.tape() as model_tape:
    _ = model_vip(*model_config.model_args)

  param_shapes = collections.OrderedDict()
  target_vip_kwargs = {}
  for param in model_tape.keys():
    if param not in model_config.observed_data.keys():
      param_shapes[param] = model_tape[param].shape
    else:
      target_vip_kwargs[param] = model_config.observed_data[param]

  def target_vip(*param_args):
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_vip_kwargs[param] = param_args[i]
        i = i + 1
    return log_joint_vip(*model_config.model_args, **target_vip_kwargs)

  full_kwargs = collections.OrderedDict(model_config.observed_data.items())
  full_kwargs['parameterisation'] = collections.OrderedDict()
  for k in learnable_parameters.keys():
    full_kwargs['parameterisation'][k] = learnable_parameters[k]

  if use_iaf_posterior:
    elbo = util.get_iaf_elbo(
        target_vip,
        num_mc_samples=num_mc_samples,
        param_shapes=param_shapes)
    variational_parameters = {}
  else:
    elbo, variational_parameters = util.get_mean_field_elbo(
        model_vip,
        target_vip,
        num_mc_samples=num_mc_samples,
        model_args=model_config.model_args,
        vi_kwargs=full_kwargs)
    vip_step_size_approx = util.get_approximate_step_size(
        variational_parameters, num_leapfrog_steps)

  ##############################################################################

  best_elbo = None
  model_dir = os.path.join(experiments_dir,
                           str(description + '_' + model_config.model.__name__))

  if not tf.gfile.Exists(model_dir):
    tf.gfile.MakeDirs(model_dir)

  saver = tf.train.Saver()
  dir_save = os.path.join(model_dir, 'saved_params_{}'.format(gen_id()))

  if not tf.gfile.Exists(dir_save):
    tf.gfile.MakeDirs(dir_save)

  best_lr = None
  best_init_loc = None
  best_init_scale = None

  learning_rate_ph = tf.placeholder(shape=[], dtype=tf.float32)
  learning_rate = tf.Variable(learning_rate_ph, trainable=False)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train = optimizer.minimize(-elbo)
  init = tf.global_variables_initializer()

  learning_rates = [0.003, 0.01, 0.01, 0.1, 0.003, 0.01]
  if use_iaf_posterior:
    learning_rates = [3e-5, 1e-4, 3e-4, 1e-4]

  start_time = time.time()
  for learning_rate_val in learning_rates:
    for init_loc in [0.]:  #, 10., -10.]:
      for init_scale in [init_loc]:

        timeline = []

        with tf.Session() as sess:

          init.run(feed_dict={init_val_loc: init_loc,
                              init_val_scale: init_scale,
                              learning_rate_ph: learning_rate_val})

          this_timeline = []
          for i in range(num_optimization_steps):
            _, e = sess.run([train, elbo])

            if np.isnan(e):
              util.print('got NaN in ELBO optimization, stopping...')
              break

            this_timeline.append(e)

          this_elbo = np.mean(this_timeline[-100:])
          info_str = ('finished cVIP optimization with elbo {} vs '
                      'best ELBO {}'.format(this_elbo, best_elbo))
          util.print(info_str)
          if best_elbo is None or best_elbo < this_elbo:
            best_elbo = this_elbo
            timeline = this_timeline

            vals = sess.run(list(learnable_parameters.values()))
            learned_reparam = collections.OrderedDict(
                zip(learnable_parameters.keys(), vals))
            vals = sess.run(list(variational_parameters.values()))
            learned_variational_params = collections.OrderedDict(
                zip(variational_parameters.keys(), vals))

            util.print('learned params {}'.format(learned_reparam))
            util.print('learned variational params {}'.format(
                learned_variational_params))

            _ = saver.save(sess, dir_save)
            best_lr = learning_rate
            best_init_loc = init_loc
            best_init_scale = init_scale

  vi_time = time.time() - start_time

  util.print('BEST: LR={}, init={}, {}'.format(best_lr, best_init_loc,
                                               best_init_scale))
  util.print('ELBO: {}'.format(best_elbo))

  to_centered = model_config.make_to_centered(**learned_reparam)

  results = collections.OrderedDict()
  results['elbo'] = best_elbo

  with tf.Session() as sess:

    saver.restore(sess, dir_save)
    results['vp'] = learned_variational_params

    if do_sample:

      vip_step_size_init = sess.run(vip_step_size_approx)

      vip_step_size = [tf.get_variable(
          name='step_size_vip'+str(i),
          initializer=np.array(vip_step_size_init[i], dtype=np.float32),
          use_resource=True,  # For TFE compatibility.
          trainable=False) for i in range(len(vip_step_size_init))]

      kernel_vip = mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_vip,
          step_size=vip_step_size,
          num_leapfrog_steps=num_leapfrog_steps,
          step_size_update_fn=mcmc.make_simple_step_size_update_policy(
              num_adaptation_steps=num_adaptation_steps, target_rate=0.85))

      states, kernel_results_vip = mcmc.sample_chain(
          num_results=num_samples,
          num_burnin_steps=burnin,
          current_state=[
              tf.zeros(param_shapes[param]) for param in param_shapes.keys()
          ],
          kernel=kernel_vip,
          num_steps_between_results=1)

      states_vip = transform_mcmc_states(states, to_centered)

      init_again = tf.global_variables_initializer()
      init_again.run(feed_dict={
          init_val_loc: best_init_loc, init_val_scale: best_init_scale,
          learning_rate_ph: 1.0})  # learning rate doesn't matter for HMC.

      ess_vip = tfp.mcmc.effective_sample_size(states_vip)

      start_time = time.time()
      samples, is_accepted, ess, ss_vip, log_accept_ratio = sess.run(
          (states_vip, kernel_results_vip.is_accepted, ess_vip,
           kernel_results_vip.extra.step_size_assign,
           kernel_results_vip.log_accept_ratio))

      sampling_time = time.time() - start_time

      results['samples'] = collections.OrderedDict()
      results['is_accepted'] = is_accepted
      results['acceptance_rate'] = np.sum(is_accepted) * 100. / float(
          num_samples)
      results['ess'] = ess
      results['sampling_time'] = sampling_time
      results['log_accept_ratio'] = log_accept_ratio
      results['step_size'] = [s[0] for s in ss_vip]

      i = 0
      for param in param_shapes.keys():
        results['samples'][param] = samples[i]
        i = i + 1

    # end if

    results['parameterisation'] = collections.OrderedDict()

    i = 0
    for param in param_shapes.keys():
      name_a = param[:-5] + 'a'
      name_b = param[:-5] + 'b'
      try:
        results['parameterisation'][name_a] = learned_reparam[name_a]
        results['parameterisation'][name_b] = learned_reparam[name_b]
      except KeyError:
        continue
      i = i + 1

    results['elbo_timeline'] = timeline
    results['vi_time'] = vi_time

    results['init_pos'] = best_init_loc

    return results

##############################################################################


def run_vip_hmc_discrete(model_config,
                         parameterisation,
                         num_samples=2000,
                         burnin=1000,
                         num_leapfrog_steps=4,
                         num_adaptation_steps=500,
                         num_optimization_steps=2000):

  tf.reset_default_graph()

  (_,
   insightful_parametrisation,
   _) = ed_transforms.make_learnable_parametrisation(
       learnable_parameters=parameterisation)

  results = run_parametrised_hmc(
      model_config=model_config,
      interceptor=insightful_parametrisation,
      num_samples=num_samples,
      burnin=burnin,
      num_leapfrog_steps=num_leapfrog_steps,
      num_adaptation_steps=num_adaptation_steps,
      num_optimization_steps=num_optimization_steps)

  results['parameterisation'] = parameterisation

  return results


algs_names = ['HMC-CP', 'HMC-NCP', 'iHMC', 'c-VIP-HMC', 'd-VIP-HMC']
