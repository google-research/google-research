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

"""Train and test a robust model with the implicit bias of an optimizer."""

import copy

from absl import app
from absl import flags
from absl import logging
import cvxpy as cp
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from ml_collections.config_flags import config_flags
import numpy as np
import scipy.linalg

import robust_optim.adversarial as adversarial
import robust_optim.data as data_loader
import robust_optim.model as model
from robust_optim.norm import norm_f
from robust_optim.norm import norm_type_dual
import robust_optim.optim as optim
import robust_optim.summary as summary_tools

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Config file name.')


def evaluate_risks(data, predict_f, loss_f, model_param):
  """Returns the risk of a model for various loss functions.

  Args:
    data: An array of data samples for approximating the risk.
    predict_f: Function that predicts labels given input.
    loss_f: Function that outputs model's specific loss function.
    model_param: Model parameters.

  Returns:
    Dictionary of risks for following loss functions:
        (model's loss, 0/1, adversarial risk wrt a single norm-ball).
  """
  inputs, labels = data
  pred = predict_f(model_param, inputs)
  loss = loss_f(model_param, inputs, labels)
  zero_one_risk = (1 - (pred == labels)).mean()
  return {
      'loss': loss,
      'zero_one': zero_one_risk,
  }


def evaluate_adversarial_risk(data, predict_f, loss_adv_f, dloss_adv_dx,
                              model_param, normalize_f, config, rng_key):
  """Evaluating adversarial risk by looping over epsilon.

  Args:
    data: An array of data samples for approximating the risk.
    predict_f: Function that predicts labels given input.
    loss_adv_f: The loss function. This loss has to be specific to the model to
      tackle gradient masking.
    dloss_adv_dx: The gradient function of the adversarial loss w.r.t. the
      input. Ideally, we will have multiple loss functions even on different
      layers of network. This loss has to be specific to the model to tackle
      gradient masking.
    model_param: Model parameters.
    normalize_f: A function to normalize the weights of the model.
    config: Dictionary of hyperparameters.
    rng_key: JAX random number generator key.

  Returns:
    Dictionary adversarial risk wrt a range of norm-balls.
  """
  _, labels = data
  # If config.adv.eps_from_cvxpy, eps is reset after min-norm solution is found
  eps_iter, eps_tot = config.adv.eps_iter, config.adv.eps_tot
  config_new = copy.deepcopy(config.adv)
  adv_risk = []
  adv_eps = []
  for i in jnp.arange(0, 1.05, 0.05):
    config_new.eps_iter = float(eps_iter * i)
    config_new.eps_tot = float(eps_tot * i)
    x_adv_multi = adversarial.find_adversarial_samples_multi_attack(
        data, loss_adv_f, dloss_adv_dx,
        model_param, normalize_f, config_new, rng_key)
    correct_label = jnp.zeros(1)
    for x_adv in x_adv_multi:
      pred_adv = predict_f(model_param, x_adv)
      correct_label += (pred_adv == labels) / len(x_adv_multi)
    adv_risk += [float((1 - correct_label).mean())]
    adv_eps += [config_new.eps_tot]
  return {'adv/%s' % config.adv.norm_type: (adv_eps, adv_risk)}


def train(model_param, train_test_data, predict_f, loss_f, loss_adv_f,
          linearize_f, normalize_f, loss_and_prox_op, summary, config, rng_key):
  """Train a model and log risks."""
  dloss_dw = jax.grad(loss_f, argnums=0)
  dloss_adv_dx = jax.grad(loss_adv_f, argnums=1)

  train_data = train_test_data[0]
  xtrain, ytrain = train_data

  # Precompute min-norm solutions
  if config.enable_cvxpy:
    min_norm_w = {}
    for norm_type in config.available_norm_types:
      min_norm_w[norm_type] = compute_min_norm_solution(xtrain, ytrain,
                                                        norm_type)
    if config.adv.eps_from_cvxpy:
      dual_norm = norm_type_dual(config.adv.norm_type)
      wcomp = min_norm_w[dual_norm]
      wnorm = norm_f(wcomp, dual_norm)
      margin = 1. / wnorm
      config.adv.eps_tot = config.adv.eps_iter = float(2 * margin)

  if config['optim']['name'] == 'cvxpy':
    norm_type = config['optim']['norm']
    cvxpy_sol = compute_min_norm_solution(xtrain, ytrain, norm_type)
    model_param = jnp.array(cvxpy_sol)

  # Train loop
  optim_step, optim_options = optim.get_optimizer_step(config['optim'])
  niters = optim_options['niters']
  for step in range(1, niters):
    # Take one optimization step
    if config['optim']['name'] != 'cvxpy':
      if config['optim']['adv_train']['enable']:
        # Adversarial training
        rng_key, rng_subkey = jax.random.split(rng_key)
        x_adv = adversarial.find_adversarial_samples(train_data, loss_adv_f,
                                                     dloss_adv_dx,
                                                     model_param, normalize_f,
                                                     config.optim.adv_train,
                                                     rng_key)
        train_data_new = x_adv, ytrain
      else:
        # Standard training
        train_data_new = train_data
      if config['optim']['name'] == 'fista':
        model_param, optim_options = optim_step(train_data_new,
                                                loss_and_prox_op,
                                                model_param,
                                                optim_options)
      else:
        model_param, optim_options = optim_step(train_data_new,
                                                loss_f,
                                                model_param, optim_options)

    # Log risks and other statistics
    if (step + 1) % config.log_interval == 0:
      # Evaluate risk on train/test sets
      for do_train in [True, False]:
        data = train_test_data[0] if do_train else train_test_data[1]
        prefix = 'risk/train' if do_train else 'risk/test'
        risk = evaluate_risks(data, predict_f, loss_f, model_param)
        for rname, rvalue in risk.items():
          summary.scalar('%s/%s' % (prefix, rname), rvalue, step=step)
        rng_key, rng_subkey = jax.random.split(rng_key)
        risk = evaluate_adversarial_risk(data, predict_f, loss_adv_f,
                                         dloss_adv_dx,
                                         model_param, normalize_f, config,
                                         rng_subkey)
        for rname, rvalue in risk.items():
          summary.array('%s/%s' % (prefix, rname), rvalue, step=step)

      grad = dloss_dw(model_param, xtrain, ytrain)
      grad_ravel, _ = ravel_pytree(grad)
      model_param_ravel, _ = ravel_pytree(model_param)
      for norm_type in config.available_norm_types:
        # Log the norm of the gradient w.r.t. various norms
        if not norm_type.startswith('dft'):
          summary.scalar(
              'grad/norm/' + norm_type,
              norm_f(grad_ravel, norm_type),
              step=step)

        # Log weight norm
        if not norm_type.startswith('dft'):
          wnorm = norm_f(model_param_ravel, norm_type)
          summary.scalar('weight/norm/' + norm_type, wnorm, step=step)

        # Log margin for the equivalent linearized single layer model
        linear_param = linearize_f(model_param)
        min_loss = jnp.min(ytrain * (linear_param.T @ xtrain))
        wcomp = linear_param / min_loss
        wnorm = norm_f(wcomp, norm_type)
        margin = jnp.sign(min_loss) * 1 / wnorm
        summary.scalar('margin/' + norm_type, margin, step=step)
        summary.scalar('weight/linear/norm/' + norm_type, wnorm, step=step)

        # Cosine similarity between the current params and min-norm solution
        if config.enable_cvxpy:

          def cos_sim(a, b):
            return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))

          min_norm_w_ravel, _ = ravel_pytree(min_norm_w[norm_type])
          cs = cos_sim(linear_param.flatten(), min_norm_w_ravel)
          summary.scalar('csim_to_wmin/' + norm_type, cs, step=step)

      if 'step_size' in optim_options:
        summary.scalar('optim/step_size', optim_options['step_size'], step=step)

      logging.info('Epoch: [%d/%d]\t%s', step + 1, niters,
                   summary.last_scalars_to_str(config.log_keys))
      logging.flush()

  summary.flush()


def compute_min_norm_solution(x, y, norm_type):
  """Compute the min-norm solution using a convex-program solver."""
  w = cp.Variable((x.shape[0], 1))
  if norm_type == 'linf':
    # compute minimal L_infinity solution
    constraints = [cp.multiply(y, (w.T @ x)) >= 1]
    prob = cp.Problem(cp.Minimize(cp.norm_inf(w)), constraints)
  elif norm_type == 'l2':
    # compute minimal L_2 solution
    constraints = [cp.multiply(y, (w.T @ x)) >= 1]
    prob = cp.Problem(cp.Minimize(cp.norm2(w)), constraints)
  elif norm_type == 'l1':
    # compute minimal L_1 solution
    constraints = [cp.multiply(y, (w.T @ x)) >= 1]
    prob = cp.Problem(cp.Minimize(cp.norm1(w)), constraints)
  elif norm_type[0] == 'l':
    # compute minimal Lp solution
    p = float(norm_type[1:])
    constraints = [cp.multiply(y, (w.T @ x)) >= 1]
    prob = cp.Problem(cp.Minimize(cp.pnorm(w, p)), constraints)
  elif norm_type == 'dft1':
    w = cp.Variable((x.shape[0], 1), complex=True)
    # compute minimal Fourier L1 norm (||F(w)||_1) solution
    dft = scipy.linalg.dft(x.shape[0]) / np.sqrt(x.shape[0])
    constraints = [cp.multiply(y, (cp.real(w).T @ x)) >= 1]
    prob = cp.Problem(cp.Minimize(cp.norm1(dft @ w)), constraints)
  prob.solve(verbose=True)
  logging.info('Min %s-norm solution found (norm=%.4f)', norm_type,
               float(norm_f(w.value, norm_type)))
  return cp.real(w).value


def main_with_config(config):
  logging.info(str(config.log_dir))
  summary = summary_tools.SummaryWriter(config.log_dir,
                                        config.available_norm_types)
  logging.info(str(config))
  summary.object('config', config)

  rng_key = jax.random.PRNGKey(config.seed)
  rng_subkey = jax.random.split(rng_key, 3)

  model_ret = model.get_model_functions(rng_subkey[0], config.dim,
                                        **config.model)
  (model_param, predict_f, loss_f, loss_adv_f, linearize_f, normalize_f,
   loss_and_prox_op) = model_ret
  train_test_generator = data_loader.get_train_test_generator(config.dataset)
  train_test_data = train_test_generator(config, rng_subkey[1])

  train(model_param, train_test_data, predict_f, loss_f, loss_adv_f,
        linearize_f, normalize_f, loss_and_prox_op, summary, config,
        rng_subkey[2])


def main(_):
  config = FLAGS.config

  main_with_config(config)


if __name__ == '__main__':
  app.run(main)
