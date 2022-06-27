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

"""Utils for debiasing ML models."""

import math

import numpy as np


class RandomizedThreshold:
  """Threshold optimizer (RTO) to debias models via postprocessing.

  See: https://arxiv.org/abs/2106.12887.

  This is a solver to the following optimiation problem:
    minimize   gamma/2 ||x||^2 - y^Tx
    s.t.       x satisfies DP constraint with tolerance eps and parameter rho.

  There are no assumptions about y in this code but, in general, y should be the
  predictions of the original classifier.
  """

  def __init__(self, gamma=1.0, eps=0.0, rho=None):
    """Instantiate object.

    Args:
      gamma: The regularization parameter gamma (for randomization). Set this to
        1 if the goal is to minmize changes to the original scores.
      eps: Tolerance parameter for bias between 0 and 1 inclusive.
      rho: The rho parameter in the post-hoc rule. If None, rho = E[y].
    """
    if eps < 0:
      raise ValueError('eps must be non-negative.')

    if gamma <= 0:
      raise ValueError('gamma must be a strictly positive number.')

    if rho is not None and rho <= 0:
      raise ValueError('rho must be either None or a strictly positive number.')

    self.num_groups = 1
    self.gamma = gamma
    self.eps = eps
    self.rho = rho
    self.avrg_y_score = 0

    # model paramters (Lagrange dual variables)
    self.lambdas = []
    self.mus = []

  def fit(self, y_orig, group_feature, sgd_steps,
          full_gradient_epochs=1_000, verbose=True, batch_size=256,
          ignore_warnings=False):
    """Debias predictions w.r.t. the sensitive class in each demographic group.

    This procedure takes as input a vector y=y_orig and solves the optimization
    problem subject to the statistical parity constraint.
      minimize_x   gamma/2 ||x||^2 - y^Tx
      s.t.      x satisfies DP constraints with tolerance eps and parameter rho.

    IMPORTANT: If this is used for postprocessing a classifier,
      the scores y_orig need to be rescaled linearly to [-1, +1].

    Training proceeds in two rounds. First is SGD. Second is full gradient
    descent. Full gradient descent is recommended when debiasing deep neural
    nets because the scores are concentrated around the extremes
    so high preciseion might be needed. Because the loss is smooth, the lr
    in full gradient method does not need tuning. It can be set to gamma / 2.0.

    Args:
      y_orig: A vector of the original probability scores. If this is used for
        debiasing binary classifiers, y_orig = 2 * p(y=1) -1.
      group_feature: An array containing the group id of each instance starting
        from group 0 to group K-1.
      sgd_steps: Number of minibatch steps in SGD.
      full_gradient_epochs: Number of epochs in full gradient descent phase.
      verbose: Set to True to display progress.
      batch_size: Size of minibatches in SGD.
      ignore_warnings: Set to True to suppress warnings.

    Returns:
      None.
    """
    if min(y_orig) >= 0:
      self.yscale = 'positive'
    else:
      self.yscale = 'negative'

    y_orig = np.array(y_orig)
    num_groups = len(set(group_feature))  # number of demographic groups

    if (min(y_orig) < -1 or max(y_orig) > 1) and not ignore_warnings:
      print('Warning: the scores y_orig are not in the range [-1, +1].'
            'To suppress this message, set ignore_warnings=True.')

    if self.yscale == 'positive' and not ignore_warnings:
      print('Warning: if this is for postprocessing a binary classifier, '
            'the scores need to be rescaled to [-1, +1]. To suppress this '
            'message, set ignore_warnings=True.')
    if min(group_feature) != 0 or (max(group_feature) != num_groups - 1):
      raise ValueError('group_feature should be in {0, 1, .. K-1} where '
                       'K is the nubmer of groups. Some groups are missing.')

    self.num_groups = num_groups
    eps0 = self.eps / 2.0
    gamma = self.gamma

    # Store group membership ids in a dictionary.
    xk_groups = {}
    for k in range(num_groups):
      xk_groups[k] = []
    for i in range(len(group_feature)):
      xk_groups[group_feature[i]].append(i)

    for k in xk_groups:
      assert xk_groups[k]  # All groups must be non-empty.

    self.avrg_y_score = float(sum(y_orig))/len(y_orig)
    if self.rho is None:
      if self.yscale == 'positive':
        self.rho = self.avrg_y_score
      else:
        self.rho = self.avrg_y_score / 2.0 + 0.5

    # The parameters we optimize in the algorithm are lambdas and mus.
    # lambdas_final and mus_final are running averages (final output).
    lambdas = np.zeros((num_groups,))
    mus = np.zeros((num_groups,))
    lambdas_final = np.zeros((num_groups,))  # running averages
    mus_final = np.zeros((num_groups,))  # running averages

    # SGD is carried out in each group separately due to decomposition of the
    # optimization problem.
    num_samples_sgd = sgd_steps * batch_size
    lr = gamma * math.sqrt(1.0 / num_samples_sgd)

    # Begin the projected SGD phase.
    if verbose:
      print('SGD phase started:')
    for k in range(num_groups):
      if verbose:
        print('Group %d.\t\t%02d%%'%(k, int(100*k/num_groups)), end='\r')

      idx = np.array(list(xk_groups[k]))  # instance IDs in group k
      group_size = len(idx)
      for _ in range(sgd_steps):
        # Using random.randint is 10x faster than random.choice.
        batch_ids = np.random.randint(0, group_size, batch_size)
        batch_ids = idx[batch_ids]

        # The code below is a faster implementation of:
        # xi_arg = y_orig[batch_ids] - (lambdas[k] - mus[k])
        # xi_gradient = xi_arg/gamma
        # xi_gradient = np.maximum(xi_gradient, 0.)
        # xi_gradient = np.minimum(xi_gradient, 1.)

        lambda_minus_mu = lambdas[k] - mus[k]
        xi_arg = np.maximum(y_orig[batch_ids], lambda_minus_mu)
        xi_arg = np.minimum(xi_arg, gamma + lambda_minus_mu)
        mean_xi = (np.mean(xi_arg) - lambda_minus_mu) / gamma

        lambda_gradient = eps0 + self.rho - mean_xi
        mu_gradient = eps0 - self.rho + mean_xi

        # stochastic gradient descent
        if eps0 > 1e-3:
          lambdas[k] = max(0, lambdas[k] - lr * batch_size * lambda_gradient)
          mus[k] = max(0, mus[k] - lr * batch_size * mu_gradient)
        else:
          # If self.eps=0, we can drop mus and optimize lambdas only but
          # lambdas will not be constrained to be non-negative in this case.
          lambdas[k] = lambdas[k] - lr * batch_size * lambda_gradient

        # lambdas_final and mus_final are running averages.
        lambdas_final[k] += lambdas[k] / sgd_steps
        mus_final[k] += mus[k] / sgd_steps

    # Now switch to full gradient descent.
    # Because the objective is smooth, lr=gamma/2 works.
    if verbose and full_gradient_epochs:
      print('\nFull gradient descent phase started:')
    for k in range(num_groups):
      if verbose:
        print('Group {}.'.format(k))

      idx = np.array(list(xk_groups[k]))
      for _ in range(full_gradient_epochs):
        lambda_minus_mu = lambdas_final[k] - mus_final[k]
        xi_arg = np.maximum(y_orig[idx], lambda_minus_mu)
        xi_arg = np.minimum(xi_arg, gamma + lambda_minus_mu)
        mean_xi = (np.mean(xi_arg) - lambda_minus_mu) / gamma

        full_grad_lambda = eps0 + self.rho - mean_xi
        full_grad_mu = eps0 - self.rho + mean_xi

        if eps0 > 1e-3:
          lambdas_final[k] = max(0,
                                 lambdas_final[k] - 0.5*gamma*full_grad_lambda)
          mus_final[k] = max(0, mus_final[k] - 0.5*gamma*full_grad_mu)
        else:
          lambdas_final[k] = lambdas_final[k] - 0.5*gamma*full_grad_lambda

    self.lambdas = lambdas_final
    self.mus = mus_final

  def predict(self, y_orig, group_feature, ignore_warnings=False):
    """Debiases the predictions.

    Given the original scores y, post-process them according to the learned
    model such that the predictions satisfy the desired fairness criteria.

    Args:
      y_orig: Original classifier scores. If this is for postprocessing binary
        classifiers, y_orig = 2 * p(y=1) -1.
      group_feature: An array containing the group id of each instance starting
        from group 0 to group K-1.
      ignore_warnings: Set to True to suppress warnings.

    Returns:
      y_new_prob: y_new_prob[i] is the probability of predicting the positive
        class for the instance i.
    """
    if (((min(y_orig) >= 0 and self.yscale == 'negative') or
         (min(y_orig) < 0 and self.yscale == 'positive')) and
        not ignore_warnings):
      print('Warning: the scores seem to have a difference scale from the '
            'training data. '
            'If the data is scaled in [0, 1], e.g. for preprocessing, or '
            'in [-1, +1], e.g. for postprocessing, make sure the test labels '
            'are scaled similarly.')

    num_examples = len(y_orig)  # number of training examples
    gamma = self.gamma
    lambdas = self.lambdas
    mus = self.mus

    y_new_prob = np.zeros((num_examples,))
    for i in range(num_examples):
      k = group_feature[i]
      if y_orig[i] < (lambdas[k]-mus[k]):
        y_new_prob[i] = 0
      elif y_orig[i] < (lambdas[k]-mus[k]) + gamma:
        y_new_prob[i] = (1.0/gamma)*(y_orig[i]-(lambdas[k]-mus[k]))
      else:
        y_new_prob[i] = 1.0

    return y_new_prob
