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

"""Stochastic Gradient MCMC methods.

Experimental implementations of advanced SG-MCMC methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import scipy.stats as stats

import tensorflow.compat.v1 as tf


DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM_DECAY = 0.7

DEFAULT_PRECOND_RUNNING_AVG_FACTOR = 0.99
DEFAULT_PRECOND_REGULARIZATION = 1.0e-7


class SGMCMCOptimizer(tf.keras.optimizers.Optimizer):
  """SG-MCMC Optimizer base class."""

  def __init__(
      self,
      name,
      total_sample_size=0,
      temp=1.0,
      timestep_factor=1.0,
      preconditioner='identity',
      preconditioner_running_average_factor=DEFAULT_PRECOND_RUNNING_AVG_FACTOR,
      preconditioner_regularization=DEFAULT_PRECOND_REGULARIZATION,
      preconditioner_update=True,
      preconditioner_mdict=None,
      **kwargs):
    r"""SG-MCMC Base class, providing preconditioning functionality.

    All SG-MCMC methods can derive from this class to obtain pre-conditioning
    functionality for free.

    [1] Geoffrey Hinton, Nitish Srivastava, Kevin Swersky, slide 29 in
        "RMSprop",
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    [2] John Duchi, Elad Hazan, Yoram Singer,
        "Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization", JMLR, 2011,
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    Args:
      name: string, name of the Optimizer object.
      total_sample_size: int, total number of training samples in the dataset.
        Must be provided.
      temp: float, > 0, temperature
      timestep_factor: float, scaling of the SDE discretization timestepping.
        We use a deep learning parameterization for our SG-MCMC
        methods which has the benefit of directly matching the common SGD
        parameterization used in deep learning.  However, in terms of the SDE
        we would like to directly control time-stepping.  We therefore allow
        adjustment of timestepping through an additional 'timestep_factor'
        variable which can be used for learning rate decay.
        A 'timestep_factor' of 0.5 would halve the SDE discretization
        time step.
      preconditioner: string, select the preconditioner; one of
        'identity': no preconditioning, M=I,
        'fixed': fixed preconditioning matrix M.  When using 'fixed', you must
          provide either the preconditioner_mdict keyword argument or call the
          set_preconditioner_dict method before using the optimizer.
        'adagrad': AdaGrad preconditioning, [2],
        'rmsprop': RMSprop preconditioner, [1].
      preconditioner_running_average_factor: float, >= 0.0, < 1.0, the running
        average factor, where a value close to one leads to large time horizon
        averages.  Used for 'rmsprop' only.
      preconditioner_regularization: float, >= 0.0, the diagonal constant added
        to the preconditioner:
        M = diag(preconditioner_regularization*I + sqrt(V)),
        where V is the diagonal statistics maintained by the preconditioning
        method.  Used by both 'adagrad' and 'rmsprop'.
      preconditioner_update: boolean, whether to update the preconditioner
        during optimization.  This property can be controlled during
        optimization through the self.preconditioner_update variable, which must
        be set using
        tf.keras.backend.set_value(optimizer.preconditioner_update, False)
      preconditioner_mdict: dictionary of var_name => scalar maps containing the
        diagonal scalar elements of the block-structured preconditioner matrix
        M.  The var_name is the tf.Variable.name property of the model
        variables.  Any variable not found in the dictionary will raise an
        exception.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.

    Raises:
      ValueError: invalid argument value.
    """
    super(SGMCMCOptimizer, self).__init__(name, **kwargs)

    # Total sample size (required)
    if total_sample_size <= 0:
      raise ValueError('You must provide the "total_sample_size" argument '
                       'to any SG-MCMC method.')

    self._set_hyper('total_sample_size', total_sample_size)

    # Target system temperature
    if temp < 0:
      raise ValueError('temp %.3f must be positive' % temp)

    self._set_hyper('temp', temp)
    self._set_hyper('timestep_factor', timestep_factor)

    # Preconditioning parameters
    self._set_hyper('preconditioner_update',
                    tf.cast(preconditioner_update, tf.bool))

    if preconditioner not in ['identity', 'fixed', 'adagrad', 'rmsprop']:
      raise ValueError('preconditioner must be "adagrad", "rmsprop", "fixed", '
                       'or "identity".')

    self.preconditioner = preconditioner
    self.preconditioner_mdict = {}
    if preconditioner_mdict:
      self.set_preconditioner_dict(preconditioner_mdict)

    if (preconditioner_running_average_factor < 0.0 or
        preconditioner_running_average_factor >= 1.0):
      raise ValueError('preconditioner_running_average_factor of %.3f must '
                       'be in [0,1).' % preconditioner_running_average_factor)
    self.preconditioner_running_average_factor = \
        preconditioner_running_average_factor

    if preconditioner_regularization < 0.0:
      raise ValueError('preconditioner_regularization of %.3f must be '
                       'non-negative.' % preconditioner_regularization)
    self.preconditioner_regularization = preconditioner_regularization

  # TODO(nowozin): think about making this a property,
  # https://www.python-course.eu/python3_properties.php, with the only problem
  # being in the derived LangevinDynamicsOptimizer class, where we need access
  # to the model variables; should we index by tf.Variables instead of variable
  # names?
  def set_preconditioner_dict(self, preconditioner_mdict, var_list=None):
    """Set new values for the fixed preconditioner.

    This method is intended to allow updating of the preconditioner during
    optimization.  The current preconditioner must be 'fixed' in order for this
    method to be used.

    Args:
      preconditioner_mdict: dictionary of var_name => scalar maps containing the
        diagonal scalar elements of the block-structured preconditioner matrix
        M.  The var_name is the tf.Variable.name property of the model
        variables.  Any variable not found in the dictionary will raise an
        exception.
      var_list: list of tf.Variables, typically model.trainable_variables.  If
        provided we additionally check that all elements have preconditioning
        values.

    Raises:
      RuntimeError: if current preconditioner is not 'fixed' or if a variable is
        missing from the supplied preconditioner_mdict.
    """
    if self.preconditioner != 'fixed':
      raise RuntimeError('Attempting to set preconditioner dictionary but '
                         'current preconditioner is "%s".  Preconditioner '
                         'dictionaries only make sense for preconditioner '
                         'type "fixed".' % (
                             self.preconditioner))

    # If we currently have a preconditioner dictionary, then check that all keys
    # are present in the updated dictionary
    if self.preconditioner_mdict:
      missing_keys = set(self.preconditioner_mdict.keys()) - \
          set(preconditioner_mdict.keys())
      if missing_keys:
        raise RuntimeError(
            'Variables present in previous preconditioner dictionary are '
            'not present in new preconditioner_mdict.')

    if var_list:
      for var in var_list:
        if var.name not in preconditioner_mdict:
          raise RuntimeError('Variable "%s" not present in given '
                             'preconditioner_mdict.' % var.name)

    if self.preconditioner_mdict:
      # previously not empty: assign
      for varname in self.preconditioner_mdict:
        tf.keras.backend.set_value(
            self.preconditioner_mdict[varname],
            tf.convert_to_tensor(preconditioner_mdict[varname]))
    else:
      # previously empty: rebuild dict with tf.Variable so the created Keras
      # computation graph can be updated across iterations
      self.preconditioner_mdict = {
          varname: tf.Variable(tf.convert_to_tensor(
              preconditioner_mdict[varname]))
          for varname in preconditioner_mdict}

  def _create_slots(self, var_list):
    """Create slots required for preconditioning.

    This method should be called by derived classes.

    Args:
      var_list: list of tf.Variable's that require preconditioning.
    """
    if self.preconditioner == 'identity':
      return
    elif self.preconditioner == 'fixed':
      return
    elif self.preconditioner == 'adagrad' or self.preconditioner == 'rmsprop':
      for var in var_list:
        self.add_slot(var, 'precond_v', initializer='ones')

  def configurational_temperature(self, grad_and_var_list):
    r"""Compute the configurational temperature of the Langevin dynamics.

    See Section 6.1.5 in (Leimkuhler and Matthews, "Molecular Dynamics", 2016).
    The configurational temperature T_conf is given as

      T_conf = E[<q, \nabla_q U(q)] / d,

    where q is the parameter vector, d is the number of parameters, and
    U(q) = n G(q) is the energy defining the target posterior.
    Here we compute the instantaneous temperature and replace the expectation
    E[<q, \nabla_q U(q)>] with the current value <q, \nabla_q U(q)>.  When
    the dimensionality of q is large this will be an accurate approximation of
    the expectation.

    Args:
      grad_and_var_list: list of (grad,var) pairs that are simulated using
        Langevin dynamics, typically model variable gradients and
        model.trainable_variables.

    Returns:
      ctemp: float, instantaneous configurational temperature of the simulated
        system.
    """
    total_sample_size = self._get_hyper('total_sample_size', tf.float32)

    ctemp = 0.0
    dof = 0
    for grad, var in grad_and_var_list:
      ctemp += tf.reduce_sum(total_sample_size*grad*var)
      dof += tf.size(var)

    ctemp = float(ctemp) / float(dof)

    return ctemp

  def perform_preconditioner_update(self, var, grad):
    """Update preconditioner based on current average gradient.

    This method only has an effect if the 'preconditioner_update' hyper
    parameter is True.

    Args:
      var: the model tf.Variable to look up the slots with.
      grad: tf.Tensor, minibatch gradient (average per-sample gradient plus
        (1/n) times the gradient of log-prior.

    Returns:
      updates: list of updates to variables.
    """
    # Check if preconditioner is disabled.
    preconditioner_update = self._get_hyper('preconditioner_update', tf.bool)
    update_factor = tf.cond(preconditioner_update,
                            lambda: tf.constant(1.0),
                            lambda: tf.constant(0.0))

    if self.preconditioner == 'identity':
      return []
    elif self.preconditioner == 'fixed':
      return []
    elif self.preconditioner == 'adagrad':
      v = self.get_slot(var, 'precond_v')
      vnext = v + tf.square(grad)
      vnext = update_factor*vnext + (1.0-update_factor)*v
      return [v.assign(vnext, read_value=False)]
    elif self.preconditioner == 'rmsprop':
      v = self.get_slot(var, 'precond_v')
      vnext = self.preconditioner_running_average_factor*v + \
          (1.0 - self.preconditioner_running_average_factor)*tf.square(grad)
      vnext = update_factor*vnext + (1.0-update_factor)*v
      return [v.assign(vnext, read_value=False)]

  def _get_mscalar(self, var):
    """Return the preconditioning scalar value for the fixed preconditioner.

    Args:
      var: the tf.Variable that is being preconditioned.

    Returns:
      mscalar: float, the scalar to replicate on the diagonal block of M for the
        respective variable.

    Raises:
      KeyError: variable is not present in preconditioner dictionary.
    """
    if var.name not in self.preconditioner_mdict:
      raise KeyError('Variable "%s" not in preconditioner M-dictionary.' %
                     var.name)

    mscalar = self.preconditioner_mdict[var.name]
    mscalar = tf.cast(mscalar, tf.float32)

    return mscalar

  def preconditioner_multiply_minv(self, var, vec):
    """Compute M^{-1} vec.

    Args:
      var: the model tf.Variable to look up the slots with.
      vec: tf.Tensor, vector to apply preconditioning to.

    Returns:
      minv_vec: tf.Tensor with dtype and shape matching grad, M^{-1} vec.
    """
    if self.preconditioner == 'identity':
      minv_vec = vec
    elif self.preconditioner == 'fixed':
      minv_vec = vec / self._get_mscalar(var)
    elif self.preconditioner == 'adagrad' or self.preconditioner == 'rmsprop':
      # M = lambda 1 + sqrt(V)
      v = self.get_slot(var, 'precond_v')
      minv = tf.reciprocal(self.preconditioner_regularization + tf.sqrt(v))
      minv_vec = minv*vec

    return minv_vec

  def preconditioner_multiply_m12(self, var, vec):
    """Compute M^{1/2} vec.

    Args:
      var: the model tf.Variable to look up the slots with.
      vec: tf.Tensor, vector to apply preconditioning to.

    Returns:
      m12_vec: tf.Tensor with dtype and shape matching grad, M^{1/2} vec.
    """
    if self.preconditioner == 'identity':
      m12_vec = vec
    elif self.preconditioner == 'fixed':
      m12_vec = tf.sqrt(self._get_mscalar(var)) * vec
    elif self.preconditioner == 'adagrad' or self.preconditioner == 'rmsprop':
      # M = lambda 1 + sqrt(V)
      v = self.get_slot(var, 'precond_v')
      m12 = tf.sqrt(self.preconditioner_regularization + tf.sqrt(v))
      m12_vec = m12*vec

    return m12_vec

  def preconditioner_multiply_minv12(self, var, vec):
    """Compute M^{-1/2} vec.

    Args:
      var: the model tf.Variable to look up the slots with.
      vec: tf.Tensor, vector to apply preconditioning to.

    Returns:
      minv12_vec: tf.Tensor with dtype and shape matching grad, M^{-1/2} vec.
    """
    if self.preconditioner == 'identity':
      minv12_vec = vec
    elif self.preconditioner == 'fixed':
      minv12_vec = tf.rsqrt(self._get_mscalar(var)) * vec
    elif self.preconditioner == 'adagrad' or self.preconditioner == 'rmsprop':
      # M = lambda 1 + sqrt(V)
      v = self.get_slot(var, 'precond_v')
      minv12 = tf.rsqrt(self.preconditioner_regularization + tf.sqrt(v))
      minv12_vec = minv12*vec

    return minv12_vec

  def get_config(self):
    """Return a dictionary containing the configuration settings."""
    config = super(SGMCMCOptimizer, self).get_config()
    config.update({
        'total_sample_size':
            self._serialize_hyperparameter('total_sample_size'),
        'temp': self._serialize_hyperparameter('temp'),
        'timestep_factor': self._serialize_hyperparameter('timestep_factor'),
        'preconditioner': self.preconditioner,
        'preconditioner_running_average_factor':
            self.preconditioner_running_average_factor,
        'preconditioner_regularization': self.preconditioner_regularization,
        'preconditioner_mdict': self.preconditioner_mdict,
        'preconditioner_update': bool(self.preconditioner_update),
    })
    return config


###
### Brownian dynamics methods: parameters
###


class StochasticGradientLangevinMCMC(SGMCMCOptimizer):
  r"""Stochastic Gradient Langevin Dynamics (SGLD).

  Implementation of the Stochastic Gradient Langevin Dynamics algorithm as
  described in [1].  The loss is assumed to be the average minibatch loss, hence
  of the form

      -(1/m) \sum_{i=1}^m log p(y_i|x_i,w) - (1/n) log p(w),

  where m is the batch size and n is the total number of iid training samples.

  #### References
  [1] _Max Welling_ and _Yee Whye Teh_, NOTYPO
      "bayesian Learning via Stochastic Gradient Langevin Dynamics",
      ICML 2011,
      [PDF](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)
  """

  def __init__(self,
               learning_rate=DEFAULT_LEARNING_RATE,
               name='StochasticGradientLangevinMCMC',
               **kwargs):
    """Initialize the Stochastic Gradient Langevin Dynamics optimizer.

    Args:
      learning_rate: float, the learning rate; here the learning rate is the
        instantaneous effect of the average-minibatch-gradient on the parameter.
        Must be positive and typically is in a range from 1.0e-6 to 0.1.
      name: name of the optimizer.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(StochasticGradientLangevinMCMC, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr = self._get_hyper('learning_rate', var_dtype)
    lr = lr*self._get_hyper('timestep_factor', var_dtype)
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)
    grad = self.preconditioner_multiply_minv(var, grad)

    noise_sigma = tf.sqrt(2.0*temp*lr/total_sample_size)
    noise = noise_sigma * self.preconditioner_multiply_minv12(
        var, tf.random.normal(var.shape))
    var1 = var - lr*grad + noise

    return tf.group(*(pupdates + [var.assign(var1)]))

  def get_config(self):
    config = super(StochasticGradientLangevinMCMC, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
    })
    return config


class LMMCMC(SGMCMCOptimizer):
  r"""Leimkuhler-Matthews method (LM-MCMC).

  Implementation of the Leimkuhler-Matthews method [2] as described on page
  308 of [1].  The loss is assumed to be the average minibatch loss, hence of
  the form

      -(1/m) \sum_{i=1}^m log p(y_i|x_i,w) - (1/n) log p(w),

  where m is the batch size and n is the total number of iid training samples.

  This method is a simple modification of the SGLD method but achieves
  O((learning_rate/total_sample_size)^2) accurate averages if no gradient
  noise is present, see [1], Section 7.10.

  #### References
  [1] _Ben Leimkuhler and Charles Matthews_,
      "Molecular Dynamics", 2015,
      [Publisher link](https://www.springer.com/de/book/9783319163741)
  [2] _Ben Leimkuhler_ and _Charles Matthews_,
      "Rational construction of stochastic numerical methods for molecular
      sampling",
      Applied Mathematical Reesarch Express, Vol. 1, 4--56, 2013.
      doi:10.1093/amrx/abs010
  """

  def __init__(self,
               learning_rate=DEFAULT_LEARNING_RATE,
               name='LMMCMC',
               **kwargs):
    """Initialize the Leimkuhler-Matthews method.

    Args:
      learning_rate: float, the learning rate; here the learning rate is the
        instantaneous effect of the average-minibatch-gradient on the parameter.
        Must be positive and typically is in a range from 1.0e-6 to 0.1.
      name: string, the name of the optimizer object.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(LMMCMC, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

  def _create_slots(self, var_list):
    super(LMMCMC, self)._create_slots(var_list)
    for var in var_list:
      self.add_slot(var, 'rand')

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr = self._get_hyper('learning_rate', var_dtype)
    lr = lr*self._get_hyper('timestep_factor', var_dtype)
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)
    grad = self.preconditioner_multiply_minv(var, grad)

    # R_n
    tau = lr / total_sample_size
    noise_sigma = tf.sqrt(2.0*temp*tau)
    rand = self.get_slot(var, 'rand')
    rand_next = tf.random.normal(var.shape)
    noise = 0.5 * noise_sigma * \
        self.preconditioner_multiply_minv12(var, rand + rand_next)

    var_updated = var - lr*grad + noise

    return tf.group(*(pupdates + [var.assign(var_updated),
                                  rand.assign(rand_next)]))

  def get_config(self):
    config = super(LMMCMC, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
    })
    return config


###
### Langevin dynamics methods: parameters + momentum variables
###


class LangevinDynamicsOptimizer(SGMCMCOptimizer):
  """Base class for any SG-MCMC method which uses moments."""

  def __init__(self,
               name='LangevinDynamicsOptimizer',
               learning_rate=DEFAULT_LEARNING_RATE,
               momentum_decay=DEFAULT_MOMENTUM_DECAY,
               **kwargs):
    """Initialize the Langevin Dynamics base class object.

    Args:
      name: string, the name of the optimizer object.
      learning_rate: float, the learning rate; here the learning rate is the
        instantaneous effect of the average-minibatch-gradient on the parameter.
        Must be positive and typically is in a range from 1.0e-6 to 0.1.
      momentum_decay: float, momentum decay factor, in the range [0,1),
        typically close to 1, e.g. values such as 0.75, 0.8, 0.9, 0.95.
      **kwargs: arguments passed the SGMCMCOptimizer base class.
    """
    super(LangevinDynamicsOptimizer, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('momentum_decay', momentum_decay)

  def _create_slots(self, var_list):
    """Create moments."""
    super(LangevinDynamicsOptimizer, self)._create_slots(var_list)
    for var in var_list:
      self.add_slot(var, 'moments')

  def set_preconditioner_dict(self, preconditioner_mdict, var_list=None):
    """Set new values for the fixed preconditioner.

    For LangevinDynamicsOptimizer implementations, the fixed preconditioner
    specifies a linear invertible map between the Euclidean geometry and the
    preconditioned geometry.  Therefore, when we update the preconditioner we
    need to transform the current moments from their previous geometry to the
    new one.  This method performs the change.

    Args:
      preconditioner_mdict: dictionary of var_name => scalar maps containing the
        diagonal scalar elements of the block-structured preconditioner matrix
        M.  The var_name is the tf.Variable.name property of the model
        variables.  Any variable not found in the dictionary will raise an
        exception.
      var_list: list of tf.Variables, typically model.trainable_variables.
    """
    if not var_list:
      raise ValueError(
          'When using set_preconditioner_dict with LangevinDynamicsOptimizer '
          'classes you must provide the "var_list" keyword argument in order '
          'to allow the moments to be adjusted accordingly.')

    if self.preconditioner_mdict:
      # Transform moments:
      #   moment_updated = M_{new}^{1/2} M_{old}^{-1/2} moment
      for var in var_list:
        mold = self.preconditioner_mdict[var.name]
        mnew = preconditioner_mdict[var.name]
        mom = self.get_slot(var, 'moments')
        mom_updated = tf.sqrt(mnew / mold) * mom
        tf.keras.backend.set_value(mom, mom_updated)
        # mom.assign(mom_updated)

    # Update the dictionary entries
    super(LangevinDynamicsOptimizer, self).set_preconditioner_dict(
        preconditioner_mdict, var_list=var_list)

  def print_diagnostics(self, model, print_fn=None):
    """Print temperature diagnostics.

    This method should be called after or during optimization to print
    diagnostic information regarding related to kinetic temperature.

    Args:
      model: tf.keras.Model that is used for optimization.
      print_fn: function, the function used to print.  Defaults to print, but
        can be overridden, for example with logging.info.
    """
    if print_fn is None:
      print_fn = print

    def kstr(name, ktemp, ktemp_range):
      ok_str = 'in     '
      if (ktemp < ktemp_range[0]) or (ktemp > ktemp_range[1]):
        ok_str = 'OUTSIDE'
      res = '%-40.38s %7.2f  %s  [%7.2f, %7.2f]' % (
          name, ktemp, ok_str, ktemp_range[0], ktemp_range[1])
      return res

    print_fn('________________________________________________________________'
             '_____________')
    print_fn('Variable name                kinetic temperature               '
             '99% hpd region')
    print_fn('================================================================'
             '=============')

    # Compute kinetic temperature for each variable and its hpd region
    for var in model.trainable_variables:
      ktemp = self.kinetic_temperature([var])
      ktemp_range = self.kinetic_temperature_region(tf.size(var))
      print_fn('%s' % kstr(var.name, ktemp, ktemp_range))

    print_fn('================================================================'
             '=============')

    # Compute overall kinetic temperature of the system
    ktemp_total = self.kinetic_temperature(model.trainable_variables)
    total_size = sum(map(tf.size, model.trainable_variables))
    ktemp_total_region = self.kinetic_temperature_region(total_size)
    print_fn('%s' % kstr('<all variables>', ktemp_total, ktemp_total_region))
    print_fn('________________________________________________________________'
             '_____________')

    config = self.get_config()
    print_fn('SGMCMCOptimizer: %s' % config['name'])
    print_fn('temp: %f' % config['temp'])
    print_fn('learning_rate: %f' % config['learning_rate'])
    print_fn('momentum_decay: %f' % config['momentum_decay'])
    print_fn('timestep_factor: %f' % config['timestep_factor'])
    print_fn('preconditioner: %s' % config['preconditioner'])
    print_fn('________________________________________________________________'
             '_____________')

  def dynamics_parameters(self, dtype):
    """Return the current Langevin dynamics discretization parameters.

    Each class deriving from LangevinDynamicsOptimizer has a specific bijective
    map between (h,gamma) and (learning_rate,momentum_decay), where h is the
    continuous-time discretization step length used in the dicretization of the
    SDE. This method performs the following sequence of translations:

      1. Map (learning_rate, momentum_decay) into (h, gamma).
      2. Scale: h_scaled = timestep_factor * h
      3. Map (h_scaled, gamma) into
         (learning_rate_updated, momentum_decay_updated).

    Doing this translation allows the user to think purely in terms of learning
    rate and momentum decay, compatible with current deep learning optimization
    methods.  If we were to scale only the learning_rate, i.e. use
    "learning_rate_updated = timestep_factor * learning_rate", this would
    implicitly also change the friction coefficient gamma, and cannot guarantee
    an improvement in SDE discretization accuracy.

    Args:
      dtype: Data type to return, typically tf.float32.

    Returns:
      lr_updated: scalar learning rate, > 0, deep learning parameterization.
      momentum_decay_updated: scalar momentum decay, >= 0, < 1.
    """
    raise NotImplementedError('Derived classes need to override '
                              'LangevinDynamicsOptimizer dynamics_parameters '
                              'method.')

  def moment_inner_product(self, var):
    """Compute the Euclidean squared norm of moments, using preconditioner.

    Compute p^T M^{-1} p, where p are the moments.

    Args:
      var: the model tf.Variable to look up the moments with.

    Returns:
      inner_product: scalar tf.Tensor containing p^T M^{-1} p.

    Raises:
      RuntimeError: moments slot not found.
    """
    if 'moments' not in self.get_slot_names():
      raise RuntimeError('Optimizer derived from LangevinDynamicsOptimizer '
                         'does not provide "moments" slot for variable "%s".' %
                         var.name)

    mom = self.get_slot(var, 'moments')
    minv_mom = self.preconditioner_multiply_minv(var, mom)
    inner_product = tf.reduce_sum(mom*minv_mom)

    return inner_product

  def kinetic_temperature_region(self, size, hpd_level=0.99):
    """Compute the high-probability-density region of the kinetic temperature.

    The returned interval guarantees that if the temperature stems from a
    perfect Langevin SDE discretization (i.e. the moments are perfectly
    Normal(0,I_d)-distributed), then That in [lb,ub] with probability
    hpd_level.

    Args:
      size: int or tf.int of the number of dimensions.  Typically tf.size(var).
      hpd_level: float, >0.0 and <1.0, the fraction of realizations covered by
        the interval.

    Returns:
      lb: float, >0.0, lower-bound for the hpd-region.
      ub: float, ub > lb, upper-bound for the hpd-region.
    """
    if hpd_level <= 0.0 or hpd_level >= 1.0:
      raise ValueError('hpd_level %f is outside (0,1).' % hpd_level)

    temp = self._get_hyper('temp', tf.float32)

    # chi2.interval evaluates the inverse cumulative distribution function, also
    # known as quantile function, F^{-1}(p) at p=0.5*hpd_level and
    # p=1-0.5*hpd_level.
    lb, ub = stats.chi2.interval(hpd_level, int(size))
    size = tf.cast(size, tf.float32)
    lb = float(temp * lb / size)
    ub = float(temp * ub / size)

    return lb, ub

  def kinetic_temperature(self, var_list):
    """Compute the kinetic temperature of the Langevin dynamics.

    See Section 6.1.5 in (Leimkuhler and Matthews, "Molecular Dynamics", 2016).

    Args:
      var_list: list of variables that are simulated using Langevin dynamics,
        typically model.trainable_variables.

    Returns:
      ktemp: float, kinectic temperature of the simulated system.
    """
    ktemp = 0.0
    dof = 0
    for var in var_list:
      ktemp += self.moment_inner_product(var)
      dof += tf.size(var)

    ktemp = float(ktemp) / float(dof)

    return ktemp

  def get_config(self):
    """Return a dictionary containing the configuration settings."""
    config = super(LangevinDynamicsOptimizer, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'momentum_decay': self._serialize_hyperparameter('momentum_decay'),
    })
    return config


class BAOABMCMC(LangevinDynamicsOptimizer):
  r"""BAOAB Integrator for the Langevin dynamics (BAOAB-MCMC).

  Implementation of the BAOAB method as described on page 271 of [1].  The loss
  is assumed to be the average minibatch loss, hence of the form

      -(1/m) \sum_{i=1}^m log p(y_i|x_i,w) - (1/n) log p(w),

  where m is the batch size and n is the total number of iid training samples.

  We use a special parametrization of the optimizer parameters in terms of the
  instantaneous average-minibatch-gradient effect (`learning_rate`), and the
  unit-free momentum decay.  These parameters are compatible with the parameters
  of SGD-with-momentum methods and are easier to set.

  #### References
  [1] _Ben Leimkuhler and Charles Matthews_,
      "Molecular Dynamics", 2015,
      [Publisher link](https://www.springer.com/de/book/9783319163741)
  """

  def __init__(self,
               name='BAOABMCMC',
               **kwargs):
    """Initialize the BAOAB integrator.

    Args:
      name: name of the optimizer.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(BAOABMCMC, self).__init__(name, **kwargs)

  def dynamics_parameters(self, dtype):
    """Return the dynamics parameters for the BAOAB scheme."""
    lr = self._get_hyper('learning_rate', dtype)
    momentum_decay = self._get_hyper('momentum_decay', dtype)

    # Scale the learning rate and momentum decay using the timestep_factor.
    timestep_factor = self._get_hyper('timestep_factor', dtype)
    lr = (timestep_factor**2.0) * (
        (momentum_decay**timestep_factor + 1.0)/(momentum_decay + 1.0)) * lr
    momentum_decay_updated = momentum_decay**timestep_factor

    return lr, momentum_decay_updated

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    lr, momentum_decay = self.dynamics_parameters(var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)

    mom = self.get_slot(var, 'moments')

    # Compute SDE discretization step size
    h = tf.sqrt(2.0*lr / (total_sample_size*(1.0 + momentum_decay)))

    # scaled R_n
    noise_sigma = tf.sqrt(temp*(1.0 - momentum_decay**2.0))
    noise = noise_sigma * tf.random.normal(var.shape)

    # nabla_U
    nabla_u = total_sample_size*grad

    mom_n = mom - h*nabla_u                                             # BB
    var_n = var + 0.5*h*self.preconditioner_multiply_minv(var, mom_n)   # A
    mom_updated = momentum_decay*mom_n + \
        self.preconditioner_multiply_m12(var, noise)                    # O
    var_updated = var_n + \
        0.5*h*self.preconditioner_multiply_minv(var, mom_updated)       # A

    return tf.group(*(pupdates + [var.assign(var_updated),
                                  mom.assign(mom_updated)]))


class NaiveSymplecticEulerMCMC(LangevinDynamicsOptimizer):
  r"""Symplectic Euler Integrator for the Langevin dynamics.

  Also known as Semi-Implicit Euler Integrator
  https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

  The method is "naively" obtained by using the updated momentum 'mom_updated'
  in the variable update 'var_updated' of the Euler-Maruyama method below.

  The NaiveSymplecticEuler is similar to the OBA integrator except that the
  noise_sigma is computed slightly differently.

  We use a special parametrization of the optimizer parameters in terms of the
  instantaneous average-minibatch-gradient effect (`learning_rate`), and the
  unit-free momentum decay. These parameters are compatible with the parameters
  of SGD-with-momentum methods and are easier to set.
  """

  def __init__(self,
               name='NaiveSymplecticEulerMCMC',
               **kwargs):
    """Initialize the NaiveSymplecticEuler integrator.

    Args:
      name: name of the optimizer.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(NaiveSymplecticEulerMCMC, self).__init__(name, **kwargs)

  def dynamics_parameters(self, dtype):
    """Return the dynamics parameters for the Symplectic Euler scheme."""
    lr = self._get_hyper('learning_rate', dtype)
    momentum_decay = self._get_hyper('momentum_decay', dtype)

    # Scale the learning rate and momentum decay using the timestep_factor.
    # A decrease in timestep_factor is guaranteed to improve SDE simulation
    # accuracy, whereas a simple scaling of learning_rate may not have a
    # beneficial effect because the friction is in effect increased by a factor
    # of (1/sqrt(timestep_factor)).  These relationships are derived from the
    # symplectic Euler deep learning parameterization by expanding
    #
    # h' = timestep_factor * h
    #   ==> learning_rate' = timestep_factor^2 * learning_rate
    #   ==> momentum_decay' = 1 - timestep_factor*(1-momentum_decay)
    timestep_factor = self._get_hyper('timestep_factor', dtype)
    lr = (timestep_factor**2.0) * lr
    momentum_decay_updated = 1.0 - timestep_factor*(1.0 - momentum_decay)

    return lr, momentum_decay_updated

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    lr, momentum_decay = self.dynamics_parameters(var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)

    mom = self.get_slot(var, 'moments')

    # Compute SDE discretization step size
    h = tf.sqrt(lr / total_sample_size)

    # scaled R_n
    noise_sigma = tf.sqrt(2.0*temp*(1.0 - momentum_decay))
    noise = noise_sigma * tf.random.normal(var.shape)

    # nabla_U
    nabla_u = total_sample_size*grad

    mom_updated = momentum_decay*mom - h*nabla_u + \
                  self.preconditioner_multiply_m12(var, noise)
    var_updated = var + h*self.preconditioner_multiply_minv(var, mom_updated)

    return tf.group(*(pupdates + [var.assign(var_updated),
                                  mom.assign(mom_updated)]))


class MultivariateNoseHooverMCMC(LangevinDynamicsOptimizer):
  r"""Multivariate Stochastic Gradient Nose-Hoover Thermostat (mSGNHT) dynamics.

  We use a special parametrization of the optimizer parameters in terms of the
  instantaneous average-minibatch-gradient effect (`learning_rate`), and the
  unit-free momentum decay. These parameters are compatible with the parameters
  of SGD-with-momentum methods and are easier to set.

  To translate the parameters we assume that E[thermostats] = D/(h T).

  #### References
  [1] Chunyuan Li, Changyou Chen, Kai Fan, and Lawrence Carin,
      "High-Order Stochastic Gradient Thermostats for Bayesian Learning of Deep
      Models", AAAI 2016,
      [PDF](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/11834/11806)

  """

  def __init__(self,
               name='MultivariateNoseHooverMCMC',
               **kwargs):
    """Initialize the MultivariateNoseHooverMCMC integrator.

    Args:
      name: name of the optimizer.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(MultivariateNoseHooverMCMC, self).__init__(name, **kwargs)

  def dynamics_parameters(self, dtype):
    """Return the dynamics parameters for the mSGNHT method."""
    # The update is identical to the symplectic Euler scheme.
    lr = self._get_hyper('learning_rate', dtype)
    momentum_decay = self._get_hyper('momentum_decay', dtype)

    # Scale the learning rate and momentum decay using the timestep_factor.
    timestep_factor = self._get_hyper('timestep_factor', dtype)
    lr = (timestep_factor**2.0) * lr
    momentum_decay_updated = 1.0 - timestep_factor*(1.0 - momentum_decay)

    return lr, momentum_decay_updated

  @property
  def mean_thermostat(self):
    """Return the estimated mean thermostat value.

    The thermostat mean is independent of the target temperature.

    Returns:
      mean: float, E[thermostat] estimate, assuming no noise.
    """
    # E[thermostat] = D/(h T), with "D = (1-momentum_decay) T" we have
    # E[thermostat] = (1-momentum_decay) / h,
    # and with h = sqrt(learning_rate/total_sample_size), we have
    # E[thermostat] = sqrt(total_sample_size/learning_rate)*(1-momentum_decay)
    total_sample_size = self._get_hyper('total_sample_size', tf.float32)
    lr, momentum_decay = self.dynamics_parameters(tf.float32)

    sqrt_h_inv = tf.sqrt(total_sample_size / lr)
    mean = sqrt_h_inv * (1.0 - momentum_decay)
    mean = float(mean)

    return mean

  def mean_thermostat_var(self, var):
    """Compute the average current thermostat value for the given variable.

    Args:
      var: tf.Variable that is in model.trainable_variables.

    Returns:
      mean_thermostat_value: float, average thermostat value among all scalar
        thermostats associated to the given variable.
    """
    thm = self.get_slot(var, 'thermostats')
    mean_thermostat_value = float(tf.reduce_mean(thm))

    return mean_thermostat_value

  def _create_slots(self, var_list):
    super(MultivariateNoseHooverMCMC, self)._create_slots(var_list)
    for var in var_list:
      # Initialize thermostat values to expected value
      thermostats_init = tf.keras.initializers.Constant(self.mean_thermostat)
      self.add_slot(var, 'thermostats', initializer=thermostats_init)

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    lr, momentum_decay = self.dynamics_parameters(var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)

    mom = self.get_slot(var, 'moments')
    thm = self.get_slot(var, 'thermostats')

    # Compute SDE discretization step size
    h = tf.sqrt(lr / total_sample_size)
    hg = tf.sqrt(lr * total_sample_size)

    # Additive noise, sqrt(2 (1-momentum_decay) T) M^{1/2} R_n
    noise_sigma = tf.sqrt(2.0*temp*(1.0 - momentum_decay))
    noise = tf.random.normal(var.shape)
    noise = noise_sigma * self.preconditioner_multiply_m12(var, noise)

    # grad = nabla G
    mom_updated = mom - hg*grad - h*(thm*mom) + noise
    momu_pc = self.preconditioner_multiply_minv(var, mom_updated)
    thm_updated = thm + h*(mom_updated*momu_pc - temp)
    var_updated = var + h*momu_pc

    return tf.group(*(pupdates + [var.assign(var_updated),
                                  thm.assign(thm_updated),
                                  mom.assign(mom_updated)]))


class BBKMCMC(LangevinDynamicsOptimizer):
  r"""Brunger-Brooks-Karplus method (BBK-MCMC).

  Implementation of the Brunger-Brooks-Karplus method [2] as described on page
  277 of [1].  The loss is assumed to be the average minibatch loss, hence of
  the form

      -(1/m) \sum_{i=1}^m log p(y_i|x_i,w) - (1/n) log p(w),

  where m is the batch size and n is the total number of iid training samples.

  We use a special parametrization of the optimizer parameters in terms of the
  instantaneous average-minibatch-gradient effect (`learning_rate`), and the
  unit-free momentum decay.  These parameters are compatible with the parameters
  of SGD-with-momentum methods and are easier to set.

  #### References
  [1] _Ben Leimkuhler and Charles Matthews_,
      "Molecular Dynamics", 2015,
      [Publisher link](https://www.springer.com/de/book/9783319163741)
  [2] _Axel Brunger, Charles L. Brooks, and Martin Karplus_,
      "Stochastic boundary conditions for molecular dynamics simulations of ST2
      water",
      Chemical physics letters, Vol. 105, No 5, 1984, 495--500.
  """

  def __init__(self,
               name='BBKMCMC',
               **kwargs):
    """Initialize the Brunger-Brooks-Karplus optimizer.

    Args:
      name: name of the optimizer.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(BBKMCMC, self).__init__(name, **kwargs)

  def dynamics_parameters(self, dtype):
    """Return the dynamics parameters for the BBK scheme."""
    lr = self._get_hyper('learning_rate', dtype)
    momentum_decay = self._get_hyper('momentum_decay', dtype)

    # Scale the learning rate and momentum decay using the timestep_factor.
    timestep_factor = self._get_hyper('timestep_factor', dtype)
    momentum_decay_updated = \
        (1.0 + momentum_decay - timestep_factor*(1.0 - momentum_decay)) / \
        (1.0 + momentum_decay + timestep_factor*(1.0 - momentum_decay))
    lr = (timestep_factor**2.0) * (
        (momentum_decay_updated + 1.0) / (momentum_decay + 1.0)) * lr

    return lr, momentum_decay_updated

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    lr, momentum_decay = self.dynamics_parameters(var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)

    mom = self.get_slot(var, 'moments')

    # Compute SDE discretization step size
    h = tf.sqrt(2.0*lr / (total_sample_size*(1.0 + momentum_decay)))
    gamma = (2.0/h) * (1.0 - momentum_decay) / (1.0 + momentum_decay)

    # R_n
    noise_sigma = 0.5*tf.sqrt(2.0*h*gamma*temp)
    noise = noise_sigma * self.preconditioner_multiply_m12(
        var, tf.random.normal(var.shape))

    # nabla_U
    nabla_u = total_sample_size*grad

    mom_n = (1.0/(1.0+0.5*h*gamma))*(mom - 0.5*h*nabla_u + noise)  # p_n
    mom_updated = (1.0-0.5*h*gamma)*mom_n - 0.5*h*nabla_u + noise  # p_{n+1/2}
    var_updated = var + h*self.preconditioner_multiply_minv(var, mom_updated)

    return tf.group(*(pupdates + [var.assign(var_updated),
                                  mom.assign(mom_updated)]))


class StochasticPositionVerletMCMC(LangevinDynamicsOptimizer):
  r"""Stochastic Position Verlet (SPV-MCMC).

  Implementation of the Stochastic Position Verlet method as described in
  described in Section 7.3.1 of [1].  The loss is assumed to be the average
  minibatch loss, hence of the form

      -(1/m) \sum_{i=1}^m log p(y_i|x_i,w) - (1/n) log p(w),

  where m is the batch size and n is the total number of iid training samples.

  We use a special parametrization of the optimizer parameters in terms of the
  instantaneous average-minibatch-gradient effect (`learning_rate`), and the
  unit-free momentum decay.  These parameters are compatible with the parameters
  of SGD-with-momentum methods and are easier to set.

  #### References
  [1] _Ben Leimkuhler and Charles Matthews_,
      "Molecular Dynamics", 2015,
      [Publisher link](https://www.springer.com/de/book/9783319163741)
  """

  def __init__(self,
               name='StochasticPositionVerletMCMC',
               **kwargs):
    """Initialize the Stochastic Position Verlet optimizer.

    Args:
      name: name of the optimizer.
      **kwargs: arguments passed the tf.keras.optimizers.Optimizer base class.
    """
    super(StochasticPositionVerletMCMC, self).__init__(name, **kwargs)

  def dynamics_parameters(self, dtype):
    """Return the dynamics parameters for the SPV scheme."""
    lr = self._get_hyper('learning_rate', dtype)
    momentum_decay = self._get_hyper('momentum_decay', dtype)

    # Scale the learning rate and momentum decay using the timestep_factor.
    timestep_factor = self._get_hyper('timestep_factor', dtype)
    momentum_decay_updated = momentum_decay**timestep_factor
    lr = (timestep_factor**2.0) * (
        (1.0 - 1.0/(timestep_factor*tf.log(momentum_decay))) / \
            (1.0 - 1.0/tf.log(momentum_decay))) * lr

    return lr, momentum_decay_updated

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    total_sample_size = self._get_hyper('total_sample_size', var_dtype)
    lr, momentum_decay = self.dynamics_parameters(var_dtype)
    temp = self._get_hyper('temp', var_dtype)

    pupdates = self.perform_preconditioner_update(var, grad)

    mom = self.get_slot(var, 'moments')

    # Compute SDE discretization step size
    h = tf.sqrt(2.0*lr / (total_sample_size *
                          (1.0-1.0/tf.math.log(momentum_decay))))
    gamma = -tf.math.log(momentum_decay) / h
    eta = (1.0-tf.exp(-h*gamma))/gamma
    zeta = tf.sqrt(temp * (1.0 - tf.exp(-2.0*h*gamma)))

    # q_{n+1/2} = q_n + (h/2) p_n
    var_half = var + 0.5*h*self.preconditioner_multiply_minv(var, mom)

    noise = zeta*self.preconditioner_multiply_m12(
        var, tf.random.normal(var.shape))  # \zeta M^{1/2} R_n

    # p_{n+1} = e^{-h \gamma} p_n - \eta \nabla U(q_n) + \zeta R_n,
    # here \nabla U(q_n) is equal to (total_sample_size \nabla G(q_n)), where
    # G is the minibatch mean and (1/total_sample_size)-scaled log-prior.
    mom_updated = tf.exp(-h*gamma)*mom - (eta*total_sample_size)*grad + noise
    var_updated = var_half + \
        0.5*h*self.preconditioner_multiply_minv(var, mom_updated)

    return tf.group(*(pupdates + [var.assign(var_updated),
                                  mom.assign(mom_updated)]))


def cyclical_rate(period_length, step, schedule='cosine', min_value=0.001):
  """Compute cyclical learning rate schedule.

  The cyclical cosine learning rate schedule due to
  [(Loshchilov and Hutter, ICLR 2017)](https://arxiv.org/pdf/1608.03983.pdf)
  and used in an SG-MCMC context by
  [(Zhang et al., 2019)](https://arxiv.org/pdf/1902.03932.pdf).

  The glide schedule spends around 10 percent of the time very close to
  min_value.

  Args:
    period_length: number of iterations within one period.  Must be >= 1.
    step: int or tf.int32, step within a period, any value >= 1 is allowed.
    schedule: str, 'cosine', 'glide', or 'flat'.
      'cosine': the cosine schedule used in (Zhang et al., 2019).
      'glide': a decreasing schedule which spends ~20 percent of iterations at
        very small values.
      'flat': always returns 1.0.
    min_value: float, the minimum rate returned by this method.

  Returns:
    stepsize: tf.float32, learning rate multiplier.  Range is (0.0, 1.0].
    is_end_of_period_iteration: tf.bool, True if this is the final iteration
      within one period (every period_length's iteration), i.e. the iteration
      with the smallest stepsize.

  Raises:
    ValueError: invalid value for step parameter.
  """
  if step <= 0:
    raise ValueError('step must be 1 or larger')

  pfraction = tf.cast(tf.math.mod(step-1, period_length),
                      dtype=tf.float32) / tf.cast(period_length, tf.float32)
  if schedule == 'cosine':
    stepsize = min_value + (1.0-min_value)*0.5*(
        tf.math.cos(math.pi*pfraction) + 1.0)
  elif schedule == 'glide':
    stepsize = min_value + (1.0-min_value)*(
        tf.math.exp(-pfraction / (1.0 - pfraction)))
  elif schedule == 'flat':
    stepsize = 1.0
  else:
    raise ValueError('Invalid schedule value "%s".' % schedule)

  is_end_of_period_iteration = tf.equal(tf.math.mod(step, period_length), 0)

  return stepsize, is_end_of_period_iteration
