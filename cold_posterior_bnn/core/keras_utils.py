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
"""Support for Ensemble api using a keras.Model.fit() loop."""

import warnings

from absl import logging
import numpy as np
import tensorflow as tf   # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow import keras
from cold_posterior_bnn.core import diagnostics
from cold_posterior_bnn.core import sgmcmc

K = tf.keras.backend


class WarmupScheduler(tf.keras.callbacks.Callback):
  """Learning rate scheduler with warm-up."""

  def __init__(self, init_lr, warmup_iterations):
    """Create a new warm-up schedule with specified decays.

    Args:
      init_lr: float, the initial base learning rate, e.g. 0.1
      warmup_iterations: int, number of initial linear-ramp warmup iterations.
    """
    self.warmup_factor = 0.0
    self.global_step = 0
    self.warmup_iterations = warmup_iterations
    self.init_lr = init_lr

  def on_batch_begin(self, batch, logs=None):
    self.global_step += 1
    if self.global_step >= self.warmup_iterations:
      self.warmup_factor = 1.0
    else:
      self.warmup_factor = float(self.global_step) / \
          float(self.warmup_iterations)

    lr = self.init_lr * self.warmup_factor
    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


class TemperatureMetric(tf.keras.callbacks.Callback):
  """Report the SG-MCMC target temperature.

  This callback only reports the target temperature for optimizers that derive
  from sgmcmc.SGMCMCOptimizer.  If this is not the case, this callback does
  nothing.
  """

  def on_epoch_end(self, epoch, logs):
    if isinstance(self.model.optimizer, sgmcmc.SGMCMCOptimizer):
      logs['temp'] = self.model.optimizer.temp.numpy()


class SamplerTemperatureMetric(tf.keras.callbacks.Callback):
  """Report the SG-MCMC kinetic temperatures per variable.

  This callback reports temperatures for optimizers that derive from
  sgmcmc.LangevinDynamicsOptimizer.  It will not output any kinetic temperatures
  for other optimizers.

  The callback adds the following logs:
    1. For each variable in model.trainable_variables it adds a
       'ktemp/<var.name>' log entry with the kinetic temperature.
    2. It adds a global 'ktemp_fraction_in_hpd' log entry which gives the
       fraction of variables which meet their 99% hpd region, see the
       documentation of the
       sgmcmc.LangevinDynamicsOptimizer.kinetic_temperature_region method.
  """

  def on_epoch_end(self, epoch, logs):
    if not isinstance(self.model.optimizer, sgmcmc.LangevinDynamicsOptimizer):
      return

    var_count = 0
    var_in_hpd = 0
    for var in self.model.trainable_variables:
      # Kinetic temperature of momentum variable
      ktemp = self.model.optimizer.kinetic_temperature([var])
      logs['ktemp/' + var.name] = ktemp

      # Keep statistics of how many variables are in hpd region
      ktemp_lb, ktemp_ub = self.model.optimizer.kinetic_temperature_region(
          tf.size(var))
      if ktemp >= ktemp_lb and ktemp <= ktemp_ub:
        var_in_hpd += 1
      var_count += 1

    var_fraction_in_hpd = float(var_in_hpd) / float(var_count)
    logs['ktemp_fraction_in_hpd'] = var_fraction_in_hpd


class TemperatureRampScheduler(tf.keras.callbacks.Callback):
  """Temperature ramp scheduler.

  This callback controls the 'temp' parameter of a sgmcmc.SGMCMCOptimizer object
  according to a linear ramp (up or down) from a specified start iteration to
  and a specified duration.
  """

  def __init__(self, init_temp, final_temp,
               begin_ramp_iteration, ramp_iterations):
    """Create a new warm-up schedule with specified decays.

    Args:
      init_temp: float, the initial system temperature, e.g. 0.0
      final_temp: float, the final system temperature, e.g. 1.0
      begin_ramp_iteration: int, number of iterations before beginning ramp.
      ramp_iterations: int, number of initial linear-ramp temperature
        iterations.
    """
    self.global_step = 0

    self.init_temp = init_temp
    self.final_temp = final_temp
    self.begin_ramp_iteration = begin_ramp_iteration
    self.ramp_iterations = ramp_iterations

  def on_batch_begin(self, batch, logs=None):
    # Callback only applies to SG-MCMC methods
    if not isinstance(self.model.optimizer, sgmcmc.SGMCMCOptimizer):
      return

    self.global_step += 1
    if self.global_step <= self.begin_ramp_iteration:
      temp = self.init_temp
    elif self.global_step >= (self.begin_ramp_iteration + self.ramp_iterations):
      temp = self.final_temp
    else:
      temp = (float(self.global_step-self.begin_ramp_iteration) /
              float(self.ramp_iterations)) * \
          (self.final_temp - self.init_temp) + self.init_temp

    tf.keras.backend.set_value(self.model.optimizer.temp, temp)


class TemperatureZeroOneZeroScheduler(tf.keras.callbacks.Callback):
  """Temperature scheduler for a 0-1-0 temperature profile.

  This callback controls the 'temp' parameter of a sgmcmc.SGMCMCOptimizer object
  according to a 0-1-0 profile.
  """

  def __init__(self, t0_iteration, t1_iteration):
    """Create a 0-1-0 temperature profile.

    * if (it <= T0_iteration) or (it > T0_iteration + T1_iteration):
        T=0
    * else: # it in [T0_iteration+1, T0_iteration + T1_iteration]
        T=1

    Args:
      t0_iteration: int, number of iterations of the first phase with T=0.
      t1_iteration: int, number of iterations of the phase with T=1
    """
    self.global_step = 0

    self.t0_iteration = t0_iteration
    self.t1_iteration = t1_iteration

  def on_batch_begin(self, batch, logs=None):
    # Callback only applies to SG-MCMC methods
    if not isinstance(self.model.optimizer, sgmcmc.SGMCMCOptimizer):
      return

    self.global_step += 1
    if self.global_step <= self.t0_iteration:
      temp = 0.
    elif self.global_step <= (self.t0_iteration + self.t1_iteration):
      temp = 1.
    else:
      temp = 0.

    tf.keras.backend.set_value(self.model.optimizer.temp, temp)


class PrintDiagnosticsCallback(keras.callbacks.Callback):
  """Print SG-MCMC diagnostics for sgmcmc.LangevinDynamicsOptimizer classes.

  This callback only prints diagnostics for optimizers which derive from the
  sgmcmc.LangevinDynamicsOptimizer class, and otherwise outputs nothing.
  """

  def __init__(self, every_nth_epoch=1, print_fn=None):
    """Create a new PrintDiagnosticsCallback object.

    This callback is only applicable for optimizer classes that derive from
    sgmcmc.LangevinDynamicsOptimizer.

    Args:
      every_nth_epoch: int, print diagnostics every n'th epoch.
      print_fn: Print function to use.
    """
    self.every_nth_epoch = every_nth_epoch
    if print_fn is None:
      print_fn = print
    self.print_fn = print_fn

  def on_epoch_end(self, epoch, logs=None):
    # Callback only applies to SG-MCMC methods
    if not isinstance(self.model.optimizer, sgmcmc.LangevinDynamicsOptimizer):
      return

    if epoch % self.every_nth_epoch != 0:
      return

    self.print_fn('.')
    self.print_fn('SG-MCMC diagnostics at end of epoch %d' % epoch)
    self.model.optimizer.print_diagnostics(self.model)


class EstimatePreconditionerCallback(tf.keras.callbacks.Callback):
  """Re-estimate a diagonal preconditioner.

  This callback can only be used with optimizers that derive from
  sgmcmc.SGMCMCOptimizer class.
  """

  def __init__(self, train_fn, train_iter, every_nth_epoch=1, batch_count=64,
               raw_second_moment=False, log_ctemp=True, update_precond=True):
    """Create a new preconditioner estimation callback.

    Example usage:

    >>> def gradest_train_fn():
          @tf.function
          def gest_step(grad_est, model, images, labels):
            with tf.GradientTape(persistent=True) as tape:
              labels = tf.squeeze(labels)
              logits = model(images)
              ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                 logits=logits, labels=labels)
              obj = tf.reduce_mean(ce) + sum(model.losses)

            gradients = tape.gradient(obj, model.trainable_variables)
            grad_est.apply_gradients(zip(gradients, model.trainable_variables))

          def train_step(grad_est, model, data):
            images, labels = data
            gest_step(grad_est, model, images, labels)

          return train_step
    >>> precond_estimator_cb = keras_utils.EstimatePreconditionerCallback(
            gradest_train_fn, iter(dataset_train))

    Args:
      train_fn: function with signature train_fn(grad_est, model, data), where
        grad_est is an object of type diagnostics.GradientNoiseEstimator and
        model a tf.keras.models.Model that can be called as model(*data).
      train_iter: iterator that can be used as data = next(train_iter) to obtain
        a data set batch.
      every_nth_epoch: int, estimate a preconditioner every n'th epoch.  A value
        of 1 means that a preconditioner will be estimated in every single
        epoch.
      batch_count: int, number of batches to use to estimate the noise.
      raw_second_moment: bool, if True we use RMSprop like raw second moment
        estimates to compute the preconditioner; if False we only use the
        gradient noise variance estimates.
      log_ctemp: bool, if True then we will output configurational temperature
        statistics, see sgmcmc.py.
      update_precond: bool, if True, then we will write the estimated
        preconditioner to the optimizer object (this is the desired behavior in
        most cases).
    """
    self.train_fn = train_fn
    self.train_iter = train_iter
    self.batch_count = batch_count
    self.every_nth_epoch = every_nth_epoch
    self.raw_second_moment = raw_second_moment
    self.log_ctemp = log_ctemp
    self.update_precond = update_precond

    self.ctemp = None
    self.ctemp_all = None

  def on_epoch_begin(self, epoch, logs=None):
    if epoch % self.every_nth_epoch != 0:
      return

    print('Epoch %d, estimating new preconditioner using %d batches' %
          (epoch, self.batch_count))

    precond_dict, ctemp, ctemp_all = self.estimate_new_preconditioner()
    self.ctemp = ctemp
    self.ctemp_all = ctemp_all
    if self.update_precond:
      self.set_new_preconditioner(precond_dict)

  def on_epoch_end(self, epoch, logs):
    if self.log_ctemp and \
        isinstance(self.model.optimizer, sgmcmc.SGMCMCOptimizer):
      if self.ctemp_all:
        logs['ctemp'] = self.ctemp_all

      if self.ctemp:
        for vname, ctemp in self.ctemp.items():
          logs[vname] = ctemp

  def estimate_new_preconditioner(self):
    grad_est = diagnostics.GradientNoiseEstimator()
    train_step = self.train_fn()
    for _ in range(self.batch_count):
      data = next(self.train_iter)
      train_step(grad_est, self.model, data)

    precond_dict = grad_est.estimate_fixed_preconditioner(
        self.model, raw_second_moment=self.raw_second_moment)

    ctemp = dict()
    ctemp_all = None
    if isinstance(self.model.optimizer, sgmcmc.SGMCMCOptimizer):
      # Per-variable configurational temperatures
      for var in self.model.trainable_variables:
        ctemp_var = self.model.optimizer.configurational_temperature(
            [(var, grad_est.get_slot(var, 'mean'))])
        ctemp['ctemp/' + var.name] = ctemp_var

      # Overall configurational temperature
      grad_and_var_list = [(var, grad_est.get_slot(var, 'mean'))
                           for var in self.model.trainable_variables]
      ctemp_all = self.model.optimizer.configurational_temperature(
          grad_and_var_list)

    return precond_dict, ctemp, ctemp_all

  def set_new_preconditioner(self, precond_dict):
    self.model.optimizer.set_preconditioner_dict(
        precond_dict, self.model.trainable_variables)


class CyclicSamplerCallback(keras.callbacks.Callback):
  """Cyclical learning rate sampler for use with model.fit.
  """

  def __init__(self, ensemble, cycle_period_length, sampling_start_epoch,
               schedule='cosine', min_value=0.005):
    """Create a new CyclicSamplerCallback object.

    This callback is only applicable for optimizer classes that derive from
    sgmcmc.LangevinDynamicsOptimizer.

    Args:
      ensemble: an ensemble container class from bnn.ensemble, for example an
        ensemble.FreshReservoirEnsemble object.
      cycle_period_length: int, number of iterations per sampling cycle.
      sampling_start_epoch: int, the epoch number at which to start sampling.
        The first epoch is numbered 1.
      schedule: str, 'cosine', 'glide', or 'flat'.  See sgmcmc.cyclical_rate for
        details.
      min_value: float, the minimum rate returned by this method.
    """
    super(CyclicSamplerCallback, self).__init__()

    self.ensemble = ensemble
    self.cycle_period_length = cycle_period_length
    self.sampling_start_epoch = sampling_start_epoch
    self.epoch_count = 0
    self.batch_count = 0
    self.schedule = schedule
    self.min_value = min_value

  def on_train_batch_end(self, batch, logs=None):
    self.batch_count += 1
    if (self.epoch_count+1) < self.sampling_start_epoch:
      return
    elif (self.epoch_count+1) == self.sampling_start_epoch and batch == 0:
      print('# Starting sampling phase')

    timestep_factor, is_cycle_end = sgmcmc.cyclical_rate(
        self.cycle_period_length, self.batch_count,
        schedule=self.schedule, min_value=self.min_value)
    if isinstance(self.model.optimizer, sgmcmc.SGMCMCOptimizer):
      tf.keras.backend.set_value(self.model.optimizer.timestep_factor,
                                 timestep_factor)

    if is_cycle_end:
      print('# Taking ensemble member sample')
      self.ensemble.append_maybe(self.model.get_weights)

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_count = epoch + 1


class TempScheduler(tf.keras.callbacks.Callback):
  """Optimizer temperature scheduler for SG-MCMC optimizers."""

  def __init__(self, schedule, verbose=0):
    """Optimizer temperature scheduler for SG-MCMC optimizers.

    Args:
      schedule: a function that takes an epoch index (integer, indexed from 0)
        and current temperature (float) as input and returns a new learning rate
        as output (float).
      verbose: int. 0: quiet, 1: update messages.
    """
    super(TempScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'temp'):
      raise ValueError('Optimizer must have a "temp" attribute.')
    try:  # new API
      temp = float(K.get_value(self.model.optimizer.temp))
      temp = self.schedule(epoch, temp)
    except TypeError:  # Support for old API for backward compatibility
      temp = self.schedule(epoch)
    if not isinstance(temp, (tf.Tensor, float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    if isinstance(temp, tf.Tensor) and not temp.dtype.is_floating:
      raise ValueError('The dtype of Tensor should be float')
    K.set_value(self.model.optimizer.temp, K.get_value(temp))
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler reducing temp scale '
            'to %s.' % (epoch + 1, temp))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['temp'] = K.get_value(self.model.optimizer.temp)


class AppendToEnsemble(keras.callbacks.Callback):
  """Append current model weights to an Ensemble every x iterations.

  Allows for maintaining a set of samples from a SG-MCMC trajectory optimized
  in a keras model.fit() loop. Can be used in combination with
  EvaluateEnsemblePartial to evaluate the ensemble at regular intervals. See
  EvaluateEnsemblePartial for an example.
  """

  def __init__(self, ensemble, every_x_batch=None, every_x_epoch=None):
    """Ensemble appending callback.

    Args:
      ensemble: bnn.ensemble.Ensemble object
      every_x_batch: integer, >=1 or None. append to ensemble every xth batch.
      every_x_epoch: integer, >=1 or None. append to ensemble every xth epoch.

    Note: batches are counted internally as keras counts batches
        from the start of an epoch.
    Note: Only one of every_x_* can be set.
    """

    both_none = every_x_epoch is None and every_x_batch is None
    both_not_none = every_x_epoch is not None and every_x_batch is not None
    if both_none or both_not_none:
      raise ValueError('Either every_x_batches or every_x_epochs must be set.')

    super(AppendToEnsemble, self).__init__()

    self.ensemble = ensemble

    self._every_x_batch = every_x_batch
    self._every_x_epoch = every_x_epoch
    self.batch_count = 0

  def on_train_batch_end(self, batch, logs=None):
    self.batch_count += 1
    if self._every_x_batch:
      batch_one_off = self._every_x_batch - 1
      log_this_batch = (self.batch_count % self._every_x_batch) == batch_one_off
      if log_this_batch:
        self.ensemble.append_maybe(self.model.get_weights)

  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: integer, index of epoch.
        logs: dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
    if self._every_x_epoch:
      log_this_epoch = (epoch % self._every_x_epoch) == self._every_x_epoch - 1
      if log_this_epoch:
        self.ensemble.append_maybe(self.model.get_weights)


class EvaluateEnsemblePartial(keras.callbacks.Callback):
  """Evaluate Ensemble after every epoch.

  Enables ensemble evaluation inside a keras model.fit() loop.
  """

  def __init__(self, evaluate_ensemble_partial, result_labels, every_x_epoch=1):
    """Evaluate ensemble using a ensemble.evaluate partial.

    Can be used in combination with EvaluateEnsemblePartial to evaluate the
    ensemble at regular intervals inside a keras model.fit() loop.

    Note: only ScalarStatistic's are supported.

    Args:
      evaluate_ensemble_partial: bnn.ensemble.Ensemble.evaluate partial.
      result_labels: list of names for statistic results in order.
      every_x_epoch: Integer

    Examples:
      ```python
      import itertools

      ens_partial_val = itertools.partial(
          ens.evaluate_ensemble, dataset=val_data,
          statistics=[
              MeanStatistic(Accuracy()),
              MeanStatistic(GibbsAccuracy())
          ])
      ens_partial_train = itertools.partial(
          ens.evaluate_ensemble, dataset=train_data,
          statistics=[
              MeanStatistic(Accuracy())
          ])
      model.fit(..., callbacks=[
          ...,
          AppendToEnsemble(ens, every_x_batch=1),
          EvaluateEnsemblePartial(ens_partial_val,
                                  ['ens_val_accuracy',
                                   'ens_val_gibbs_accuracy']),
          EvaluateEnsemblePartial(ens_partial_train,
                                  ['ens_train_accuracy']), ])
      ```
    """
    super(EvaluateEnsemblePartial, self).__init__()
    self.evaluate_ensemble_partial = evaluate_ensemble_partial
    self.result_labels = result_labels
    self.every_x_epoch = every_x_epoch

  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: integer, index of epoch.
        logs: dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """

    if epoch % self.every_x_epoch == self.every_x_epoch - 1:
      results = self.evaluate_ensemble_partial()

      if results is not None:
        # logs is a stateful object.
        logs.update({k: v for k, v in zip(self.result_labels, results)})
      else:
        # if ensemble is empty write a default value to logs
        logs.update({k: float('nan') for k in self.result_labels})


