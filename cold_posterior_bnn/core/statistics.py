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

# Lint as: python3
"""Statistics to compute on ensemble models.

Statistics are functions of models and data.  For example, aggregating the
predictions across all member models of an ensemble is a statistics.  Another
example would be a performance metric that evaluates the predictive performance
against a known ground truth.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import math
import tensorflow as tf    # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tensorflow_probability.python.stats import calibration

tfd = tfp.distributions


class Statistic(object):
  """Statistic object used for computation across an ensemble.

  A `Statistic` is an arbitrary object that is a function of:
    1. Data, either inputs, or inputs and labels; and
    2. Model, either an ensemble member or the full ensemble.

  A statistic can take the function of a performance metric, of a predicted
  value or of an ensemble member model.  It is thus a quite general concept used
  for both model prediction, model performance evaluation and model analytics.

  The statistic is operated in the following manner: a batch of data is fixed
  and the following operations are performed:
    1. A single call to `reset`.
    2. One or multiple calls to `update`.  The statistic may use one or both
       arguments provided to `update`.
    3. One or multiple calls to `result` which returns an arbitrary result
       object to the end user.  Typically this will be a Tensor or list of the
       same size as the number of instances in the batch.  Each call successive
       call to result should return the same data provided there is no call to
       reset or update between.
  """

  def reset(self):
    """Initialize the state of the statistic before accumulation."""
    raise NotImplementedError("Users must override reset method in Statistic")

  def update(self, pred, labels=None):
    """Update the state of the statistic using predictions or labels.

    Args:
      pred: output of the model produced on the batch.  Typically a tf.Tensor or
        a tfd.Distribution.
      labels: (optional), the ground truth labels on the batch.
    """
    raise NotImplementedError("Users must override update method in Statistic")

  def result(self):
    """Return the statistic result on the batch."""
    raise NotImplementedError("Users must override result method in Statistic")


class SampleStatistic(Statistic):
  """A Statistic that provides per-example results, see `Statistic`.

  Suitable for any measure that is specific per datapoint/example/sample.

  For all inputs and outputs of this object, the sample dimension is expected to
  be the first axis.
  """
  pass


class MeanStatistic(Statistic):
  """A Statistic that provides sample-averaged results, see `Statistic`.

  MeanStatistic be used to provide the sample-mean of a SampleStatistic by
  providing a SampleStatistic at initialization. This is useful for reporting
  metrics on ensemble models. MeanStatistic can alternatively be subclassed to
  implement a sample-mean statistic directly.
  """

  def __init__(self, sample_statistic):
    """Initializes a sample-mean wrapper around a SampleStatistic.

    Args:
      sample_statistic: A Statistic to compute the mean of.
    """
    if not isinstance(sample_statistic, SampleStatistic):
      raise ValueError("Expecting a SampleStatistic instance.")
    self.sample_statistic = sample_statistic

  def reset(self):
    """Initialize the state of the statistic before accumulation."""
    self.sample_statistic.reset()

  def update(self, pred, labels=None):
    """Update the state of the statistic using predictions or labels.

    Args:
      pred: output of the model produced on the batch.  Typically a tf.Tensor or
        a tfd.Distribution. First axis must be the sample dimension.
      labels: (optional), the ground truth labels on the batch. First axis must
        be the sample dimension.
    """
    self.sample_statistic.update(pred, labels=labels)

  def result(self):
    """Returns sample-mean of the wrapped SampleStatistic."""
    res = self.sample_statistic.result()
    return tf.reduce_mean(res, axis=0)


class MaximumStatistic(Statistic):
  """A Statistic that provides the maximum over per-sample results.

  MaximumStatistic be used to provide the maximum value of the per-sample values
  produced by a SampleStatistic.
  """

  def __init__(self, sample_statistic):
    """Initializes a sample-maximum wrapper around a SampleStatistic.

    Args:
      sample_statistic: A Statistic to compute the mean of.
    """
    if not isinstance(sample_statistic, SampleStatistic):
      raise ValueError("Expecting a SampleStatistic instance.")
    self.sample_statistic = sample_statistic

  def reset(self):
    """Initialize the state of the statistic before accumulation."""
    self.sample_statistic.reset()

  def update(self, pred, labels=None):
    """Update the state of the statistic using predictions or labels.

    Args:
      pred: output of the model produced on the batch.  Typically a tf.Tensor or
        a tfd.Distribution. First axis must be the sample dimension.
      labels: (optional), the ground truth labels on the batch. First axis must
        be the sample dimension.
    """
    self.sample_statistic.update(pred, labels=labels)

  def result(self):
    """Returns sample-maximum of the wrapped SampleStatistic."""
    res = self.sample_statistic.result()
    return tf.reduce_max(res, axis=0)


class StandardDeviation(Statistic):
  """Compute the standard deviation of a SampleStatistic result.

  Given a vector of per-instance statistics (output of a `SampleStatistic`) this
  statistic computes the sample standard deviation of the vector.

    stddev = sqrt(mean((x-mean(x))^2) / (n-1))
  """

  def __init__(self, sample_statistic):
    """Initializes a standard deviation wrapper around a SampleStatistic.

    Args:
      sample_statistic: A Statistic to compute the mean of.
    """
    if not isinstance(sample_statistic, SampleStatistic):
      raise ValueError("Expecting a SampleStatistic instance.")
    self.sample_statistic = sample_statistic

  def reset(self):
    """Initialize the state of the statistic before accumulation."""
    self.sample_statistic.reset()

  def update(self, pred, labels=None):
    """Update the state of the statistic using predictions or labels.

    Args:
      pred: output of the model produced on the batch.  Typically a tf.Tensor or
        a tfd.Distribution. First axis must be the sample dimension.
      labels: (optional), the ground truth labels on the batch. First axis must
        be the sample dimension.
    """
    self.sample_statistic.update(pred, labels=labels)

  def result(self):
    """Returns standard deviation of the wrapped SampleStatistic."""
    res = self.sample_statistic.result()
    stddev = tf.math.reduce_std(res)

    # Correct sample estimate
    n = tf.size(res, out_type=tf.float32)
    stddev = tf.sqrt(tf.square(stddev)*(n/(n-1)))
    return stddev


class StandardError(Statistic):
  """Compute the standard error of a SampleStatistic result.

  Given a vector of per-instance statistics (output of a `SampleStatistic`) this
  statistic computes the standard error of the mean (SEM) of the vector.
  The SEM is defined as

    SEM = stddev(X) / sqrt(len(X)).
  """

  def __init__(self, sample_statistic):
    """Initializes a standard error wrapper around a SampleStatistic.

    Args:
      sample_statistic: A Statistic to compute the mean of.
    """
    if not isinstance(sample_statistic, SampleStatistic):
      raise ValueError("Expecting a SampleStatistic instance.")
    self.sample_statistic = sample_statistic

  def reset(self):
    """Initialize the state of the statistic before accumulation."""
    self.sample_statistic.reset()

  def update(self, pred, labels=None):
    """Update the state of the statistic using predictions or labels.

    Args:
      pred: output of the model produced on the batch.  Typically a tf.Tensor or
        a tfd.Distribution. First axis must be the sample dimension.
      labels: (optional), the ground truth labels on the batch. First axis must
        be the sample dimension.
    """
    self.sample_statistic.update(pred, labels=labels)

  def result(self):
    """Returns standard error of the mean of the wrapped SampleStatistic."""
    res = self.sample_statistic.result()
    n = tf.size(res, out_type=tf.float32)

    # Correct sample estimate
    stddev = tf.math.reduce_std(res)
    sem = tf.sqrt(tf.square(stddev)/(n-1))
    return sem


class ClassificationLogProb(SampleStatistic):
  """Compute the posterior predictive log-probability over the ensemble."""

  def reset(self):
    self.logp = None
    self.count = 0

  def update(self, logits, labels=None):  # pylint: disable=unused-argument
    """Update the class probabilities using the given batch logits.

    We update the probabilities using a running mean in log-domain, where P_M is
    the ensemble prediction for ensemble members 1,..,M, and p_M is the
    individual ensemble member prediction.

    The update reads

      log p_{M+1}(y) = log( (M/(M+1)) P_{M}(y_i) + (1/(M+1)) p_{M+1}(y_i) )
        = log( exp( log p_{M}(y) + log(M/(M+1)) )
               + exp( log p_{M+1}(y) - log(M+1) ) ).

    Args:
      logits: Tensor, shape (ninstances, nlabels), classification logits
        predicted by the model on the batch.
      labels: (optional), not used.
    """
    self.count += 1
    logits = tf.math.log_softmax(logits)  # log p(y^* | x)

    if self.count <= 1:
      self.logp = logits
    else:
      running_average_factor = float(self.count - 1) / float(self.count)
      self.logp = tf.reduce_logsumexp(
          tf.stack([
              self.logp + math.log(running_average_factor),
              logits - math.log(float(self.count))
          ],
                   axis=0),
          axis=0)

  def result(self):
    """Return the posterior predictive log-probabilities on the batch.

    Returns:
      logpp: Tensor, shape (ninstances, nclasses), the log-probabilities for
        `ninstances` instances and `nclasses` class labels.
    """
    return self.logp


class ClassificationCrossEntropy(SampleStatistic):
  """Compute the posterior predictive cross entropy over the ensemble.

  The cross entropy over the ensemble is defined as

    CE_i = - log p(y_i) = - log sum_{i=1}^M exp(log p(y_i|x_i,theta_m)) + log M.
  """

  def reset(self):
    self.logp = None
    self.count = 0

  def update(self, logits, labels=None):
    """Update the cross entropy using the given batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses).
    """
    if labels is None:
      raise ValueError("ClassificationCrossEntropy requires labels.")

    self.count += 1
    logits = tf.math.log_softmax(logits)  # log p(y^* | x)
    logpy = tf.gather_nd(
        logits,
        tf.stack([tf.range(logits.shape[0]),
                  tf.cast(labels, tf.int32)], axis=1))
    if self.count <= 1:
      self.logp = logpy
    else:
      running_average_factor = float(self.count - 1) / float(self.count)
      self.logp = tf.reduce_logsumexp(
          tf.stack([
              self.logp + math.log(running_average_factor),
              logpy - math.log(float(self.count))
          ],
                   axis=0),
          axis=0)

  def result(self):
    """Return the posterior predictive cross entropy on the batch.

    Returns:
      ce: Tensor, shape (ninstances,), the cross-entropy for `ninstances`
        instances.

    Raises:
      RuntimeError: result method called before any update call.
    """
    if self.logp is None:
      raise RuntimeError("result method called before any update call.")

    return -self.logp  # pylint: disable=invalid-unary-operand-type


class ClassificationGibbsCrossEntropy(SampleStatistic):
  """Compute the average cross entropy of ensemble members (Gibbs CE).

  The Gibbs cross entropy over the ensemble is defined as

    GCE_i = - (1/M) sum_{m=1,..,M} log p(y_i|x_i,theta_m),

  where there are M models in the ensemble, represented by parameters
  theta_m, m=1,..,M.

  The Gibbs cross entropy approximates the average cross entropy of a single
  model drawn from the (Gibbs) ensemble.
  """

  def reset(self):
    self.ce = None
    self.count = 0

  def update(self, logits, labels=None):
    """Update the Gibbs cross entropy using a batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses).
    """
    if labels is None:
      raise ValueError("ClassificationGibbsCrossEntropy requires labels.")

    self.count += 1
    ce_cur = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    if self.count <= 1:
      self.ce = ce_cur
    else:
      running_average_factor = float(self.count - 1) / float(self.count)
      self.ce = running_average_factor * self.ce + ce_cur / float(self.count)

  def result(self):
    """Return the Gibbs cross entropy on the batch.

    Returns:
      ce: Tensor, shape (ninstances,), the Gibbs cross entropy for `ninstances`
        instances.
    Raises:
      RuntimeError: Result method called before any update call.
    """
    if self.ce is None:
      raise RuntimeError("result method called before any update call")

    return self.ce


class Accuracy(SampleStatistic):
  """Compute the classification accuracy of the posterior predictive.

  The accuracy over the ensemble is defined as

    ACC_i = 1_{y_i == argmax_k p(k|x_i)},

  where p(k|x) is the ensemble average class probability prediction.
  """

  def __init__(self):
    self.cprob = ClassificationLogProb()

  def reset(self):
    self.cprob.reset()
    self.correct = None

  def update(self, logits, labels=None):
    """Update the class accuracy using the given batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses).
    """
    if labels is None:
      raise ValueError("Accuracy requires ground truth labels.")

    self.cprob.update(logits, labels)

    correct = tf.equal(
        tf.cast(tf.argmax(self.cprob.logp, axis=1), labels.dtype), labels)
    correct = tf.where(correct, tf.ones_like(labels), tf.zeros_like(labels))
    correct = tf.cast(correct, tf.float32)
    self.correct = correct

  def result(self):
    """Return the accuracy vector over the batch.

    Returns:
      accuracy: Tensor, shape (ninstances,), tf.float32, with elements either
        zero or one, where one denotes a correct prediction for the
        corresponding instance.
    """
    return self.correct


class GibbsAccuracy(SampleStatistic):
  """Compute the average accuracy of ensemble members (Gibbs accuracy).

  The average accuracy over the ensemble (Gibbs accuracy) is defined as

    GACC_i = (1/M) sum_{m=1,..,M} 1_{y_i == argmax_k p(k|x_i,theta_m)},

  where p(k|x,theta_m) is the class posterior of the m'th ensemble member.
  The name "Gibbs accuracy" stems from the Gibbs ensemble, i.e. this statistic
  approximates the average accuracy that a single model drawn from the (Gibbs)
  ensemble achieves.
  """

  def reset(self):
    self.accuracy = None
    self.count = 0

  def update(self, logits, labels=None):
    """Update the Gibbs accuracy using the given batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses).
    """
    if labels is None:
      raise ValueError("GibbsAccuracy requires ground truth labels.")

    correct = tf.equal(tf.cast(tf.argmax(logits, axis=1), labels.dtype), labels)
    correct = tf.where(correct, tf.ones_like(labels), tf.zeros_like(labels))
    correct = tf.cast(correct, tf.float32)

    self.count += 1
    if self.count <= 1:
      self.accuracy = correct
    else:
      running_average_factor = float(self.count - 1) / float(self.count)
      self.accuracy = running_average_factor*self.accuracy + \
          correct/float(self.count)

  def result(self):
    """Return the average accuracy vector over the batch (Gibbs accuracy).

    Returns:
      accuracy: Tensor, shape (ninstances,), tf.float32, with elements between
        zero and one.
    """
    return self.accuracy


class BrierScore(SampleStatistic):
  """Compute the classification Brier score of the posterior predictive.

  The [Brier score][1] is a loss function for probabilistic predictions over a
  number of discrete outcomes.  For a probability vector `p` and a realized
  outcome `k` the Brier score is `sum_i p[i]*p[i] - 2*p[k]`.  Smaller values are
  better in terms of prediction quality.  The Brier score can be negative.

  Compared to the cross entropy (aka logarithmic scoring rule) the Brier score
  does not strongly penalize events which are deemed unlikely but do occur,
  see [2].  The Brier score is a strictly proper scoring rule and therefore
  yields consistent probabilistic predictions.

  #### References
  [1]: G.W. Brier.
       Verification of forecasts expressed in terms of probability.
       Monthley Weather Review, 1950.
  [2]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  """

  def __init__(self):
    self.cprob = ClassificationLogProb()

  def reset(self):
    self.cprob.reset()
    self.brier = None

  def update(self, logits, labels=None):
    """Update the Brier score using the given batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses-1].
    """
    if labels is None:
      raise ValueError("BrierScore requires ground truth labels.")

    self.cprob.update(logits, labels)

    prob = tf.math.softmax(self.cprob.result(), axis=1)
    _, nlabels = prob.shape
    plabel = tf.reduce_sum(tf.one_hot(labels, nlabels) * prob, axis=1)

    self.brier = tf.reduce_sum(tf.square(prob), axis=1) - 2.0*plabel

  def result(self):
    """Return the Brier score vector over the batch.

    Returns:
      brier: Tensor, shape (ninstances,), tf.float32, with each element being
        the Brier score for an instance.
    """
    return self.brier


class BrierDecompositionStatistic(MeanStatistic):
  r"""Brier decomposition into uncertainty, resolution, and reliability.

  Note: you would not use this class directly but instead use one of the derived
  classes.

  This statistic only applies to classification.

  This method implements Broecker's decomposition [1] for the Brier score.
  The Brier score is decomposed as

    Brier = Uncertainty - Resolution + Reliability.

  #### References
  [1]: Jochen Broecker,
       Reliability, sufficiency, and the decomposition of proper scores.
       Quarterly Journal of the Royal Meteorological Society, Vol. 135, 2009.
       https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.456
  """

  def __init__(self, statistic_name):
    """Create a Brier decomposition statistic.

    Args:
      statistic_name: one of "uncertainty", "resolution", or "reliability".
    """
    super(BrierDecompositionStatistic, self).__init__(SampleStatistic())
    if statistic_name not in ["uncertainty", "resolution", "reliability"]:
      raise ValueError("BrierDecompositionStatistic's statistic_name "
                       "argument must be one of 'uncertainty', 'resolution', "
                       "or 'reliability'.")

    self.statistic_name = statistic_name
    self.cprob = ClassificationLogProb()

  def reset(self):
    self.cprob.reset()
    self.labels = None

  def update(self, logits, labels=None):
    """Update the Brier decomposition using the given batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses).
    """
    if labels is None:
      raise ValueError("BrierDecompositionStatistic requires ground truth "
                       "labels.")

    self.cprob.update(logits, labels)
    self.labels = labels

  def result(self):
    """Return a scalar Brier decomposition statistic.

    Returns:
      stat: Tensor, scalar, tf.float32, containing the Brier decomposition
        statistic selected using the statistic_name argument.
    """
    uncert, resol, reliab = calibration.brier_decomposition(
        labels=self.labels, logits=self.cprob.result())

    # TODO(nowozin): find a way to only compute the brier decomposition once and
    # return all three quantities.
    if self.statistic_name == "uncertainty":
      return uncert
    elif self.statistic_name == "resolution":
      return resol
    elif self.statistic_name == "reliability":
      return reliab


class BrierUncertainty(BrierDecompositionStatistic):
  """Brier decomposition: uncertainty value.

  Brier = Uncertainty - Resolution + Reliability.

  The uncertainty is a generalized entropy of the average predictive
  distribution; it can both be positive or negative.
  """

  def __init__(self):
    super(BrierUncertainty, self).__init__("uncertainty")


class BrierResolution(BrierDecompositionStatistic):
  """Brier decomposition: resolution value.

  Brier = Uncertainty - Resolution + Reliability.

  Resolution is a generalized variance of individual predictive distributions;
  it is always non-negative.  Difference in predictions reveal information, that
  is why a larger resolution improves the predictive score and the resolution
  enters the decomposition with a negative sign.
  """

  def __init__(self):
    super(BrierResolution, self).__init__("resolution")


class BrierReliability(BrierDecompositionStatistic):
  """Brier decomposition: reliability value.

  Brier = Uncertainty - Resolution + Reliability.

  Reliability is a measure of calibration of predictions against the true
  frequency of events.  It is always non-negative and a lower value here
  indicates better calibration.
  """

  def __init__(self):
    super(BrierReliability, self).__init__("reliability")


class ECE(MeanStatistic):
  r"""Compute the expected calibration error of the posterior predictive.

  This statistic only applies to classification.

  This method implements equation (3) in [1].  In this equation the probability
  of the decided label being correct is used to estimate the calibration
  property of the predictor.

  Note: a trade-off exist between using a small number of `num_bins` and the
  estimation reliability of the ECE.  In particular, this method may produce
  unreliable ECE estimates in case there are few samples available in some bins.

  #### References
  [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
       On Calibration of Modern Neural Networks.
       Proceedings of the 34th International Conference on Machine Learning
       (ICML 2017).
       arXiv:1706.04599
       https://arxiv.org/pdf/1706.04599.pdf
  """

  def __init__(self, num_bins):
    """Create an ECE statistic.

    Args:
      num_bins: int, >= 2, the number of probability bins to use.
        Typical choices are small integers, e.g. 5 or 10.  Higher choices can
        be used if there are few classes and many test examples.
    """
    super(ECE, self).__init__(SampleStatistic())
    self.num_bins = num_bins
    self.cprob = ClassificationLogProb()

  def reset(self):
    self.cprob.reset()
    self.ece = None
    self.labels = None

  def update(self, logits, labels=None):
    """Update the ECE score using the given batch logits and labels.

    Args:
      logits: Tensor, shape (ninstances,nclasses), as used in
        tf.nn.sparse_softmax_cross_entropy_with_logits.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0,nclasses).
    """
    if labels is None:
      raise ValueError("ECE requires ground truth labels.")

    self.cprob.update(logits, labels)
    self.labels = labels

  def result(self):
    """Return the scalar ECE score.

    Returns:
      ece: Tensor, scalar, tf.float32, containing the expected calibration
        error score.
    """
    ece = calibration.expected_calibration_error(
        self.num_bins, labels_true=self.labels, logits=self.cprob.result())

    return ece


class RegressionOutputs(SampleStatistic):
  r"""Compute the ensemble regression outputs.

  Compute the regression outputs for posterior predictive of a mixture based on
  regression outputs of the mixture components. We input the component
  regression outputs through the `update` function. The component regression
  outputs can be either:
  - tf.Tensor with only means (homoscedastic regression) in format [means].
  - tf.Tensor with both means and logarithms of standard deviations
      (heteroscedastic regression) in format [means, log_stddevs] (concatenated
      together).
  - tfp.distributions.Normal, with means() and stddevs() functions (both for
      homoscedastic and heteroscedastic regression).

  NOTE: While for the tf.Tensor inputs (`update` function) we accept logarithms
  of standard deviations for the mixture components (this is the format usually
  returned by the neural network models that is later combined with an
  exponential activation function to ensure positive stddevs), we return
  (`result` function) the variance of the mixture posterior predictive.

  The mean and variance of the mixture posterior predictive are calculated as
  below:

  mu_pred      = M^-1 \Sum_m mu_m
  sigma_pred^2 = M^-1 \Sum_m (sigma_m^2 + mu_m^2) - mu_pred^2.

  We calculate the mixture regression outputs (means and variances) for multiple
  targets separately.

  TODO(swiat): this statistic currently assumes full dataset evaluation and
  does not support evaluation with mini-batches.
  """

  def __init__(self, outputs_with_log_stddevs=False, stddev=1.0):
    """Constructor for RegressionOutputs.

    Args:
      outputs_with_log_stddevs: boolean, applies to the inputs of the `update`
          function in the tf.Tensor format. Decides whether only means
          (homoscedastic regression) or both means and log standard deviations
          (heteroscedastic regression) are contained in each row of the
          tf.Tensors passed to the `update` function (in
          format [means, log_stddevs] concatenated). All regression outputs
          in the tf.Tensor format passed to the `update` function must follow
          this selected strategy. When the `outputs_with_log_stddevs` is False
          then we interpret all the columns as means and use the homoscedastic
          `stddev` specified in the constructor. When the
          `outputs_with_log_stddevs` is True then we interprep the second half
          of columns as the log stddevs matching the means in the first half of
          the columns and the `stddev` parameter is ignored.
      stddev: float, stddev used for homoscedastic regression outputs.
    """
    self.outputs_with_log_stddevs = outputs_with_log_stddevs
    self.stddev = stddev
    self.reset()

  def reset(self):
    self.m1_sum = None
    self.m2_sum = None
    self.count = 0

  @staticmethod
  def _convert_regression_outputs_to_means_and_log_stddevs(regression_outputs):
    """Converts regression model outputs to a means and log stddevs.

    Args:
      regression_outputs: Either:
          - tensor, shape (ninstances, 1x or 2x ntargets), regression model
              outputs for a single ensemble member/mixture component. Depending
              on the value of outputs_with_log_stddevs the columns are
              interpreted either as regression output means for each target or
              both the means and log stddevs for each target. In the latter
              case, the first half of the colums is interpreted as means for
              each of the targets and the second half of the columns is
              iterpreted as log stddevs matching the means in the first half.
          - tfp.distributions.Normal, shape (ninstances, ntargets), regression
              model outputs in the form of an independent 1d tfd.Normal
              distributions for each target.
    Returns:
      moments: tensor, shape (ninstances, nlabels), predicted regression
          moments.
      model_with_dist_outputs: boolean, whether the model outputs where of type
          tfd.Normal.
    Raises:
      Exception: unknown type of regression_outputs.
    """
    model_with_dist_outputs = False
    if isinstance(regression_outputs, tfd.Normal):
      model_with_dist_outputs = True
      means = regression_outputs.mean()
      # TODO(swiat): We can currently only obtain stddevs not log stddevs.
      # This can be numerically unstable due to low floating point precision.
      stddevs = regression_outputs.stddev()
      log_stddevs = tf.math.log(stddevs)
      converted_regression_outputs = tf.concat([means, log_stddevs],
                                               len(means.shape) - 1)
    elif isinstance(regression_outputs, tf.Tensor):
      converted_regression_outputs = regression_outputs
    else:
      raise Exception("Unknown type: {}".format(type(regression_outputs)))
    return converted_regression_outputs, model_with_dist_outputs

  def update(self, regression_outputs, labels=None):  # pylint: disable=unused-argument
    """Update the ensemble means and variances with new member outputs.

    Args:
      regression_outputs: Either:
         - tensor, shape (ninstances, 1x or 2x ntargets), regression model
              outputs for a single ensemble member/mixture component. Depending
              on the value of outputs_with_log_stddevs the columns are
              interpreted either as regression output means for each target or
              both the means and log stddevs for each target. In the latter
              case, the first half of the colums is interpreted as means for
              each of the targets and the second half of the columns is
              iterpreted as log stddevs matching the means in the first half.
          - tfp.distributions.Normal, shape (ninstances, ntargets), regression
              model outputs in the form of an independent 1d tfd.Normal
              distributions for each target.
      labels: (optional), not used.
    """
    self.count += 1
    regression_outputs, model_with_dist_outputs = self._convert_regression_outputs_to_means_and_log_stddevs(
        regression_outputs)
    if self.outputs_with_log_stddevs or model_with_dist_outputs:
      n_targets = regression_outputs.shape[1] // 2
      means = regression_outputs[:, :n_targets]
      stddevs = tf.math.exp(regression_outputs[:, n_targets:])
    else:
      means = regression_outputs
      stddevs = self.stddev

    if self.count <= 1:
      self.m1_sum = means
      self.m2_sum = means**2 + stddevs**2
    else:
      self.m1_sum += means
      self.m2_sum += means**2 + stddevs**2

  def result(self):
    """Return the ensemble means and variances.

    Returns:
      means: tensor, shape (ninstances, ntargets), ensemble posterior predictive
          mean for each of the targets of each data instance/example.
      variances: tensor, shape (ninstances, ntargets), ensemble posterior
          predictive variance for each of the targets of each data
          instance/example.
    """
    means = self.m1_sum / self.count
    variances = ((self.m2_sum / self.count) - means**2)
    return means, variances


class RegressionNormalLogProb(SampleStatistic):
  """Approximates posterior predictive with a Normal and computes the log prob.

  Following [0] we approximate the ensamble posterior predictive with Normal
  which has its mean and variance equal to the mean and variance of the ensemble
  mixture.

  We assume independence of multiple outputs of regression and compute their
  negative log probability separately.

  [0] Lakshminarayanan, B., Pritzel, A. and Blundell, C. Simple and scalable
  predictive uncertainty estimation using deep ensembles. NeurIPS 2017.
  """

  def __init__(self, outputs_with_log_stddevs=False, stddev=1.0):
    """RegressionNormalLogProb constructor.

    Args:
      outputs_with_log_stddevs: boolean, applies to the inputs of the `update`
          function in the tf.Tensor format. Decides whether only means
          (homoscedastic regression) or both means and log standard deviations
          (heteroscedastic regression) are contained in each row of the
          tf.Tensors passed to the `update` function (in
          format [means, log_stddevs] concatenated). All regression outputs
          in the tf.Tensor format passed to the `update` function must follow
          this selected strategy. When the `outputs_with_log_stddevs` is False
          then we interpret all the columns as means and use the homoscedastic
          `stddev` specified in the constructor. When the
          `outputs_with_log_stddevs` is True then we interprep the second half
          of columns as the log stddevs matching the means in the first half of
          the columns and the `stddev` parameter is ignored.
      stddev: float, stddev used for homoscedastic regression outputs.
    """
    self.ensemble_regression_outputs = RegressionOutputs(
        outputs_with_log_stddevs, stddev)
    self.reset()

  def reset(self):
    self.logp = None
    self.count = 0
    self.ensemble_regression_outputs.reset()

  def update(self, regression_outputs, labels):
    """Update the ensemble log prob with new member outputs and labels.

    Args:
      regression_outputs: Either:
         - tensor, shape (ninstances, 1x or 2x ntargets), regression model
              outputs for a single ensemble member/mixture component. Depending
              on the value of outputs_with_log_stddevs the columns are
              interpreted either as regression output means for each target or
              both the means and log stddevs for each target. In the latter
              case, the first half of the colums is interpreted as means for
              each of the targets and the second half of the columns is
              iterpreted as log stddevs matching the means in the first half.
          - tfp.distributions.Normal, shape (ninstances, ntargets), regression
              model outputs in the form of an independent 1d tfd.Normal
              distributions for each target.
      labels: Tensor, shape (ninstances,), tf.int32 or tf.int64, with elements
        in the range [0, ntargets - 1].
    """
    self.count += 1
    self.ensemble_regression_outputs.update(regression_outputs)
    means, variances = self.ensemble_regression_outputs.result()
    normals = tfd.Normal(means, variances**0.5)
    self.logp = normals.log_prob(labels)

    # TODO(swiat): instead of performing the above approximation, we could store
    # all the mixture components, which could be the represented as
    # tfd.Distributions for each ensemble member. Then we could evaluate the
    # likelihood on the mixture exactly. The disadvantage of this approach would
    # be the M times slower log prob evaluation and M times more memory.
    # However, this should be negligible. We leave that as a next step.

  def result(self):
    """Return the ensemble posterior predictive negative log probability.

    Returns:
      neg_logp: Tensor, shape (ninstances, ntargets), the ensemble posterior
          predictive negative log probability for `ninstances` instances with
          `noutputs`.

    Raises:
      RuntimeError: result method called before any update call.
    """
    if self.logp is None:
      raise RuntimeError("result method called before any update call.")

    neg_logp = -self.logp  # pylint: disable=invalid-unary-operand-type
    return neg_logp

