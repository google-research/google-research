"""Experiments for paper on private sampling sketches."""

import math
import pickle
import os
import time

# Imports matplotlib and sets parameters
import matplotlib.pyplot as plt
plt.rc('font', size=13)
plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text

import numpy as np
import scipy.integrate as integrate

from collections import defaultdict
from private_sampling import private_sampling

DEFAULT_DIR_FOR_PRECOMPUTED = "precomputed_pickle_files"

class PrecomputePrivateThresholdSampling(object):
  """Precomputes and stores values to avoid recomputation each time a plot is generated."""
  SAVE_EVERY = 1000

  def __init__(self, threshold, eps, delta, sampling_method, pickle_dir, print_progress=True):
    self.sample = private_sampling.PrivateThresholdSampleWithFrequencies(threshold, eps, delta, sampling_method, store_every=1)
    file_name = "private_sampling_%s_t%s_e%s_d%s" % (sampling_method.__name__, threshold, eps, 1 / delta)
    self.file_path = os.path.join(pickle_dir, file_name)
    self.print_progress = print_progress

    if os.path.exists(self.file_path):
      self._load()

  def _load(self):
    f = open(self.file_path, "rb")
    precomputed = pickle.load(f)
    if precomputed["threshold"] != self.sample.threshold or \
      precomputed["eps"] != self.sample.eps or \
      precomputed["delta"] != self.sample.delta or \
      precomputed["sampling_method"] != self.sample.sampling_method.__name__:
      raise Exception("Tried to load precomputed values for wrong parameters")
    self.sample._reported_weight_dist = precomputed["reported_weight_dist"]
    self.sample._mle_estimators = precomputed["mle_estimators"]
    self.sample._biased_down_estimators = precomputed["biased_down_estimators"]
    f.close()

  def _save(self):
    precomputed = {}
    precomputed["threshold"] = self.sample.threshold
    precomputed["eps"] = self.sample.eps
    precomputed["delta"] = self.sample.delta
    precomputed["sampling_method"] = self.sample.sampling_method.__name__
    precomputed["timestamp"] = time.time()
    precomputed["reported_weight_dist"] = self.sample._reported_weight_dist
    precomputed["mle_estimators"] = self.sample._mle_estimators
    precomputed["biased_down_estimators"] = self.sample._biased_down_estimators
    f = open(self.file_path, "wb")
    pickle.dump(precomputed, f)
    f.close()

  def precompute(self, max_freq):
    start_time = time.time()
    for i in range(1, max_freq + 1, self.SAVE_EVERY):
      self.sample.compute_reported_frequency_dist(i)
      for j in range(i, min(i + self.SAVE_EVERY, max_freq + 1)):
        self.sample.mle_estimator(j)
        self.sample.biased_down_estimator(j)
      self._save()
      if self.print_progress:
        print("Finished %d, time: %f" % (min(i + self.SAVE_EVERY - 1, max_freq), time.time() - start_time))


def inclusion_prob_vec_for_private_sampling_keys_only(max_freq, threshold, eps, delta, sampling_method):
  """Computes the vector of inclusion probabilities for private sampling."""
  s = private_sampling.PrivateThresholdSampleKeysOnly(threshold, eps, delta, sampling_method, store_every=1)
  s.compute_inclusion_prob(max_freq)
  return s._inclusion_prob.copy()

# Functions used to compute the inclusion probability, bias, and MSE when first
# generating a private histrogram and then sampling.

# Auxiliary functions: pdf and cdf of the Laplace distribution, integrals

def laplace_pdf(x, b, mean=0):
  return 0.5 * math.exp(-1.0 * abs(x - mean) / b) / b

def laplace_cdf(x, b, mean=0):
  if x <= mean:
    return 0.5 * math.exp((x - mean) / b)
  return 1 - 0.5 * math.exp((mean - x) / b)

def indef_int_x_exp_minus_epsx(eps, x):
  """Integral of x * exp(-eps * x)dx."""
  return -1.0 * (1.0 / eps ** 2) * math.exp(-1.0 * eps * x) * (eps * x + 1)

def indef_int_x_exp_epsx(eps, x):
  """Integral of x * exp(eps * x)dx."""
  return (1.0 / eps ** 2) * math.exp(eps * x) * (eps * x - 1)

def eps_indef_int_x_exp_minus_epsx(eps, x, add_to_exp=0.0):
  """Integral of x * exp(-eps * x)dx times eps * exp(add_to_exp)."""
  return -1.0 * (1.0 / eps) * math.exp(-1.0 * eps * x + add_to_exp) * (eps * x + 1)

def eps_indef_int_x_exp_epsx(eps, x, add_to_exp=0.0):
  """Integral of x * exp(eps * x)dx times eps * exp(add_to_exp)."""
  return (1.0 / eps) * math.exp(eps * x + add_to_exp) * (eps * x - 1)

def eps_indef_int_x_sqr_exp_epsx(eps, x, add_to_exp=0.0):
  """Integral of x^2 * exp(eps * x)dx times eps * exp(add_to_exp)."""
  return (1.0 / eps ** 2) * math.exp(eps * x + add_to_exp) * ((eps * x) ** 2 - 2 * eps * x + 2)

def eps_indef_int_x_sqr_exp_minus_epsx(eps, x, add_to_exp=0.0):
  """Integral of x^2 * exp(-eps * x)dx times eps * exp(add_to_exp)."""
  return (-1.0 / eps ** 2) * math.exp(-1.0 * eps * x + add_to_exp) * ((eps * x) ** 2 + 2 * eps * x + 2)

def inclusion_prob_using_private_histogram_numerical(freq, threshold, eps, delta, sampling_method, err=10**-6):
  """Computes the inclusion probability of a key when sampling from a private histogram.
  
  This function uses numerical integration."""
  # TODO: for some sampling methods, we can solve the integral and compute exactly.
  laplace_param_b = 1.0 / eps
  histogram_inclusion_threshold = (1.0 / eps) * np.log(1.0 / delta) + 1
  # To make the integration output stable
  max_noise = max(-1.0 * laplace_param_b * np.log(2 * err), 100)
  derivative = lambda x: laplace_pdf(x, laplace_param_b) * sampling_method.inclusion_prob(freq + x, threshold)
  return integrate.quad(derivative, max(histogram_inclusion_threshold - freq, -1 * max_noise), max_noise)[0]

def inclusion_prob_using_private_histogram(freq, threshold, eps, delta, sampling_method):
  """Computes the inclusion probability of a key when sampling from a private histogram."""
  laplace_param_b = 1.0 / eps
  histogram_inclusion_threshold = (1.0 / eps) * np.log(1.0 / delta) + 1
  if sampling_method == private_sampling.AlwaysIncludeSamplingMethod:
    return 1.0 - laplace_cdf(histogram_inclusion_threshold - freq, laplace_param_b)
  elif sampling_method == private_sampling.PpsworSamplingMethod:
    if freq <= histogram_inclusion_threshold:
      return 1.0 - laplace_cdf(histogram_inclusion_threshold - freq, laplace_param_b) - (eps / (2 * (eps + threshold))) * math.exp(freq * eps - (eps + threshold) * histogram_inclusion_threshold)
    else:
      est = 1.0 - laplace_cdf(histogram_inclusion_threshold - freq, laplace_param_b)
      est -= (eps / (2 * (eps + threshold))) * math.exp(-1.0 * threshold * freq)
      if eps != threshold:
        est -= (eps / (2 * (eps - threshold))) * (math.exp(-1.0 * threshold * freq) - math.exp(histogram_inclusion_threshold * (eps - threshold) - freq * eps))
      else:
        est -= 0.5 * eps * (freq - histogram_inclusion_threshold) * math.exp(-1.0 * freq * eps)
      return est
  elif sampling_method == private_sampling.PrioritySamplingMethod:
    if histogram_inclusion_threshold >= 1.0 / threshold:
      return 1.0 - laplace_cdf(histogram_inclusion_threshold - freq, laplace_param_b)
    if freq <= histogram_inclusion_threshold:
      add_to_exp = eps * freq
      return 0.5 * threshold * (eps_indef_int_x_exp_minus_epsx(eps, 1.0 / threshold, add_to_exp) - eps_indef_int_x_exp_minus_epsx(eps, histogram_inclusion_threshold, add_to_exp)) + 1.0 - laplace_cdf(1.0 / threshold - freq, laplace_param_b)
    elif freq >= 1 / threshold:
      add_to_exp = -1.0 * eps * freq
      return 0.5 * threshold * (eps_indef_int_x_exp_epsx(eps, 1.0 / threshold, add_to_exp) - eps_indef_int_x_exp_epsx(eps, histogram_inclusion_threshold, add_to_exp)) + 1.0 - laplace_cdf(1.0 / threshold - freq, laplace_param_b)
    else:
      add_to_exp = eps * freq
      part_one = 0.5 * threshold * (eps_indef_int_x_exp_minus_epsx(eps, 1.0 / threshold, add_to_exp) - eps_indef_int_x_exp_minus_epsx(eps, freq, add_to_exp))
      add_to_exp = -1.0 * eps * freq
      part_two = 0.5 * threshold * (eps_indef_int_x_exp_epsx(eps, freq, add_to_exp) - eps_indef_int_x_exp_epsx(eps, histogram_inclusion_threshold, add_to_exp))
      return part_one + part_two + 1.0 - laplace_cdf(1.0 / threshold - freq, laplace_param_b)
  raise Exception("Unknown sampling method")

def mse_always_sample(freq, eps, delta):
  """MSE when there is no sampling (inclusion probability = 1.0)."""
  histogram_inclusion_threshold = (1.0 / eps) * np.log(1.0 / delta) + 1
  est_of_freq = expected_estimator_using_private_histogram(freq, eps, delta)
  # inc_prob = inclusion_prob_using_private_histogram(freq, 1.0, eps, delta, private_sampling.AlwaysIncludeSamplingMethod)
  if freq >= histogram_inclusion_threshold:
    int_of_sq_range1 = 0.5 * (freq ** 2) + freq / eps + 1 / (eps ** 2)
    int_of_sq_range2 = 0.5 * math.exp(-1.0 * eps * freq) * (eps_indef_int_x_sqr_exp_epsx(eps, freq) - eps_indef_int_x_sqr_exp_epsx(eps, histogram_inclusion_threshold))
    int_of_sq = int_of_sq_range1 + int_of_sq_range2
  else:
    int_of_sq = 0.5 * (1 / eps ** 2) * math.exp((freq - histogram_inclusion_threshold) * eps) * ((histogram_inclusion_threshold * eps) ** 2 + 2 * eps * histogram_inclusion_threshold + 2)
  return int_of_sq - 2 * freq * est_of_freq + (freq ** 2) # * inc_prob

def mse_priority_sampling(freq, eps, delta, tau):
  """MSE when using priority sampling on a private histogram."""
  est_of_freq = expected_estimator_using_private_histogram(freq, eps, delta)
  parts_except_int_of_sq = -2 * freq * est_of_freq + (freq ** 2)
  histogram_inclusion_threshold = (1.0 / eps) * np.log(1.0 / delta) + 1
  if freq <= histogram_inclusion_threshold:
    if 1.0 / tau <= histogram_inclusion_threshold:
      return parts_except_int_of_sq - 0.5 * eps_indef_int_x_sqr_exp_minus_epsx(eps, histogram_inclusion_threshold, eps * freq)
    return parts_except_int_of_sq + (0.5 / tau) * (eps_indef_int_x_exp_minus_epsx(eps, 1.0 / tau, eps * freq) - eps_indef_int_x_exp_minus_epsx(eps, histogram_inclusion_threshold, eps * freq)) - 0.5 * eps_indef_int_x_sqr_exp_minus_epsx(eps, 1.0 / tau, eps * freq)
  # freq > histogram_inclusion_threshold
  if 1.0 / tau <= histogram_inclusion_threshold:
    return parts_except_int_of_sq - 0.5 * eps_indef_int_x_sqr_exp_minus_epsx(eps, freq, eps * freq) + 0.5 * eps_indef_int_x_sqr_exp_epsx(eps, freq, -1.0 * freq * eps) - 0.5 * eps_indef_int_x_sqr_exp_epsx(eps, histogram_inclusion_threshold, -1.0 * freq * eps)
  elif 1.0 / tau <= freq:
    int1 = -0.5 * eps_indef_int_x_sqr_exp_minus_epsx(eps, freq, eps * freq)
    int2 = (0.5 / tau) * (eps_indef_int_x_exp_epsx(eps, 1.0 / tau, -1.0 * freq * eps) - eps_indef_int_x_exp_epsx(eps, histogram_inclusion_threshold, -1.0 * freq * eps))
    int3 = 0.5 * (eps_indef_int_x_sqr_exp_epsx(eps, freq, -1.0 * freq * eps) - eps_indef_int_x_sqr_exp_epsx(eps, 1.0 / tau, -1.0 * freq * eps))
    return parts_except_int_of_sq + int1 + int2 + int3
  int1 = -0.5 * eps_indef_int_x_sqr_exp_minus_epsx(eps, 1.0 / tau, eps * freq)
  int2 = (0.5 / tau) * (eps_indef_int_x_exp_minus_epsx(eps, 1.0 / tau, eps * freq) - eps_indef_int_x_exp_minus_epsx(eps, freq, eps * freq))
  int3 = (0.5 / tau) * (eps_indef_int_x_exp_epsx(eps, freq, -1.0 * eps * freq) - eps_indef_int_x_exp_epsx(eps, histogram_inclusion_threshold, -1.0 * eps * freq))
  return parts_except_int_of_sq + int1 + int2 + int3

def mse_using_private_histogram(freq, eps, delta, sampling_method=private_sampling.AlwaysIncludeSamplingMethod, threshold=1.0):
  """Computes the MSE when sampling from a private histogram using various sampling methods."""
  if sampling_method == private_sampling.AlwaysIncludeSamplingMethod:
    return mse_always_sample(freq, eps, delta)
  if sampling_method == private_sampling.PrioritySamplingMethod:
    return mse_priority_sampling(freq, eps, delta, threshold)
  raise NotImplementedError("Unsupported sampling method (for MSE computation)")

def inclusion_probability_priority(i, tau, eps, delta):
  """Computes the inclusion probability when using priority sampling on a private histogram using explicit/simplified expressions."""
  T = (1.0 / eps) * math.log(1.0 / delta) + 1
  if T >= 1.0 / tau:
    if i >= T:
      return 1.0 - (0.5 / delta) * math.exp(-1.0 * (i - 1) * eps)
    return 0.5 * delta * math.exp(eps * (i - 1))
  if i <= T:
    return 0.5 * tau * ((T + 1 / eps) * math.exp((i - T) * eps) - (1 / eps) * math.exp((i - 1 / tau) * eps))
  if i >= 1.0 / tau:
    return 1.0 - 0.5 * (tau / eps) * math.exp(eps * (1 / tau - i)) - 0.5 * tau * (T - 1 / eps) * math.exp(eps * (T - i))
  return tau * (i - (0.5 / eps) * math.exp(eps * (i - 1 / tau)) - 0.5 * (T - 1 / eps) * math.exp(eps * (T - i)))

def bias_and_variance_using_private_histogram_on_freq_vector(freq_vec, eps, delta, sampling_method=private_sampling.AlwaysIncludeSamplingMethod, threshold=1.0):
  """Computes the bias and variance on an entire dataset/frequency distribution when sampling from a private histogram."""
  var_sum = 0.0
  bias_sum = 0.0
  bias_and_mse_by_freq = {}
  for freq in freq_vec:
    if freq in bias_and_mse_by_freq:
      bias, mse = bias_and_mse_by_freq[freq]
    else:
      bias = bias_using_private_histogram(freq, eps, delta)
      mse = mse_using_private_histogram(freq, eps, delta, sampling_method, threshold)
      bias_and_mse_by_freq[freq] = (bias, mse)
    bias_sum += bias
    var_sum += mse - (bias ** 2)
  return bias_sum, var_sum

def bias_and_variance_using_precomputed_sample_on_freq_vector(freq_vec, sample, estimator_func):
  """Computes the bias and variance on an entire dataset/frequency distribution using a precomputed private weighted sample."""
  var_sum = 0.0
  bias_sum = 0.0
  bias_and_mse_by_freq = {}
  for freq in freq_vec:
    if freq not in bias_and_mse_by_freq:
      bias_and_mse_by_freq[freq] = sample.bias_and_mean_square_error(freq, estimator_func)
    bias, mse = bias_and_mse_by_freq[freq]
    bias_sum += bias
    var_sum += mse - (bias ** 2)
  return bias_sum, var_sum

def expected_estimator_using_private_histogram(freq, eps, delta):
  """Computes the expected estimator when sampling a key with a given (non-private) frequency from a private histogram."""
  laplace_param_b = 1.0 / eps
  histogram_inclusion_threshold = (1.0 / eps) * np.log(1.0 / delta) + 1
  if freq >= histogram_inclusion_threshold:
    # est = 0.5 * eps * (math.exp(-1.0 * freq * eps) * (indef_int_x_exp_epsx(eps, freq) - indef_int_x_exp_epsx(eps, histogram_inclusion_threshold)) - \
    #                    math.exp(freq * eps) * indef_int_x_exp_minus_epsx(eps, freq))
    # if abs(est - freq + (0.5 / delta) * math.exp((1 - freq) * eps) * (histogram_inclusion_threshold - (1.0 / eps))) > 0.1**10:
    #   raise Exception("Incorrect expected estimate", freq - (0.5 / delta) * math.exp((1 - freq) * eps) * (histogram_inclusion_threshold - (1.0 / eps)) - est)
    return freq - (0.5 / delta) * math.exp((1 - freq) * eps) * (histogram_inclusion_threshold - (1.0 / eps))
  # est = -0.5 * eps * math.exp(eps * freq) * indef_int_x_exp_minus_epsx(eps, histogram_inclusion_threshold)
  # if abs(est + 0.5 * eps * math.exp(eps * freq) * indef_int_x_exp_minus_epsx(eps, histogram_inclusion_threshold)) > 0.1**10:
  #   raise Exception("Incorrect expected estimate", -0.5 * eps * math.exp(eps * freq) * indef_int_x_exp_minus_epsx(eps, histogram_inclusion_threshold) - est)
  return 0.5 * delta * math.exp((freq - 1) * eps) * (histogram_inclusion_threshold + (1.0 / eps))

def bias_using_private_histogram(freq, eps, delta):
  """Computes the bias of the estimator of a key with a given (non-private) frequency when sampling from a private histogram."""
  return expected_estimator_using_private_histogram(freq, eps, delta) - freq

def inclusion_prob_vec_using_private_histogram(max_freq, threshold, eps, delta, sampling_method):
  """Computes the inclusion probability for each frequency when sampling from a private histogram."""
  return [inclusion_prob_using_private_histogram(i, threshold, eps, delta, sampling_method) for i in range(1, int(max_freq) + 1)]

# Functions used to produce plots

def plot_inclusion_prob_using_precompute(max_freq, sample, output_path):
  """Inclusion probability plots."""
  eps = sample.eps
  delta = sample.delta
  sampling_method = sample.sampling_method
  threshold = sample.threshold
  log_threshold = math.log10(threshold)
  if int(log_threshold) == log_threshold:
    log_threshold = int(log_threshold)
  plt.clf()
  log1_delta = math.log10(delta)
  if log1_delta == int(log1_delta):
    log1_delta = int(log1_delta)
  include_non_private = True
  if sampling_method == private_sampling.AlwaysIncludeSamplingMethod or (sampling_method == private_sampling.PrioritySamplingMethod and threshold == 1.0):
    include_non_private = False
    title = "Inclusion Probability: No Sampling, $\\varepsilon=%s, \\delta=10^{%s}$" % (eps, log1_delta)
  elif sampling_method == private_sampling.PrioritySamplingMethod:
    title = "Inclusion Probability: Priority Sampling $\\tau=10^{%s}, \\varepsilon=%s, \\delta=10^{%s}$" % (log_threshold, eps, log1_delta)
  elif sampling_method == private_sampling.PpsworSamplingMethod:
    title = "Inclusion Probability: PPSWOR $\\tau=10^{%s}, \\varepsilon=%s, \\delta=10^{%s}$" % (log_threshold, eps, log1_delta)
  else:
    raise NotImplementedError("Sampling method not supported")
  plt.xlabel("Frequency")
  plt.ylabel("Inclusion Probability")
  # plt.yscale("log", basey=10)
  # prob_vec_our = [1.0 - sample.compute_reported_frequency_dist(i)[0] for i in range(1, max_freq + 1)]
  sample = private_sampling.PrivateThresholdSampleKeysOnly(threshold, eps, delta, sampling_method)
  prob_vec_our = [sample.compute_inclusion_prob(i) for i in range(1, max_freq + 1)]
  prob_vec_histogram = inclusion_prob_vec_using_private_histogram(max_freq, threshold, eps, delta, sampling_method)
  if include_non_private:
    plt.loglog(range(1, max_freq + 1), [sampling_method.inclusion_prob(i, threshold) for i in range(1, int(max_freq) + 1)], color="tab:green", label="Non-private", marker="d", markevery=0.25)
  plt.loglog(range(1, max_freq + 1), prob_vec_our, color="tab:blue", label="PWS", marker='s', markevery=0.25)
  plt.loglog(range(1, max_freq + 1), prob_vec_histogram, color="tab:orange", label="SbH", marker='.', markevery=0.25)
  plt.title(title)
  plt.legend()
  plt.savefig(output_path)

MARKERS = ["d", "s", "v", "^", "D", "<", ">"]
COLORS = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def plot_bias_using_precompute_and_private_histogram(max_freq, samples, output_path):
  """Bias plots."""
  eps = samples[0].eps
  delta = samples[0].delta
  sampling_method = samples[0].sampling_method
  if len(samples) > len(MARKERS) or len(samples) > len(COLORS):
    raise ValueError("Tried to plot more samples than colors")
  plt.clf()
  for sample, color, marker in zip(samples, COLORS, MARKERS):
    if sample.sampling_method != sampling_method or sample.eps != eps or sample.delta != delta:
      raise ValueError("Mismatch in sample parameters")
    if sampling_method == private_sampling.PrioritySamplingMethod and sample.threshold == 1.0:
      label = "MLE, no sampling"
    else:
      log_threshold = math.log10(sample.threshold)
      if int(log_threshold) == log_threshold:
        log_threshold = int(log_threshold)
      label = "MLE, $\\tau = 10^{%s}$" % log_threshold
    bias_and_mse_mle_est = [sample.bias_and_mean_square_error(i, lambda x, y: sample.mle_estimator(x, y)) for i in range(1, max_freq + 1)]
    plt.plot(range(1, max_freq + 1), [x[0] / (i + 1) for i, x in enumerate(bias_and_mse_mle_est)], color=color, label=label, marker=marker, markevery=0.25)
  plt.xscale("log", basex=2)
  plt.xlabel("Frequency")
  plt.ylabel("Bias / Frequency")
  # bias_and_mse_biased_down_est = [sample.bias_and_mean_square_error(i, lambda x, y: sample.biased_down_estimator(x, y)) for i in range(1, max_freq + 1)]
  # plt.plot(range(1, max_freq + 1), [x[0] / (i + 1) for i, x in enumerate(bias_and_mse_biased_down_est)], color="tab:blue", label="Biased Down", marker = 'd', markevery= [0, -1])
  private_histogram_bias = [bias_using_private_histogram(i, eps, delta) for i in range(1, max_freq + 1)]
  plt.plot(range(1, max_freq + 1), [x / (i + 1) for i, x in enumerate(private_histogram_bias)], color="tab:orange", label="SbH", marker = '.', markevery=0.25)
  log1_delta = math.log2(delta)
  if log1_delta == int(log1_delta):
    log1_delta = int(log1_delta)
  plt.title("Normalized Bias: %s, $\\varepsilon=%s, \\delta=2^{%s}$" % (sampling_method.__name__.strip("SamplingMethod") + " Sampling", eps, log1_delta))
  plt.legend()
  plt.savefig(output_path)

def plot_variance_using_precompute_and_private_histogram(max_freq, sample, output_path, normalized=True, short_name=False):
  """Variance plots."""
  eps = sample.eps
  delta = sample.delta
  sampling_method = sample.sampling_method
  threshold = sample.threshold
  # bias_and_mse_biased_down_est = [sample.bias_and_mean_square_error(i, lambda x, y: sample.biased_down_estimator(x, y)) for i in range(1, max_freq + 1)]
  bias_and_mse_mle_est = [sample.bias_and_mean_square_error(i, lambda x, y: sample.mle_estimator(x, y)) for i in range(1, max_freq + 1)]
  private_histogram_expected = [expected_estimator_using_private_histogram(i, eps, delta) for i in range(1, max_freq + 1)]
  private_histogram_bias = [bias_using_private_histogram(i, eps, delta) for i in range(1, max_freq + 1)]
  non_private_var = [non_private_variance(i, sampling_method, threshold) for i in range(1, max_freq + 1)]
  include_non_private = True
  if sampling_method == private_sampling.AlwaysIncludeSamplingMethod or (sampling_method == private_sampling.PrioritySamplingMethod and threshold == 1.0):
    private_histogram_mse = [mse_always_sample(i, eps, delta) for i in range(1, max_freq + 1)]
    include_non_private = False
    method_text = "No Sampling"
  elif sampling_method == private_sampling.PrioritySamplingMethod:
    private_histogram_mse = [mse_priority_sampling(i, eps, delta, threshold) for i in range(1, max_freq + 1)]
    log_threshold = math.log10(sample.threshold)
    if int(log_threshold) == log_threshold:
      log_threshold = int(log_threshold)
    method_text = "Priority Sampling $\\tau=10^{%s}$" % log_threshold
    if short_name:
      method_text = method_text.replace(" Sampling", "")
  else:
    raise NotImplementedError("Sampling method not supported")
  plt.clf()
  plt.tight_layout()
  plt.xlabel("Frequency")
  log1_delta = math.log2(delta)
  if log1_delta == int(log1_delta):
    log1_delta = int(log1_delta)
  if normalized:
    plt.ylabel("Variance / Frequency$^2$")
    plt.yscale("log", basey=10)
    # plt.plot(range(1, max_freq + 1), [(mse - bias ** 2) / ((i + 1) ** 2) for i, (bias, mse) in enumerate(bias_and_mse_biased_down_est)], color="tab:blue", label="Biased Down", marker = 'd', markevery= [0, -1])
    if include_non_private:
      plt.plot(range(1, max_freq + 1), [x / ((i + 1) ** 2) for i, x in enumerate(non_private_var)], color="tab:green", label="Non-private", marker="d", markevery=0.25)
    plt.plot(range(1, max_freq + 1), [(mse - bias ** 2) / ((i + 1) ** 2) for i, (bias, mse) in enumerate(bias_and_mse_mle_est)], color="tab:blue", label="MLE", marker='s', markevery=0.25)
    plt.plot(range(1, max_freq + 1), [(mse - bias ** 2) / ((i + 1) ** 2) for i, (bias, mse) in enumerate(zip(private_histogram_bias, private_histogram_mse))], color="tab:orange", label="SbH", marker='.', markevery=0.25)
    plt.title("Normalized Variance: %s, $\\varepsilon=%s, \\delta=2^{%s}$" % (method_text, eps, log1_delta))
  else:
    plt.ylabel("Variance")
    # plt.plot(range(1, max_freq + 1), [mse - bias ** 2 for bias, mse in bias_and_mse_biased_down_est], color="tab:blue", label="Biased Down", marker = 'd', markevery= [0, -1])
    if include_non_private:
      plt.plot(range(1, max_freq + 1), non_private_var, color="tab:green", label="Non-private", marker="d", markevery=0.25)
    plt.plot(range(1, max_freq + 1), [mse - bias ** 2 for bias, mse in bias_and_mse_mle_est], color="tab:blue", label="MLE", marker='s', markevery=0.25)
    plt.plot(range(1, max_freq + 1), [mse - bias ** 2 for bias, mse in zip(private_histogram_bias, private_histogram_mse)], color="tab:orange", label="SbH", marker='.', markevery=0.25)
    plt.title("Variance: %s, $\\varepsilon=%s, \\delta=2^{%s}$" % (method_text, eps, log1_delta))
  plt.legend()
  plt.savefig(output_path)

def non_private_variance(freq, sampling_method, threshold):
  """The variance of non-private sampling."""
  return freq * freq * ((1.0 / sampling_method.inclusion_prob(freq, threshold)) - 1)

def plot_nrmse_on_freq_vector(freq_vec, eps, delta, thresholds, output_path, dataset_name="", sampling_method=private_sampling.PrioritySamplingMethod, precomputed_pickle_dir=DEFAULT_DIR_FOR_PRECOMPUTED):
  """Plots of the error on an entire dataset/frequency vector."""
  sum_of_freq = sum(freq_vec)
  max_freq = max(freq_vec)
  thresholds = sorted(thresholds)
  nrmse_non_private = []
  nrmse_sbh = []
  nrmse_mle = []
  nrmse_biased_down = []
  for t in thresholds:
    var_non_private = sum([non_private_variance(x, sampling_method, t) for x in freq_vec])
    nrmse_non_private.append((var_non_private ** 0.5) / sum_of_freq)
    bias_sbh, var_sbh = bias_and_variance_using_private_histogram_on_freq_vector(freq_vec, eps, delta, sampling_method, t)
    nrmse_sbh.append(((var_sbh + bias_sbh ** 2) ** 0.5) / sum_of_freq)
    pre = PrecomputePrivateThresholdSampling(t, eps, delta, sampling_method, precomputed_pickle_dir, print_progress=False)
    pre.precompute(max_freq)
    sample = pre.sample
    bias_mle, var_mle = bias_and_variance_using_precomputed_sample_on_freq_vector(freq_vec, sample, lambda x, y: sample.mle_estimator(x, y))
    nrmse_mle.append(((var_mle + bias_mle ** 2) ** 0.5) / sum_of_freq)
    # bias_biased_down, var_biased_down = bias_and_variance_using_precomputed_sample_on_freq_vector(freq_vec, sample, lambda x, y: sample.biased_down_estimator(x, y))
    # nrmse_biased_down.append(((var_biased_down + bias_biased_down ** 2) ** 0.5) / sum_of_freq)
  plt.clf()
  plt.xlabel("Sampling Threshold")
  plt.xscale("log", basex=10)
  plt.ylabel("NRMSE")
  # plt.yscale("log", basey=10)
  log1_delta = math.log2(delta)
  if log1_delta == int(log1_delta):
    log1_delta = int(log1_delta)
  plt.title("%s on %s: $\\varepsilon=%s, \\delta=2^{%s}$" % (sampling_method.__name__.strip("SamplingMethod") + " Sampling", dataset_name, eps, log1_delta))
  plt.plot(thresholds, nrmse_sbh, color="tab:orange", label="SbH", marker = '.', markevery=0.25)
  plt.plot(thresholds, nrmse_mle, color="tab:blue", label="MLE", marker = 's', markevery=0.25)
  # plt.plot(thresholds, nrmse_biased_down, color="tab:blue", label="Biased Down", marker = 'd', markevery=0.25)
  plt.plot(thresholds, nrmse_non_private, color="tab:green", label="Non-private", marker = "^", markevery=0.25)
  plt.legend()
  plt.savefig(output_path)

def compute_fraction_reported_non_private(freq_vec, sampling_method, threshold):
  """For a given vector of key frequencies, computes the expected number of keys to reported in a non-private sample."""
  expected_sample = 0.0
  for freq in freq_vec:
    expected_sample += sampling_method.inclusion_prob(freq, threshold)
  return expected_sample / len(freq_vec)

def compute_fraction_reported_pws(freq_vec, eps, delta, sampling_method=private_sampling.AlwaysIncludeSamplingMethod, threshold=1.0):
  """For a given vector of key frequencies, computes the expected number of keys to reported in a private weighted sample."""
  s = private_sampling.PrivateThresholdSampleKeysOnly(threshold, eps, delta, sampling_method)
  expected_sample = 0.0
  for freq in freq_vec:
    expected_sample += s.compute_inclusion_prob(freq)
  return expected_sample / len(freq_vec)

def compute_fraction_reported_sbh(freq_vec, eps, delta, sampling_method=private_sampling.AlwaysIncludeSamplingMethod, threshold=1.0):
  """For a given vector of key frequencies, computes the expected number of keys to reported when sampling from a stability-based histogram."""
  expected_sample = 0.0
  for freq in freq_vec:
    expected_sample += inclusion_prob_using_private_histogram(freq, threshold, eps, delta, sampling_method)
  return expected_sample / len(freq_vec)

def plot_gains_by_delta(dataset_name, freq_vec, eps, deltas, output_path, sampling_method=private_sampling.AlwaysIncludeSamplingMethod, threshold=1.0):
  """Plots the fraction of reported keys (comparing PWS and SbH) for different delta values."""
  plt.clf()
  pws_fraction_reported = [compute_fraction_reported_pws(freq_vec, eps, delta, sampling_method, threshold) for delta in deltas]
  sbh_fraction_reported = [compute_fraction_reported_sbh(freq_vec, eps, delta, sampling_method, threshold) for delta in deltas]
  plt.loglog(deltas, pws_fraction_reported, label='PWS',marker = 'd', markevery= [0, -1] )
  plt.loglog(deltas, sbh_fraction_reported, label='SbH',marker = '.', markevery= [0, -1] )
  plt.xlabel("$\\delta$", fontsize=18)
  plt.ylabel("Fraction", fontsize=18)
  plt.title("Keys reported: %s, $\\varepsilon=$%s" % (dataset_name, eps), fontsize=18)
  plt.legend(prop={"size":20})
  plt.savefig(output_path)

def plot_gains_by_tau(dataset_name, freq_vec, eps, delta, output_path, thresholds, sampling_method=private_sampling.PpsworSamplingMethod):
  """Plots the fraction of reported keys (comparing PWS, SbH, and non-private) for different sampling threshold values."""
  plt.clf()
  pws_fraction_reported = [compute_fraction_reported_pws(freq_vec, eps, delta, sampling_method, threshold) for threshold in thresholds]
  sbh_fraction_reported = [compute_fraction_reported_sbh(freq_vec, eps, delta, sampling_method, threshold) for threshold in thresholds]
  non_private = [compute_fraction_reported_non_private(freq_vec, sampling_method, threshold) for threshold in thresholds]
  plt.loglog(thresholds, non_private, label='Non-private',marker = 'o',markersize=11, color = 'red', markevery= [0, -1] )
  plt.loglog(thresholds, pws_fraction_reported, label='PWS',marker = 'd', markevery= [0, -1] )
  plt.loglog(thresholds, sbh_fraction_reported, label='SbH',marker = '.', markevery= [0, -1] )
  plt.xlabel("$\\tau$", fontsize=18)
  plt.ylabel("Fraction", fontsize=18)
  plt.title("Keys reported: %s, %s, (%s,%s)" % (dataset_name, sampling_method.__name__.strip("SamplingMethod"), eps, delta), fontsize=18)
  plt.legend(prop={"size":20})
  plt.savefig(output_path)

def plot_gain_ratio_by_delta(datasets, eps, deltas, output_path, sampling_method=private_sampling.AlwaysIncludeSamplingMethod, threshold=1.0, markers=MARKERS):
  """Plots the gain in the number of reported keys (the ratio of PWS/SbH) for different delta values."""
  plt.clf()
  for (dataset_name, freq_vec), marker in zip(datasets, markers):
    pws_fraction_reported = [compute_fraction_reported_pws(freq_vec, eps, delta, sampling_method, threshold) for delta in deltas]
    sbh_fraction_reported = [compute_fraction_reported_sbh(freq_vec, eps, delta, sampling_method, threshold) for delta in deltas]
    ratio = [x / y for x, y in zip(pws_fraction_reported, sbh_fraction_reported)]
    plt.semilogx(deltas, ratio, label=dataset_name, marker=marker, markevery= [0, -1] )
  plt.xlabel("$\\delta$", fontsize=18)
  plt.ylabel("$\\times$Gain", fontsize=18)
  plt.title("Reporting gain: PWS/SbH, $\\varepsilon=$"+str(eps)+" ", fontsize=18)
  plt.legend(prop={"size":20})
  plt.savefig(output_path)

def plot_gain_ratio_by_tau(datasets, eps, delta, output_path, thresholds, sampling_method=private_sampling.PpsworSamplingMethod, markers=MARKERS):
  """Plots the gain in the number of reported keys (the ratio of PWS/SbH) for different sampling threshold values."""
  plt.clf()
  for (dataset_name, freq_vec), marker in zip(datasets, markers):
    pws_fraction_reported = [compute_fraction_reported_pws(freq_vec, eps, delta, sampling_method, threshold) for threshold in thresholds]
    sbh_fraction_reported = [compute_fraction_reported_sbh(freq_vec, eps, delta, sampling_method, threshold) for threshold in thresholds]
    ratio = [x / y for x, y in zip(pws_fraction_reported, sbh_fraction_reported)]
    plt.semilogx(thresholds, ratio, label=dataset_name, marker=marker, markevery= [0, -1] )
  log1_delta = math.log10(delta)
  if log1_delta == int(log1_delta):
    log1_delta = int(log1_delta)
  plt.xlabel("$\\tau$", fontsize=18)
  plt.ylabel("$\\times$Gain", fontsize=18)
  plt.title("Reporting Gain: PWS/SbH, %s, $\\varepsilon=$%s, $\\delta=10^{%s}$" % (sampling_method.__name__.strip("SamplingMethod"), eps, log1_delta), fontsize=18)
  plt.legend(prop={"size":20})
  plt.savefig(output_path)

# Main functions used to generate plots for the paper

def main_precompute():
  """Precomputes and stores values."""
  EPS_LIST = [1.0, 0.5, 0.25, 0.1]
  DELTA = 0.5**20
  SAMPLING_METHODS_AND_THRESHOLDS = [
    (private_sampling.PpsworSamplingMethod, 0.1),
    (private_sampling.PpsworSamplingMethod, 0.01),
    (private_sampling.PpsworSamplingMethod, 0.001),
    (private_sampling.PpsworSamplingMethod, 0.0001),
    (private_sampling.PrioritySamplingMethod, 0.1),
    (private_sampling.PrioritySamplingMethod, 0.01),
    (private_sampling.PrioritySamplingMethod, 0.001),
    (private_sampling.PrioritySamplingMethod, 0.0001),
    (private_sampling.PrioritySamplingMethod, 10 ** -5),
    (private_sampling.PrioritySamplingMethod, 10 ** -6),
    (private_sampling.AlwaysIncludeSamplingMethod, 1.0),
  ]
  MAX_FREQ = 10000

  for sampling_method, threshold in SAMPLING_METHODS_AND_THRESHOLDS:
    for eps in EPS_LIST:
      pre = PrecomputePrivateThresholdSampling(threshold, eps, DELTA, sampling_method, DEFAULT_DIR_FOR_PRECOMPUTED)
      pre.precompute(10 * int((1 / eps) * np.log(1.0 / DELTA) + 1))


def main_plot_bias_using_precompute():
  """Generates bias plots."""
  EPS_LIST = [1.0, 0.5, 0.25, 0.1]
  DELTA = 0.5**20
  SAMPLING_METHODS_AND_THRESHOLDS = [
    # (private_sampling.PpsworSamplingMethod, 0.1),
    # (private_sampling.PpsworSamplingMethod, 0.01),
    # (private_sampling.PpsworSamplingMethod, 0.001),
    # (private_sampling.PpsworSamplingMethod, 0.0001),
    (private_sampling.PrioritySamplingMethod, 1.0),
    (private_sampling.PrioritySamplingMethod, 0.1),
    (private_sampling.PrioritySamplingMethod, 0.01),
    (private_sampling.PrioritySamplingMethod, 0.001),
    (private_sampling.PrioritySamplingMethod, 0.0001),
    (private_sampling.PrioritySamplingMethod, 10 ** -5),
    (private_sampling.PrioritySamplingMethod, 10 ** -6),
    # (private_sampling.AlwaysIncludeSamplingMethod, 1.0),
  ]

  for eps in EPS_LIST:
    samples = []
    for sampling_method, threshold in SAMPLING_METHODS_AND_THRESHOLDS:
      pre = PrecomputePrivateThresholdSampling(threshold, eps, DELTA, sampling_method, DEFAULT_DIR_FOR_PRECOMPUTED, print_progress=False)
      pre.precompute(10 * int((1 / eps) * np.log(1.0 / DELTA) + 1))
      samples.append(pre.sample)
    output_path = "norm_bias_e%s_full.pdf" % eps
    plot_bias_using_precompute_and_private_histogram(10 * int((1 / eps) * np.log(1.0 / DELTA) + 1), samples, output_path)
    output_path = "norm_bias_e%s.pdf" % eps
    plot_bias_using_precompute_and_private_histogram(10 * int((1 / eps) * np.log(1.0 / DELTA) + 1), samples[::2], output_path)

def main_plot_variance():
  """Generates variance plots."""
  EPS_LIST = [1.0, 0.5, 0.25, 0.1]
  DELTA = 0.5**20
  MAX_FREQ = 1000
  SAMPLING_METHODS_AND_THRESHOLDS = [
    (private_sampling.PrioritySamplingMethod, 0.1),
    (private_sampling.PrioritySamplingMethod, 0.01),
    (private_sampling.PrioritySamplingMethod, 0.001),
    (private_sampling.PrioritySamplingMethod, 0.0001),
    (private_sampling.PrioritySamplingMethod, 10 ** -5),
    (private_sampling.PrioritySamplingMethod, 10 ** -6),
    (private_sampling.AlwaysIncludeSamplingMethod, 1.0),
  ]
  for sampling_method, threshold in SAMPLING_METHODS_AND_THRESHOLDS:
    for eps in EPS_LIST:
      pre = PrecomputePrivateThresholdSampling(threshold, eps, DELTA, sampling_method, DEFAULT_DIR_FOR_PRECOMPUTED, print_progress=False)
      pre.precompute(4 * int((1 / eps) * np.log(1.0 / DELTA) + 1))
      output_path = ("variance_%s_t%s_e%s" % (sampling_method.__name__, threshold, eps)).replace(".", "") + ".pdf"
      plot_variance_using_precompute_and_private_histogram(2 * int((1 / eps) * np.log(1.0 / DELTA) + 1), pre.sample, "norm_" + output_path, normalized=True)
      plot_variance_using_precompute_and_private_histogram(2 * int((1 / eps) * np.log(1.0 / DELTA) + 1), pre.sample, output_path, normalized=False, short_name=True)


def main_plot_inclusion_probability():
  """Generates plots of the inclusion probability."""
  EPS_LIST = [1.0, 0.5, 0.25, 0.1]
  DELTA = 0.5**20
  SAMPLING_METHODS_AND_THRESHOLDS = [
    (private_sampling.PrioritySamplingMethod, 0.1),
    (private_sampling.PrioritySamplingMethod, 0.01),
    (private_sampling.PrioritySamplingMethod, 0.001),
    (private_sampling.PrioritySamplingMethod, 0.0001),
    (private_sampling.PrioritySamplingMethod, 10 ** -5),
    (private_sampling.PrioritySamplingMethod, 10 ** -6),
    (private_sampling.AlwaysIncludeSamplingMethod, 1.0),
    (private_sampling.PpsworSamplingMethod, 0.1),
    (private_sampling.PpsworSamplingMethod, 0.01),
    (private_sampling.PpsworSamplingMethod, 0.001),
    (private_sampling.PpsworSamplingMethod, 0.0001),
  ]
  for sampling_method, threshold in SAMPLING_METHODS_AND_THRESHOLDS:
    for eps in EPS_LIST:
      pre = PrecomputePrivateThresholdSampling(threshold, eps, DELTA, sampling_method, DEFAULT_DIR_FOR_PRECOMPUTED, print_progress=False)
      pre.precompute(2 * int((1 / eps) * np.log(1.0 / DELTA) + 1))
      output_path = "inclusion_prob_%s_t%s_e%s.pdf" % (sampling_method.__name__, threshold, eps)
      plot_inclusion_prob_using_precompute(2 * int((1 / eps) * np.log(1.0 / DELTA) + 1), pre.sample, output_path)

def int_zipf_distribution(size, a=1.0, mult=1):
  """Generates a synthetic dataset according to the Zipf distribution."""
  # integer entries and minimum equal to 1
  d = [(i+1)**(-a) for i in range(size)]
  return [int(mult * d[i]/d[size-1]) for i in range(size)]

def main_plot_nrmse_on_dist_by_threshold():
  """Plots the error of the various methods on synthethic datasets."""
  EPS_LIST = [0.5, 0.25, 0.1]
  DELTA = 0.5**20
  THRESHOLDS = [1.0, 0.1, 0.01, 0.001, 0.0001, 10 ** -5, 10 ** -6]
  THRESHOLDS_BY_EPS = defaultdict(lambda: THRESHOLDS)
  THRESHOLDS_BY_EPS[0.5] = [1.0, 0.1, 0.01, 0.001, 0.0001, 10 ** -5]
  SAMPLING_METHOD = private_sampling.PrioritySamplingMethod

  datasets = []

  UNIFORM_PARAMS = [(200, 1000)]
  for max_freq, mult in UNIFORM_PARAMS:
    datasets.append((list(range(1, max_freq + 1)) * mult, "$[1,\\ldots,%d] \\cdot %d$" % (max_freq, mult), "uniform_range%d_mult%d" % (max_freq, mult)))

  ZIPF_PARAMS = []
  for support_size, alpha, mult in ZIPF_PARAMS:
    dist = int_zipf_distribution(support_size, alpha, mult)
    dataset_name = "Zipf($10^%d$, %s, %s)" % (math.log10(support_size), alpha, mult)
    dataset_filename = "zipf_%s_%s_%s" % (support_size, alpha, mult, eps)
    datasets.append(dist, dataset_name, dataset_filename)

  for dist, dataset_name, dataset_filename in datasets:
    for eps in EPS_LIST:
      output_name = "nrmse_on_%s_e%s.pdf" % (dataset_filename, eps)
      plot_nrmse_on_freq_vector(dist, eps, DELTA, THRESHOLDS_BY_EPS[eps], output_name, dataset_name, SAMPLING_METHOD, DEFAULT_DIR_FOR_PRECOMPUTED)

def main_zipf_plots():
  """Experiments on real-world datasets."""
  REAL_WORLD_DATASETS = [
    ("ABC", list(map(int, open("abcnews_freq_vec.txt", "r")))),
    ("SO", list(map(int, open("stackoverflow_freq_vec.txt", "r")))),
  ]
  for dataset_name, freq_vec in REAL_WORLD_DATASETS:
    plot_gains_by_delta(dataset_name, freq_vec, 0.1, [0.1**i for i in range(9)], "gains_%s.pdf" % dataset_name)
    plot_gains_by_tau(dataset_name, freq_vec, 0.1, 0.001, "gains_with_ppswor_%s.pdf" % dataset_name, [0.1**power for power in range(-1, 4)])
  plot_gain_ratio_by_delta(REAL_WORLD_DATASETS, 0.1, [0.1**i for i in range(9)], "real_world_datasets_gain_ratio.pdf")
  plot_gain_ratio_by_tau(REAL_WORLD_DATASETS, 0.1, 0.001, "real_world_gain_with_sampling_01_3.pdf", [0.1**power for power in range(-1, 5)])
  plot_gain_ratio_by_tau(REAL_WORLD_DATASETS, 0.1, 0.1**5, "real_world_gain_with_sampling_01_5.pdf", [0.1**power for power in range(-1, 5)])

if __name__ == "__main__":
  main_precompute()
  main_plot_bias_using_precompute()
  main_plot_variance()
  main_plot_nrmse_on_dist_by_threshold()
  main_plot_inclusion_probability()
  main_zipf_plots()
