# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# python3
"""Evaluation."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
# pylint: disable=no-value-for-parameter
import os
import gin
from absl import app
from absl import flags
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.evaluation.evaluate import evaluate
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time

from weak_disentangle import networks, datasets, viz
from weak_disentangle import utils as ut


def get_eval_bindings_list():
  # To run each of the metrics used in disentanglement_lib, we iterate through the gin-config settings specified in
  # https://github.com/google-research/disentanglement_lib/tree/master/disentanglement_lib/config/unsupervised_study_v1/metric_configs
  all_evaluation_bindings = """
  get_evaluation.evaluation_fn = @factor_vae_score
  factor_vae_score.num_variance_estimate=10000
  factor_vae_score.num_train=10000
  factor_vae_score.num_eval=5000
  factor_vae_score.batch_size=64
  prune_dims.threshold = 0.05

  get_evaluation.evaluation_fn = @mig
  mig.num_train=10000
  discretizer.discretizer_fn = @histogram_discretizer
  discretizer.num_bins = 20

  get_evaluation.evaluation_fn = @beta_vae_sklearn
  beta_vae_sklearn.batch_size=64
  beta_vae_sklearn.num_train=10000
  beta_vae_sklearn.num_eval=5000

  get_evaluation.evaluation_fn = @dci
  dci.num_train=10000
  dci.num_test=5000

  get_evaluation.evaluation_fn = @modularity_explicitness
  modularity_explicitness.num_train=10000
  modularity_explicitness.num_test=5000
  discretizer.discretizer_fn = @histogram_discretizer
  discretizer.num_bins = 20

  get_evaluation.evaluation_fn = @sap_score
  sap_score.num_train=10000
  sap_score.num_test=5000
  sap_score.continuous_factors=False
  """
  return [bindings.split("\n") for bindings in all_evaluation_bindings.strip().split("\n\n")]


@gin.configurable
def get_evaluation(evaluation_fn):
  return evaluation_fn


def negative_pida(factor_string, s_dim, enc_np, dset, sample_size=10000, random_state=None):
  """Estimates restrictiveness, consistency, or ranking accuracy.

  This function estimates restrictiveness via change-pairing (c=), consistency
  via share-pairing (s=), or ranking accuracy (r=) depending on the
  factor_string. Example factor string: c=0 means change-pairing on factor 0
  only, which in turn means estimating restrictiveness on factor 0. This
  method is named "pida" for Post-Interventional DisAgreement (see
  https://arxiv.org/abs/1811.00007) due to its similarity.

  Args:
    factor_string: string that determines the pairing procedure.
    s_dim: number of non-nuisance dimensions of the latent space.
    enc_np: an encoding function that takes in and returns numpy arrays.
    dset: a disentanglement_lib dataset.
    sample_size: number of samples used for monte carlo estimate.
    random_state: a numpy RandomState object. Used by dset for sampling.

  Returns:
    A dictionary of scores.
  """
  # We're going to store the sqrt-norm, the norm, the sqrt-normalizier, and the normalizer.
  def compute_loss(m1, m2, masks, sqrt=False):
    m1 = m1[:, masks.reshape(-1) == 0]
    m2 = m2[:, masks.reshape(-1) == 0]
    if sqrt:
      return np.sqrt(((m1 - m2) ** 2).sum(-1)).mean(0)
    else:
      return ((m1 - m2) ** 2).sum(-1).mean(0)

  if "s=" in factor_string or "c=" in factor_string:
    masks = datasets.make_masks(factor_string, s_dim, mask_type="match")
    x1, x2, _ = datasets.sample_match_images(dset, sample_size, masks, random_state)
    m1 = enc_np(x1)
    m2 = enc_np(x2)
    loss_unnorm = compute_loss(m1, m2, masks, sqrt=False)
    loss_unnorm_sqrt = compute_loss(m1, m2, masks, sqrt=True)

    x1 = datasets.sample_images(dset, sample_size, random_state)
    x2 = datasets.sample_images(dset, sample_size, random_state)
    m1 = enc_np(x1)
    m2 = enc_np(x2)
    loss = loss_unnorm / compute_loss(m1, m2, masks, sqrt=False)
    loss_sqrt = loss_unnorm_sqrt / compute_loss(m1, m2, masks, sqrt=True)

    scores = {
        factor_string: -loss,  # legacy code: original loss used in all previous experiments
        "sqrt_" + factor_string: -loss_sqrt,
        "unnorm_" + factor_string: -loss_unnorm,
        "unnorm_sqrt_" + factor_string: -loss_unnorm_sqrt
    }

  elif "r=" in factor_string:
    masks = datasets.make_masks(factor_string, s_dim, mask_type="rank")
    x1, x2, y = datasets.sample_rank_images(dset, sample_size, masks, random_state)
    m1 = enc_np(x1)[:, masks]
    m2 = enc_np(x2)[:, masks]
    acc = np.mean((m1 > m2) == y)
    scores = {factor_string: acc}

  return scores


def evaluate_enc(enc, dset, s_dim, original_file, original_bindings, pida_sample_size=10000, dlib_metrics=True):
  """Evaluates an encoder on multiple disentanglement metrics.

  Given an encoder and an oracle generator, this function computes encoder-
  based metrics on consistency, restrictiveness, ranking accuracy, and six
  additional disentanglement metrics used in disentanglement_lib. The
  disentanglement_lib metrics are set by modifying the global gin-config. We
  require the user provide the original gin-file and gin-bindings so that they
  can be re-established at the end of this call.

  Args:
    enc: an encoding function that takes in and returns numpy arrays.
    dset: a disentanglement_lib dataset.
    s_dim: number of non-nuisance dimensions of the latent space.
    original_file: path to original gin file
    original_bindings: list of original gin bindings
    pida_sample_size: number of samples for monte carlo estimate pida metrics.
    dlib_metrics: flag for using disentanglement_lib metrics

  Returns:
    A dictionary of scores.
  """
  # enc takes in and outputs numpy arrays
  # Consistency/Restrictiveness Metrics
  itypes = ["{}={}".format(t, i)
            for t, i in itertools.product(("s", "c", "r"), range(s_dim))]

  evals = {}
  for it in itypes:
    scores = negative_pida(it, s_dim, enc, dset,
                           sample_size=pida_sample_size,
                           random_state=np.random.RandomState(0))
    evals.update(scores)

    for k in scores:
      ut.log(k, ":", scores[k])

  evals["s_mean"] = np.mean([evals[k] for k in evals if "s=" == k[:2]])
  evals["c_mean"] = np.mean([evals[k] for k in evals if "c=" == k[:2]])

  if not dlib_metrics:
    return evals

  # Disentanglement Lib Metrics
  eval_bindings_list = get_eval_bindings_list()
  metrics = ("factor", "mig", "beta", "dci", "modularity", "sap")

  for metric, eval_bindings in zip(metrics, eval_bindings_list):
    gin.parse_config_files_and_bindings([], eval_bindings, finalize_config=False)
    evaluation_fn = get_evaluation()
    tf.logging.info("Reset eval func to {}".format(evaluation_fn.__name__))
    result = evaluation_fn(dset, enc, np.random.RandomState(0))
    ut.log(result)

    if metric == "factor":
      evals[metric] = result["eval_accuracy"]
    elif metric == "mig":
      evals[metric] = result["discrete_mig"]
    elif metric == "beta":
      evals[metric] = result["eval_accuracy"]
    elif metric == "dci":
      evals[metric] = result["disentanglement"]
    elif metric == "modularity":
      evals[metric] = result["modularity_score"]
    elif metric == "sap":
      evals[metric] = result["SAP_score"]

  # Clean up: resetting gin configs to original bindings
  gin.parse_config_files_and_bindings([original_file],
                                      original_bindings,
                                      finalize_config=False)

  return evals


# pylint: disable=unused-argument
def evaluate_enc_on_targets(enc, dset, s_dim, original_file, original_bindings, target_metrics):
  # Disentanglement Lib Metrics
  evals = {}
  eval_bindings_list = get_eval_bindings_list()
  metrics = ("factor", "mig", "beta", "dci", "modularity", "sap")

  for metric, eval_bindings in zip(metrics, eval_bindings_list):
    if metric in target_metrics:
      gin.parse_config_files_and_bindings([], eval_bindings, finalize_config=False)
      evaluation_fn = get_evaluation()
      tf.logging.info("Reset eval func to {}".format(evaluation_fn.__name__))
      result = evaluation_fn(dset, enc, np.random.RandomState(0))
      ut.log(result)

      if metric == "factor":
        evals[metric] = result["eval_accuracy"]
      elif metric == "mig":
        evals[metric] = result["discrete_mig"]
      elif metric == "beta":
        evals[metric] = result["eval_accuracy"]
      elif metric == "dci":
        evals[metric] = result["disentanglement"]
      elif metric == "modularity":
        evals[metric] = result["modularity_score"]
      elif metric == "sap":
        evals[metric] = result["SAP_score"]

  # Clean up: resetting gin configs to original bindings
  gin.parse_config_files_and_bindings([original_file],
                                      original_bindings,
                                      finalize_config=False)

  return evals
