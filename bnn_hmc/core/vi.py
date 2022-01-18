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
"""Implementation of Variational Inference."""

import jax
import copy
from jax import numpy as jnp

from bnn_hmc.utils import tree_utils


def inv_softplus(x):
  return jnp.log(jnp.exp(x) - 1)


def get_mfvi_model_fn(net_fn, params, net_state, seed=0, sigma_init=0.):
  """Convert model, parameters and net state to use MFVI.

  Convert the model to fit a Gaussian distribution to each of the weights
  following the Mean Field Variational Inference (MFVI) procedure.

  Args:
    net_fn: neural network function.
    params: parameters of the network; we intialize the mean in MFVI with
      params.
    net_state: state of the network.
    seed: random seed; used for generating random samples when computing MFVI
      predictions (default: 0).
    sigma_init: initial value of the standard deviation of the per-prarameter
      Gaussians.
  """
  #  net_fn(params, net_state, None, batch, is_training)
  mean_params = jax.tree_map(lambda p: p.copy(), params)
  sigma_isp = inv_softplus(sigma_init)
  std_params = jax.tree_map(lambda p: jnp.ones_like(p) * sigma_isp, params)
  mfvi_params = {"mean": mean_params, "inv_softplus_std": std_params}
  mfvi_state = {
      "net_state": copy.deepcopy(net_state),
      "mfvi_key": jax.random.PRNGKey(seed)
  }

  def sample_parms_fn(params, state):
    mean = params["mean"]
    std = jax.tree_map(jax.nn.softplus, params["inv_softplus_std"])
    noise, new_key = tree_utils.normal_like_tree(mean, state["mfvi_key"])
    params_sampled = jax.tree_multimap(lambda m, s, n: m + n * s, mean, std,
                                       noise)
    new_mfvi_state = {
        "net_state": copy.deepcopy(state["net_state"]),
        "mfvi_key": new_key
    }
    return params_sampled, new_mfvi_state

  def mfvi_apply_fn(params, state, _, batch, is_training):
    params_sampled, new_mfvi_state = sample_parms_fn(params, state)
    predictions, new_net_state = net_fn(params_sampled, state["net_state"],
                                        None, batch, is_training)
    new_mfvi_state = {
        "net_state": copy.deepcopy(new_net_state),
        "mfvi_key": new_mfvi_state["mfvi_key"]
    }
    return predictions, new_mfvi_state

  def mfvi_apply_mean_fn(params, state, _, batch, is_training):
    """Predict with the variational mean."""
    mean = params["mean"]
    predictions, new_net_state = net_fn(mean, state["net_state"], None, batch,
                                        is_training)
    new_mfvi_state = {
        "net_state": copy.deepcopy(new_net_state),
        "mfvi_key": state["mfvi_key"]
    }
    return predictions, new_mfvi_state

  return (mfvi_apply_fn, mfvi_apply_mean_fn, sample_parms_fn, mfvi_params,
          mfvi_state)


def make_kl_with_gaussian_prior(weight_decay, temperature=1.):
  """Implements the prior KL term in the ELBO.

  Args:
    weight_decay: weight decay corresponding to the prior distribution.
    temperature: temperature of the posterior, corresponds to the weight of the
      KL term in the ELBO.  Returns a function that takes the MFVI parameters
      and returns the KL divergence between the posterior and the prior weighted
      by the temperature.
  """

  def kl_fn(params):
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    sigma_prior = jnp.sqrt(1 / weight_decay)

    mu_vi_tree = params["mean"]
    sigma_vi_tree = jax.tree_map(jax.nn.softplus, params["inv_softplus_std"])

    def get_parameter_kl(mu_vi, sigma_vi):
      return (jnp.log(sigma_prior / sigma_vi) +
              (sigma_vi**2 + mu_vi**2) / 2 / sigma_prior**2 - 1 / 2)

    kl_tree = jax.tree_multimap(get_parameter_kl, mu_vi_tree, sigma_vi_tree)
    kl = sum([p_kl.sum() for p_kl in jax.tree_leaves(kl_tree)])

    return -kl * temperature

  return kl_fn
