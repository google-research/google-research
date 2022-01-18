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

"""Module containing the get_ece_bias function."""
import numpy as np
from caltrain.run_calibration import estimate_ece
from caltrain.utils import get_hash_key


def get_ece_bias(config,
                 n_samples,
                 ce_types,
                 params,
                 cached_result=None,
                 data_dir=None):
  """Get ece bias for given sample sizes, ce_types, and parametric datasets.

  Args:
      config (dict): Configuration dictionary:
        * num_reps (str): The number of repititions
        * num_bins (int): The number of bins for binning scheme
        * split (string): train/test split (default="")
        * a (float): Coefficient for logistic model: E[Y | f(x)] =
          1/(1+exp(-a*(fx-b)))
        * b (float): Coefficient for logistic model: E[Y | f(x)] =
          1/(1+exp(-a*(fx-b)))
        * d (float): Exponent for polynomial model: E[Y | f(x)] = f(x)^d
        * alpha (float): Parameter for Beta distribution: f(x)~Beta(alpha, beta)
        * beta (float): Parameter for Beta distribution: f(x)~Beta(alpha, beta)
      n_samples (int): Number of samples from the model
      ce_types (list[str]): = list of calibration error types: 'em_ece_bin',
        'ew_ece_bin', 'em_ece_sweep', 'ew_ece_sweep'
      params (dict): Dictionary of dataset configurations; each value is of
        len(num_datasets)
      cached_result (bool): Use cached result (default=True)
      data_dir (str): location of data dir

  Returns:
      array (n_samples, ce_types, num_datasets): Computed ECE Biases.

  """

  num_datasets = len(params['a'])
  ece_bias = np.zeros((len(n_samples), len(ce_types), num_datasets))
  for i in range(len(n_samples)):
    config['num_samples'] = n_samples[i]
    for ce_idx in range(len(ce_types)):
      config['ce_type'] = ce_types[ce_idx]
      for j in range(num_datasets):
        config['a'] = params['a'][j]
        config['b'] = params['b'][j]
        config['alpha'] = params['alpha'][j]
        config['beta'] = params['beta'][j]
        config['d'] = params['d'][j]
        config['dataset'] = params['dataset'][j]
        hash_key = get_hash_key(config)
        if cached_result:
          if hash_key in cached_result:
            # print(f'{hash_key} already computed, loading cached result.')
            mean = cached_result[hash_key]['bias']
          else:
            mean, _, _ = estimate_ece(config, data_dir=data_dir)
        ece_bias[i, ce_idx, j] = mean
  return ece_bias
