# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Example use case for all the accountants developed in this directory.

This file provides a simple example to use the accountant class for each
batch sampler to compute bounds on the corresponding hockey stick divergence.
The source also supports other methods such as computing `epsilon` for a given
`delta`, or computing the smallest noise multiplier `sigma` for a single
`(epsilon, delta)`. These are not demonstrated in the examples below. The user
is encouraged to review the source code for the classes in `dpsgd_bounds.py`
and `balls_and_bins.py` to understand how to use them.
"""

from collections.abc import Sequence

from absl import app
import pandas as pd

from dpsgd_batch_sampler_accounting import balls_and_bins
from dpsgd_batch_sampler_accounting import dpsgd_bounds


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  num_steps_per_epoch = 1000
  num_epochs = 1
  sigma = 0.35
  epsilons = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0)

  print(f"""
For DP-SGD with noise multiplier {sigma}, run for {num_epochs} epochs and {num_steps_per_epoch} steps per epoch,
the following bounds are provided for the hockey stick divergence corresponding to
various batch samplers.
""")

  data = {'epsilon': epsilons}
  # 1. Deterministic batch sampler.
  print('1. delta_D corresponding to the Deterministic batch sampler.')
  det_accountant = dpsgd_bounds.DeterministicAccountant()
  data['delta_D'] = det_accountant.get_deltas(sigma, epsilons, num_epochs)

  # 2. Poisson batch sampler.
  # (a) Upper bound using RDP analysis.
  print('2. delta_P corresponding to the Poissson subsampling batch sampler.')
  print('   (a) Upper bound using RDP analysis, and')
  poisson_rdp_accountant = dpsgd_bounds.PoissonRDPAccountant()
  data['delta_P (RDP: upper)'] = poisson_rdp_accountant.get_deltas(
      sigma, epsilons, num_steps_per_epoch, num_epochs
  )
  # (b) Upper and lower bound using PLD analysis.
  discretization = 1e-3
  print('   (b) Upper and lower bounds using PLD analysis '
        f'(with {discretization=}).')
  poisson_pld_ub_accountant = dpsgd_bounds.PoissonPLDAccountant(
      pessimistic_estimate=True
  )
  poisson_pld_lb_accountant = dpsgd_bounds.PoissonPLDAccountant(
      pessimistic_estimate=False
  )
  data['delta_P (PLD: upper)'] = poisson_pld_ub_accountant.get_deltas(
      sigma, epsilons, num_steps_per_epoch, num_epochs, discretization
  )
  data['delta_P (PLD: lower)'] = poisson_pld_lb_accountant.get_deltas(
      sigma, epsilons, num_steps_per_epoch, num_epochs, discretization,
  )

  # 3. Shuffle batch sampler
  print('3. delta_S corresponding to the Shuffle batch sampler '
        '(lower bound only).')
  shuffle_lb_accountant = dpsgd_bounds.ShuffleAccountant()
  data['delta_S (lower)'] = shuffle_lb_accountant.get_deltas(
      sigma, epsilons, num_steps_per_epoch, num_epochs
  )

  # 4. Balls and Bins batch sampler.
  print('4. delta_B corresponding to the Balls and Bins batch sampler.')
  bnb_accountant = balls_and_bins.BnBAccountant()
  # (a) Lower bound.
  print('   (a) Lower bound using a similar approach as for the Shuffle batch '
        'sampler.')
  data['delta_B (lower)'] = bnb_accountant.get_deltas_lower_bound(
      sigma, epsilons, num_steps_per_epoch, num_epochs
  )
  # (b) Estimate and upper confidence bound using importance sampling.
  sample_size = 100_000
  error_prob = 0.01
  adjacency_type = balls_and_bins.AdjacencyType.REMOVE
  print(f"""\
   (b) Monte Carlo estimate using importance sampling (denoted 'imp' in the table below) is performed
       with sample size {sample_size}. Both the mean estimate and the upper confidence bound are
       provided, the latter is computed for error probability {error_prob}.""")
  delta_estimates = bnb_accountant.estimate_deltas(
      sigma, epsilons, num_steps_per_epoch, sample_size, num_epochs,
      adjacency_type, use_importance_sampling=True,
  )
  data['delta_B (imp: mean)'] = [e.mean for e in delta_estimates]
  data['delta_B (imp: ucb)'] = [
      e.get_upper_confidence_bound(error_prob) for e in delta_estimates
  ]
  # (c) Estimate and upper confidence bound using order statistics sampling.
  sample_size = 500_000
  order_stats_encoding = (1, 100, 1, 100, 500, 10, 500, 1000, 50)
  print(f"""\
   (c) Monte Carlo estimate using order statistics sampling (denoted 'os' in the table below) is
       performed with sample size {sample_size}. Both the mean estimate and the upper confidence
       bound are provided, the latter is computed for error probability {error_prob}. The order
       statistics used are given by the encoding {order_stats_encoding}.
       See the source code for `balls_and_bins.get_order_stats_seq_from_encoding` for more details.
   In this example, we only demonstrate the Monte Carlo estimation for REMOVE adjacency,
   since the value for ADD adjacency is only smaller. But in practice, one would have to
   compute both directions and take the maximum.\n""")
  order_stats_seq = balls_and_bins.get_order_stats_seq_from_encoding(
      order_stats_encoding, num_steps_per_epoch
  )
  delta_estimates = bnb_accountant.estimate_order_stats_deltas(
      sigma, epsilons, num_steps_per_epoch, sample_size, order_stats_seq,
      num_epochs, adjacency_type,
  )
  data['delta_B (os: mean)'] = [e.mean for e in delta_estimates]
  data['delta_B (os: ucb)'] = [
      e.get_upper_confidence_bound(error_prob) for e in delta_estimates
  ]

  # Print the data as a table.
  df = pd.DataFrame(data)
  print(df.to_string(index=False))


if __name__ == '__main__':
  app.run(main)
