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

"""Tests for molecule_sampler."""

import collections
import math

from absl.testing import absltest
from absl.testing import parameterized
from graph_sampler import molecule_sampler
from graph_sampler import stoichiometry
import igraph
import numpy as np
from rdkit import Chem

# For debugging
edge_set = lambda g: set(e.tuple for e in g.es)
element_list = lambda g: g.vs['element']


class MoleculeSamplerTest(parameterized.TestCase):

  def test_against_known_counts(self):
    # This is a random-ish sample of stoichiometries for which the
    # exact count (lumping together any number of hydrogens) is known.
    # pyformat: disable
    known_counts = [
        (4449, {'C': 7}),
        (12580, {'C': 5, 'N+': 1, 'O-': 1}),
        (4920, {'C': 4, 'N+': 1, 'O-': 1, 'F': 1}),
        (29841, {'C': 3, 'N': 3, 'O': 1}),
        (7293, {'C': 3, 'N': 1, 'O': 3}),
        (6544, {'C': 2, 'N': 5}),
        (17978, {'C': 2, 'N': 1, 'N+': 1, 'O': 2, 'O-': 1}),
        (10, {'C': 2, 'O': 1, 'F': 4}),
        (1, {'C': 1, 'N': 1, 'F': 5}),
        (4440, {'C': 1, 'N': 4, 'O': 2}),
        (1187, {'C': 1, 'N': 2, 'O': 3, 'F': 1}),
        (237, {'C': 1, 'N': 1, 'O': 5}),
        (604, {'N': 5, 'O': 2}),
        (107, {'N': 3, 'O': 2, 'F': 2}),
        (115, {'N+': 3, 'O': 1, 'O-': 3}),
        (80, {'N': 2, 'O': 5}),
        (31, {'N+': 1, 'O': 5, 'O-': 1}),
    ]
    # pyformat: enable

    # Setting rng_seed to 0 produces two big-ish zscores (-2.85 and -3.17), but
    # this is just bad luck. I ran all these multiple times (rng_seed in
    # range(10, 20)) and the distribution of z-scores looked very much like a
    # standard normal (KL divergence of 0.04, which is very reasonable for 170
    # samples).
    rng_seed = 1

    relative_precision = 0.2
    print('\n%13s: %8s %8s %8s %8s %8s' %
          ('stoich', 'true', 'estimate', 'error', 'std', 'z-score'))

    fmt = lambda x: f'{x:.2f}'
    for true_num, counts in known_counts:
      stoich = stoichiometry.Stoichiometry(counts)
      max_h = stoich.max_hydrogens()
      total_estimate, total_variance = 0.0, 0.0
      for num_h in range(max_h, -1, -2):
        est, std = molecule_sampler.estimate_num_molecules(
            stoich.replace(H=num_h),
            relative_precision=relative_precision,
            rng_seed=rng_seed)
        total_estimate += est
        total_variance += std * std
      error = total_estimate - true_num
      total_std = math.sqrt(total_variance)
      print('%13s: %8s %8s %8s %8s %8s' %
            (''.join(stoich.to_element_list()), true_num, fmt(total_estimate),
             fmt(error), fmt(total_std), fmt(error / total_std)))
      self.assertLess(abs(error), 2.5 * total_std)

  @parameterized.parameters([
      Chem.MolFromSequence('R'),  # Arginine
      Chem.MolFromSmiles('[O-][N+](F)(F)F'),
      # An example where Chem.MolStandardize.rdMolStandardize.Cleanup would
      # apply the rule "Recombine1,3-separatedcharges", undoing two ionizations.
      Chem.MolFromSmiles('[H]N1N([O-])[N+]1=N[N+](OF)=C([O-])F'),
      # An example where Chem.MolStandardize.rdMolStandardize.Cleanup would
      # apply the rule "Normalize1,3conjugatedcation", changing the graph.
      Chem.MolFromSmiles('[H]N1[N+](F)=[N+]1C1(F)N([O-])ON1[O-]'),
  ])
  def test_to_from_mol(self, mol):
    graph = molecule_sampler.to_graph(mol)
    recovered_graph = molecule_sampler.to_graph(molecule_sampler.to_mol(graph))
    self.assertTrue(molecule_sampler.is_isomorphic(graph, recovered_graph))

  def test_implicit_hydrogens(self):
    smiles = 'C'
    graph = igraph.Graph([(0, 1), (1, 2), (1, 3), (1, 4)])
    graph.vs['element'] = ['H', 'C', 'H', 'H', 'H']

    # Check that recovered graphs include implicit hydrogens.
    recovered_graph = molecule_sampler.to_graph(Chem.MolFromSmiles(smiles))
    self.assertTrue(molecule_sampler.is_isomorphic(graph, recovered_graph))

    # Check that canonicalized SMILES strings omit implicit hydrogens.
    recovered_smiles = molecule_sampler.to_smiles(
        molecule_sampler.to_mol(graph))
    self.assertEqual(smiles, recovered_smiles)

  def test_to_from_mol_with_symbol_dict(self):
    graph = igraph.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    graph.vs['element'] = ['Np', 'H', 'H', 'H', 'On']
    symbol_dict = {'Np': 'N+', 'On': 'O-'}

    mol = molecule_sampler.to_mol(graph, symbol_dict)
    recovered_graph = molecule_sampler.to_graph(mol)

    graph.vs['element'] = [symbol_dict.get(x, x) for x in graph.vs['element']]
    self.assertTrue(molecule_sampler.is_isomorphic(graph, recovered_graph))

  def test_reject_to_uniform(self):
    # We sample uniformly from the interval [0, 1] and reject to "uniform" under
    # an importance weighting that gives us the "triangle distribution" with
    # pdf(x) = 2 * x. We expect half the samples to be rejected (the part of the
    # uniform square "above the triangle"), and the mean of the remainder to be
    # 2/3.
    seed1, seed2 = np.random.SeedSequence().spawn(2)
    size = 10000
    values = np.random.default_rng(seed1).random(size=size)
    importance_fn = lambda x: x  # bigger values are "more important"
    rejector = molecule_sampler.RejectToUniform(
        base_iter=values,
        max_importance=1.0,
        importance_fn=importance_fn,
        rng_seed=seed2)
    accepted_values = list(rejector)

    expected_size = size / 2
    expected_mean = 2.0 / 3

    self.assertLess(abs(len(accepted_values) - expected_size), np.sqrt(size))
    self.assertLess(
        abs(np.mean(accepted_values) - expected_mean), 1 / np.sqrt(size))

  def test_aggregate_uniform(self):
    # Suppose we have three buckets with sizes 100, 1000, 10000, from which we
    # have uniform samples of size 100, 500, 1000, respectively. Then the union
    # of the buckets has size 10000, and the best uniform sample we can get will
    # have (1000 / 10000) = 10% of that union.
    bucket_sizes = [100, 1000, 10000]
    sample_sizes = [100, 500, 1000]

    # We'll make every element of the i'th bucket be the number i.
    def dumb_iter(value, num):
      for _ in range(num):
        yield value

    base_iters = [dumb_iter(i, num) for i, num in enumerate(sample_sizes)]

    aggregator = molecule_sampler.AggregateUniformSamples(
        bucket_sizes, sample_sizes, base_iters, rng_seed=0)
    accepted_values = list(aggregator)
    count = collections.Counter(accepted_values)

    expected_count = {0: 10, 1: 100, 2: 1000}
    for i in count:
      acceptance_ratio = expected_count[i] / sample_sizes[i]
      std = np.sqrt(sample_sizes[i] * acceptance_ratio * (1 - acceptance_ratio))
      self.assertLessEqual(abs(count[i] - expected_count[i]), 2.5 * std)


if __name__ == '__main__':
  absltest.main()
