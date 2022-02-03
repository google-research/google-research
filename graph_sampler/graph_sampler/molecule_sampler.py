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

"""Sample and count molecules with a given stoichiometry.

This file is the "glue" between graph_sampler, igraph, and rdkit.
* graph_sampler provides the importance sampling algorithm.
* igraph counts graph automorphisms, so provides the weight function. It's also
  our storage format (via graph_io).
* rdkit translates to the lingua franca, in particular SMILES strings.
"""

import copy
import dataclasses
import math
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

from . import graph_sampler
from . import stoichiometry
import igraph
import numpy as np
from rdkit import Chem

Stoichiometry = stoichiometry.Stoichiometry
Graph = igraph.Graph

BOND_TYPES = [
    Chem.rdchem.BondType.ZERO,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.QUADRUPLE,
    Chem.rdchem.BondType.QUINTUPLE,
    Chem.rdchem.BondType.HEXTUPLE,
]


def to_mol(graph, symbol_dict = None):
  """Convert the graph to an rdkit.Chem.Mol.

  Args:
    graph: an igraph.Graph for which vertices have an "element" property.
    symbol_dict: optional lookup table for converting your symbols into
      rdkit-digestible symbols. E.g. if you used "On" to represent a negative
      oxygen ion, include {"On": "O-"} so rdkit understands what you meant.

  Returns:
    The graph as an rdkit.Chem.Mol.
  """
  simple_graph = convert_to_simple(graph)
  editable_mol = Chem.RWMol(Chem.Mol())
  atom_indices = {}  # Map from our indices to rdkit's.
  for i, element in enumerate(simple_graph.vs['element']):
    if symbol_dict and element in symbol_dict:
      element = symbol_dict[element]
    x = stoichiometry.as_atom(element)
    atom = Chem.Atom(x.GetSymbol())
    atom.SetFormalCharge(x.GetFormalCharge())
    atom_indices[i] = editable_mol.AddAtom(atom)
  for e in simple_graph.es:
    editable_mol.AddBond(atom_indices[e.source], atom_indices[e.target],
                         BOND_TYPES[e['order']])
  mol = editable_mol.GetMol()
  # UpdatePropertyCache calculates valence states. Without this, we get the
  # following error if we immediately call to_graph:
  # getNumImplicitHs() called without preceding call to calcImplicitValence()
  mol.UpdatePropertyCache()
  return mol


def to_symbol(atom):
  """Converts a Chem.Atom to a string like "C", "O-", or "Fe+2"."""
  symbol = atom.GetSymbol()
  charge = atom.GetFormalCharge()
  if charge == 0:
    return symbol
  suffix = '+' if charge > 0 else '-'
  if abs(charge) > 1:
    suffix += str(abs(charge))
  return symbol + suffix


def to_graph(mol):
  """Convert a rdkit.Chem.Mol to a graph."""
  mol = Chem.AddHs(mol)  # Include any implicit hydrogens.
  elements = []
  for atom in mol.GetAtoms():
    elements.append(to_symbol(atom))
  edges = []
  for bond in mol.GetBonds():
    edge = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    order = BOND_TYPES.index(bond.GetBondType())
    edges.extend([edge] * order)

  graph = igraph.Graph(edges)
  graph.vs['element'] = elements
  return graph


def to_smiles(mol):
  """Convert the graph to a canonicalized SMILES string."""
  return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def edge_list_to_graph(stoich, edges,
                       importance, weight):
  """Package an edge list and importance weights into an igraph.Graph."""
  graph = igraph.Graph(edges)
  graph.vs['element'] = stoich.to_element_list()
  graph['importance'] = importance
  graph['weight'] = weight
  return graph


def convert_to_simple(graph):
  """Drop duplicate edges, recording multiplicity in the "order" property."""
  g = copy.copy(graph)
  g.es['order'] = 1
  g.simplify(multiple=True, loops=False, combine_edges='sum')
  g.es['order'] = [int(x) for x in g.es['order']]
  return g


def is_isomorphic(graph1, graph2):
  """True if given graphs are isomorphic."""
  simple1 = convert_to_simple(graph1)
  simple2 = convert_to_simple(graph2)
  elements = list(set(simple1.vs['element'] + simple2.vs['element']))
  color1 = [elements.index(v['element']) for v in simple1.vs]
  color2 = [elements.index(v['element']) for v in simple2.vs]
  # This is kind of annoying. This bug
  # https://github.com/igraph/igraph/issues/1010 while fixed a while ago, the
  # fix isn't in the version that easily installable via pip.  So we're going to
  # do the simple edge counting trick to error out early if the number of edge
  # is not the same.
  if (simple1.ecount() != simple2.ecount() or
      simple1.vcount() != simple2.vcount()):
    return False
  return simple1.isomorphic_vf2(
      simple2,
      color1=color1,
      color2=color2,
      edge_color1=simple1.es['order'],
      edge_color2=simple2.es['order'])


# An unlabelled graph g with a given underlying element list corresponds to
# prod(factorial(num(elem)))/|Aut(g)| labelled graphs. So to count the number of
# unlabelled graphs with given stoichiometry, we can count the number of
# labelled graphs, but weight each labelled graph by
# |Aut(g)|/prod(factorial(num(elem)))
def weight_connected_up_to_iso(
    stoich):
  """A weight function for enumerating connected molecules."""
  prod_of_factorials = 1
  for n in stoich.counts.values():
    prod_of_factorials *= math.factorial(n)
  colors = sum([[i] * cnt for i, cnt in enumerate(stoich.counts.values())], [])

  def weight_func(graph):
    # igraph sometimes gets confused by multiple edges when computing
    # automorphisms (github.com/igraph/python-igraph/issues/166), so we throw
    # out multiple edges and color edges according to their multiplicity.
    edges = list(set(graph))
    orders = [graph.count(e) for e in edges]
    graph = igraph.Graph(edges)
    if len(graph.components()) > 1:
      return 0.0
    return graph.count_automorphisms_vf2(
        color=colors, edge_color=orders) / prod_of_factorials

  return weight_func


class MoleculeSampler:
  """Yields molecular graphs, tracking various stats and stopping conditions.

  See graph_sampler.GraphsSampler for available kwargs. This class is a thin
  wrapper which uses the given stoichiometry to specify the correct degree
  vector and weight function, and converts the sampled graphs into igraph.Graph
  objects.

  Yields:
    igraph.Graph objects representing molecular graphs, with weight and
    importance encoded in graph properties.
  """

  def __init__(self, stoich, **kwargs):
    """Initializer.

    Args:
      stoich: the stoichiometry from which to sample.
      **kwargs: keyword arguments passed to graph_sampler.GraphSampler.
    """
    self.stoich = stoich
    self.graph_sampler = graph_sampler.GraphSampler(
        degrees=self.stoich.to_degree_list(),
        prune_disconnected=True,
        weight_func=weight_connected_up_to_iso(self.stoich),
        **kwargs)

  def sample(self):
    edges, importance, weight = self.graph_sampler.sample()
    return edge_list_to_graph(self.stoich, edges, importance, weight)

  def __iter__(self):
    for edges, importance, weight in self.graph_sampler:
      yield edge_list_to_graph(self.stoich, edges, importance, weight)

  def stats(self):
    return self.graph_sampler.stats()


def estimate_num_molecules(stoich,
                           **kwargs):
  """Estimates the of molecular graphs with given stoichiometry.

  Args:
    stoich: the stoichiometry.
    **kwargs: keyword arguments to pass along to MoleculeSampler.

  Returns:
    A pair (num_graphs, std_err), where num_graphs is an estimate of the
    number of graphs (up to isomorphism) with given stoichiometry and std_err is
    the standard error of the estimate.
  """
  sampler = MoleculeSampler(stoich, **kwargs)
  for _ in sampler:
    # Ignore the samples, we'll just return the estimated size.
    pass
  stats = sampler.stats()
  return stats['estimated_num_graphs'], stats['num_graphs_std_err']


T = TypeVar('T')


@dataclasses.dataclass
class RejectToUniform(Generic[T]):
  """Given weighted samples and a maximum importance, yields uniform samples.

  Attributes:
    base_iter: an iterable of importance-weighted samples.
    max_importance: an upper bound on importance.
    importance_fn: a callable which returns the importance weight of a given
      item.
    rng_seed: an integer to seed random number generation.

  Yields:
    A subset of base_iter, where each object x is emitted with probability
    importance_fn(x)/max_importance.
  """
  base_iter: Iterable[T]
  max_importance: float
  importance_fn: Callable[[T], float] = lambda x: x['importance'] * x['weight']
  rng_seed: Optional[int] = None

  def __post_init__(self):
    if self.rng_seed is None:
      # Record an explicit seed to save out for reproducibility.
      self.rng_seed = graph_sampler.rng_seed_int32()
    self._rng = np.random.default_rng(self.rng_seed)
    self.num_processed = 0
    self.num_accepted = 0

    # There's some floating point rounding when reading/writing files, so add an
    # epsilon for purposes of checking that max_importance is not exceeded.
    self._max_plus_eps = self.max_importance * (1 + 1e-10)

  def __iter__(self):
    for x in self.base_iter:
      self.num_processed += 1
      importance = self.importance_fn(x)
      assert importance <= self._max_plus_eps, (
          f'Invididual importances must be no greater than max_importance. '
          f'Got importance {importance} and max_importance {self.max_importance}'
      )
      if self._rng.random() <= importance / self.max_importance:
        self.num_accepted += 1
        yield x


@dataclasses.dataclass
class AggregateUniformSamples(Generic[T]):
  """Converts uniform samples from buckets to uniform samples from their union.

  Attributes:
    bucket_sizes: a list of estimated bucket sizes.
    sample_sizes: how many samples you have from each bucket.
    base_iters: iterators through your samples for each bucket.
    target_num_samples: optional target total number of samples. If not
      provided, we'll try to get as many as possible while still sampling
      uniformly from the union of the buckets.
    rng_seed: an integer to seed random number generation.
  """
  bucket_sizes: List[float]
  sample_sizes: List[int]
  base_iters: Iterable[Iterable[T]]
  target_num_samples: Optional[float] = None
  rng_seed: Optional[int] = None

  def __post_init__(self):
    if self.rng_seed is None:
      # Record an explicit seed to save out for reproducibility.
      self.rng_seed = graph_sampler.rng_seed_int32()
    self._rng = np.random.default_rng(self.rng_seed)

    self.num_accepted = 0
    self.num_proccessed = 0
    self.num_iters_started = 0

    self.bucket_sizes = np.array(self.bucket_sizes)
    self.sample_sizes = np.array(self.sample_sizes)
    self.weights = self.bucket_sizes / self.bucket_sizes.sum()

    # Set or validate target_num_samples. If we try to produce N samples, the
    # expected number of samples from bucket i is N * weights[i], which must be
    # no greater than sample_sizes[i], the number of available samples from
    # bucket i.
    max_target_num_samples = np.min(self.sample_sizes / self.weights)
    if self.target_num_samples is None:
      self.target_num_samples = max_target_num_samples
    elif max_target_num_samples < self.target_num_samples:
      print('Sample pool is too small to provide {self.target_num_samples} '
            'samples. Shooting for {max_target_num_samples} instead.')
      self.target_num_samples = max_target_num_samples

  def __iter__(self):
    for base_iter, weight, sample_size in zip(self.base_iters, self.weights,
                                              self.sample_sizes):
      self.num_iters_started += 1
      acceptance_prob = self.target_num_samples * weight / sample_size
      for x in base_iter:
        self.num_proccessed += 1
        if self._rng.random() <= acceptance_prob:
          self.num_accepted += 1
          yield x
