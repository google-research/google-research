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

"""A class representing molecular stoichiometries.

Also, a method for enumerating neutral stoichiometries with a given number of
heavy atoms.
"""

import csv
import dataclasses
from typing import Dict, Iterable, Iterator, List, Optional, TextIO

from . import graph_sampler
from rdkit import Chem
from rdkit import RDLogger


def parse_dict_flag(lst):
  """Converts a list of strings like "foo=5" to a str->int dictionary."""
  result = {}
  for item in lst:
    k, v = item.split('=')
    result[k] = int(v)
  return result


def as_atom(symbol):
  # Temporarily disable rdkit's logging to avoid spamming with
  # "WARNING: not removing hydrogen atom without neighbors"
  RDLogger.DisableLog('rdApp.warning')
  mol = Chem.MolFromSmiles(f'[{symbol}]')
  RDLogger.EnableLog('rdApp.warning')
  return mol.GetAtoms()[0]


def get_valence(symbol):
  """Returns the guessed valence of a given atom type."""
  atom = as_atom(symbol)
  return atom.GetNumRadicalElectrons()


def get_valences(symbol_list):
  valence = {}
  for symbol in symbol_list:
    try:
      valence[symbol] = get_valence(symbol)
    except AttributeError as e:
      raise ValueError(f'Failed to infer valence for symbol "{symbol}". '
                       f'Please specify a valence explicitly.') from e
  return valence


def get_charge(symbol):
  """Returns the charge of a given atom type."""
  atom = as_atom(symbol)
  return atom.GetFormalCharge()


def get_charges(symbol_list):
  charge = {}
  for symbol in symbol_list:
    try:
      charge[symbol] = get_charge(symbol)
    except AttributeError as e:
      raise ValueError(f'Failed to infer charge for symbol "{symbol}". '
                       f'Please specify a charge explicitly.') from e
  return charge


@dataclasses.dataclass
class Stoichiometry:
  """Counts and properties of a set of atoms."""
  counts: Dict[str, int]
  valences: Dict[str, int] = dataclasses.field(default_factory=dict)
  charges: Dict[str, int] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    # Guess any charges and valences not explicitly specified.
    self.valences.update(get_valences(set(self.counts) - set(self.valences)))
    self.charges.update(get_charges(set(self.counts) - set(self.charges)))

    # Drop count/valence/charge information for elements we don't have.
    self.counts = {k: v for k, v in self.counts.items() if v != 0}
    self.valences = {k: self.valences[k] for k in self.counts}
    self.charges = {k: self.charges[k] for k in self.counts}

  def replace(self, new_counts = None, **kwargs):
    """Returns a copy of this stoichiometry with given counts overwritten."""
    valences = self.valences.copy()
    valences.update(kwargs.pop('valences', {}))
    charges = self.charges.copy()
    charges.update(kwargs.pop('charges', {}))
    counts = self.counts.copy()
    if new_counts:
      counts.update(new_counts)
    if kwargs:
      counts.update(kwargs)
    return Stoichiometry(counts, valences, charges)

  def max_hydrogens(self):
    total_valence = 0
    total_atoms = 0
    for symbol, count in self.counts.items():
      total_atoms += count
      total_valence += count * self.valences[symbol]
    return total_valence - 2 * (total_atoms - 1)

  def charge(self):
    return sum(
        count * self.charges[symbol] for symbol, count in self.counts.items())

  def to_degree_list(self):
    return sum([[self.valences[symbol]] * count
                for symbol, count in self.counts.items()], [])

  def to_element_list(self):
    return sum([[symbol] * count for symbol, count in self.counts.items()], [])

  def write(self, file_obj):
    writer = csv.DictWriter(file_obj, self.counts.keys())
    writer.writeheader()
    writer.writerow(self.counts)
    writer.writerow(self.valences)
    writer.writerow(self.charges)


def is_valid(stoich):
  degrees = stoich.to_degree_list()
  return (graph_sampler.valid_degrees(degrees) and
          graph_sampler.can_be_connected(degrees))


def read(file_obj):
  reader = csv.DictReader(file_obj)
  counts = {k: int(v) for k, v in next(reader).items()}
  valences = {k: int(v) for k, v in next(reader).items()}
  charges = {k: int(v) for k, v in next(reader).items()}
  return Stoichiometry(counts, valences, charges)


def enumerate_stoichiometries(
    num_heavy,
    heavy_elements,
    valences = None,
    charges = None):
  """Yields all valid stoichiometries with a given number of heavy atoms."""

  if valences is None:
    valences = {}
  if charges is None:
    charges = {}
  valences.update(get_valences(set(heavy_elements) - set(valences)))
  charges.update(get_charges(set(heavy_elements) - set(charges)))

  def recursive_enumerate_stoichiometries(stoich, num_heavy, heavy_elements):
    if num_heavy == 0:
      # We are out of heavy atoms. First, check that we are neutral, then
      # allocate hydrogens.
      if stoich.charge() != 0:
        return
      # Parity of the number of hydrogens is determined, so use step size 2.
      for num_h in range(stoich.max_hydrogens(), -1, -2):
        x = stoich.replace(H=num_h, valences=valences, charges=charges)
        if is_valid(x):
          yield x
    elif len(heavy_elements) == 1:
      for x in recursive_enumerate_stoichiometries(
          stoich.replace({heavy_elements[0]: num_heavy},
                         valences=valences,
                         charges=charges), 0, heavy_elements[1:]):
        if is_valid(x):
          yield x
    else:
      for this_count in range(num_heavy + 1):
        for x in recursive_enumerate_stoichiometries(
            stoich.replace({heavy_elements[0]: this_count},
                           valences=valences,
                           charges=charges), num_heavy - this_count,
            heavy_elements[1:]):
          if is_valid(x):
            yield x

  for x in recursive_enumerate_stoichiometries(
      Stoichiometry({}), num_heavy, heavy_elements):
    yield x
