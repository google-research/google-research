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

"""Python wrappers for gen_molecule_graph_parsing_ops."""
import collections
import os

import tensorflow.compat.v1 as tf  # tf

try:
  from gigamol.molecule_graph_parsing_ops.py import gen_molecule_graph_parsing_ops  # pylint: disable=g-import-not-at-top
  kernels = gen_molecule_graph_parsing_ops
except ImportError:
  kernels = tf.load_op_library(
      os.path.join(
          os.path.dirname(
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
          'bazel-bin/molecule_graph_parsing_ops/cc/molecule_graph_parsing_ops.so'
      ))


class MoleculeGraphInput(
    collections.namedtuple(
        'MGInput', [
            'example_ids', 'atoms', 'pairs', 'atom_mask', 'pair_mask'])):
  """A named tuple used to organize MoleculeGraphParser output.

  The named tuple contains the following tensors:
    example_ids: The example ids.
    atoms: Atom feature Tensor with dimensions
      batch_size x max_atoms x num_atom_features.
    pairs: AtomPair feature Tensor with dimensions
      batch_size x max_atoms x max_atoms x num_pair_features.
    atom_mask: A boolean tensor with dimensions batch_size x max_atoms
      indicating which atoms are valid.
    pair_mask: A boolean tensor with dimensions
      batch_size x max_atoms x max_atoms indicating which pairs are valid.
  """


def MoleculeGraphParser(example_ids,
                        values,
                        max_atoms,
                        max_pair_distance = -1,
                        allow_overflow = True):
  """MoleculeGraph input from MoleculeGraphParser.

  Args:
    example_ids: A tensor containing example ids.
    values: A tensor containing serialized MoleculeGraph protos.
    max_atoms: Maximum number of atoms in a molecule.
    max_pair_distance: Maximum distance between atoms in pairs.
    allow_overflow: If true, allow molecules with more than max_atoms atoms.
      Only the first max_atoms atoms will be used.

  Returns:
    A MGInput named tuple.
  """
  return MoleculeGraphInput(*kernels.molecule_graph_parser(
      example_ids,
      values,
      max_atoms=max_atoms,
      max_pair_distance=max_pair_distance,
      allow_overflow=allow_overflow))
