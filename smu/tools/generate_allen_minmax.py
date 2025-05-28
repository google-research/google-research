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
"""Generates python code to be pasted for the Allen et al bond lengths."""

import itertools

from absl import app

from smu import dataset_pb2
from smu.geometry import bond_length_distribution


def main(argv):
  # Shortcuts for below
  atom_str = {
      dataset_pb2.BondTopology.ATOM_C: "ATOM_C",
      dataset_pb2.BondTopology.ATOM_N: "ATOM_N",
      dataset_pb2.BondTopology.ATOM_O: "ATOM_O",
      dataset_pb2.BondTopology.ATOM_F: "ATOM_F"
  }
  bond_str = {
      dataset_pb2.BondTopology.BOND_SINGLE: "BOND_SINGLE",
      dataset_pb2.BondTopology.BOND_DOUBLE: "BOND_DOUBLE",
      dataset_pb2.BondTopology.BOND_TRIPLE: "BOND_TRIPLE"
  }

  allen_dists = bond_length_distribution.AllAtomPairLengthDistributions()
  allen_dists.add_from_gaussians_file(argv[1], 3)

  for (atom_a, atom_b), bond in itertools.product(
      itertools.combinations_with_replacement([
          dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
          dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_F
      ], 2), [
          dataset_pb2.BondTopology.BOND_SINGLE,
          dataset_pb2.BondTopology.BOND_DOUBLE,
          dataset_pb2.BondTopology.BOND_TRIPLE
      ]):
    try:
      mn = allen_dists[(atom_a, atom_b)][bond].min()
      mx = allen_dists[(atom_a, atom_b)][bond].max()
      print(
          f"  (dataset_pb2.BondTopology.{atom_str[atom_a]},\n"
          f"   dataset_pb2.BondTopology.{atom_str[atom_b]},\n"
          f"   dataset_pb2.BondTopology.{bond_str[bond]}): ({mn:0.3f}, {mx:.03f}),"
      )
    except KeyError:
      pass


if __name__ == "__main__":
  app.run(main)
