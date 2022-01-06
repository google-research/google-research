# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Topology from Geometry."""
from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from smu.geometry import topology_from_geom

from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.geometry import smu_molecule

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None,
                    "TFDataRecord file containg Conformer protos")
flags.DEFINE_string("bonds", None,
                    "File name stem for bond length distributions")
flags.DEFINE_string("output", None, "Output file")
flags.DEFINE_boolean("xnonbond", False, "Exclude non bonded interactions")
flags.DEFINE_boolean("neutral", False,
                     "Do the topology search against neutral forms")


class SummaryData(beam.DoFn):
  """Given BondTopologies as input, yield summary data."""

  def process(self, topology_matches):
    # Some variables written with each record.
    starting_smiles = topology_matches.starting_smiles
    conformer_id = topology_matches.conformer_id
    nbt = len(topology_matches.bond_topology)
    fate = topology_matches.fate

    if not topology_matches.bond_topology:
      yield f".,{starting_smiles},{conformer_id},{fate},0,.,.,."
      return

    result = ""
    for bt in topology_matches.bond_topology:
      gscore = round(bt.geometry_score, 1)
      tscore = round(bt.topology_score, 4)
      result += f"{bt.smiles},{starting_smiles},{conformer_id},{fate},{nbt},{bt.ring_atom_count},{bt.is_starting_topology},{tscore},{gscore}\n"

    yield result.rstrip("\n")


def ReadConFormer(bond_lengths, input_string, output):
  """Reads conformer.

  Args:
    bond_lengths:
    input_string:
    output:

  Returns:
  """

  #   class GetAtoms(beam.DoFn):

  #     def process(self, item):
  #       yield item.optimized_geometry.atom_positions[0].x

  options = PipelineOptions(
      direct_num_workers=6, direct_running_mode="multi_processing")
  # options = PipelineOptions()
  with beam.Pipeline(options=options) as p:
    protos = (
        p | beam.io.tfrecordio.ReadFromTFRecord(
            input_string,
            coder=beam.coders.ProtoCoder(dataset_pb2.Conformer().__class__))
        | beam.ParDo(topology_from_geom.TopologyFromGeom(bond_lengths))
        | beam.ParDo(SummaryData()) | beam.io.textio.WriteToText(output))

    return protos


class TopologyFromGeom(beam.DoFn):
  """Beam class for extracting BondTopology from Conformer protos."""

  def __init__(self, bond_lengths):
    super().__init__()
    self._bond_lengths = bond_lengths

  def process(self, conformer):
    """Called by Beam.

      Returns a TopologyMatches for the plausible BondTopology's in `conformer`.
    Args:
      conformer:

    Yields:
      dataset_pb2.TopologyMatches
    """
    # Adjust as needed...
    #   if conformer.fate != dataset_pb2.Conformer.FATE_SUCCESS:
    #     return
    matching_parameters = smu_molecule.MatchingParameters()
    matching_parameters.neutral_forms_during_bond_matching = True
    matching_parameters.must_match_all_bonds = True
    matching_parameters.consider_not_bonded = True
    matching_parameters.ring_atom_count_cannot_decrease = False
    yield topology_from_geom.bond_topologies_from_geom(
        self._bond_lengths, conformer.conformer_id, conformer.fate,
        conformer.bond_topologies[0], conformer.optimized_geometry,
        matching_parameters)


def TopologyFromGeometryMain(unused_argv):
  del unused_argv

  bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
  bond_lengths.add_from_files(FLAGS.bonds, 0.0, FLAGS.xnonbond)
  protos = ReadConFormer(bond_lengths, FLAGS.input, FLAGS.output)
  print(protos)


if __name__ == "__main__":
  flags.mark_flag_as_required("input")
  flags.mark_flag_as_required("bonds")
  flags.mark_flag_as_required("output")
  app.run(TopologyFromGeometryMain)
