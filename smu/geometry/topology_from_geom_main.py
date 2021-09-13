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
import topology_from_geom

from smu import dataset_pb2
from smu.geometry import bond_length_distribution

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None,
                    "TFDataRecord file containg Conformer protos")
flags.DEFINE_string("bonds", None,
                    "File name stem for bond length distributions")
flags.DEFINE_string("output", None, "Output file")
flags.DEFINE_boolean("xnonbond", False, "Exclude non bonded interactions")


class SummaryData(beam.DoFn):
  """Given BondTopologies as input, yield summary data."""

  def process(self, topology_matches):
    result = f"{len(topology_matches.bond_topology)}"

    for bt in topology_matches.bond_topology:
      result += f",{bt.score:.3f},{bt.smiles}"
      if len(topology_matches.bond_topology) == 1:
        break
      if bt.is_starting_topology:
        result += ",T"
      else:
        result += ",F"

    yield result


def ReadConFormer(
    bond_lengths,
    input_string, output):
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

  with beam.Pipeline(options=PipelineOptions()) as p:
    protos = (
        p | beam.io.tfrecordio.ReadFromTFRecord(
            input_string,
            coder=beam.coders.ProtoCoder(dataset_pb2.Conformer().__class__))
        | beam.ParDo(topology_from_geom.TopologyFromGeom(bond_lengths))
        | beam.ParDo(SummaryData()) | beam.io.textio.WriteToText(output))

    return protos


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
