from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.tfrecordio import ReadFromTFRecord

from smu import dataset_pb2
from smu.geometry import bond_length_distribution

import topology_from_geom

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None, "TFDataRecord file containg Conformer protos")
flags.DEFINE_string("bonds", None, "File name stem for bond length distributions")
flags.DEFINE_string("output", None, "Output file")
flags.DEFINE_boolean("xnonbond", False, "Exclude non bonded interactions")

class SummaryData(beam.DoFn):
  """Given BondTopologies as input, yield summary data"""
  def process(self, topology_matches:dataset_pb2.TopologyMatches):
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


def ReadConFormer(bond_lengths: bond_length_distribution.AllAtomPairLengthDistributions, input: str,
                  output: str):
  """
  Args:
  Returns:
  """

  class GetAtoms(beam.DoFn):

    def process(self, item):
      yield item.optimized_geometry.atom_positions[0].x

  with beam.Pipeline(options=PipelineOptions()) as p:
    protos = (p | beam.io.tfrecordio.ReadFromTFRecord(
        input, coder=beam.coders.ProtoCoder(dataset_pb2.Conformer().__class__)) |
              beam.ParDo(topology_from_geom.TopologyFromGeom(bond_lengths)) |
              beam.ParDo(SummaryData()) |
              beam.io.textio.WriteToText(output))

    return protos


def topology_from_geometry_main(unused_argv):
  del unused_argv

  bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
  bond_lengths.add_from_files(FLAGS.bonds, 0.0, FLAGS.xnonbond)
  protos = ReadConFormer(bond_lengths, FLAGS.input, FLAGS.output)
  print(protos)


if __name__ == "__main__":
  flags.mark_flag_as_required("input")
  flags.mark_flag_as_required("bonds")
  flags.mark_flag_as_required("output")
  app.run(topology_from_geometry_main)
