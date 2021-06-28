"""Extracts bond length distributions from existing Conformer protos."""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from absl import app
from absl import flags
from absl import logging

from smu import dataset_pb2

import bond_lengths

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None, "TFDataRecord file containg Conformer protos")
flags.DEFINE_string("output", None, "Output file")

class BondDistToString(beam.DoFn):
  """Generate the dot separated form of a bond length distribution component """
  def process(self, bond_dist):
    key, value = bond_dist
    print(f"BondDistToString: key {key} value {value}")
    yield f"{key[0]}.{key[1]}.{key[2]}.{key[3]}.{value}"

class GroupBondTypes(beam.DoFn):
  def process(self, bond_dist):
    key, value = bond_dist
    print(f"GroupBondTypes: key #{key} value {value}")
    yield (key[0], key[1], key[2]), (key[3], value)

def get_bond_length_distribution_inner(input_fname: str, output_fname: str):
  """Generate bond length distibutions.

  Args:
    input_fname: An existing TFRecord file containing Conformer protos.
    output_fname: An output file that will be created that contains
      all bond length distributions - all bond types, all atom types.
      Requires post-processing to generate bond length distribution files.
  """
  print("Reading from {input_fname} output to {output_fname}")
  with beam.Pipeline(options=PipelineOptions()) as p:
    protos = (p
      | beam.io.tfrecordio.ReadFromTFRecord(input_fname, coder=beam.coders.ProtoCoder(dataset_pb2.Conformer().__class__))
      | beam.ParDo(bond_lengths.GetBondLengthDistribution())
      | beam.CombinePerKey(sum)
#     | beam.ParDo(GroupBondTypes())
#     | beam.GroupByKey()
      | beam.ParDo(BondDistToString())
      | beam.io.WriteToText(output_fname)
    )
    print(protos)

def get_bond_length_distribution(unused_argv):
  """Scan Conformer protos to extract bond length distributions."""
  del unused_argv

  get_bond_length_distribution_inner(FLAGS.input, FLAGS.output)


if __name__ == "__main__":
  app.run(get_bond_length_distribution)
