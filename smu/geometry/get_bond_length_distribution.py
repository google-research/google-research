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

"""Extracts bond length distributions from existing Conformer protos."""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import bond_lengths

from smu import dataset_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None,
                    "TFDataRecord file containg Conformer protos")
flags.DEFINE_string("output", None, "Output file")


class BondDistToString(beam.DoFn):
  """Generate the dot separated form of a bond length distribution component."""

  def process(self, bond_dist):
    key, value = bond_dist
    print(f"BondDistToString: key {key} value {value}")
    yield f"{key[0]}.{key[1]}.{key[2]}.{key[3]}.{value}"


class GroupBondTypes(beam.DoFn):

  def process(self, bond_dist):
    key, value = bond_dist
    print(f"GroupBondTypes: key #{key} value {value}")
    yield (key[0], key[1], key[2]), (key[3], value)


def get_bond_length_distribution_inner(input_fname, output_fname):
  """Generate bond length distibutions.

  Args:
    input_fname: An existing TFRecord file containing Conformer protos.
    output_fname: An output file that will be created that contains all bond
      length distributions - all bond types, all atom types. Requires
      post-processing to generate bond length distribution files.
  """
  print("Reading from {input_fname} output to {output_fname}")
  options = PipelineOptions(
      direct_num_workers=6, direct_running_mode="multi_processing")
  # options = PipelineOptions()
  with beam.Pipeline(options=options) as p:
    protos = (
        p
        | beam.io.tfrecordio.ReadFromTFRecord(
            input_fname,
            coder=beam.coders.ProtoCoder(dataset_pb2.Conformer().__class__))
        | beam.ParDo(bond_lengths.GetBondLengthDistribution())
        | beam.CombinePerKey(sum)
        #     | beam.ParDo(GroupBondTypes())
        #     | beam.GroupByKey()
        | beam.ParDo(BondDistToString())
        | beam.io.WriteToText(output_fname))
    print(protos)


def get_bond_length_distribution(unused_argv):
  """Scan Conformer protos to extract bond length distributions."""
  del unused_argv

  get_bond_length_distribution_inner(FLAGS.input, FLAGS.output)


if __name__ == "__main__":
  app.run(get_bond_length_distribution)
