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

r"""Reads in one pair of fastq files and creates a single sstable.

Notes for the Aptamers Manuscript:
The usage description below is accurate. For the flags, the values were pulled
from learning/config.py (for example: MIN_BASE_QUALITY = 20) and used when
this module was called from our pipeline harness.
This module would be called one time for each pair of fastq files.


Usage:
 (for actual run, replaced [dir] with an actual directory)

run xxx/fastq_to_sstable \
 -- --fastq1=[dir]/R1-TAAGGCGA_S1_R1_001.fastq \
 --fastq2=[dir]/R1-TAAGGCGA_S1_R2_001.fastq \
 --measurement_id=1 \
 --output_name=[dir]/count_tables/2016-07-28T21-16-29.024029/R1.sstable
 --alsologtostderr

"""
import collections
import gzip


# Google Internal
import app
import flags
import gfile
import sstable


from ..util import measurement_pb2
from ..preprocess import utils


class Error(Exception):
  pass


FLAGS = flags.FLAGS

flags.DEFINE_string("fastq1", None, "Path to the first fastq file.")
flags.DEFINE_string("fastq2", None,
                    "Path to the second fastq file for paired-end "
                    "sequencing, or None for single end")
flags.DEFINE_integer("measurement_id", None,
                     "The measurement data set ID for this fastq pair, from "
                     "the experiment proto")
flags.DEFINE_integer("sequence_length", 40,
                     "Expected length of each sequence read")
flags.DEFINE_string("output_name",
                    "xxx"
                    "aptitude", "Path and name for the output sstable")
flags.DEFINE_integer("base_qual_threshold", 20, "integer indicating the "
                     "lowest quality (on scale from 0 to 40) for a single "
                     "base to be considered acceptable")
flags.DEFINE_integer("bad_base_threshold", 5, "integer indicating the maximum "
                     "number of bad bases before a read is bad quality")
flags.DEFINE_float("avg_qual_threshold", 30.0, "float indicating the mean "
                   "quality across the whole read to be considered good")
flags.DEFINE_integer("num_reads", 99999999999, "The number of reads to include "
                     "from each fastq file.")


def main(unused_argv):

  # read in the fastq file(s)
  make_zeros = lambda: [0]
  count_table = collections.defaultdict(make_zeros)

  # The readahead speeds up this process about 10x
  # While it is technically possible to pre-pend '/gzip' to automagically
  # un-gzip files, this doesn't play nice with the readahead prepend.
  # See b/63985459 for more information
  # As a result, we do not pre-pend '/gzip' anymore and instead do this
  # double loop.
  with gfile.FastGFile("/readahead/256M/" + FLAGS.fastq1) as gzinput1:
    with gzip.GzipFile(mode="rb", fileobj=gzinput1) as in1:
      if FLAGS.fastq2:
        with gfile.FastGFile("/readahead/256M" + FLAGS.fastq2) as gzinput2:
          with gzip.GzipFile(mode="rb", fileobj=gzinput2) as in2:
            stats = utils.read_one_fastq_pair(
                in1,
                in2,
                count_table,
                0,
                expected_sequence_length=FLAGS.sequence_length,
                base_qual_threshold=FLAGS.base_qual_threshold,
                bad_base_threshold=FLAGS.bad_base_threshold,
                avg_qual_threshold=FLAGS.avg_qual_threshold,
                num_reads=FLAGS.num_reads)
      else:
        stats = utils.read_one_fastq_pair(
            in1,
            None,
            count_table,
            0,
            expected_sequence_length=FLAGS.sequence_length,
            base_qual_threshold=FLAGS.base_qual_threshold,
            bad_base_threshold=FLAGS.bad_base_threshold,
            avg_qual_threshold=FLAGS.avg_qual_threshold,
            num_reads=FLAGS.num_reads)

  # write out the statistics
  stats.fastq_read1_name = FLAGS.fastq1
  stats.fastq_read2_name = FLAGS.fastq2
  with gfile.FastGFile(FLAGS.output_name + ".statistics.pbtxt", "w") as out:
    out.write(str(stats))

  # write to an SSTable of measurement protos
  with sstable.SortingBuilder(FLAGS.output_name) as builder:
    for sequence, counts in count_table.iteritems():
      key = sequence
      proto = measurement_pb2.Measurement()
      proto.counts[FLAGS.measurement_id] = counts[0]
      builder.Add(key, proto.SerializeToString())


if __name__ == "__main__":
  # mark required flags here so unit test doesn't need the sys.argv.extend hack
  flags.mark_flag_as_required("fastq1")
  flags.mark_flag_as_required("measurement_id")

  app.run()
