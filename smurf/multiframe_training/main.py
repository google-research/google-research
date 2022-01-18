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

"""SMURF multi-frame self-supervision data generation pipeline."""

import os
from typing import Generator, Sequence, Tuple

from absl import app
from absl import flags
import tensorflow as tf

from smurf.data import simple_dataset
from smurf.data_conversion_scripts import conversion_utils
from smurf.multiframe_training import smurf_multi_frame_fusion

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'worker', 0, 'Worker id. This specifies which shard of the'
    'the total_shards will be created.')
flags.DEFINE_integer('total_shards', 800, 'How many shards to produce.')
flags.DEFINE_string('input_dir', '', 'Directory holding the input tfrecord '
                    'files.')
flags.DEFINE_string('output_dir', '', 'Directory where the new tfrecord files '
                    'will be saved.')
flags.DEFINE_bool(
    'boundaries_occluded', False, 'If false pixels moving outside'
    ' the image frame will be considered non-occluded.')
flags.DEFINE_string(
    'ckpt_dir', '', 'Checkpoint directory of the model that '
    'should be used to compute the required flow fields.')
flags.DEFINE_integer('height', None, 'Height used for the flow computation.')
flags.DEFINE_integer('width', None, 'Which used for the flow computation.')


def triplet_generator(
):
  """A generator that yields image triplets."""
  dataset_manager = simple_dataset.SimpleDataset()
  dataset = dataset_manager.make_dataset(
      path=FLAGS.input_dir,
      mode='multiframe',
  )
  for triplet in iter(dataset):
    if triplet['images'].shape[0] < 3:
      continue
    yield triplet['images']


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  worker_id = FLAGS.worker
  number_of_shards = FLAGS.total_shards

  # Create the directory to save the tfrecords.
  output_directory = FLAGS.output_dir
  if not os.path.exists(output_directory):
    os.mkdir(output_directory)

  out_filenames = conversion_utils.generate_sharded_filenames(
      os.path.join(output_directory,
                   'smurf_multriframe@{}'.format(number_of_shards)))

  infer_flow = smurf_multi_frame_fusion.get_flow_inference_function(
      FLAGS.ckpt_dir, FLAGS.height, FLAGS.width)
  infer_mask = smurf_multi_frame_fusion.get_occlusion_inference_function(
      FLAGS.boundaries_occluded)

  with tf.io.TFRecordWriter(out_filenames[worker_id]) as record_writer:
    for counter, images in enumerate(triplet_generator()):
      # Skip if triplet will be processed by another worker.
      if counter % number_of_shards != worker_id:
        continue

      flow, mask = smurf_multi_frame_fusion.run_multiframe_fusion(
          images, infer_flow, infer_mask)
      example = smurf_multi_frame_fusion.create_output_sequence_example(
          images, flow, mask)
      record_writer.write(example.SerializeToString())


if __name__ == '__main__':
  app.run(main)
