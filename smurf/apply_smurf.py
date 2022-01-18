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

r"""Produces videos of flow predictions from a directory of ordered images.

Run with a directory of images using apply_smurf \
  --data_dir=<directory with images> \
  --plot_dir=<directory to output results> \
  --checkpoint_dir=<directory to restore model from>
"""

from absl import app
from absl import flags

# pylint:disable=g-bad-import-order
import os
import sys
import gin

import tensorflow as tf

from smurf import smurf_flags  # pylint:disable=unused-import
from smurf import smurf_plotting
from smurf import smurf_evaluator

try:
  import cv2  # pylint:disable=g-import-not-at-top
except:  # pylint:disable=bare-except
  print('Missing cv2 dependency. Please install opencv-python.')



flags.DEFINE_string('data_dir', '', 'Directory with images to run on. Images '
                    'should be named numerically, e.g., 1.png, 2.png.')

FLAGS = flags.FLAGS


def get_image_iterator():
  """Iterate through images in the FLAGS.data_dir."""
  images = os.listdir(FLAGS.data_dir)
  images = [os.path.join(FLAGS.data_dir, i) for i in images]
  images = sorted(images, key=lambda x: int(os.path.basename(x).split('.')[0]))
  images = zip(images[:-1], images[1:])
  for image1, image2 in images:
    image1 = cv2.imread(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = tf.image.convert_image_dtype(image1, tf.float32)
    image2 = cv2.imread(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = tf.image.convert_image_dtype(image2, tf.float32)
    yield (image1, image2)


def main(unused_argv):
  if not FLAGS.plot_dir:
    raise ValueError('apply_smurf needs plot directory.')
  if not tf.io.gfile.exists(FLAGS.plot_dir):
    print('Making new plot directory', FLAGS.plot_dir)
    tf.io.gfile.makedirs(FLAGS.plot_dir)
  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  smurf = smurf_evaluator.build_network(batch_size=1)
  smurf.update_checkpoint_dir(FLAGS.checkpoint_dir)
  smurf.restore()
  for i, (image1, image2) in enumerate(get_image_iterator()):
    sys.stdout.write(':')
    sys.stdout.flush()
    flow_forward, occlusion, flow_backward = smurf.infer(
        image1, image2, input_height=FLAGS.height, input_width=FLAGS.width,
        infer_occlusion=True, infer_bw=True)
    occlusion = 1. - occlusion
    smurf_plotting.complete_paper_plot(plot_dir=FLAGS.plot_dir, index=i,
                                       image1=image1, image2=image2,
                                       flow_uv=flow_forward,
                                       ground_truth_flow_uv=None,
                                       flow_valid_occ=None,
                                       predicted_occlusion=occlusion,
                                       ground_truth_occlusion=None)

if __name__ == '__main__':
  app.run(main)
