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

"""Extract images from summaries.

During training, we save image summaries. This extracts the images and dumps
them in a log directory.

Flags: target log dir,
dump dir

We would want, for each dataset:
samples from all of the methods

each dataset should have a folder
images in the folder should have the method name + sample #, then we can
combine the images together or do that in latex/generate latex code to do that
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf


from tensorboard.backend.event_processing import event_accumulator
tf.enable_eager_execution()


flags.DEFINE_string('summary_dir', None,
                    'Directory where summaries are located.')
flags.DEFINE_string('target_file', None,
                    'Target file for the image.')
FLAGS = flags.FLAGS


def rescale(image):
  """Rescale to full [0, 255] range."""
  image = tf.cast(image, tf.float32)
  image = ((image - tf.reduce_min(image)) /
           (tf.reduce_max(image) - tf.reduce_min(image)) * 255.)
  return image


def get_image(path):
  """Get images from event file."""
  events = event_accumulator.EventAccumulator(path)
  events.Reload()
  images = []
  for image_tag in events.Tags()['images']:
    image_event = events.Images(image_tag)[-1]  # get the last step
    images.append(rescale(
        tf.image.decode_image(image_event.encoded_image_string)))

  # Concat images into a row
  return tf.concat(images, axis=1).numpy().astype(np.uint8)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  image = get_image(FLAGS.summary_dir)
  with tf.gfile.Open(FLAGS.target_file, 'w') as out:
    np.save(out, image)


if __name__ == '__main__':
  app.run(main)
