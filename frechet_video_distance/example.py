# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Example code that computes FVD for some empty frames.

The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from frechet_video_distance import frechet_video_distance as fvd

# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 16
VIDEO_LENGTH = 15


def main(argv):
  del argv
  with tf.Graph().as_default():

    first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
    second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]
                                  ) * 255

    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(first_set_of_videos,
                                                (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(second_set_of_videos,
                                                (224, 224))))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("FVD is: %.2f." % sess.run(result))


if __name__ == "__main__":
  tf.app.run(main)
