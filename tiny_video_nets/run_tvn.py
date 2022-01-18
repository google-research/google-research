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

"""Script to run TVNs."""
import json

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf  # tf

from tiny_video_nets import found_tvns
from tiny_video_nets import tiny_video_net

flags.DEFINE_enum('model_name', 'tvn1',
                  ['tvn1', 'tvn2', 'tvn3', 'tvn4',
                   'tvn_mobile_1', 'tvn_mobile_2'], 'Name of TVN to use.')
flags.DEFINE_integer('num_classes', 157,
                     'The number of classes in the dataset.')

FLAGS = flags.FLAGS


def main(_):
  # Create model.
  if FLAGS.model_name == 'tvn1':
    tvn = found_tvns.TVN1
  elif FLAGS.model_name == 'tvn2':
    tvn = found_tvns.TVN2
  elif FLAGS.model_name == 'tvn3':
    tvn = found_tvns.TVN3
  elif FLAGS.model_name == 'tvn4':
    tvn = found_tvns.TVN4
  elif FLAGS.model_name == 'tvn_mobile_1':
    tvn = found_tvns.TVN_MOBILE_1
  elif FLAGS.model_name == 'tvn_mobile_2':
    tvn = found_tvns.TVN_MOBILE_2

  batch_size = 2
  if 'image_size' in tvn:
    image_size = tvn['image_size']
    num_frames = tvn['num_frames']
  else:
    image_size = tvn['input_streams'][0]['image_size']
    num_frames = tvn['input_streams'][0]['num_frames']

  vid_placeholder = tf.placeholder(tf.float32,
                                   shape=(batch_size, num_frames,
                                          image_size, image_size, 3))
  vid_placeholder = tf.reshape(vid_placeholder, (batch_size * num_frames,
                                                 image_size, image_size, 3))

  model = tiny_video_net.tiny_video_net(json.dumps(tvn),
                                        num_classes=FLAGS.num_classes,
                                        num_frames=num_frames,
                                        data_format='channels_last',
                                        dropout_keep_prob=0.5,
                                        get_representation=False,
                                        max_pool_predictions=False)
  # The model function takes the inputs and is_training.
  outputs = model(vid_placeholder, False)

  with tf.Session() as sess:
    # Generate a random video to run on.
    # This should be replaced by a real video.
    vid = np.random.rand(*vid_placeholder.shape)
    sess.run(tf.global_variables_initializer())
    out = sess.run(outputs, feed_dict={vid_placeholder: vid})
  logits = out['logits']
  print(np.argmax(logits, axis=1))


if __name__ == '__main__':
  app.run(main)
