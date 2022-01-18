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

"""Script to run AssembleNet++ with objects."""
import json

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf  # tf

from assemblenet import assemblenet_plus

from assemblenet import model_structures


flags.DEFINE_string('precision', 'float32',
                    'Precision to use; one of: {bfloat16, float32}.')
flags.DEFINE_integer('num_frames', 64, 'Number of frames to use.')

flags.DEFINE_integer('num_classes', 157,
                     'Number of classes. 157 is for Charades')


flags.DEFINE_string('assemblenet_mode', 'assemblenet_plus',
                    '"assemblenet" or "assemblenet_plus" or "assemblenet_plus_lite"')  # pylint: disable=line-too-long

flags.DEFINE_string('model_structure', '[-1,1]',
                    'AssembleNet model structure in the string format.')
flags.DEFINE_string(
    'model_edge_weights', '[]',
    'AssembleNet model structure connection weights in the string format.')

flags.DEFINE_string('attention_mode', 'peer', '"peer" or "self" or None')

flags.DEFINE_float('dropout_keep_prob', None, 'Keep ratio for dropout.')
flags.DEFINE_bool(
    'max_pool_preditions', False,
    'Use max-pooling on predictions instead of mean pooling on features. It helps if you have more than 32 frames.')  # pylint: disable=line-too-long

flags.DEFINE_bool('use_object_input', True,
                  'Whether to use object input for AssembleNet++ or not')  # pylint: disable=line-too-long
flags.DEFINE_integer('num_object_classes', 151,
                     'Number of object classes, when using object inputs. 151 is for ADE-20k')  # pylint: disable=line-too-long


FLAGS = flags.FLAGS


def main(_):
  # Create model.

  batch_size = 2
  image_size = 256

  vid_placeholder = tf.placeholder(tf.float32,
                                   (batch_size*FLAGS.num_frames, image_size, image_size, 3))  # pylint: disable=line-too-long
  object_placeholder = tf.placeholder(tf.float32,
                                      (batch_size*FLAGS.num_frames, image_size, image_size, FLAGS.num_object_classes))  # pylint: disable=line-too-long
  input_placeholder = (vid_placeholder, object_placeholder)

  # We are using the full_asnp50_structure, since we feed both video and object.
  FLAGS.model_structure = json.dumps(model_structures.full_asnp50_structure)  # pylint: disable=line-too-long
  FLAGS.model_edge_weights = json.dumps(model_structures.full_asnp_structure_weights)  # pylint: disable=line-too-long

  network = assemblenet_plus.assemblenet_plus(
      assemblenet_depth=50,
      num_classes=FLAGS.num_classes,
      data_format='channels_last')

  # The model function takes the inputs and is_training.
  outputs = network(input_placeholder, False)

  with tf.Session() as sess:
    # Generate a random video to run on.
    # This should be replaced by a real video.
    sess.run(tf.global_variables_initializer())
    vid = np.random.rand(*vid_placeholder.shape)
    obj = np.random.rand(*object_placeholder.shape)
    logits = sess.run(outputs, feed_dict={input_placeholder: (vid, obj)})
    print(logits)
    print(np.argmax(logits, axis=1))


if __name__ == '__main__':
  app.run(main)
