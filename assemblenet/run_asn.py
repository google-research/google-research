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

"""Script to run AssembleNets without object input."""
import json

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf  # tf

from assemblenet import assemblenet
from assemblenet import assemblenet_plus
from assemblenet import assemblenet_plus_lite

from assemblenet import model_structures


flags.DEFINE_string('precision', 'float32',
                    'Precision to use; one of: {bfloat16, float32}.')
flags.DEFINE_integer('num_frames', 64, 'Number of frames to use.')

flags.DEFINE_integer('num_classes', 157,
                     'Number of classes. 157 is for Charades')


flags.DEFINE_string('assemblenet_mode', 'assemblenet',
                    '"assemblenet" or "assemblenet_plus" or "assemblenet_plus_lite"')  # pylint: disable=line-too-long

flags.DEFINE_string('model_structure', '[-1,1]',
                    'AssembleNet model structure in the string format.')
flags.DEFINE_string(
    'model_edge_weights', '[]',
    'AssembleNet model structure connection weights in the string format.')

flags.DEFINE_string('attention_mode', None, '"peer" or "self" or None')

flags.DEFINE_float('dropout_keep_prob', None, 'Keep ratio for dropout.')
flags.DEFINE_bool(
    'max_pool_preditions', False,
    'Use max-pooling on predictions instead of mean pooling on features. It helps if you have more than 32 frames.')  # pylint: disable=line-too-long

flags.DEFINE_bool('use_object_input', False,
                  'Whether to use object input for AssembleNet++ or not')  # pylint: disable=line-too-long
flags.DEFINE_integer('num_object_classes', 151,
                     'Number of object classes, when using object inputs. 151 is for ADE-20k')  # pylint: disable=line-too-long


FLAGS = flags.FLAGS


def main(_):
  # Create model.

  batch_size = 2
  image_size = 256

  vid_placeholder = tf.placeholder(tf.float32,
                                   (batch_size, FLAGS.num_frames, image_size, image_size, 3))  # pylint: disable=line-too-long

  if FLAGS.assemblenet_mode == 'assemblenet_plus_lite':
    FLAGS.model_structure = json.dumps(model_structures.asnp_lite_structure)
    FLAGS.model_edge_weights = json.dumps(model_structures.asnp_lite_structure_weights)  # pylint: disable=line-too-long

    network = assemblenet_plus_lite.assemblenet_plus_lite(
        num_layers=[3, 5, 11, 7],
        num_classes=FLAGS.num_classes,
        data_format='channels_last')
  else:
    vid_placeholder = tf.reshape(vid_placeholder,
                                 [batch_size*FLAGS.num_frames, image_size, image_size, 3])  # pylint: disable=line-too-long

    if FLAGS.assemblenet_mode == 'assemblenet_plus':
      # Here, we are using model_structures.asn50_structure for AssembleNet++
      # instead of full_asnp50_structure. By using asn50_structure, it
      # essentially becomes AssembleNet++ without objects, only requiring RGB
      # inputs (and optical flow to be computed inside the model).
      FLAGS.model_structure = json.dumps(model_structures.asn50_structure)
      FLAGS.model_edge_weights = json.dumps(model_structures.asn_structure_weights)  # pylint: disable=line-too-long

      network = assemblenet_plus.assemblenet_plus(
          assemblenet_depth=50,
          num_classes=FLAGS.num_classes,
          data_format='channels_last')
    else:
      FLAGS.model_structure = json.dumps(model_structures.asn50_structure)
      FLAGS.model_edge_weights = json.dumps(model_structures.asn_structure_weights)  # pylint: disable=line-too-long

      network = assemblenet.assemblenet_v1(
          assemblenet_depth=50,
          num_classes=FLAGS.num_classes,
          data_format='channels_last')

  # The model function takes the inputs and is_training.
  outputs = network(vid_placeholder, False)

  with tf.Session() as sess:
    # Generate a random video to run on.
    # This should be replaced by a real video.
    vid = np.random.rand(*vid_placeholder.shape)
    sess.run(tf.global_variables_initializer())
    logits = sess.run(outputs, feed_dict={vid_placeholder: vid})
    print(logits)
    print(np.argmax(logits, axis=1))


if __name__ == '__main__':
  app.run(main)
