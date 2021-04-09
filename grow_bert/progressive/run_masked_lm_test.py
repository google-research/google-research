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

"""Tests for grow_bert.progressive.run_masked_lm."""
import json
import os

from absl.testing import flagsaver
import numpy as np
import tensorflow as tf

from grow_bert.progressive import run_masked_lm
from official.common import flags as tfm_flags

tfm_flags.define_flags()


def _create_fake_dataset(output_path, seq_lengths, num_masked_tokens,
                         max_seq_length):
  """Creates a fake dataset from the given sequence lengths."""
  writer = tf.io.TFRecordWriter(output_path)

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  for seq_length in seq_lengths:
    features = {}
    padding = np.zeros(shape=(max_seq_length - seq_length), dtype=np.int32)
    input_ids = np.random.randint(low=1, high=100, size=(seq_length))
    features['new_input_ids'] = create_int_feature(
        np.concatenate((input_ids, padding)))
    features['new_input_masks'] = create_int_feature(
        np.concatenate((np.ones_like(input_ids), padding)))
    features['new_segment_ids'] = create_int_feature(
        np.concatenate((np.ones_like(input_ids), padding)))
    features['new_input_positions'] = create_int_feature(
        np.concatenate((np.ones_like(input_ids), padding)))
    features['masked_lm_positions'] = create_int_feature(
        np.random.randint(60, size=(num_masked_tokens), dtype=np.int64))
    features['masked_lm_ids'] = create_int_feature(
        np.random.randint(100, size=(num_masked_tokens), dtype=np.int64))
    features['masked_input_ids'] = create_int_feature(
        np.random.randint(100, size=(num_masked_tokens), dtype=np.int64))
    features['masked_segment_ids'] = create_int_feature(
        np.zeros(shape=(num_masked_tokens), dtype=np.int64))
    features['masked_lm_weights'] = create_float_feature(
        np.ones((num_masked_tokens,), dtype=np.float32))

    features['next_sentence_labels'] = create_int_feature(np.array([0]))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class RunMaskedLmTest(tf.test.TestCase):

  def test_intergration(self):
    model_dir = self.get_temp_dir()
    self.input_path = os.path.join(self.get_temp_dir(), 'train.tf_record')
    _create_fake_dataset(
        self.input_path,
        seq_lengths=[128] * 30,
        num_masked_tokens=20,
        max_seq_length=128)
    test_config = {
        'trainer': {
            'train_steps': 5,
            'checkpoint_interval': 5,
            'steps_per_loop': 5,
            'summary_interval': 5,
            'progressive': {
                'stage_list': [{
                    'override_train_data': None,
                    'override_valid_data': None,
                }]
            },
        },
        'task': {
            'model': {
                'encoder': {
                    'bert': {
                        'num_layers': 1
                    }
                }
            },
            'small_train_data': {
                'input_path': self.input_path,
                'global_batch_size': 2,
                'max_predictions_per_seq': 20,
                'seq_length': 128,
            },
        },
    }
    flags_dict = dict(
        mode='train',
        model_dir=model_dir,
        params_override=json.dumps(test_config))
    with flagsaver.flagsaver(**flags_dict):
      run_masked_lm.main(None)
    self.assertNotEmpty(
        tf.io.gfile.glob(os.path.join(model_dir, 'params.yaml')))
    self.assertNotEmpty(tf.io.gfile.glob(os.path.join(model_dir, 'checkpoint')))


if __name__ == '__main__':
  tf.test.main()
