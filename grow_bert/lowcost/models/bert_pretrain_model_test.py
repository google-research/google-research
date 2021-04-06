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

"""Tests for BERT trainer network."""

import tensorflow as tf

from grow_bert.lowcost.models import bert_pretrain_model
from grow_bert.lowcost.models import transformer_encoder


class BertPretrainerTest(tf.test.TestCase):

  def test_bert_pretrainer(self):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    sequence_length = 512
    test_network = transformer_encoder.TransformerEncoder(
        vocab_size=vocab_size,
        num_layers=2,
        max_sequence_length=sequence_length)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    predict_length = 2
    dummy_inputs = dict(
        input_mask=tf.zeros((1, sequence_length), dtype=tf.int32),
        input_positions=tf.zeros((1, sequence_length), dtype=tf.int32),
        input_type_ids=tf.zeros((1, sequence_length), dtype=tf.int32),
        input_word_ids=tf.zeros((1, sequence_length), dtype=tf.int32),
        masked_lm_positions=tf.zeros((1, predict_length), dtype=tf.int32),
        masked_input_ids=tf.zeros((1, predict_length), dtype=tf.int32),
        masked_segment_ids=tf.zeros((1, predict_length), dtype=tf.int32),
        masked_lm_weights=tf.zeros((1, predict_length), dtype=tf.float32))
    _ = test_network(dummy_inputs)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_pretrain_model.BertPretrainModel(test_network)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    outputs = bert_trainer_model(dummy_inputs)

    # Validate that the outputs are of the expected shape.
    expected_lm_shape = [1, predict_length, vocab_size]
    self.assertAllEqual(expected_lm_shape,
                        outputs['mlm_logits'].shape.as_list())


if __name__ == '__main__':
  tf.test.main()
