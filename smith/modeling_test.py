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

import json
import tempfile

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from smith import constants
from smith import experiment_config_pb2
from smith import modeling

FLAGS = flags.FLAGS


class ModelingTest(tf.test.TestCase):

  def setUp(self):
    super(ModelingTest, self).setUp()
    bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "type_vocab_size": 2,
        "vocab_size": 9
    }
    with tempfile.NamedTemporaryFile(delete=False) as bert_config_writer:
      bert_config_writer.write(json.dumps(bert_config).encode("utf-8"))
    # Note that in practice the bert_config_file and doc_bert_config_file can
    # be different.
    bert_config_file = bert_config_writer.name
    doc_bert_config_file = bert_config_writer.name

    # Construct a dual_encoder_config for testing purpose.
    dual_encoder_config = experiment_config_pb2.DualEncoderConfig()
    encoder_config = dual_encoder_config.encoder_config
    encoder_config.model_name = constants.MODEL_NAME_SMITH_DUAL_ENCODER
    encoder_config.max_seq_length = 6
    encoder_config.max_sent_length_by_word = 2
    encoder_config.max_doc_length_by_sentence = 3
    encoder_config.loop_sent_number_per_doc = 3
    encoder_config.max_predictions_per_seq = 1
    encoder_config.use_masked_sentence_lm_loss = True
    encoder_config.max_masked_sent_per_doc = 2
    encoder_config.bert_config_file = bert_config_file
    encoder_config.doc_bert_config_file = doc_bert_config_file
    # Set train_batch_size and eval_batch_size for the batch_size_static used
    # in the build_smith_ca function.
    train_eval_config = dual_encoder_config.train_eval_config
    train_eval_config.train_batch_size = 1
    train_eval_config.eval_batch_size = 1
    self.dual_encoder_config = dual_encoder_config
    self.train_mode = constants.TRAIN_MODE_JOINT_TRAIN

    self.model_fn = modeling.model_fn_builder(
        dual_encoder_config=dual_encoder_config,
        train_mode=self.train_mode,
        learning_rate=1e-5,
        num_train_steps=100000,
        num_warmup_steps=500,
        use_tpu=False,
        use_one_hot_embeddings=False,
        debugging=True)

    self.features = {
        "input_ids_1": tf.constant([[0, 5, 5, 7, 1, 1]], dtype=tf.int32),
        "input_mask_1": tf.constant([[1, 1, 1, 1, 1, 1]], dtype=tf.int32),
        "masked_lm_positions_1": tf.constant([[3]], dtype=tf.int32),
        "masked_lm_ids_1": tf.constant([[5]], dtype=tf.int32),
        "masked_lm_weights_1": tf.constant([[1.0]], dtype=tf.float32),
        "input_ids_2": tf.constant([[0, 4, 4, 7, 1, 1]], dtype=tf.int32),
        "input_mask_2": tf.constant([[1, 1, 1, 1, 1, 1]], dtype=tf.int32),
        "masked_lm_positions_2": tf.constant([[3]], dtype=tf.int32),
        "masked_lm_ids_2": tf.constant([[4]], dtype=tf.int32),
        "masked_lm_weights_2": tf.constant([[1.0]], dtype=tf.float32),
        "documents_match_labels": tf.constant([[1.0]], dtype=tf.float32)
    }

  def test_build_smith_dual_encoder(self):
    masked_lm_positions_1 = tf.constant([[0, 2, 5]], dtype=tf.int32)
    masked_lm_ids_1 = tf.constant([[0, 5, 1]], dtype=tf.int32)
    masked_lm_weights_1 = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)
    masked_lm_positions_2 = tf.constant([[0, 2, 5]], dtype=tf.int32)
    masked_lm_ids_2 = tf.constant([[0, 5, 1]], dtype=tf.int32)
    masked_lm_weights_2 = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)

    (masked_lm_loss_1, _,
          masked_lm_example_loss_1, _,
          _, _,
          masked_sent_lm_loss_1, _,
          _, _,
          _, _, sequence_encoding_1,
          _, _,
          _, _,
          _, siamese_loss, siamese_example_loss,
          siamese_logits) = \
        modeling.build_smith_dual_encoder(
            dual_encoder_config=self.dual_encoder_config,
            train_mode=self.train_mode,
            is_training=True,
            input_ids_1=self.features["input_ids_1"],
            input_mask_1=self.features["input_mask_1"],
            masked_lm_positions_1=masked_lm_positions_1,
            masked_lm_ids_1=masked_lm_ids_1,
            masked_lm_weights_1=masked_lm_weights_1,
            input_ids_2=self.features["input_ids_2"],
            input_mask_2=self.features["input_mask_2"],
            masked_lm_positions_2=masked_lm_positions_2,
            masked_lm_ids_2=masked_lm_ids_2,
            masked_lm_weights_2=masked_lm_weights_2,
            use_one_hot_embeddings=False,
            documents_match_labels=self.features["documents_match_labels"])
    with tf.Session() as sess:
      sess.run([tf.global_variables_initializer()])
      result_numpy = sess.run([
          masked_lm_loss_1, masked_lm_example_loss_1, sequence_encoding_1,
          siamese_loss, siamese_example_loss, siamese_logits,
          masked_sent_lm_loss_1
      ])
      self.assertEqual(result_numpy[0].shape, ())
      self.assertDTypeEqual(result_numpy[0], np.float32)

      self.assertEqual(result_numpy[1].shape, (1, 3))
      self.assertDTypeEqual(result_numpy[1], np.float32)

      self.assertEqual(result_numpy[2].shape, (1, 16))
      self.assertDTypeEqual(result_numpy[2], np.float32)

      self.assertEqual(result_numpy[3].shape, ())
      self.assertDTypeEqual(result_numpy[3], np.float32)

      self.assertEqual(result_numpy[4].shape, (1,))
      self.assertDTypeEqual(result_numpy[4], np.float32)

      self.assertEqual(result_numpy[5].shape, (1,))
      self.assertDTypeEqual(result_numpy[5], np.float32)

      self.assertEqual(result_numpy[6].shape, ())
      self.assertDTypeEqual(result_numpy[6], np.float32)

  def test_model_fn_builder_train(self):
    self.model_fn(
        features=self.features,
        labels=None,
        mode=tf.estimator.ModeKeys.TRAIN,
        params=None)

  def test_model_fn_builder_eval(self):
    self.model_fn(
        features=self.features,
        labels=None,
        mode=tf.estimator.ModeKeys.EVAL,
        params=None)

  def test_model_fn_builder_predict(self):
    self.model_fn(
        features=self.features,
        labels=None,
        mode=tf.estimator.ModeKeys.PREDICT,
        params=None)


if __name__ == "__main__":
  tf.test.main()
