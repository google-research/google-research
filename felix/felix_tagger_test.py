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

"""Tests for felix.felix_tagger."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from official.nlp.bert import configs
from official.nlp.modeling import networks

from felix import felix_tagger


class FelixTaggerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('training', False, False, True),
      ('training_with_pointing', True, False, True),
      ('training_with_pointing_transformer', True, True, True),
      ('testing', False, False, False),
      ('testing_with_pointing', True, False, False),
      ('testing_with_pointing_transformer', True, True, False))
  def test_forward_pass(self,
                        use_pointing=False,
                        query_transformer=False,
                        is_training=True):
    """Randomly generate and run different configuarations for Felix Tagger."""
    # Ensures reproducibility.

    # Setup.
    sequence_length = 7
    vocab_size = 11
    bert_hidden_size = 13
    bert_num_hidden_layers = 1
    bert_num_attention_heads = 1
    bert_intermediate_size = 4
    bert_type_vocab_size = 2
    bert_max_position_embeddings = sequence_length
    bert_encoder = networks.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=bert_hidden_size,
        num_layers=bert_num_hidden_layers,
        num_attention_heads=bert_num_attention_heads,
        intermediate_size=bert_intermediate_size,
        sequence_length=sequence_length,
        max_sequence_length=bert_max_position_embeddings,
        type_vocab_size=bert_type_vocab_size)
    bert_config = configs.BertConfig(
        vocab_size,
        hidden_size=bert_hidden_size,
        num_hidden_layers=bert_num_hidden_layers,
        num_attention_heads=bert_num_attention_heads,
        intermediate_size=bert_intermediate_size,
        type_vocab_size=bert_type_vocab_size,
        max_position_embeddings=bert_max_position_embeddings)
    batch_size = 17
    edit_tags_size = 19
    bert_config.num_classes = edit_tags_size
    bert_config.query_size = 23
    bert_config.query_transformer = query_transformer

    tagger = felix_tagger.FelixTagger(
        bert_encoder,
        bert_config=bert_config,
        seq_length=sequence_length,
        use_pointing=use_pointing,
        is_training=is_training)

    # Create inputs.
    np.random.seed(42)
    input_word_ids = np.random.randint(
        vocab_size - 1, size=(batch_size, sequence_length))
    input_mask = np.random.randint(1, size=(batch_size, sequence_length))
    input_type_ids = np.ones((batch_size, sequence_length))
    edit_tags = np.random.randint(
        edit_tags_size - 2, size=(batch_size, sequence_length))

    # Run the model.
    if is_training:
      output = tagger([input_word_ids, input_type_ids, input_mask, edit_tags])
    else:
      output = tagger([input_word_ids, input_type_ids, input_mask])

    # Check output shapes.
    if use_pointing:
      tag_logits, pointing_logits = output
      self.assertEqual(pointing_logits.shape,
                       (batch_size, sequence_length, sequence_length))
    else:
      tag_logits = output[0]
    self.assertEqual(tag_logits.shape,
                     (batch_size, sequence_length, edit_tags_size))


if __name__ == '__main__':
  absltest.main()
