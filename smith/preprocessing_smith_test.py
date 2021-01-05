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

import random
import tempfile

from absl import flags
import tensorflow.compat.v1 as tf

from smith import preprocessing_smith
from smith.bert import tokenization

FLAGS = flags.FLAGS


class PreprocessingSmithTest(tf.test.TestCase):

  def setUp(self):
    super(PreprocessingSmithTest, self).setUp()
    doc_one_text = (
        "I am in Dominick's for my dinner. OK, no problem. I am "
        "in Dominick's for my dinner which is the best dinner I have "
        "in my whole life.")
    doc_one_text = tokenization.convert_to_unicode(doc_one_text).strip()
    vocab_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "am", "in", "for",
        "my", "dinner", "ok", "no", "problem", "which", "is", "the", "be",
        "##s", "##t", ","
    ]
    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
      vocab_writer.write("".join([x + "\n" for x in vocab_tokens
                                 ]).encode("utf-8"))
      self.vocab_file = vocab_writer.name
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=self.vocab_file, do_lower_case=True)
    self.vocab_words = list(self.tokenizer.vocab.keys())
    self.rng = random.Random(12345)
    self.doc_one_tokens, _ = preprocessing_smith.get_smith_model_tokens(
        doc_one_text, self.tokenizer, [0, 0])
    self.max_sent_length_by_word = 20
    self.max_doc_length_by_sentence = 3
    self.greedy_sentence_filling = True
    self.max_predictions_per_seq = 0
    self.masked_lm_prob = 0

  def test_get_tokens_segment_ids_masks(self):
    (tokens_1, segment_ids_1, _, _, input_mask_1, _) = \
    preprocessing_smith.get_tokens_segment_ids_masks(
        max_sent_length_by_word=self.max_sent_length_by_word,
        max_doc_length_by_sentence=self.max_doc_length_by_sentence,
        doc_one_tokens=self.doc_one_tokens,
        masked_lm_prob=self.masked_lm_prob,
        max_predictions_per_seq=self.max_predictions_per_seq,
        vocab_words=self.vocab_words,
        rng=self.rng)
    self.assertEqual(tokens_1, [
        "[CLS]", "i", "am", "in", "[UNK]", "[UNK]", "[UNK]", "for", "my",
        "dinner", "[UNK]", "ok", ",", "no", "problem", "[UNK]", "[SEP]",
        "[SEP]", "[PAD]", "[PAD]", "[CLS]", "i", "am", "in", "[UNK]", "[UNK]",
        "[UNK]", "for", "my", "dinner", "which", "is", "the", "be", "##s",
        "##t", "dinner", "i", "[SEP]", "[SEP]", "[PAD]", "[PAD]", "[PAD]",
        "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]",
        "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]",
        "[PAD]"
    ])
    self.assertEqual(segment_ids_1, [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    self.assertEqual(input_mask_1, [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])


if __name__ == "__main__":
  tf.test.main()
