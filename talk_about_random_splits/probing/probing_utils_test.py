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

# lint as: python3
import os

from absl.testing import absltest
import pandas as pd
from pandas.util import testing as pandas_testing

from talk_about_random_splits.probing import probing_utils


class UtilsTest(absltest.TestCase):

  def test_split_with_wasserstein(self):
    """Tests if the wasserstein split works."""
    texts = [
        # Cluster with token overlap.
        'aa bb cc',
        'aa bb xx',

        # Cluster with token overlap.
        'dd ef ff',
        'dd ef zz'
    ]
    result = probing_utils.split_with_wasserstein(
        texts, test_set_size=2, no_of_trials=2, min_df=1, leaf_size=1)

    self.assertLen(result, 2)
    self.assertLen(result[0], 2)
    self.assertLen(result[1], 2)

    # Unless the randomly picked test centroid is exactly in the middle of the 2
    # clusters, the algorithm should cluster the first 2 and last 2 texts into
    # the same group. We don't care about the order of the elements.
    def cluster_is_correct(cluster):
      return set(cluster) == set([0, 1]) or set(cluster) == set([2, 3])

    self.assertTrue(cluster_is_correct(result[0]))
    self.assertTrue(cluster_is_correct(result[1]))

  def test_split_by_length_threshold(self):
    """Tests if splitting by threshold works."""
    # Generates text with i times "i" as token.
    data = [(' '.join([str(i)] * i), i) for i in range(1, 101)]
    df = pd.DataFrame(data, columns=['text', 'text_len'])
    expected_length_threshold = 90
    expected_test_lengths = set(range(91, 101))
    expected_test_mask = [False] * 90 + [True] * 10

    length_threshold, test_lengths, test_mask = (
        probing_utils.split_by_length_threshold(df, 10))
    self.assertEqual(expected_length_threshold, length_threshold)
    self.assertSetEqual(expected_test_lengths, test_lengths)
    self.assertListEqual(expected_test_mask, test_mask.values.tolist())

  def test_split_by_length_threshold_no_valid_split(self):
    """Tests if exception raises when there is no proper split."""
    data = [('unimportant', i) for i in range(1, 10)]
    df = pd.DataFrame(data, columns=['text', 'text_len'])

    with self.assertRaises(ValueError):
      probing_utils.split_by_length_threshold(df, 100)

  def test_split_by_random_length(self):
    """Tests if splitting by random length works."""
    data = [('1', 1), ('2 2', 2), ('2 2', 2)]
    df = pd.DataFrame(data, columns=['text', 'text_len'])
    expected_test_lengths = set([1])
    expected_test_mask = [True, False, False]

    test_lengths, test_mask = probing_utils.split_by_random_length(df, 1)
    self.assertSetEqual(expected_test_lengths, test_lengths)
    self.assertListEqual(expected_test_mask, test_mask.values.tolist())

  def test_split_by_random_length_with_tolerance(self):
    """Tests if splitting by random length works when considering tolerance."""
    data = [('1', 1), ('2 2', 2), ('2 2', 2), ('2 2', 2), ('2 2', 2),
            ('2 2', 2)]
    df = pd.DataFrame(data, columns=['text', 'text_len'])
    expected_test_lengths = set([2])
    expected_test_mask = [False, True, True, True, True, True]

    test_lengths, test_mask = probing_utils.split_by_random_length(
        df, 4, size_tolerance=0.3)
    self.assertSetEqual(expected_test_lengths, test_lengths)
    self.assertListEqual(expected_test_mask, test_mask.values.tolist())

  def test_split_by_random_length_no_valid_split(self):
    """Tests if exception raises when there is no proper split."""
    # Generate 5 examples each for 10 lengths.
    data = []
    for i in range(1, 10):
      data += [('unimportant', i)] * 5
    df = pd.DataFrame(data, columns=['text', 'text_len'])

    # We won't find a length that has only 2 examples.
    with self.assertRaises(RuntimeError):
      probing_utils.split_by_random_length(df, 2)

  def test_get_target_word_to_sentence_mapping(self):
    """Tests if sentence mapping is correct."""
    target_words = set(['tw1', 'tw2', 'tw_doesnt_exist'])
    ignore_sentences = set()
    sent_tw1 = 'contains tw1'
    sent_tw2_1 = 'first with tw2'
    sent_tw2_2 = '2nd with tw2 in the middle'
    sent_tw1_tw2 = 'contains 2 target words , tw1 and tw2 must be ignored'
    expected = {'tw1': [sent_tw1], 'tw2': [sent_tw2_1, sent_tw2_2]}
    result = probing_utils.get_target_word_to_sentence_mapping(
        target_words, ignore_sentences,
        [sent_tw1, sent_tw2_1, sent_tw2_2, sent_tw1_tw2])
    self.assertDictEqual(expected, result)

  def test_get_target_word_to_sentence_mapping_ignore_sentences(self):
    """Tests if sentence mapping ignores existing sentences."""
    target_words = set(['tw'])
    sent1 = 'one sentence with tw'
    sent_to_be_ignored = 'ignore me although I contain tw here .'
    ignore_sentences = set([sent_to_be_ignored])
    expected = {
        'tw': [sent1],
    }
    result = probing_utils.get_target_word_to_sentence_mapping(
        target_words, ignore_sentences, [sent1, sent_to_be_ignored])
    self.assertDictEqual(expected, result)

  def _write_senteval_test_data(self, task_name, sub_dir, data):
    file_dir = os.path.join(absltest.get_default_test_tmpdir(), sub_dir,
                            'senteval_task')
    # May have been created by some previous test already.
    os.makedirs(file_dir)
    file_path = os.path.join(file_dir, task_name)

    file_content = ['\t'.join(line) for line in data]
    file_content = '\n'.join(file_content)

    with open(file_path, 'w') as f:
      f.write(file_content)

    return file_dir

  def test_read_senteval_data(self):
    """Test if reading with correct input works."""
    data = [
        ('tr', 'I', 'text train'),
        ('va', 'I', 'text validation'),
        ('te', 'O', 'text test'),
    ]
    task_name = 'word_content.txt'
    file_dir = self._write_senteval_test_data(task_name, 'read_senteval_data',
                                              data)

    expected = pd.DataFrame(data, columns=['set', 'target', 'text'])
    result = probing_utils.read_senteval_data(file_dir, task_name)
    pandas_testing.assert_frame_equal(expected, result)

  def test_read_senteval_data_with_quotes(self):
    """Test if reading with quotes works."""
    data = [
        ('tr', 'I', '"text train'),
        ('va', 'I', 'text "validation'),
        ('te', 'O', 'text test""'),
    ]
    task_name = 'word_content.txt'
    file_dir = self._write_senteval_test_data(task_name,
                                              'read_senteval_data_with_quotes',
                                              data)
    expected = pd.DataFrame(data, columns=['set', 'target', 'text'])
    result = probing_utils.read_senteval_data(file_dir, task_name)
    pandas_testing.assert_frame_equal(expected, result)


if __name__ == '__main__':
  absltest.main()
