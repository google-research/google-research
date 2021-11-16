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

"""Tests for align_transforms."""

import gin
import tensorflow as tf

from dedal import vocabulary
from dedal.data import align_transforms


class AlignTransformsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    tf.random.set_seed(0)
    self.vocab = vocabulary.alternative
    self.sampler = vocabulary.Sampler(vocab=self.vocab)

  def test_project_msa_rows(self):
    token = '-'
    seq1 = tf.convert_to_tensor(self.vocab.encode('XX--X-XX-X'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('Y-Y-Y-YYYY'), tf.int32)
    mc = tf.convert_to_tensor([False] + 8 * [True] + [False], tf.bool)

    proj_msa_fn = align_transforms.ProjectMSARows(token=token, vocab=self.vocab)

    out1, out2 = proj_msa_fn.call(seq1, seq2)
    self.assertAllEqual(out1, self.vocab.encode('XX-XXX-X'))
    self.assertAllEqual(out2, self.vocab.encode('Y-YYYYYY'))

    out1, out2 = proj_msa_fn.call(seq1, seq2, mc)
    self.assertAllEqual(out1, self.vocab.encode('X-XXX-'))
    self.assertAllEqual(out2, self.vocab.encode('-YYYYY'))

  def test_pid1(self):
    token = '-'
    pid_fn = align_transforms.PID(definition=1, token=token, vocab=self.vocab)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 1.0)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('YXXXYXYX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.5)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXYYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('YXYXXYYX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.0)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('---X-XXY'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.8)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('---X-YYX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.2)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY----'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('---X-YYXYXXX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.2)

  def test_pid3(self):
    token = '-'
    pid_fn = align_transforms.PID(definition=3, token=token, vocab=self.vocab)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 1.0)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('YXXXYXYX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.5)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXYYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('YXYXXYYX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.0)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('---X-XXY'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 1.0)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('---X-YYX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.25)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XYXXYXXY----'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('---X-YYXYXXX'), tf.int32)
    pid = pid_fn.call(seq1, seq2)
    self.assertEqual(pid, 0.125)

  def test_create_alignment_targets(self):
    gap_token = '-'
    n_prepend_tokens = 0
    align_fn = align_transforms.CreateAlignmentTargets(
        gap_token=gap_token,
        n_prepend_tokens=n_prepend_tokens,
        vocab=self.vocab)

    seq1 = tf.convert_to_tensor(self.vocab.encode('XX-XXXX'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('YYYY-YY'), tf.int32)
    expected_output = tf.convert_to_tensor([[1, 2, 2, 3, 4, 5, 6],
                                            [1, 2, 3, 4, 4, 5, 6],
                                            [0, 1, 4, 2, 6, 3, 1]], tf.int32)
    output = align_fn.call(seq1, seq2)
    self.assertAllEqual(output, expected_output)

    seq1 = tf.convert_to_tensor(self.vocab.encode('--XXXXXX'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('YYYY-YY-'), tf.int32)
    expected_output = tf.convert_to_tensor([[1, 2, 3, 4, 5],
                                            [3, 4, 4, 5, 6],
                                            [0, 1, 6, 3, 1]], tf.int32)
    output = align_fn.call(seq1, seq2)
    self.assertAllEqual(output, expected_output)

    seq1 = tf.convert_to_tensor(self.vocab.encode('X-X-X-X-'), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode('-Y-Y-Y-Y'), tf.int32)
    expected_output = tf.zeros([3, 0], tf.int32)
    output = align_fn.call(seq1, seq2)
    self.assertAllEqual(output, expected_output)

  def test_create_homology_targets(self):
    hom_fn = align_transforms.CreateHomologyTargets()

    values = tf.repeat([0, 1, 2, 3], 2)
    expected_output = tf.convert_to_tensor([1, 1, 1, 1, 0, 0, 0, 0], tf.float32)
    expected_output = expected_output[:, tf.newaxis]
    self.assertAllEqual(hom_fn.call(values), expected_output)

    values = tf.repeat([0, 1, 2, 0], 2)
    expected_output = tf.convert_to_tensor([1, 1, 1, 1, 1, 0, 0, 0], tf.float32)
    expected_output = expected_output[:, tf.newaxis]
    self.assertAllEqual(hom_fn.call(values), expected_output)

    hom_fn = align_transforms.CreateHomologyTargets(process_negatives=False)
    self.assertAllEqual(hom_fn.call(values),
                        tf.ones([len(values) // 2, 1], tf.float32))

  def test_create_batched_weights(self):
    b, l = 32, 64

    mock_targets = self.sampler.sample([b])
    weights_fn = align_transforms.CreateBatchedWeights()
    weights = weights_fn.call(mock_targets)
    self.assertAllEqual(weights, tf.ones(b, tf.float32))

    mock_targets = self.sampler.sample([b, 3, l])
    weights_fn = align_transforms.CreateBatchedWeights()
    weights = weights_fn.call(mock_targets)
    self.assertAllEqual(weights, tf.ones(b, tf.float32))

  def test_pad_negative_pairs(self):
    b, l = 32, 64
    pad_fn = align_transforms.PadNegativePairs()

    mock_targets = self.sampler.sample([b])
    targets = pad_fn.call(mock_targets)
    self.assertAllEqual(targets[:b], mock_targets)
    self.assertAllEqual(targets[b:], tf.zeros_like(mock_targets))

    mock_targets = self.sampler.sample([b, 3, l])
    targets = pad_fn.call(mock_targets)
    self.assertAllEqual(targets[:b], mock_targets)
    self.assertAllEqual(targets[b:], tf.zeros_like(mock_targets))

    value = -1
    pad_fn = align_transforms.PadNegativePairs(value=value)
    mock_targets = tf.ones([b], tf.float32)
    targets = pad_fn.call(mock_targets)
    self.assertAllEqual(targets[:b], mock_targets)
    self.assertAllEqual(targets[b:], value * tf.ones_like(mock_targets))

  def test_add_random_tails(self):
    seq1 = 'ACG----AATGGCACC--CTAA---'
    seq2 = '---GGGTAA-GGTACCTACT--TCG'
    seq1 = tf.convert_to_tensor(self.vocab.encode(seq1), tf.int32)
    seq2 = tf.convert_to_tensor(self.vocab.encode(seq2), tf.int32)

    add_random_tails = align_transforms.AddRandomTails()
    out_seq1, out_seq2 = add_random_tails.call(seq1, seq2)

    start_pos1 = self.vocab.decode(out_seq1).find(self.vocab.decode(seq1))
    start_pos2 = self.vocab.decode(out_seq2).find(self.vocab.decode(seq2))

    # Verifies that seq1 (resp. seq2) is contained in out_seq1 (resp. out_seq2).
    self.assertNotEqual(start_pos1, -1)
    self.assertNotEqual(start_pos2, -1)

    # Verifies alignment targets are shifted by the right offset.
    create_alignment_targets = align_transforms.CreateAlignmentTargets()
    alg_tar = create_alignment_targets.call(seq1, seq2)
    out_alg_tar = create_alignment_targets.call(out_seq1, out_seq2)

    self.assertAllEqual(out_alg_tar[0] - alg_tar[0],
                        alg_tar.shape[1] * [start_pos1])
    self.assertAllEqual(out_alg_tar[1] - alg_tar[1],
                        alg_tar.shape[1] * [start_pos2])
    self.assertAllEqual(out_alg_tar[2] - alg_tar[2],
                        alg_tar.shape[1] * [0])  # States unchanged.

  def test_add_alignment_context(self):
    sequence_1 = 'AATGGCACC--CT'
    sequence_2 = 'AA-GGTACCTACT'
    full_sequence_1 = 'ACG' + sequence_1.replace('-', '') + 'AA'
    full_sequence_2 = 'GGGT' + sequence_2.replace('-', '') + 'TCG'

    sequence_1 = tf.convert_to_tensor(self.vocab.encode(sequence_1), tf.int32)
    sequence_2 = tf.convert_to_tensor(self.vocab.encode(sequence_2), tf.int32)
    full_sequence_1 = tf.convert_to_tensor(
        self.vocab.encode(full_sequence_1), tf.int32)
    full_sequence_2 = tf.convert_to_tensor(
        self.vocab.encode(full_sequence_2), tf.int32)
    start_1, end_1 = 4, 14
    start_2, end_2 = 5, 16

    add_alignment_context = align_transforms.AddAlignmentContext()
    sequence_with_ctx_1, sequence_with_ctx_2 = add_alignment_context.call(
        sequence_1, sequence_2, full_sequence_1, full_sequence_2,
        start_1, start_2, end_1, end_2)

    self.assertEqual(len(sequence_with_ctx_1), len(sequence_with_ctx_2))
    self.assertIn(self.vocab.decode(sequence_with_ctx_1),
                  self.vocab.decode(full_sequence_1))
    self.assertIn(self.vocab.decode(sequence_with_ctx_2),
                  self.vocab.decode(full_sequence_2))

    create_alignment_targets = align_transforms.CreateAlignmentTargets()
    targets = create_alignment_targets.call(sequence_1, sequence_2)
    targets_with_ctx = create_alignment_targets.call(
        sequence_with_ctx_1, sequence_with_ctx_2)
    find_1 = self.vocab.decode(sequence_with_ctx_1).find(
        self.vocab.decode(sequence_1))
    find_2 = self.vocab.decode(sequence_with_ctx_2).find(
        self.vocab.decode(sequence_2))
    self.assertAllEqual(targets_with_ctx[0], targets[0] + find_1)
    self.assertAllEqual(targets_with_ctx[1], targets[1] + find_2)
    self.assertAllEqual(targets_with_ctx[2], targets[2])

  def test_trim_alignment(self):
    sequence_1 = 'ACG----AATGGCACC--CTAA---'
    sequence_2 = '---GGGTAA-GGTACCTACT--TCG'
    sequence_1 = tf.convert_to_tensor(self.vocab.encode(sequence_1), tf.int32)
    sequence_2 = tf.convert_to_tensor(self.vocab.encode(sequence_2), tf.int32)

    trim_alignment = align_transforms.TrimAlignment(p_trim=1.0)
    trimmed_sequence_1, trimmed_sequence_2 = trim_alignment.call(
        sequence_1, sequence_2)

    self.assertEqual(len(trimmed_sequence_1), len(trimmed_sequence_2))
    self.assertIn(self.vocab.decode(trimmed_sequence_1),
                  self.vocab.decode(sequence_1))
    self.assertIn(self.vocab.decode(trimmed_sequence_2),
                  self.vocab.decode(sequence_2))

    def get_match_subsequences(seq1, seq2):
      mask = tf.logical_and(self.vocab.compute_mask(seq1, '-'),
                            self.vocab.compute_mask(seq2, '-'))
      match_indices = tf.reshape(tf.where(mask), [-1])
      match_seq_1 = self.vocab.decode(tf.gather(seq1, match_indices))
      match_seq_2 = self.vocab.decode(tf.gather(seq2, match_indices))
      return match_seq_1, match_seq_2

    matches_1, matches_2 = get_match_subsequences(sequence_1, sequence_2)
    trimmed_matches_1, trimmed_matches_2 = get_match_subsequences(
        trimmed_sequence_1, trimmed_sequence_2)
    self.assertIn(trimmed_matches_1, matches_1)
    self.assertIn(trimmed_matches_2, matches_2)

    trim_alignment = align_transforms.TrimAlignment(p_trim=0.0)
    trimmed_sequence_1, trimmed_sequence_2 = trim_alignment.call(
        sequence_1, sequence_2)
    self.assertAllEqual(trimmed_sequence_1, sequence_1)
    self.assertAllEqual(trimmed_sequence_2, sequence_2)


class AlignDatasetTransformsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    self.vocab = vocabulary.alternative
    self.sampler = vocabulary.Sampler(vocab=self.vocab)

  def test_stratified_sampling_pairing(self):
    n_steps = 25

    seq = self.sampler.sample([24, 30])
    cla_key = tf.repeat([0, 1, 2], 8)
    fam_key = tf.repeat([0, 1, 2, 3, 4, 5], 4)
    clu_key = tf.concat(
        [tf.tile([0, 0, 1, 1], [3]), tf.tile([0, 1, 2, 3], [3])], 0)
    ds_in = tf.data.Dataset.from_tensor_slices({
        'seq': seq,
        'cla_key': cla_key,
        'fam_key': fam_key,
        'clu_key': clu_key
    })

    pair_examples = align_transforms.StratifiedSamplingPairing(
        index_keys=('fam_key', 'clu_key'), branch_key='clu_key')
    ds_out = ds_in.apply(pair_examples)
    for ex in ds_out.take(n_steps):
      self.assertEqual(ex['cla_key_1'], ex['cla_key_2'])
      self.assertEqual(ex['fam_key_1'], ex['fam_key_2'])
      self.assertNotEqual(ex['clu_key_1'], ex['clu_key_2'])

    pair_examples = align_transforms.StratifiedSamplingPairing(
        index_keys=('cla_key', 'fam_key', 'clu_key'), branch_key='clu_key')
    ds_out = ds_in.apply(pair_examples)
    for ex in ds_out.take(n_steps):
      self.assertEqual(ex['cla_key_1'], ex['cla_key_2'])
      self.assertEqual(ex['fam_key_1'], ex['fam_key_2'])
      self.assertNotEqual(ex['clu_key_1'], ex['clu_key_2'])

    pair_examples = align_transforms.StratifiedSamplingPairing(
        index_keys=('cla_key', 'fam_key', 'clu_key'), branch_key='fam_key')
    ds_out = ds_in.apply(pair_examples)
    for ex in ds_out.take(n_steps):
      self.assertEqual(ex['cla_key_1'], ex['cla_key_2'])
      self.assertNotEqual(ex['fam_key_1'], ex['fam_key_2'])

    pair_examples = align_transforms.StratifiedSamplingPairing(
        index_keys=('cla_key', 'fam_key', 'clu_key'), branch_key='cla_key')
    ds_out = ds_in.apply(pair_examples)
    for ex in ds_out.take(n_steps):
      self.assertNotEqual(ex['cla_key_1'], ex['cla_key_2'])


if __name__ == '__main__':
  tf.test.main()
