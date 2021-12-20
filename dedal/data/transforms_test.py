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

"""Tests for transforms.py."""

import gin
import numpy as np
import tensorflow as tf

from dedal import vocabulary
from dedal.data import transforms


class TransformsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    tf.random.set_seed(0)
    self.sampler = vocabulary.Sampler()
    self.seq = self.sampler.sample((256,))

  def test_transform(self):
    crop_fn = transforms.CropOrPad(size=50, random=True)
    self.assertIsInstance(crop_fn.call(self.seq), tf.Tensor)
    self.assertIsInstance(crop_fn({'sequence': self.seq}), dict)

    on_key = 'abcd'
    out_key = 'ixaEH'
    crop_fn = transforms.CropOrPad(size=50, random=True, on=on_key, out=out_key)

    with self.assertRaises(ValueError):
      crop_fn({'sequence': self.seq})

    output = crop_fn({on_key: self.seq})
    self.assertIn(on_key, output)
    self.assertIn(out_key, output)

    # The out_key is equal to the on_key if not set.
    crop_fn = transforms.CropOrPad(size=50, random=True, on=on_key)
    self.assertLen(crop_fn._out, 1)
    self.assertEqual(crop_fn._out[0], on_key)

  def test_reshape(self):
    b, l = 16, 64
    seqs = self.sampler.sample([b, 2, l])
    reshape_fn = transforms.Reshape(shape=[-1, l])
    output = reshape_fn.call(seqs)
    self.assertAllEqual(output.shape, [2 * b, l])
    self.assertAllEqual(output[::2], seqs[:, 0])
    self.assertAllEqual(output[1::2], seqs[:, 1])

  def test_stack(self):
    seqs = self.sampler.sample([3, 256])
    split_seqs = [tf.reshape(s, [-1]) for s in tf.split(seqs, len(seqs))]
    stack_fn = transforms.Stack(axis=0)
    output = stack_fn.call(*split_seqs)
    self.assertAllEqual(output, seqs)

  def test_pop(self):
    keys = ('seq1', 'seq2', 'seq3')
    inputs = {k: v for v, k in enumerate(keys)}
    pop_fn = transforms.Pop(on=keys[1])
    outputs = pop_fn(inputs)

    self.assertIn(keys[0], outputs)
    self.assertNotIn(keys[1], outputs)
    self.assertIn(keys[2], outputs)

  def test_encode(self):
    encode_fn = transforms.Encode(vocab=vocabulary.proteins)
    text = 'AABUCCFDEFGHYBOUXZACF'
    sequence = tf.constant(text)
    encoded = encode_fn.call(sequence)
    self.assertEqual(encoded.shape[0], len(text))
    self.assertEqual(encoded.dtype, tf.int32)

    output = encode_fn({'sequence': sequence})
    self.assertIn('sequence', output)
    self.assertAllEqual(output['sequence'], encoded)

  def test_recode(self):
    source = vocabulary.proteins
    target = vocabulary.alternative
    text = 'AABUCCFDEFGHYBOUXZACF'
    sequence = tf.constant(source.encode(text))
    recode_fn = transforms.Recode(vocab=source, target=target)
    encode_fn = transforms.Encode(vocab=target)
    self.assertAllEqual(
        recode_fn.call(sequence), encode_fn.call(tf.constant(text)))

  def test_crop_or_pad(self):
    # The sequence is padded if smaller than size.
    size = 256 + 10
    pad_fn = transforms.CropOrPad(size=size)
    padded = pad_fn.call(self.seq)
    self.assertEqual(padded.shape[0], size)
    self.assertAllEqual(padded[:256], self.seq)
    self.assertAllEqual(padded[256:], tf.repeat(padded[-1:], [size - 256]))
    # Pad on the left instead.
    pad_fn = transforms.CropOrPad(size=size, right=False)
    padded = pad_fn.call(self.seq)
    self.assertEqual(padded.shape[0], size)
    self.assertAllEqual(padded[size - 256:], self.seq)
    self.assertAllEqual(padded[:size - 256], tf.repeat(padded[:1],
                                                       [size - 256]))

    # The sequence is cropped (randomly by default) if longer than size.
    size = 256 - 56
    crop_fn = transforms.CropOrPad(size=size, random=True, seed=0)
    cropped = crop_fn.call(self.seq)
    self.assertEqual(cropped.shape[0], size)
    self.assertNotAllEqual(crop_fn.call(self.seq), crop_fn.call(self.seq))
    # Not random is deterministic.
    crop_fn = transforms.CropOrPad(size=size, random=False, seed=0)
    self.assertEqual(crop_fn.call(self.seq).shape[0], size)
    self.assertAllEqual(crop_fn.call(self.seq), crop_fn.call(self.seq))

  def test_multiple(self):
    pad_fn = transforms.CropOrPad(
        size=10, on=['in1', 'in2', 'in3'], out=['out1', 'out2', 'out3'])
    padded = pad_fn.call(self.seq, self.seq, self.seq)
    self.assertLen(padded, 3)
    out = pad_fn({
        'in1': self.seq,
        'in2': self.seq,
        'in3': self.seq,
        'other': 1
    })
    self.assertLen(set(out.keys()), 7)

  def test_eos(self):
    eos_fn = transforms.EOS()
    output = eos_fn.call(self.seq)
    self.assertAllEqual(output[:-1], self.seq)
    self.assertEqual(output[-1], eos_fn._vocab.get('>'))

  def test_prepend_cls(self):
    cls_fn = transforms.PrependClass()
    output = cls_fn.call(self.seq)
    self.assertAllEqual(output[1:], self.seq)
    self.assertEqual(output[0], cls_fn._vocab.get('<'))

  def test_remove_tokens(self):
    tokens = ['<', '-', '>']
    vocab = vocabulary.alternative
    rm_fn = transforms.RemoveTokens(tokens=tokens, vocab=vocab)

    output = rm_fn.call(self.seq)
    self.assertAllEqual(output, self.seq)

    seq = tf.tensor_scatter_nd_update(self.seq, [[0], [1], [2]],
                                      [vocab.get(token) for token in tokens])
    output = rm_fn.call(seq)
    self.assertAllEqual(output, self.seq[3:])

  def test_crop_or_pad_nd(self):
    x = tf.random.uniform((5, 4, 3))
    resizer = transforms.CropOrPadND(size=8, axis=0)
    output = resizer.call(x)
    self.assertAllEqual(output[5:], tf.zeros((3, 4, 3)))

    x = tf.random.uniform((5, 4, 32))
    resizer = transforms.CropOrPadND(size=8, axis=-1)
    output = resizer.call(x)
    self.assertEqual(output.shape, (5, 4, 8))

  def test_gin(self):
    gin.clear_config()
    gin.parse_config([
        'CropOrPad.size = 100',
        'Transform.on = "key"',
        'Transform.vocab = %vocabulary.proteins',
    ])
    crop_fn = transforms.CropOrPad()
    cropped = crop_fn.call(self.seq)
    self.assertEqual(cropped.shape[0], 100)

    self.assertIn('key', crop_fn._on)
    self.assertLen(crop_fn._vocab, len(vocabulary.proteins))
    self.assertNotEqual(len(crop_fn._vocab), len(vocabulary.alternative))

  def test_contact_matrix(self):
    n = 112
    positions = (2 * tf.random.uniform((n, 3)) - 1) * 100
    contact_fn = transforms.ContactMatrix(threshold=10.0)
    contact = contact_fn.single_call(positions)
    self.assertEqual(contact.dtype, tf.float32)
    self.assertEqual(contact.shape, (n, n))
    # Along the diagonal, the distance should be zero (hence contact)
    self.assertAllClose(np.diag(contact.numpy()), np.ones(n))
    self.assertAllClose(tf.transpose(contact), contact)
    # The contact matrix is only made of zeros and ones.
    num_zeros = tf.where(contact == 0.0).shape[0]
    num_ones = tf.where(contact == 1.0).shape[0]
    self.assertEqual(num_zeros + num_ones, n * n)


class DatasetTransformsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    self.vocab = vocabulary.alternative
    self.sampler = vocabulary.Sampler(vocab=self.vocab)

  def test_filter_by_length(self):

    def seq_len(seq):
      return tf.reduce_sum(tf.cast(self.vocab.padding_mask(seq), tf.int32))

    n, len1, len2 = 10, 20, 30
    max_len = 25
    x1 = tf.pad(
        self.sampler.sample([n, len1]), [[0, 0], [0, len2 - len1]],
        constant_values=self.vocab.padding_code)
    x2 = self.sampler.sample([n, len2])
    x = tf.concat([x1, x2], 0)

    filter_fn = transforms.FilterByLength(
        on='seq', vocab=self.vocab, max_len=max_len, precomputed=False)
    ds_in = tf.data.Dataset.from_tensor_slices({'seq': x})
    ds_out = ds_in.apply(filter_fn.call)
    for ex in ds_out:
      self.assertLessEqual(seq_len(ex['seq']), max_len)

    filter_fn = transforms.FilterByLength(
        on='seq', vocab=self.vocab, max_len=max_len, precomputed=False)
    ds_in = tf.data.Dataset.from_tensor_slices(
        {'seq': [tf.convert_to_tensor(self.vocab.decode(x_i)) for x_i in x]})
    ds_out = ds_in.apply(filter_fn.call)
    for ex in ds_out:
      self.assertLessEqual(tf.strings.length(ex['seq']), max_len)

    seq_lens = tf.convert_to_tensor([seq_len(x_i) for x_i in x], tf.int32)
    filter_fn = transforms.FilterByLength(
        on='seq_len', vocab=self.vocab, max_len=max_len, precomputed=True)
    ds_in = tf.data.Dataset.from_tensor_slices({'seq': x, 'seq_len': seq_lens})
    ds_out = ds_in.apply(filter_fn.call)
    for ex in ds_out:
      self.assertLessEqual(seq_len(ex['seq']), max_len)


if __name__ == '__main__':
  tf.test.main()
