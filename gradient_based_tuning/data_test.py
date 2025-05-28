# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Tests for data.py."""

from absl.testing import absltest
import tensorflow.compat.v2 as tf

from gradient_based_tuning import data


def _get_ds_from_examples(raw_examples, data_type=tf.int32):

  def generator():
    for ex_dict in raw_examples:
      ret = {k: v for k, v in ex_dict.items()}
      yield ret

  in_types = {x: data_type for x in raw_examples[0].keys()}
  in_shapes = {x: tf.TensorShape([3]) for x in raw_examples[0].keys()}
  return tf.data.Dataset.from_generator(generator, in_types, in_shapes)


class CastDatasetTypesTest(absltest.TestCase):

  def test_converts_matching_dtype(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33]
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66]
    }]
    ds = _get_ds_from_examples(raw_examples, tf.int32)

    cast_dict = {
        'inputs': tf.uint16,
        'targets': tf.uint16,
    }
    output_ds = data.cast_dataset_types(ds, cast_dict)

    self.assertEqual(
        output_ds.element_spec['inputs'].dtype,
        tf.uint16,
    )
    self.assertEqual(
        output_ds.element_spec['targets'].dtype,
        tf.uint16,
    )

  def test_ignore_extra_types_dtype(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33]
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66]
    }]
    ds = _get_ds_from_examples(raw_examples, tf.float32)

    cast_dict = {
        'targets': tf.uint16,
        'extraneous_tag': tf.int64,
    }
    output_ds = data.cast_dataset_types(ds, cast_dict)

    self.assertEqual(
        output_ds.element_spec['inputs'].dtype,
        tf.float32,
    )
    self.assertEqual(
        output_ds.element_spec['targets'].dtype,
        tf.uint16,
    )


class PackAndBatchTest(absltest.TestCase):

  def test_pack_repeats_only_inputs_targets(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
        'edits': [0, 1, 2],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
        'edits': [0, 1, 2],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
        'edits': [0, 1, 2],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=True)
    self.assertCountEqual(
        list(pnb_ds.element_spec.keys()),
        [
            'edits', 'inputs', 'targets', 'inputs_position', 'targets_position',
            'targets_segmentation', 'inputs_segmentation'
        ],
    )

  def test_pack_pads_to_max_len(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=8,
        extend_to_fill=False,
        drop_remainder=False,
        pack=True)
    shape_of_single_batch = next(iter(pnb_ds))['inputs'].shape
    self.assertEqual(shape_of_single_batch, (2, 8))

  def test_no_drop_no_extend_yields_underfull_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=True)
    pnb_ds_iter = iter(pnb_ds)
    first_batch_inputs = next(pnb_ds_iter)['inputs']
    final_batch_inputs = next(pnb_ds_iter)['inputs']
    shape_of_first_batch = first_batch_inputs.shape
    shape_of_final_batch = final_batch_inputs.shape
    self.assertEqual(shape_of_first_batch, (2, 7))
    self.assertListEqual(first_batch_inputs.numpy().tolist(),
                         [[1, 2, 3, 4, 5, 6, 0], [7, 8, 9, 10, 20, 30, 0]])
    self.assertEqual(shape_of_final_batch, (1, 7))
    self.assertListEqual(final_batch_inputs.numpy().tolist(),
                         [[40, 50, 60, 70, 80, 90, 0]])

  def test_drop_remainder_drops_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=True,
        pack=True)
    pnb_ds_iter = iter(pnb_ds)
    shape_of_first_batch = next(pnb_ds_iter)['inputs'].shape
    self.assertEqual(shape_of_first_batch, (2, 7))
    with self.assertRaises(StopIteration):
      _ = next(pnb_ds_iter)['inputs'].shape

  def test_extend_to_fill_extends_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=True,
        drop_remainder=False,
        pack=True)
    pnb_ds_iter = iter(pnb_ds)
    first_batch_inputs = next(pnb_ds_iter)['inputs']
    final_batch_inputs = next(pnb_ds_iter)['inputs']
    shape_of_first_batch = first_batch_inputs.shape
    shape_of_final_batch = final_batch_inputs.shape
    self.assertEqual(shape_of_first_batch, (2, 7))
    self.assertListEqual(first_batch_inputs.numpy().tolist(),
                         [[1, 2, 3, 4, 5, 6, 0], [7, 8, 9, 10, 20, 30, 0]])
    self.assertEqual(shape_of_final_batch, (2, 7))
    self.assertListEqual(final_batch_inputs.numpy().tolist(),
                         [[40, 50, 60, 70, 80, 90, 0], [1, 2, 3, 4, 5, 6, 0]])


class GetUniqueExamplesTest(absltest.TestCase):

  def test_packed_and_batched(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=True)
    pnb_ds_iter = iter(pnb_ds)
    self.assertEqual(data.get_unique_examples(next(pnb_ds_iter)), 4)

  def test_not_packed_not_batched(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=1,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=False)
    pnb_ds_iter = iter(pnb_ds)
    self.assertEqual(data.get_unique_examples(next(pnb_ds_iter)), 1)

  def test_batched_not_packed_errors(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=False)
    pnb_ds_iter = iter(pnb_ds)
    self.assertEqual(data.get_unique_examples(next(pnb_ds_iter)), 2)

  def test_packed_not_batched_errors(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=False)
    pnb_ds_iter = iter(pnb_ds)
    self.assertEqual(data.get_unique_examples(next(pnb_ds_iter)), 2)


class GetFinalBatchSizeTest(absltest.TestCase):

  def test_underfull_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=True)
    self.assertEqual(data.get_ds_final_batch_size(pnb_ds), (2, 1))

  def test_extended_full_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=True,
        drop_remainder=False,
        pack=True)
    self.assertEqual(data.get_ds_final_batch_size(pnb_ds), (2, 2))

  def test_dropped_full_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=True,
        pack=True)
    self.assertEqual(data.get_ds_final_batch_size(pnb_ds), (1, 2))

  def test_empty_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=True,
        pack=True)
    self.assertEqual(data.get_ds_final_batch_size(pnb_ds), (0, 0))


class GetDsMetricsTest(absltest.TestCase):

  def test_underfull_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=True)
    self.assertEqual(data.get_ds_metrics(pnb_ds), (6, 2, 2, 7))

  def test_extended_full_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=True,
        drop_remainder=False,
        pack=True)
    self.assertEqual(data.get_ds_metrics(pnb_ds), (8, 2, 2, 7))

  def test_dropped_full_final_batch(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=True,
        pack=True)
    self.assertEqual(data.get_ds_metrics(pnb_ds), (4, 1, 2, 7))

  def test_not_packed_not_batched(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=1,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=False)
    self.assertEqual(data.get_ds_metrics(pnb_ds), (6, 6, 1, 3))

  def test_batched_not_packed_errors(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=False)
    with self.assertRaises(NotImplementedError):
      _ = data.get_ds_metrics(pnb_ds)

  def test_packed_not_batched_errors(self):
    raw_examples = [{
        'inputs': [1, 2, 3],
        'targets': [11, 22, 33],
    }, {
        'inputs': [4, 5, 6],
        'targets': [44, 55, 66],
    }, {
        'inputs': [7, 8, 9],
        'targets': [77, 88, 99],
    }, {
        'inputs': [10, 20, 30],
        'targets': [110, 220, 330],
    }, {
        'inputs': [40, 50, 60],
        'targets': [440, 550, 660],
    }, {
        'inputs': [70, 80, 90],
        'targets': [770, 880, 990],
    }]
    ds = _get_ds_from_examples(raw_examples)
    pnb_ds = data.pack_and_batch_ds(
        ds,
        batch_size=2,
        max_length=7,
        extend_to_fill=False,
        drop_remainder=False,
        pack=False)
    with self.assertRaises(NotImplementedError):
      _ = data.get_ds_metrics(pnb_ds)


if __name__ == '__main__':
  absltest.main()
