# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for tf_cnn_benchmark.allreduce."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as pycoll

import numpy as np
import tensorflow as tf
from cnn_quantization.tf_cnn_benchmarks import allreduce
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables


class AllReduceTest(tf.test.TestCase):

  def testGroupKey(self):
    d0 = ['/job:worker/replica:0/task:0/device:GPU:1',
          '/job:worker/replica:0/task:0/device:GPU:0',
          '/job:worker/replica:0/task:0/device:GPU:3',]
    d1 = ['/job:worker/replica:0/task:1/device:GPU:1',
          '/job:worker/replica:0/task:1/device:GPU:0',
          '/job:worker/replica:0/task:1/device:GPU:3',]
    d2 = ['/job:worker/replica:0/task:1/device:GPU:1',
          '/job:worker/replica:0/task:1/device:GPU:3',
          '/job:worker/replica:0/task:1/device:GPU:0',]
    d3 = ['/job:worker/replica:0/task:1/device:GPU:1',
          '/job:worker/replica:0/task:1/device:GPU:3',
          '/job:worker/replica:0/task:1/device:GPU:2',]
    d4 = ['/job:worker/task:0/device:GPU:1',
          '/job:worker/task:0/device:GPU:2',
          '/job:worker/task:0/device:GPU:3',]
    d5 = ['/job:worker/task:0/device:CPU:1',
          '/job:worker/task:0/device:CPU:2']
    d6 = ['/job:worker/task:0/device:CPU:2',
          '/job:worker/task:0/device:CPU:1']
    g0 = allreduce.collective_group_key(d0)
    g1 = allreduce.collective_group_key(d1)
    g2 = allreduce.collective_group_key(d2)
    g3 = allreduce.collective_group_key(d3)
    g4 = allreduce.collective_group_key(d4)
    g5 = allreduce.collective_group_key(d5)
    g6 = allreduce.collective_group_key(d6)
    self.assertEqual(g0, g1)
    self.assertEqual(g0, g2)
    self.assertTrue(g0 != g3)
    self.assertEqual(g3, g4)
    self.assertEqual(g5, g6)
    self.assertTrue(g4 != g5)

  def testExtractRanges(self):
    x = []
    expected_ranges = []
    expected_singles = []
    ranges, singles = allreduce.extract_ranges(x)
    self.assertEqual(expected_ranges, ranges)
    self.assertEqual(expected_singles, singles)
    x = [1, 3, 4, 6, 7, 8, 9]
    expected_ranges = [[3, 4], [6, 9]]
    expected_singles = [1]
    ranges, singles = allreduce.extract_ranges(x)
    self.assertEqual(expected_ranges, ranges)
    self.assertEqual(expected_singles, singles)
    x = [1, 2, 3, 4, 6, 7, 8, 9]
    expected_ranges = [[1, 4], [6, 9]]
    expected_singles = []
    ranges, singles = allreduce.extract_ranges(x)
    self.assertEqual(expected_ranges, ranges)
    self.assertEqual(expected_singles, singles)
    x = [1, 3, 4, 6, 7, 9]
    expected_ranges = [[3, 4], [6, 7]]
    expected_singles = [1, 9]
    ranges, singles = allreduce.extract_ranges(x)
    self.assertEqual(expected_ranges, ranges)
    self.assertEqual(expected_singles, singles)
    x = [1, 3, 6, 9]
    expected_ranges = []
    expected_singles = [1, 3, 6, 9]
    ranges, singles = allreduce.extract_ranges(x)
    self.assertEqual(expected_ranges, ranges)
    self.assertEqual(expected_singles, singles)

  def testPackRange(self):
    packing = {}
    t0 = tf.constant([0, 1, 2, 3], dtype=tf.float32)
    t1 = tf.constant([4, 5, 6, 7], dtype=tf.float32)

    gv = [(t0, 'v0'), (t1, 'v1')]
    new_t = allreduce.pack_range('0:0', packing, gv, [0, 1])
    self.assertEqual(1, new_t.shape.ndims)
    self.assertEqual(8, new_t.shape.dims[0])
    self.assertEqual(
        packing, {
            '0:0':
                allreduce.GradPackTuple(
                    indices=range(2),
                    vars=['v0', 'v1'],
                    shapes=[tf.TensorShape([4]),
                            tf.TensorShape([4])])
        })

    t2 = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    t3 = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    gv = [(t0, 'v0'), (t1, 'v1'), (t2, 'v2'), (t3, 'v3')]
    packing = {}
    new_t = allreduce.pack_range('1:0', packing, gv, [0, 3])
    self.assertEqual(1, new_t.shape.ndims)
    self.assertEqual(26, new_t.shape.dims[0])
    self.assertEqual(
        packing, {
            '1:0':
                allreduce.GradPackTuple(
                    indices=range(4),
                    vars=['v0', 'v1', 'v2', 'v3'],
                    shapes=[
                        tf.TensorShape([4]),
                        tf.TensorShape([4]),
                        tf.TensorShape([3, 3]),
                        tf.TensorShape([3, 3])
                    ])
        })

  def testUnpackGradTuple(self):
    packing = {
        '0:0':
            allreduce.GradPackTuple(
                indices=range(4),
                vars=['v0', 'v1', 'v2', 'v3'],
                shapes=[
                    tf.TensorShape([4]),
                    tf.TensorShape([4]),
                    tf.TensorShape([3, 3]),
                    tf.TensorShape([3, 3])
                ])
    }
    tc = tf.constant([0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7, 8,
                      0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.float32)
    packed_gv = [tc, 'packing_var_placeholder']
    gv = allreduce.unpack_grad_tuple(packed_gv, packing['0:0'])
    self.assertEqual(4, len(gv))
    self.assertEqual('v0', gv[0][1])
    self.assertEqual('v1', gv[1][1])
    self.assertEqual('v2', gv[2][1])
    self.assertEqual('v3', gv[3][1])
    self.assertEqual(1, gv[0][0].shape.ndims)
    self.assertEqual(4, gv[0][0].shape.dims[0])
    self.assertEqual(1, gv[1][0].shape.ndims)
    self.assertEqual(4, gv[1][0].shape.dims[0])
    self.assertEqual(2, gv[2][0].shape.ndims)
    self.assertEqual(3, gv[2][0].shape.dims[0])
    self.assertEqual(3, gv[2][0].shape.dims[1])

  def testPackSmallTensors(self):
    t0 = tf.constant([0, 1, 2, 3], dtype=tf.float32)
    t1 = tf.constant([4, 5, 6, 7], dtype=tf.float32)
    t2 = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    t3 = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    tower_grads = []
    for d in range(0, 3):
      gv = [(t0, 'v_%d_0' % d), (t1, 'v_%d_1' %d), (t2, 'v_%d_2' %d),
            (t3, 'v_%d_3' % d)]
      tower_grads.append(gv)

    # 1) Set the size limit so small that nothing gets concatenated.
    new_tower_grads, packing = allreduce.pack_small_tensors(
        tower_grads, max_bytes=12,
        max_group=10)
    self.assertEqual(tower_grads, new_tower_grads)
    self.assertTrue(packing is None)

    # 2) Set the size limit so only the first two tensors get concatenated
    new_tower_grads, packing = allreduce.pack_small_tensors(
        tower_grads, max_bytes=16,  # 16 bytes == 4 elements
        max_group=10)
    self.assertEqual(3, len(new_tower_grads))
    self.assertEqual(4, len(tower_grads[0]))
    first_tower = new_tower_grads[0]
    self.assertEqual(3, len(first_tower))
    self.assertEqual(1, first_tower[0][0].shape.ndims)
    self.assertEqual(8, first_tower[0][0].shape.dims[0])
    self.assertEqual(packing,
                     {'0:0': allreduce.GradPackTuple(
                         indices=range(2),
                         vars=['v_0_0', 'v_0_1'],
                         shapes=[tf.TensorShape([4]),
                                 tf.TensorShape([4])]),
                      '1:0': allreduce.GradPackTuple(
                          indices=range(2),
                          vars=['v_1_0', 'v_1_1'],
                          shapes=[tf.TensorShape([4]),
                                  tf.TensorShape([4])]),
                      '2:0': allreduce.GradPackTuple(
                          indices=range(2),
                          vars=['v_2_0', 'v_2_1'],
                          shapes=[tf.TensorShape([4]),
                                  tf.TensorShape([4])])})

    # 3) Set the size limit so all tensors get concatenated
    new_tower_grads, packing = allreduce.pack_small_tensors(
        tower_grads, max_bytes=256,   # bytes = 64 elements
        max_group=10)
    self.assertEqual(3, len(new_tower_grads))
    self.assertEqual(4, len(tower_grads[0]))
    self.assertEqual(1, len(new_tower_grads[0]))
    first_tower = new_tower_grads[0]
    self.assertEqual(1, first_tower[0][0].shape.ndims)
    self.assertEqual(26, first_tower[0][0].shape.dims[0])
    self.assertEqual(packing,
                     {'0:0': allreduce.GradPackTuple(
                         indices=range(4),
                         vars=['v_0_0', 'v_0_1', 'v_0_2', 'v_0_3'],
                         shapes=[tf.TensorShape([4]),
                                 tf.TensorShape([4]),
                                 tf.TensorShape([3, 3,]),
                                 tf.TensorShape([3, 3,])]),
                      '1:0': allreduce.GradPackTuple(
                          indices=range(4),
                          vars=['v_1_0', 'v_1_1', 'v_1_2', 'v_1_3'],
                          shapes=[tf.TensorShape([4]),
                                  tf.TensorShape([4]),
                                  tf.TensorShape([3, 3,]),
                                  tf.TensorShape([3, 3,])]),
                      '2:0': allreduce.GradPackTuple(
                          indices=range(4),
                          vars=['v_2_0', 'v_2_1', 'v_2_2', 'v_2_3'],
                          shapes=[tf.TensorShape([4]),
                                  tf.TensorShape([4]),
                                  tf.TensorShape([3, 3,]),
                                  tf.TensorShape([3, 3,])])})

  def testUnpackSmallTensors(self):
    packing = {'0:0': allreduce.GradPackTuple(indices=range(2),
                                              vars=['v_0_0', 'v_0_1'],
                                              shapes=[tf.TensorShape([4]),
                                                      tf.TensorShape([4])]),
               '0:1': allreduce.GradPackTuple(indices=range(3, 5),
                                              vars=['v_0_3', 'v_0_4'],
                                              shapes=[tf.TensorShape([3, 3,]),
                                                      tf.TensorShape([3, 3,])]),
               '1:0': allreduce.GradPackTuple(indices=range(2),
                                              vars=['v_1_0', 'v_1_1'],
                                              shapes=[tf.TensorShape([4]),
                                                      tf.TensorShape([4])]),
               '1:1': allreduce.GradPackTuple(indices=range(3, 5),
                                              vars=['v_1_3', 'v_1_4'],
                                              shapes=[tf.TensorShape([3, 3,]),
                                                      tf.TensorShape([3, 3,])])}
    t0 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7], dtype=tf.float32)
    t1 = tf.constant([17, 17], dtype=tf.float32)
    t2 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8,
                      0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.float32)
    t3 = tf.constant([0], dtype=tf.float32)
    tower_grads = []
    for d in range(0, 2):
      one_tower = [(t0, 'packing_var_placeholder'),
                   (t2, 'packing_var_placeholder'),
                   (t1, 'v_%d_2' % d), (t3, 'v_%d_5' %d)]
      tower_grads.append(one_tower)
    new_tower_grads = allreduce.unpack_small_tensors(tower_grads, packing)
    self.assertEqual(2, len(new_tower_grads))
    for d, tg in enumerate(new_tower_grads):
      self.assertEqual(6, len(tg))
      self.assertEqual('v_%d_0' % d, tg[0][1])
      self.assertEqual('v_%d_1' % d, tg[1][1])
      self.assertEqual('v_%d_2' % d, tg[2][1])
      self.assertEqual('v_%d_3' % d, tg[3][1])
      self.assertEqual('v_%d_4' % d, tg[4][1])
      self.assertEqual('v_%d_5' % d, tg[5][1])
      self.assertEqual(1, tg[0][0].shape.ndims)
      self.assertEqual(4, tg[0][0].shape.dims[0])
      self.assertEqual(1, tg[1][0].shape.ndims)
      self.assertEqual(4, tg[1][0].shape.dims[0])
      self.assertEqual(1, tg[2][0].shape.ndims)
      self.assertEqual(2, tg[2][0].shape.dims[0])
      self.assertEqual(2, tg[3][0].shape.ndims)
      self.assertEqual(3, tg[3][0].shape.dims[0])
      self.assertEqual(3, tg[3][0].shape.dims[1])
      self.assertEqual(2, tg[4][0].shape.ndims)
      self.assertEqual(3, tg[4][0].shape.dims[0])
      self.assertEqual(3, tg[4][0].shape.dims[1])
      self.assertEqual(1, tg[5][0].shape.ndims)
      self.assertEqual(1, tg[5][0].shape.dims[0])


class DynamicPackingTest(test_util.TensorFlowTestCase):
  """Packing/Unpacking tests that require executing a TensorFlow session."""

  def _init_tensors(self, num_towers, tensor_shapes):
    """Construct a collection of tensors across multiple devices."""
    num_tensors = len(tensor_shapes)
    consts = []
    tensors = []
    vrbls = []
    tower_grads = []
    tf.Variable([-1], dtype=tf.int32, name='packing_var_placeholder')
    for dev_idx in range(0, num_towers):
      devname = '/job:localhost/device:GPU:%d' % dev_idx
      consts.append([])
      tensors.append([])
      vrbls.append([])
      with tf.device(devname):
        base_value = 0
        gv_tuples = []
        for t_idx in range(0, num_tensors):
          shape = tensor_shapes[t_idx]
          num_elts = 0
          for d in shape:
            num_elts = (num_elts or 1) * d
          c = np.fromiter(range(base_value, base_value + num_elts),
                          dtype=np.float32).reshape(shape)
          base_value += num_elts
          consts[dev_idx].append(c)
          tensors[dev_idx].append(tf.constant(c))
          vrbls[dev_idx].append(
              tf.Variable(c, name='v_d%d_t%d' % (dev_idx, t_idx)))
          gv_tuples.append((tensors[dev_idx][-1], vrbls[dev_idx][-1]))
        tower_grads.append(gv_tuples)
    return tower_grads, consts, tensors, vrbls

  _test_tuple = pycoll.namedtuple('_test_tuple',
                                  'num_devices, in_shapes out_shapes out_i')

  def _do_pack_unpack_test(self, tt):
    """Do a single pack-unpack test.

    Args:
      tt: A _test_tuple defining the parameters of the test to do.

    This test executes a graph that performs a pack of tower_grads
    followed by an unpack and verifies that the shapes and values
    of gradient tensors are unchanged, along with paired variables.
    """
    with ops.Graph().as_default():
      tower_grads, consts, _, vrbls = self._init_tensors(
          tt.num_devices, tt.in_shapes)
      packed_tg, packing = allreduce.pack_small_tensors(
          tower_grads, max_bytes=40, max_group=10)
      unpacked_tg = allreduce.unpack_small_tensors(packed_tg, packing)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        packed = sess.run(packed_tg)
        for d in range(0, tt.num_devices):
          for t in range(0, len(tt.out_shapes)):
            num_elts = 0
            for dim in tt.out_shapes[t]:
              num_elts = (num_elts or 1) * dim
            self.assertTrue(np.array_equal(
                np.array(range(tt.out_i[t], tt.out_i[t] + num_elts),
                         dtype=np.float32).reshape(tt.out_shapes[t]),
                packed[d][t][0]))
        unpacked = sess.run(unpacked_tg)
        for d in range(0, tt.num_devices):
          for t in range(0, len(tt.in_shapes)):
            self.assertTrue(np.array_equal(consts[d][t], unpacked[d][t][0]))
            self.assertEqual(vrbls[d][t], unpacked_tg[d][t][1])

  def testPackUnpack0(self):
    self._do_pack_unpack_test(
        self._test_tuple(num_devices=3,
                         in_shapes=[[8], [3, 3], [12], [5, 5, 5]],
                         out_shapes=[[17], [12], [5, 5, 5]],
                         out_i=[0, 17, 29]))

  def testPackUnpack1(self):
    self._do_pack_unpack_test(
        self._test_tuple(num_devices=4,
                         in_shapes=[[5, 5, 5], [2, 3], [5]],
                         out_shapes=[[11], [5, 5, 5]],
                         out_i=[125, 0]))

  def testPackUnpack2(self):
    self._do_pack_unpack_test(
        self._test_tuple(num_devices=2,
                         in_shapes=[[5, 5, 5], [2, 3], [1, 5], [7], [100]],
                         out_shapes=[[18], [5, 5, 5], [100]],
                         out_i=[125, 0, 143]))

  def _do_all_reduce_pack_test(self, tt):
    """Test that all-reduce results are the same with or without packing."""
    with ops.Graph().as_default():
      tower_grads, consts, _, _ = self._init_tensors(
          tt.num_devices, tt.in_shapes)
      dev_prefixes = ['/job:localhost']
      num_workers = 1
      alg = 'xring'
      shards = 1
      single_session = True
      gpu_indices = range(0, tt.num_devices)
      assert len(gpu_indices) == len(tower_grads)
      no_pack_all_reduce = allreduce.sum_gradients_all_reduce(
          single_session,
          dev_prefixes, tower_grads, num_workers, alg, shards,
          gpu_indices,
          agg_small_grads_max_bytes=0, agg_small_grads_max_group=1)
      packed_tg, packing = allreduce.pack_small_tensors(tower_grads, 100, 100)
      packed_all_reduce = allreduce.sum_gradients_all_reduce(
          single_session,
          dev_prefixes, packed_tg, num_workers, alg, shards,
          gpu_indices,
          agg_small_grads_max_bytes=0, agg_small_grads_max_group=1)
      unpacked_tg = allreduce.unpack_small_tensors(packed_all_reduce, packing)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        no_pack_values = sess.run(no_pack_all_reduce)
        pack_unpack_values = sess.run(unpacked_tg)
        for d in range(1, tt.num_devices):
          for t in range(0, len(tt.in_shapes)):
            self.assertTrue(np.allclose(no_pack_values[d][t][0],
                                        tt.num_devices * consts[0][t]))
            self.assertTrue(np.array_equal(no_pack_values[d][t][0],
                                           pack_unpack_values[d][t][0]))

  def testAllReducePacked0(self):
    self._do_all_reduce_pack_test(
        self._test_tuple(num_devices=3,
                         in_shapes=[[8], [3, 3], [12], [5, 5, 5]],
                         out_shapes=[[17], [12], [5, 5, 5]],
                         out_i=[0, 17, 29]))

  def testAllReducePacked1(self):
    self._do_all_reduce_pack_test(
        self._test_tuple(num_devices=2,
                         in_shapes=[[8], [3, 3], [12], [5, 5, 5], [3], [4]],
                         out_shapes=[[17], [7], [12], [5, 5, 5]],
                         out_i=[0, 17, 29, 154, 157]))


if __name__ == '__main__':
  tf.test.main()
