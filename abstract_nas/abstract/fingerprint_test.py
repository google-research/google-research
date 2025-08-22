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

"""Tests for fingerprint."""

import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract import fingerprint
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType


class FingerprintTest(test.TestCase):

  def test_identical(self):
    """Tests whether the fingerprint is the same for identical graphs."""
    ops = [
        new_op(
            op_name="dense0",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="dense1",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="output",
            op_type=OpType.ADD,
            input_names=["dense0", "dense1"]),
    ]
    graph = new_graph(["input"], ["output"], ops)
    input_dict = {"input": jnp.ones((5, 5, 5))}
    fingerprint1 = fingerprint.fingerprint_graph(graph, {}, input_dict)
    fingerprint2 = fingerprint.fingerprint_graph(graph, {}, input_dict)
    self.assertEqual(fingerprint1, fingerprint2)

  def test_equal(self):
    """Tests whether the fingerprint is the same for equivalent graphs.

    The ops have different names and also have different topological sort.
    """
    ops1 = [
        new_op(
            op_name="dense",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="conv",
            op_type=OpType.CONV,
            op_kwargs={"features": 32,
                       "kernel_size": [3]},
            input_names=["input"]),
        new_op(
            op_name="output",
            op_type=OpType.ADD,
            input_names=["dense", "conv"]),
    ]
    graph1 = new_graph(["input"], ["output"], ops1)

    ops2 = [
        new_op(
            op_name="conv2",
            op_type=OpType.CONV,
            op_kwargs={"features": 32,
                       "kernel_size": [3]},
            input_names=["input"]),
        new_op(
            op_name="dense2",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="output",
            op_type=OpType.ADD,
            input_names=["dense2", "conv2"]),
    ]
    graph2 = new_graph(["input"], ["output"], ops2)

    input_dict = {"input": jnp.ones((5, 5, 5))}
    fingerprint1 = fingerprint.fingerprint_graph(graph1, {}, input_dict)
    fingerprint2 = fingerprint.fingerprint_graph(graph2, {}, input_dict)
    self.assertEqual(fingerprint1, fingerprint2)

  def test_not_equal(self):
    """Tests whether the fingerprint is different for non-equivalent graphs."""
    ops1 = [
        new_op(
            op_name="dense0",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="dense1",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="output",
            op_type=OpType.ADD,
            input_names=["dense0", "dense1"]),
    ]
    graph1 = new_graph(["input"], ["output"], ops1)

    ops2 = [
        new_op(
            op_name="conv2",
            op_type=OpType.CONV,
            op_kwargs={"features": 32,
                       "kernel_size": [3]},
            input_names=["input"]),
        new_op(
            op_name="dense2",
            op_type=OpType.DENSE,
            op_kwargs={"features": 32},
            input_names=["input"]),
        new_op(
            op_name="output",
            op_type=OpType.ADD,
            input_names=["dense2", "conv2"]),
    ]
    graph2 = new_graph(["input"], ["output"], ops2)

    input_dict = {"input": jnp.ones((5, 5, 5))}
    fingerprint1 = fingerprint.fingerprint_graph(graph1, {}, input_dict)
    fingerprint2 = fingerprint.fingerprint_graph(graph2, {}, input_dict)
    self.assertNotEqual(fingerprint1, fingerprint2)

if __name__ == "__main__":
  test.main()
