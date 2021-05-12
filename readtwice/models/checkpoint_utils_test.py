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

"""Tests for checkpoint_utils."""
import tempfile
from typing import Text, Tuple

import tensorflow.compat.v1 as tf

from readtwice.models import checkpoint_utils


def _create_test_variables(
    outer_scope, inner_scope, var_c_name, var_e_name,
    var_n_name):
  # Keras layers can cause problems for `tf.train.init_from_checkpoint`
  # if not handled properly. Here we intentionally use Dense layers
  # to test whether the ckpt loading logic works.
  dense_layer = tf.keras.layers.Dense(10, name="dense")
  with tf.variable_scope(outer_scope):
    var_c = tf.get_variable(
        var_c_name, shape=[2, 4], initializer=tf.truncated_normal_initializer())
    var_d = dense_layer(var_c)
    with tf.variable_scope(inner_scope):
      var_e = tf.get_variable(
          var_e_name,
          shape=[2, 3],
          initializer=tf.truncated_normal_initializer())
      _ = tf.get_variable(
          var_n_name,
          shape=[3, 5],
          initializer=tf.truncated_normal_initializer())
  return var_c, var_d, var_e


class CheckpointUtilsTest(tf.test.TestCase):

  def _create_test_checkpoint(self, outer_scope, inner_scope,
                              var_c_name, var_e_name,
                              var_n_name):
    with tempfile.NamedTemporaryFile(suffix="ckpt_test") as ckpt_file:
      with self.session() as sess:
        var_c, var_d, var_e = _create_test_variables(outer_scope, inner_scope,
                                                     var_c_name, var_e_name,
                                                     var_n_name)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, ckpt_file.name)
    return ckpt_file.name, var_c, var_d, var_e

  def test_get_assignment_map_from_checkpoint(self):
    ckpt_path, expected_c, expected_d, expected_e = (
        self._create_test_checkpoint("scope_a", "scope_b", "var_c", "var_e",
                                     "var_f"))
    with self.cached_session() as sess:
      var_c, var_d, var_e = _create_test_variables("another_scope_a", "scope_b",
                                                   "var_c", "var_e", "var_g")

      (assignment_map, initialized_variable_names
      ) = checkpoint_utils.get_assignment_map_from_checkpoint(
          variables=sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
          ckpt_path=ckpt_path,
          variable_scope="another_scope_a/",
          ckpt_variable_scope="scope_a/")

      self.assertCountEqual(initialized_variable_names, [
          "another_scope_a/var_c:0", "another_scope_a/dense/bias:0",
          "another_scope_a/dense/kernel:0", "another_scope_a/scope_b/var_e:0"
      ])

      tf.train.init_from_checkpoint(ckpt_path, assignment_map)

      sess.run(tf.global_variables_initializer())
      self.assertAllClose(var_c, expected_c)
      self.assertAllClose(var_d, expected_d)
      self.assertAllClose(var_e, expected_e)

      # When require_all_variables_initialized = True, an error is raised
      # since a checkpoint variable corresponding to the variable
      # `another_scope_a/scope_b/var_g` cannot be found
      # in the ckpt_variable_scope `scope_a/`.
      with self.assertRaisesRegex(ValueError, "cannot be mapped"):
        (assignment_map, initialized_variable_names
        ) = checkpoint_utils.get_assignment_map_from_checkpoint(
            variables=sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
            ckpt_path=ckpt_path,
            variable_scope="another_scope_a/",
            ckpt_variable_scope="scope_a/",
            require_all_variables_initialized=True)

  def test_init_from_checkpoint_init_checkpoint_none(self):
    self.assertIsNone(checkpoint_utils.get_scaffold_fn(None, True))

  def test_init_from_checkpoint_single_scope_pair(self):
    ckpt_path, expected_c, expected_d, expected_e = (
        self._create_test_checkpoint("scope_a", "scope_b", "var_c", "var_e",
                                     "var_f"))
    with self.cached_session() as sess:
      var_c, var_d, var_e = _create_test_variables("scope_a_1", "scope_b",
                                                   "var_c", "var_e", "var_g")

      scaffold_fn = checkpoint_utils.get_scaffold_fn(
          ckpt_path, True, variable_scope_pairs=[("scope_a_1/", "scope_a/")])

      scaffold = scaffold_fn()
      self.assertIsInstance(scaffold, tf.train.Scaffold)

      sess.run(tf.global_variables_initializer())
      self.assertAllClose(var_c, expected_c)
      self.assertAllClose(var_d, expected_d)
      self.assertAllClose(var_e, expected_e)

  def test_init_from_checkpoint_multiple_scope_pairs(self):
    ckpt_path, expected_c, expected_d, expected_e = (
        self._create_test_checkpoint("scope_a", "scope_b", "var_c", "var_e",
                                     "var_f"))
    with self.cached_session() as sess:
      var_c_1, var_d_1, var_e_1 = _create_test_variables(
          "scope_a_1", "scope_b", "var_c", "var_e", "var_g")
      var_c_2, var_d_2, var_e_2 = _create_test_variables(
          "scope_a_2", "scope_b", "var_c", "var_e", "var_g")

      scaffold_fn = checkpoint_utils.get_scaffold_fn(
          ckpt_path,
          True,
          variable_scope_pairs=[("scope_a_1/", "scope_a/"),
                                ("scope_a_2/", "scope_a/")])

      scaffold = scaffold_fn()
      self.assertIsInstance(scaffold, tf.train.Scaffold)

      sess.run(tf.global_variables_initializer())
      self.assertAllClose(var_c_1, expected_c)
      self.assertAllClose(var_d_1, expected_d)
      self.assertAllClose(var_e_1, expected_e)
      self.assertAllClose(var_c_2, expected_c)
      self.assertAllClose(var_d_2, expected_d)
      self.assertAllClose(var_e_2, expected_e)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
