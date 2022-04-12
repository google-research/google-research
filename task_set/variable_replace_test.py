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

"""Tests for task_set.variable_replace."""

import sonnet as snt

from task_set import variable_replace
import tensorflow.compat.v1 as tf


class VaraibleReplaceTest(tf.test.TestCase):

  def test_variable_replace_getter(self):
    with self.test_session() as sess:
      context = variable_replace.VariableReplaceGetter()
      mod = snt.Linear(1, custom_getter=context)

      inp_data = tf.ones([10, 1])
      with context.use_variables():
        y1 = mod(inp_data)
        sess.run(tf.initialize_all_variables())
        np_y1 = sess.run(y1)

      values = context.get_variable_dict()

      new_values = {k: v + 1 for k, v in values.items()}

      with context.use_value_dict(new_values):
        np_y2 = mod(inp_data).eval()

      self.assertNear((np_y2 - np_y1)[0], 2, 1e-8)

      for v in values.values():
        v.assign(v + 1).eval()

      with context.use_variables():
        np_y3 = mod(inp_data).eval()
        self.assertNear((np_y3 - np_y2)[0], 0, 1e-8)


if __name__ == '__main__':
  tf.test.main()
