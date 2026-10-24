# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Checks the residual connections in summae/transformer.py.

`_PostProcess.__call__` performs the residual add itself:

    x += y
    return x

so a caller that writes `x += self._postprocess(x, y, training)` adds the
residual stream twice and gets `2 * x + dropout(y)`. Every call site must
therefore assign, not accumulate.

The module imports `tensorflow.contrib`, which only exists in TensorFlow 1.x,
so these checks parse the source with `ast` rather than importing it. That
keeps them runnable on current TensorFlow versions.
"""

import ast
import os

from absl.testing import absltest

_TRANSFORMER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'transformer.py'
)


def _is_postprocess_call(node):
  """True for a call like `self._ffn_postprocess(...)`."""
  return (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr.endswith('_postprocess')
  )


class ResidualConnectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    with open(_TRANSFORMER) as f:
      self._tree = ast.parse(f.read(), filename=_TRANSFORMER)

  def test_no_postprocess_call_uses_augmented_assignment(self):
    """A `+=` on a postprocess call doubles the residual stream."""
    offenders = []
    for node in ast.walk(self._tree):
      if not isinstance(node, ast.AugAssign):
        continue
      if _is_postprocess_call(node.value):
        offenders.append(
            f'{node.target.id if isinstance(node.target, ast.Name) else "?"} '
            f'{ast.unparse(node.value.func)} at line {node.lineno}'
        )
    self.assertEqual(
        offenders,
        [],
        'these call sites use `+=` on a postprocess helper that already '
        'performs the residual add, which counts the residual twice: '
        + str(offenders),
    )

  def test_every_postprocess_call_is_assigned(self):
    """Postprocess results must be stored, not discarded."""
    calls, assigned = 0, 0
    for node in ast.walk(self._tree):
      if isinstance(node, ast.Assign) and _is_postprocess_call(node.value):
        assigned += 1
    for node in ast.walk(self._tree):
      if isinstance(node, ast.Call) and _is_postprocess_call(node):
        calls += 1
    self.assertGreater(assigned, 0, 'no postprocess call sites found')
    self.assertEqual(
        calls, assigned,
        f'{calls - assigned} postprocess call(s) are neither assigned nor '
        'accumulated, so their residual contribution is discarded',
    )

  def test_postprocess_performs_the_residual_add(self):
    """Documents the helper's contract that the call sites rely on."""
    postprocess = None
    for node in self._tree.body:
      if isinstance(node, ast.ClassDef) and node.name == '_PostProcess':
        postprocess = node
    self.assertIsNotNone(postprocess)
    aug = [n for n in ast.walk(postprocess) if isinstance(n, ast.AugAssign)]
    self.assertTrue(
        aug,
        '_PostProcess is expected to add the residual itself; if it stopped '
        'doing so, the `x =` call sites would need a separate add',
    )


if __name__ == '__main__':
  absltest.main()
