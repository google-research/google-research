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

"""Static checks for the SiFeR entry point.

`misc` imports TensorFlow and torch, so these checks parse the sources with
`ast` instead of importing them. They catch calls to helpers that do not
exist, which is how `misc.download_files()` went unnoticed.
"""

import ast
import os

from absl.testing import absltest

_SIFER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN = os.path.join(_SIFER_DIR, 'train.py')
_MISC = os.path.join(_SIFER_DIR, 'utils', 'misc.py')


def _module_name(node):
  """Returns the module name for `misc` or `misc.x` attribute access."""
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    return _module_name(node.value)
  return None


def _attributes_used(tree, module_name):
  """Collects attribute names read off `module_name` in the given tree."""
  used = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Attribute):
      if _module_name(node) == module_name:
        used.add(node.attr)
  return used


def _module_scope_names(tree):
  """Top-level names bound in a module: functions, classes and imports."""
  names = set()
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      names.add(node.name)
    elif isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name):
          names.add(target.id)
    elif isinstance(node, ast.Import):
      for alias in node.names:
        names.add(alias.asname or alias.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
      for alias in node.names:
        names.add(alias.asname or alias.name)
  return names


class SiferMiscUsageTest(absltest.TestCase):

  def test_train_only_uses_helpers_defined_in_misc(self):
    """Every `misc.<name>` used in train.py must be defined in misc.py."""
    with open(_TRAIN) as f:
      train_tree = ast.parse(f.read(), filename=_TRAIN)
    with open(_MISC) as f:
      misc_tree = ast.parse(f.read(), filename=_MISC)

    used = _attributes_used(train_tree, 'misc')
    defined = _module_scope_names(misc_tree)

    self.assertNotEmpty(used, 'expected train.py to reference misc')
    missing = sorted(used - defined)
    self.assertEqual(
        missing,
        [],
        f'train.py calls misc.{missing} but misc.py defines only '
        f'{sorted(defined)}',
    )

  def test_train_parses(self):
    """train.py must be syntactically valid."""
    with open(_TRAIN) as f:
      ast.parse(f.read(), filename=_TRAIN)


if __name__ == '__main__':
  absltest.main()
