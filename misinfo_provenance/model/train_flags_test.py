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

"""Checks the flag definitions in misinfo_provenance/model/train.py.

`train.py` imports TensorFlow, so these tests do not import it. They lift the
`flags.DEFINE_*` statements out of the module with `ast` and execute only
those, which is enough to catch a duplicated flag: that raises
`DuplicateFlagError` as soon as the module is imported.
"""

import ast
import os

from absl import flags
from absl.testing import absltest

_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')


def _is_flag_definition(node):
  """True for a `flags.DEFINE_*` call, assigned or bare.

  `DEFINE_alias` is not assigned to a module-level name, so bare expression
  statements have to be included too.
  """
  if isinstance(node, ast.Assign):
    call = node.value
  elif isinstance(node, ast.Expr):
    call = node.value
  else:
    return False
  return (
      isinstance(call, ast.Call)
      and isinstance(call.func, ast.Attribute)
      and isinstance(call.func.value, ast.Name)
      and call.func.value.id == 'flags'
      and call.func.attr.startswith('DEFINE_')
  )


def _flag_name(node):
  """Returns the flag name string of a DEFINE_* node, if it has one."""
  call = node.value
  if isinstance(call.func, ast.Attribute) and call.func.attr == 'DEFINE_alias':
    # An alias intentionally re-registers the historical spelling.
    return None
  first = call.args[0] if call.args else None
  if isinstance(first, ast.Constant) and isinstance(first.value, str):
    return first.value
  return None


class TrainFlagsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    with open(_TRAIN) as f:
      self._source = f.read()
    self._tree = ast.parse(self._source, filename=_TRAIN)

  def _execute_flag_definitions(self):
    """Runs the module's flag definitions in a freshly cleared registry."""
    # `flags.DEFINE_*` writes to the live `flags.FLAGS` object, so reset that
    # object rather than rebinding the module attribute.
    flags.FLAGS.__init__()
    statements = [node for node in self._tree.body if _is_flag_definition(node)]
    self.assertNotEmpty(statements, 'no flag definitions found in train.py')
    module = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, _TRAIN, 'exec'), {'flags': flags})  # pylint: disable=exec-used

  def test_module_parses(self):
    self.assertIsInstance(self._tree, ast.Module)

  def test_flag_definitions_execute_without_duplicates(self):
    """Executing the DEFINE_* statements must not raise DuplicateFlagError."""
    self._execute_flag_definitions()
    self.assertIn('conv_output_layer', flags.FLAGS)

  def test_no_misspelled_flag_names(self):
    """Flag names should not contain the 'lenght' misspelling."""
    names = []
    for node in self._tree.body:
      if not _is_flag_definition(node):
        continue
      name = _flag_name(node)
      if name is not None:
        names.append(name)
    misspelled = [name for name in names if 'lenght' in name]
    self.assertEqual(misspelled, [], f'misspelled flag names: {misspelled}')

  def test_old_flag_spelling_still_resolves(self):
    """`dataset_length` exists and `--dataset_lenght` still sets it."""
    self._execute_flag_definitions()
    self.assertIn('dataset_length', flags.FLAGS)
    self.assertIn('dataset_lenght', flags.FLAGS)
    flags.FLAGS(['train', '--dataset_lenght=7'])
    self.assertEqual(flags.FLAGS['dataset_length'].value, 7)


if __name__ == '__main__':
  absltest.main()
