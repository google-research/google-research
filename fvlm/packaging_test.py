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

"""Checks the F-VLM package layout.

Several directories shipped an empty file named `__init__` instead of
`__init__.py`. A directory without `__init__.py` is only a namespace package
portion, and Python prefers any regular package found elsewhere on `sys.path`
over a namespace portion found earlier. The `fvlm/utils` directory therefore
lost to an unrelated `utils` package installed in site-packages, giving

  ImportError: cannot import name 'clip_utils' from 'utils'
  (/usr/local/lib/python3.10/dist-packages/utils/__init__.py)

These tests keep the directories regular packages.
"""

import os

from absl.testing import absltest

_FVLM_DIR = os.path.dirname(os.path.abspath(__file__))


def _walk_dirs():
  for dirpath, dirnames, _ in os.walk(_FVLM_DIR):
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    yield dirpath


class PackageLayoutTest(absltest.TestCase):

  def test_no_init_file_is_missing_its_extension(self):
    """No file may be named `__init__` without the `.py` suffix."""
    stray = []
    for dirpath in _walk_dirs():
      if os.path.exists(os.path.join(dirpath, '__init__')):
        stray.append(os.path.relpath(dirpath, _FVLM_DIR))
    self.assertEqual(
        sorted(stray),
        [],
        'these directories have a file named "__init__" instead of '
        '"__init__.py", so they are not regular packages: ' + str(sorted(stray)),
    )


if __name__ == '__main__':
  absltest.main()
