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

import filecmp
from os import path
import shutil
import unittest
from absl.testing import parameterized

import arxiv_latex_cleaner
from PIL import Image


class UnitTests(parameterized.TestCase):

  @parameterized.named_parameters({
      'testcase_name': 'no_comment',
      'line_in': 'Foo\n',
      'true_output': 'Foo\n'
  }, {
      'testcase_name': 'auto_ignore',
      'line_in': '%auto-ignore\n',
      'true_output': '%auto-ignore\n'
  }, {
      'testcase_name': 'percent',
      'line_in': r'100\% accurate\n',
      'true_output': r'100\% accurate\n'
  }, {
      'testcase_name': 'comment',
      'line_in': '  % Comment\n',
      'true_output': ''
  }, {
      'testcase_name': 'comment_inline',
      'line_in': 'Foo %Comment\n',
      'true_output': 'Foo %\n'
  })
  def test_remove_comments(self, line_in, true_output):
    self.assertEqual(arxiv_latex_cleaner._remove_comments(line_in), true_output)

  @parameterized.named_parameters({
      'testcase_name': 'all_pass',
      'inputs': ['abc', 'bca'],
      'patterns': ['a'],
      'true_outputs': ['abc', 'bca'],
  }, {
      'testcase_name': 'not_all_pass',
      'inputs': ['abc', 'bca'],
      'patterns': ['a$'],
      'true_outputs': ['bca'],
  })
  def test_keep_pattern(self, inputs, patterns, true_outputs):
    self.assertEqual(
        list(arxiv_latex_cleaner._keep_pattern(inputs, patterns)), true_outputs)

  @parameterized.named_parameters({
      'testcase_name': 'all_pass',
      'inputs': ['abc', 'bca'],
      'patterns': ['a'],
      'true_outputs': [],
  }, {
      'testcase_name': 'not_all_pass',
      'inputs': ['abc', 'bca'],
      'patterns': ['a$'],
      'true_outputs': ['abc'],
  })
  def test_remove_pattern(self, inputs, patterns, true_outputs):
    self.assertEqual(
        list(arxiv_latex_cleaner._remove_pattern(inputs, patterns)),
        true_outputs)


class IntegrationTests(unittest.TestCase):

  def _compare_files(self, filename, filename_true):
    if path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']:
      # We check only the sizes of the images, checking pixels would be too
      # complicated in case the resize implementations change.
      self.assertEqual(
          Image.open(filename).size,
          Image.open(filename_true).size,
          'Images {:s} was not resized properly.'.format(filename))
    else:
      self.assertTrue(
          filecmp.cmp(filename, filename_true),
          '{:s} and {:s} are not equal.'.format(filename, filename_true))

  def test_complete(self):
    out_path_true = path.join('test_data', 'tex_arXiv_true')
    self.out_path = path.join('test_data', 'tex_arXiv')

    # Make sure the folder does not exist, since we erase it in the test.
    if path.isdir(self.out_path):
      raise RuntimeError('The folder {:s} should not exist.'.format(
          self.out_path))

    arxiv_latex_cleaner._run_arxiv_cleaner({
        'input_folder': path.join('test_data', 'tex'),
        'images_whitelist': {
            'images/im2_included.jpg': 200
        },
        'im_size': 100
    })

    # Checks the set of files is the same as in the true folder.
    out_files = set(arxiv_latex_cleaner._list_all_files(self.out_path))
    out_files_true = set(arxiv_latex_cleaner._list_all_files(out_path_true))
    self.assertEqual(out_files, out_files_true)

    # Compares the contents of each file against the true value.
    for f1 in out_files:
      self._compare_files(
          path.join(self.out_path, f1), path.join(out_path_true, f1))

  def tearDown(self):
    shutil.rmtree(self.out_path)
    super(IntegrationTests, self).tearDown()


if __name__ == '__main__':
  unittest.main()
