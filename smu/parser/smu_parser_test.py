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

# Lint as: python3
"""Tests for parsing and writing code."""
import os

from absl.testing import absltest

from smu.parser import smu_parser_lib
from smu.parser import smu_writer_lib

MAIN_DAT_FILE = 'x07_sample.dat'
SMU1_DAT_FILE = 'x01_sample.dat'
SMU2_DAT_FILE = 'x02_sample.dat'
STAGE1_DAT_FILE = 'x07_stage1.dat'
SMU1_STAGE1_DAT_FILE = 'x01_stage1.dat'
MINIMAL_DAT_FILE = 'x07_minimal.dat'
GOLDEN_PROTO_FILE = 'x07_sample.pbtxt'
TESTDATA_PATH = 'testdata'


class SmuParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.parser = smu_parser_lib.SmuParser(
        os.path.join(TESTDATA_PATH, MAIN_DAT_FILE))

  def test_expand_atom_types(self):
    self.assertEqual('ccccoofhhhhhhh', self.parser.expand_atom_types('c4o2fh7'))
    self.assertEqual('nnnnnooh', self.parser.expand_atom_types('n5o2h'))

  def test_roundtrip(self):
    """Tests a conversion from a SMU .dat file to protocol buffer and back."""
    smu_writer = smu_writer_lib.SmuWriter(annotate=False)
    for conformer, orig_contents in self.parser.process_stage2():
      smu_writer_lib.check_dat_formats_match(
          orig_contents,
          smu_writer.process_stage2_proto(conformer).splitlines())


class RoundtripTest(absltest.TestCase):
  """Test roundtrip of several files."""

  def try_roundtrip(self, filename, stage):
    parser = smu_parser_lib.SmuParser(
        os.path.join(TESTDATA_PATH, filename))
    writer = smu_writer_lib.SmuWriter(annotate=False)

    if stage == 'stage1':
      process_fn = parser.process_stage1
      writer_fn = writer.process_stage1_proto
    elif stage == 'stage2':
      process_fn = parser.process_stage2
      writer_fn = writer.process_stage2_proto
    else:
      raise ValueError(stage)

    for maybe_conformer, orig_contents in process_fn():
      if isinstance(maybe_conformer, Exception):
        raise maybe_conformer
      self.assertGreater(maybe_conformer.bond_topologies[0].bond_topology_id,
                         0)
      smu_writer_lib.check_dat_formats_match(
          orig_contents,
          writer_fn(maybe_conformer).splitlines())

  def test_minimal_input_stage2(self):
    self.try_roundtrip(MINIMAL_DAT_FILE, 'stage2')

  def test_smu1_stage2(self):
    self.try_roundtrip(SMU1_DAT_FILE, 'stage2')

  def test_smu2_stage2(self):
    self.try_roundtrip(SMU2_DAT_FILE, 'stage2')

  def test_stage1(self):
    self.try_roundtrip(STAGE1_DAT_FILE, 'stage1')

  def test_smu1_stage1(self):
    self.try_roundtrip(SMU1_STAGE1_DAT_FILE, 'stage1')


class ParseLongIdentifierTest(absltest.TestCase):

  def test_success_smu7(self):
    num_heavy_atoms, stoich, btid, cid = smu_parser_lib.parse_long_identifier(
        'x07_c4o2fh7.618451.001')
    self.assertEqual(7, num_heavy_atoms)
    self.assertEqual('c4o2fh7', stoich)
    self.assertEqual(618451, btid)
    self.assertEqual(1, cid)

  def test_success_smu2(self):
    num_heavy_atoms, stoich, btid, cid = smu_parser_lib.parse_long_identifier(
        'x02_c2h2.123.456')
    self.assertEqual(2, num_heavy_atoms)
    self.assertEqual('c2h2', stoich)
    self.assertEqual(123, btid)
    self.assertEqual(456, cid)

  def test_failure(self):
    with self.assertRaises(ValueError):
      smu_parser_lib.parse_long_identifier(
          'Im a little teapot, short and stout')


if __name__ == '__main__':
  absltest.main()
