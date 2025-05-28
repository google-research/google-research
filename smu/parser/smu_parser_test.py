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
"""Tests for parsing and writing code."""
import copy
import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
from tensorflow.io import gfile


from smu import dataset_pb2
from smu.parser import smu_parser_lib
from smu.parser import smu_writer_lib

MAIN_DAT_FILE = 'x07_sample.dat'
SMU1_DAT_FILE = 'x01_sample.dat'
SMU2_DAT_FILE = 'x02_sample.dat'
STAGE1_DAT_FILE = 'x07_stage1.dat'
SMU1_STAGE1_DAT_FILE = 'x01_stage1.dat'
MINIMAL_DAT_FILE = 'x07_minimal.dat'
GOLDEN_PROTO_FILE = 'x07_sample.pbtxt'
ATOMIC_INPUT = 'x07_first_atomic2_input.inp'
FINAL_MOL_STEM = 'final_mol'
TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


# Helper function to handle file access.
def get_file_contents(file_path):
  if not gfile.exists(file_path):
    raise FileNotFoundError
  with gfile.GFile(file_path) as f:
    return f.readlines()


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
    for molecule, orig_contents in self.parser.process_stage2():
      smu_writer_lib.check_dat_formats_match(
          orig_contents,
          smu_writer.process_stage2_proto(molecule).splitlines())

  def test_roundtrip_tweaked_bt(self):
    """Tests a conversion from a SMU .dat file to protocol buffer and back."""
    smu_writer = smu_writer_lib.SmuWriter(annotate=False)
    for molecule, orig_contents in self.parser.process_stage2():
      # We're going to mess with the molecule by perturbing the bond_toplogies.
      # The .dat format shoudl only ever use the starting topology, so we are
      # going to add some wrong bond topologies to make sure they are ignored.
      molecule.bond_topo.append(molecule.bond_topo[0])
      molecule.bond_topo.append(molecule.bond_topo[0])
      molecule.bond_topo[0].info = dataset_pb2.BondTopology.SOURCE_DDT
      molecule.bond_topo[1].info = dataset_pb2.BondTopology.SOURCE_CSD
      for bt in molecule.bond_topo[0:2]:
        bt.bond[0].bond_type = dataset_pb2.BondTopology.BOND_TRIPLE
        bt.topo_id += 9999
      smu_writer_lib.check_dat_formats_match(
          orig_contents,
          smu_writer.process_stage2_proto(molecule).splitlines())


class RoundtripTest(absltest.TestCase):
  """Test roundtrip of several files."""

  def try_roundtrip(self, filename, stage):
    parser = smu_parser_lib.SmuParser(os.path.join(TESTDATA_PATH, filename))
    writer = smu_writer_lib.SmuWriter(annotate=False)

    if stage == 'stage1':
      process_fn = parser.process_stage1
      writer_fn = writer.process_stage1_proto
    elif stage == 'stage2':
      process_fn = parser.process_stage2
      writer_fn = writer.process_stage2_proto
    else:
      raise ValueError(stage)

    for maybe_molecule, orig_contents in process_fn():
      if isinstance(maybe_molecule, Exception):
        raise maybe_molecule
      self.assertGreater(maybe_molecule.bond_topo[0].topo_id, 0)
      smu_writer_lib.check_dat_formats_match(
          orig_contents,
          writer_fn(maybe_molecule).splitlines())

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


class GoldenTest(parameterized.TestCase):

  def compare_list_items(self, list1, list2):
    self.assertLen(list1, len(list2))
    for i in range(len(list1)):
      line1 = list1[i]
      line2 = list2[i]
      if len(line2) > len(line1):
        line1 = line1[:-1]
        line2 = line2[:len(line1)]
      self.assertEqual(line1, line2)

  @parameterized.parameters(
      (SMU1_DAT_FILE, 'x01_sample.pbtxt'),
      (SMU2_DAT_FILE, 'x02_sample.pbtxt'),
      (MAIN_DAT_FILE, 'x07_sample.pbtxt'),
  )
  def test_dat_to_pbtxt(self, input_fn, expected_fn):
    # Note that this is partially a copy and paste from smu_parser (which is
    # what is used to regenerate the golden)
    full_input_fn = os.path.join(TESTDATA_PATH, input_fn)
    full_expected_fn = os.path.join(TESTDATA_PATH, expected_fn)

    multiple_molecules = dataset_pb2.MultipleMolecules()
    parser = smu_parser_lib.SmuParser(full_input_fn)
    for e, unused_orig_contents in parser.process_stage2():
      if isinstance(e, Exception):
        raise e
      multiple_molecules.molecules.append(e)

    got = ('# proto-file: '
           'third_party/google_research/google_research/smu/dataset.proto\n# '
           'proto-message: MultipleMolecules\n')
    got += text_format.MessageToString(multiple_molecules)

    expected = get_file_contents(full_expected_fn)

    print('Command line to regenerate:\npython3 parser/smu_parser.py '
          '--input_file {} --output_file {}'.format(full_input_fn,
                                                    full_expected_fn))

    self.compare_list_items([l.rstrip('\n') for l in expected],
                            got.splitlines())

  @parameterized.parameters(
      ('x01_sample.pbtxt', 'x01_sample_annotated.dat'),
      ('x02_sample.pbtxt', 'x02_sample_annotated.dat'),
      ('x07_sample.pbtxt', 'x07_sample_annotated.dat'),
  )
  def test_pbtxt_to_annotated_dat(self, input_fn, expected_fn):
    # Note that this is partially a copy and paste from smu_writer (which is
    # what is used to regenerate the golden)
    full_input_fn = os.path.join(TESTDATA_PATH, input_fn)
    full_expected_fn = os.path.join(TESTDATA_PATH, expected_fn)

    smu_proto = dataset_pb2.MultipleMolecules()
    raw_proto = '\n'.join(get_file_contents(full_input_fn))
    text_format.Parse(raw_proto, smu_proto)
    smu_writer = smu_writer_lib.SmuWriter(True)
    got = ''.join(
        smu_writer.process_stage2_proto(molecule)
        for molecule in smu_proto.molecules)

    expected = get_file_contents(full_expected_fn)

    print('Command line to regenerate:\npython3 parser/smu_writer.py '
          '--input_file {} --output_file {} --annotate True'.format(
              full_input_fn, full_expected_fn))

    self.assertEqual([l.rstrip('\n') for l in expected], got.splitlines())


class ParseLongIdentifierTest(absltest.TestCase):

  def test_success_smu7(self):
    num_heavy_atoms, stoich, btid, mid = smu_parser_lib.parse_long_identifier(
        'x07_c4o2fh7.618451.001')
    self.assertEqual(7, num_heavy_atoms)
    self.assertEqual('c4o2fh7', stoich)
    self.assertEqual(618451, btid)
    self.assertEqual(1, mid)

  def test_success_smu2(self):
    num_heavy_atoms, stoich, btid, mid = smu_parser_lib.parse_long_identifier(
        'x02_c2h2.123.456')
    self.assertEqual(2, num_heavy_atoms)
    self.assertEqual('c2h2', stoich)
    self.assertEqual(123, btid)
    self.assertEqual(456, mid)

  def test_failure(self):
    with self.assertRaises(ValueError):
      smu_parser_lib.parse_long_identifier(
          'Im a little teapot, short and stout')


class Atomic2InputTest(absltest.TestCase):

  def test_simple(self):
    parser = smu_parser_lib.SmuParser(
        os.path.join(TESTDATA_PATH, MAIN_DAT_FILE))
    molecule, _ = next(parser.process_stage2())
    expected = get_file_contents(os.path.join(TESTDATA_PATH, ATOMIC_INPUT))
    writer = smu_writer_lib.Atomic2InputWriter()

    smu_writer_lib.check_dat_formats_match(
        expected,
        writer.process(molecule, 0).splitlines())

  def test_error_cases(self):
    parser = smu_parser_lib.SmuParser(
        os.path.join(TESTDATA_PATH, MAIN_DAT_FILE))
    orig_molecule, _ = next(parser.process_stage2())
    writer = smu_writer_lib.Atomic2InputWriter()

    with self.assertRaises(ValueError):
      molecule = copy.deepcopy(orig_molecule)
      molecule.prop.calc.status = -1
      writer.process(molecule, 0)

    with self.assertRaises(ValueError):
      molecule = copy.deepcopy(orig_molecule)
      molecule.prop.calc.status = 19
      writer.process(molecule, 0)

    with self.assertRaises(ValueError):
      molecule = copy.deepcopy(orig_molecule)
      molecule.prop.ClearField('spe_std_hf_3')
      writer.process(molecule, 0)

    with self.assertRaises(ValueError):
      molecule = copy.deepcopy(orig_molecule)
      molecule.prop.ClearField('spe_std_mp2_3')
      writer.process(molecule, 0)


class CleanTextTest(absltest.TestCase):

  def test_final_mol(self):
    proto_fn = os.path.join(TESTDATA_PATH, FINAL_MOL_STEM + '.pbtxt')
    clean_text_fn = os.path.join(TESTDATA_PATH, FINAL_MOL_STEM + '.txt')

    smu_proto = dataset_pb2.MultipleMolecules()
    raw_proto = '\n'.join(get_file_contents(proto_fn))
    text_format.Parse(raw_proto, smu_proto)
    self.assertLen(smu_proto.molecules, 1)

    expected = get_file_contents(clean_text_fn)

    writer = smu_writer_lib.CleanTextWriter()
    got = writer.process(smu_proto.molecules[0])

    self.assertEqual([l.rstrip('\n') for l in expected], got.splitlines())


if __name__ == '__main__':
  absltest.main()
