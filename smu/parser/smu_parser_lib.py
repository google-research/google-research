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
"""A parser for Small Molecule Universe (SMU) files in custom Uni Basel format.

Used to read contents from SMU files and populates a corresponding protocol
buffer.
"""

import bz2
import collections
import enum
import math
import re
import traceback
from absl import logging
import numpy as np


from smu import dataset_pb2
from smu.parser import smu_utils_lib

# Number of items in BOND entry.
NUM_ITEMS_IN_BOND_ENTRY = 5

# We distinguish between the initial (cheap) coordinates and the optimized ones.
VALID_COORDINATE_LABELS = ['Initial Coords', 'Optimized Coords']

# Encoding order for quadru- and octopole moments.
RANK2_ENCODING_ORDER = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
RANK3_ENCODING_ORDER = [
    'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'xyy', 'yyz', 'xzz', 'yzz', 'xyz'
]

SEPARATOR_LINE = '=' * 80

EXCITATION_HEADER = ('Excitation energies and length rep. osc. strengths at '
                     'CC2/TZVP')

# All the possible single value properties and their associated fields
PROPERTIES_LABEL_FIELDS = collections.OrderedDict([
    ['PBE0D3/6-311Gd', 'spe_std_pbe0d3_6311gd'],
    ['PBE0/6-311Gd', 'spe_check_pbe0_6311gd_tmol'],
    ['PBE0/6-311GdMRCC', 'spe_check_pbe0_6311gd_mrcc'],
    ['PBE0/6-311GdORCA', 'spe_check_pbe0_6311gd_orca'],
    ['PBE0/6-311Gd(CAT)', 'spe_stdcat_pbe0_6311gd_tmol'],
    ['PBE0/6-311Gd(CAT)MRCC', 'spe_stdcat_pbe0_6311gd_mrcc'],
    ['PBE0/6-311Gd(CAT)ORCA', 'spe_stdcat_pbe0_6311gd_orca'],
    ['PBE0/aug-pc-1', 'spe_std_pbe0_augpc1'],
    ['HF/6-31Gd', 'spe_std_hf_631gd'],
    ['B3LYP/6-31++Gdp', 'spe_std_b3lyp_631ppgdp'],
    ['B3LYP/aug-pcS-1', 'spe_std_b3lyp_augpcs1'],
    ['PBE0/6-31++Gdp', 'spe_std_pbe0_631ppgdp'],
    ['PBE0/aug-pcS-1', 'spe_std_pbe0_augpcs1'],
    ['HF/TZVP', 'spe_std_hf_tzvp'],
    ['MP2/TZVP', 'spe_std_mp2_tzvp'],
    ['CC2/TZVP', 'spe_std_cc2_tzvp'],
    ['HF/3', 'spe_std_hf_3'],
    ['MP2/3', 'spe_std_mp2_3'],
    ['HF/4', 'spe_std_hf_4'],
    ['MP2/4', 'spe_std_mp2_4'],
    ['HF/(34)', 'spe_std_hf_34'],
    ['MP2/(34)', 'spe_std_mp2_34'],
    ['HF/CVTZ', 'spe_std_hf_cvtz'],
    ['MP2ful/CVTZ', 'spe_std_mp2full_cvtz'],
    ['HF/2sp', 'spe_std_hf_2sp'],
    ['MP2/2sp', 'spe_std_mp2_2sp'],
    ['CCSD/2sp', 'spe_std_ccsd_2sp'],
    ['CCSD(T)/2sp', 'spe_std_ccsd_t_2sp'],
    ['HF/2sd', 'spe_std_hf_2sd'],
    ['MP2/2sd', 'spe_std_mp2_2sd'],
    ['CCSD/2sd', 'spe_std_ccsd_2sd'],
    ['CCSD(T)/2sd', 'spe_std_ccsd_t_2sd'],
    ['HF/3Psd', 'spe_std_hf_3psd'],
    ['MP2/3Psd', 'spe_std_mp2_3psd'],
    ['CCSD/3Psd', 'spe_std_ccsd_3psd'],
    ['NUCREP', 'enuc'],  # special cased to refer to opt_geo
    ['NUM_OPT', 'number_of_optimization_runs'],
    ['NIMAG', 'number_imaginary_frequencies'],
    ['ZPE_unscaled', 'vib_zpe'],
])


class Atomic2FieldTypes(enum.Enum):
  STRING = 1
  SCALAR = 2
  TRIPLE = 3


ATOMIC_LABEL_FIELDS = collections.OrderedDict([
    ['AT2_BSR_LEFT', ('at2_gen_bsr_left', Atomic2FieldTypes.STRING)],
    ['AT2_BSR_RIGHT', ('at2_gen_bsr_right', Atomic2FieldTypes.STRING)],
    ['AT2_T1mol', ('wf_diag_t1_2sd', Atomic2FieldTypes.SCALAR)],
    ['AT2_T1exc', ('at2_gen_t1_exc', Atomic2FieldTypes.SCALAR)],
    ['AT2_ZPE', ('at2_std_zpe', Atomic2FieldTypes.TRIPLE)],
    ['AT2_ENE_B5', ('spe_comp_b5', Atomic2FieldTypes.SCALAR)],
    ['AT2_ENE_B6', ('spe_comp_b6', Atomic2FieldTypes.SCALAR)],
    ['AT2_ENE_ECCSD', ('spe_comp_eccsd', Atomic2FieldTypes.SCALAR)],
    ['AT2_BSE_B5', ('at2_std_b5_ereac', Atomic2FieldTypes.TRIPLE)],
    ['AT2_BSE_B6', ('at2_std_b6_ereac', Atomic2FieldTypes.TRIPLE)],
    ['AT2_BSE_ECCSD', ('at2_std_eccsd_ereac', Atomic2FieldTypes.TRIPLE)],
    ['AT2_AEe_B5', ('at2_std_b5_eae', Atomic2FieldTypes.TRIPLE)],
    ['AT2_AEe_B6', ('at2_std_b6_eae', Atomic2FieldTypes.TRIPLE)],
    ['AT2_AEe_ECCSD', ('at2_std_eccsd_eae', Atomic2FieldTypes.TRIPLE)],
    ['AT2_AE0_B5', ('at2_std_b5_ea0', Atomic2FieldTypes.TRIPLE)],
    ['AT2_AE0_B6', ('at2_std_b6_ea0', Atomic2FieldTypes.TRIPLE)],
    ['AT2_AE0_ECCSD', ('at2_std_eccsd_ea0', Atomic2FieldTypes.TRIPLE)],
    ['AT2_HF0_B5', ('at2_std_b5_hf0', Atomic2FieldTypes.TRIPLE)],
    ['AT2_HF0_B6', ('at2_std_b6_hf0', Atomic2FieldTypes.TRIPLE)],
    ['AT2_HF0_ECCSD', ('at2_std_eccsd_hf0', Atomic2FieldTypes.TRIPLE)],
    ['AT2_HF298_B5', ('at2_std_b5_hf298', Atomic2FieldTypes.TRIPLE)],
    ['AT2_HF298_B6', ('at2_std_b6_hf298', Atomic2FieldTypes.TRIPLE)],
    ['AT2_HF298_ECCSD', ('at2_std_eccsd_hf298', Atomic2FieldTypes.TRIPLE)],
])

PARTIAL_CHARGES_LABEL_FIELDS = collections.OrderedDict([
    ['MUL-PBE0/aug-pc-1', 'chg_mul_pbe0_augpc1'],
    ['LOE-PBE0/aug-pc-1', 'chg_loe_pbe0_augpc1'],
    ['NAT-PBE0/aug-pc-1', 'chg_nat_pbe0_augpc1'],
    ['PON-PBE0/aug-pc-1', 'partial_charges_paboon_pbe0_aug_pc_1'],
    ['ESP-PBE0/aug-pc-1', 'chg_esp_pbe0_augpc1'],
    ['MUL-HF/6-31Gd', 'chg_mul_hf_631gd'],
    ['LOE-HF/6-31Gd', 'chg_loe_hf_631gd'],
    ['NAT-HF/6-31Gd', 'chg_nat_hf_631gd'],
    ['PON-HF/6-31Gd', 'partial_charges_paboon_hf_6_31gd'],
    ['ESP-HF/6-31Gd', 'chg_esp_hf_631gd'],
])

NMR_ISOTROPIC_SHIELDINGS_LABEL_FIELDS = collections.OrderedDict([
    ['B3LYP/6-31++Gdp', 'nmr_b3lyp_631ppgdp'],
    ['B3LYP/aug-pcS-1', 'nmr_b3lyp_augpcs1'],
    ['PBE0/6-31++Gdp', 'nmr_pbe0_631ppgdp'],
    ['PBE0/aug-pcS-1', 'nmr_pbe0_augpcs1'],
])

# The Gaussian sanity check section is a label with 1 or 2 floats after it.
# These are tuples of the prefix followed by a list of the proto fields from
# GaussianSanityCheck.
GAUSSIAN_SANITY_CHECK_LINES = [
    ('Dev. Energy PBE0/6-311G*', ['energy_pbe0_6_311gd_diff']),
    ('Max. Force ', ['max_force']),
    ('Max. and Avg Dev. Frequencies',
     ['max_frequencies_diff', 'mean_frequencies_diff']),
    ('Max. and Avg Dev. Intensities',
     ['max_intensities_diff', 'mean_intensities_diff']),
    ('Dev. Energy HF/6-31G* ', ['energy_hf_6_31gd_diff']),
    ('Max. Dev. Dipole Components', ['max_dipole_components_diff']),
    ('Max. Dev. Quadrup. Components', ['max_quadropole_components_diff']),
    ('Max. Dev. Octupole Components', ['max_octopole_components_diff']),
]


class SmuKnownError(Exception):
  pass


class SmuBondTypeZeroError(SmuKnownError):
  pass


class SmuOverfullFloatFieldError(SmuKnownError):
  pass


class ParseModes(enum.Enum):
  INITIALIZE = 1
  SKIP = 2
  KEYVALUE = 3
  ALTERNATING = 4
  INTERLEAVED = 5
  RAW = 6
  BYLABEL = 7
  SKIP_BLANK_LINES = 8


_LONG_IDENTIFIER_RE = re.compile(r'^x(\d+)_(\w+)\.(\d+)\.(\d+)$')


def parse_long_identifier(identifier):
  """Parses a long style identifier into its components.

  Long identifiers look like:
  x07_c4o2fh7.618451.001

  Args:
    identifier: string

  Returns:
    heavy atom count(int), stoichiometry(string), bond topology id (int),
    molecule id (int)

  Raises:
    ValueError: If the string could not be parsed
  """
  match = _LONG_IDENTIFIER_RE.match(identifier)
  if not match:
    raise ValueError('Could not interpret "{}"'.format(identifier))

  return (int(match.group(1)), match.group(2), int(match.group(3)),
          int(match.group(4)))


class SmuParser:
  """A class to handle parsing of a single SMU file."""

  def __init__(self, input_file):
    """Initialize the parser class.

    Args:
      input_file: The path to the SMU input file.
    """
    self.input_file = input_file
    self._raw_contents = None
    self._molecule = None
    self.line_num = None

  def _input_generator(self):
    """Yields lines from from input_file."""
    # This import is here to avoid dependency on gfile except while essential.
    # This function is the only one that uses gfile.
    from tensorflow.io import gfile  # pylint: disable=g-import-not-at-top

    if not gfile.exists(self.input_file):
      raise FileNotFoundError

    if self.input_file.endswith('.bz2'):
      logging.info('Opening %s as bzip2 file', self.input_file)
      with gfile.GFile(self.input_file, 'rb') as compressed_f:
        with bz2.BZ2File(compressed_f, 'rb') as f:
          for bin_line in f:
            yield bin_line.decode('utf-8').rstrip('\n')
    else:
      logging.info('Opening %s via gfile', self.input_file)
      with gfile.GFile(self.input_file, 'r') as f:
        for line in f:
          yield line.rstrip('\n')

  def _read_next_chunk(self, line_generator):
    """Reads a chunk of input from line_generator and sets _raw_contents.

    Args:
      line_generator: generator for lines of input, from _input_generator()

    Returns:
      bool on whether we have a next chunk
    """
    self._raw_contents = [SEPARATOR_LINE]
    self.line_num = 0
    try:
      line = next(line_generator)
    except StopIteration:
      return False
    # We have to handle the case where the separator line is the first thing
    # and where the separator line has already been consumed from the last call
    # to _read_next_chunk
    if line != SEPARATOR_LINE:
      self._raw_contents.append(line)

    while True:
      try:
        line = next(line_generator)
      except StopIteration:
        return True
      if line == SEPARATOR_LINE:
        return True
      self._raw_contents.append(line)

  def _next_line(self):
    if self.line_num >= len(self._raw_contents):
      return None
    return self._raw_contents[self.line_num]

  def _next_line_startswith(self, prefix):
    """Whether the next line to parse starts with prefix."""
    next_line = self._next_line()
    return next_line and next_line.startswith(prefix)

  def parse(self, mode, label=None, num_lines=0, allowed_keys=None):
    """Parses a number of lines by using a given mode.

    This is the main parsing routine for the Basel SMU file format. It supports
    a number of modes to tell the parser how data is stored in the file, which
    changes frequently throughout and needs to be specified for each section of
    data. The length of each section (in number of lines) has to be specified
    as well where available. A mode like 'BYLABEL' can be used to parse sections
    of unknown length.

    Args:
      mode: Mode used for parsing.
      label: A label used to identify lines belonging to a section.
      num_lines: The number of lines to parse next.
      allowed_keys: if mode=KEYVALUE, the list of allowed keys. Only lines
        starting with one of these keys will be included up until the first line
        that does not split into two values or has a disallowed key, and
        num_lines becomes a maximum to consider.'

    Returns:
      Depending on the mode, the method returns nothing (INITIALIZE, SKIP), a
      section of file (RAW, BYLABEL), a dict (KEYVALUE) or a tuple of labels and
      values (all others).
    """
    first_line = self.line_num

    if mode == ParseModes.SKIP_BLANK_LINES:
      while (self.line_num < len(self._raw_contents) and
             not str(self._raw_contents[self.line_num]).strip()):
        self.line_num += 1
      return

    # If only label is known, determine size of section.
    if mode == ParseModes.BYLABEL:
      assert label in self._raw_contents[
          first_line], 'Label %s not found in line %s.' % (
              label, self._raw_contents[first_line])
      last_line = first_line + 1
      while (last_line < len(self._raw_contents) and
             label in self._raw_contents[last_line]):
        last_line += 1
      num_lines = last_line - first_line
    if mode is not ParseModes.INITIALIZE:
      message = 'Parser was called with invalid number of lines %d.' % num_lines
      assert num_lines > 0, message

    last_line = self.line_num + num_lines
    section = self._raw_contents[first_line:last_line]
    self.line_num += num_lines

    if mode == ParseModes.INITIALIZE:
      # Resets the line number and returns.
      self.line_num = 0
      return
    elif mode == ParseModes.RAW or mode == ParseModes.BYLABEL:
      return section
    elif mode == ParseModes.SKIP:
      # Skip over a number of lines by updating the index.
      return
    elif mode == ParseModes.KEYVALUE:
      # Lines with simple label-value pairs separated by a delimiter.
      out = {}
      for line_idx, line in enumerate(section):
        splits = line.split()
        if len(splits) != 2 or (allowed_keys and splits[0] not in allowed_keys):
          # line_num was already advanced. We have to fix it up for the actual
          # number of lines consumed.
          self.line_num += line_idx - num_lines
          return out
        out[splits[0]] = splits[1]
      return out
    elif mode == ParseModes.ALTERNATING:
      # Alternating lines, where odd lines contain (a number of) labels and even
      # lines contain the corresponding values (one value per label).
      labels = [line.strip() for i, line in enumerate(section) if i % 2 == 0]
      labels = ' '.join(labels).split()
      values = [line for i, line in enumerate(section) if i % 2 == 1]
      values = [int(value) for value in ' '.join(values).split()]
      return labels, values
    elif mode == ParseModes.INTERLEAVED:
      # Labels and values are interleaved in whitespace-separated columns. Odd
      # columns contain the keys and even columns the corresponding values.
      # KEYVALUE is a special case of this with only one pair.
      serialized_section = ' '.join(section)
      labels = [
          label for i, label in enumerate(serialized_section.split())
          if i % 2 == 0
      ]
      values = [
          value for i, value in enumerate(serialized_section.split()) if i % 2
      ]
      return labels, values
    else:
      logging.error('Unknown parse mode %s.', mode)

  def parse_stage1_header(self):
    """The first line after a divider describes a new molecule."""
    header = str(self.parse(ParseModes.RAW, num_lines=1)[0])
    vals = header.split()
    if len(vals) != 7:
      raise ValueError('Stage 1 header line %s invalid, need 7 values.' %
                       header)
    if vals[0] == '*****':
      # This is a fortran numeric overflow. We don't actually care about this
      # info, so we just drop an invalid value in and regenerate '*****' in the
      # writer.
      self._molecule.original_molecule_index = -1
    else:
      self._molecule.original_molecule_index = int(vals[0])
    errors = self._molecule.prop.calc
    for field, val in zip(smu_utils_lib.STAGE1_ERROR_FIELDS, vals[1:5]):
      setattr(errors, field, int(val))
    # Note that vals[6] is the molecule identifier which we ignore in favor of
    # the "ID" line.
    # vals[5] is the number of atoms
    return int(vals[5])

  def parse_stage2_header(self):
    """The first line after a divider describes a new molecule."""
    header = str(self.parse(ParseModes.RAW, num_lines=1)[0])
    if len(header.split()) != 3:
      raise ValueError(
          'Header line %s invalid, need molecule, #atoms, and id.' % header)
    molecule_index, num_atoms, unused_identifier = header.split()
    if molecule_index == '*****':
      # This is a fortran numeric overflow. We don't actually care about this
      # info, so we just drop an invalid value in and regenerate '*****' in the
      # writer.
      self._molecule.original_molecule_index = -1
    else:
      self._molecule.original_molecule_index = int(molecule_index)
    return int(num_atoms)

  def parse_database(self):
    """Parse the line indicating what database the molecule should go to.

    This line looks like:
    Database   standard
    """
    line = str(self.parse(ParseModes.RAW, num_lines=1)[0])
    parts = line.split()
    if len(parts) != 2:
      raise ValueError('Expected database line, got: {}'.format(line))
    if parts[0] != 'Database':
      raise ValueError('Bad keyword on database line, got: {}'.format(parts[0]))
    if parts[1] == 'standard':
      self._molecule.prop.calc.which_database = dataset_pb2.STANDARD
    elif parts[1] == 'complete':
      self._molecule.prop.calc.which_database = dataset_pb2.COMPLETE
    else:
      raise ValueError('Expected database indicator, got: {}'.format(parts[1]))

  def parse_error_codes(self):
    """Parses the error section with the warning flags."""
    lines = iter(self.parse(ParseModes.RAW, num_lines=6))
    errors = self._molecule.prop.calc

    parts = str(next(lines)).split()
    assert (len(parts) == 2 and parts[0]
            == 'Status'), ('Expected Status line, got {}'.format(parts))
    errors.status = int(parts[1])

    parts = str(next(lines)).split()
    assert (len(parts) == 3 and parts[0]
            == 'Warn_T1'), ('Expected Status line, got {}'.format(parts))
    errors.warn_t1 = int(parts[1])
    errors.warn_delta_t1 = int(parts[2])

    parts = str(next(lines)).split()
    assert (len(parts) == 3 and parts[0]
            == 'Warn_BSE'), ('Expected Status line, got {}'.format(parts))
    errors.warn_bse_b6 = int(parts[1])
    errors.warn_bse_eccsd = int(parts[2])

    parts = str(next(lines)).split()
    assert (len(parts) == 4 and parts[0]
            == 'Warn_EXC'), ('Expected Status line, got {}'.format(parts))
    errors.warn_exc_ene = int(parts[1])
    errors.warn_exc_osmin = int(parts[2])
    errors.warn_exc_osmax = int(parts[3])

    parts = str(next(lines)).split()
    assert (len(parts) == 3 and parts[0]
            == 'Warn_VIB'), ('Expected Status line, got {}'.format(parts))
    errors.warn_vib_linear = int(parts[1])
    errors.warn_vib_imag = int(parts[2])

    parts = str(next(lines)).split()
    assert (len(parts) == 2 and parts[0]
            == 'Warn_NEG'), ('Expected Status line, got {}'.format(parts))
    errors.warn_bsr_neg = int(parts[1])

  def parse_bond_topology(self):
    """Parse region with adjancy matrix, hydrogen count, smiles, and atom types."""
    adjacency_code = str(self.parse(ParseModes.RAW, num_lines=1)[0]).strip()
    hydrogen_counts = [
        int(count)
        for count in str(self.parse(ParseModes.RAW, num_lines=1)[0]).strip()
    ]

    smiles = self.parse(ParseModes.RAW, num_lines=1)[0]

    entry_id = str(self.parse(ParseModes.RAW, num_lines=1)[0]).strip()
    assert entry_id.startswith('x'), 'Expected line like x02_c2h2'
    atom_types = entry_id[4:].lower()
    expanded_atom_types = self.expand_atom_types(atom_types)
    self._molecule.bond_topo.add()
    self._molecule.bond_topo[-1].CopyFrom(
        smu_utils_lib.create_bond_topology(expanded_atom_types, adjacency_code,
                                           hydrogen_counts))
    self._molecule.bond_topo[-1].smiles = str(smiles).replace('\'', '').strip()
    # Note that we only set source to STARTING and not DDT. This is because this
    # geometry may not actually pass the DDT criteria. We let the later geometry
    # detection take care of this.
    self._molecule.bond_topo[-1].info = (
        dataset_pb2.BondTopology.SOURCE_STARTING)

  def expand_atom_types(self, atom_types):
    """Takes an abbreviated atom composition, such as c4o2fh7, and expands it.

    In the provided example, c4o2fh7 will be expanded to ccccoofhhhhhhh. The
    sequence is significant, since it determines the order in which atom
    coordinates and properties are stored in a SMU file.

    Args:
      atom_types: A sorted string of atoms in the molecule and their count.

    Returns:
      The expanded atom types with one letter per atom.
    """
    expanded_atom_types = ''
    current_type = atom_types[0]
    character_count = 0
    for i, character in enumerate(atom_types):
      if character.isalpha():
        if i > 0:
          if not character_count:
            character_count = 1
          expanded_atom_types += current_type * character_count
        current_type = character
        character_count = 0
      else:
        character_count = character_count * 10 + int(character)

    # Catches the case where the last character has no digits afterward
    if not character_count:
      character_count = 1

    return expanded_atom_types + current_type * character_count

  def parse_identifier(self):
    """Extracts and sets the bond topology and molecule identifier."""
    line = str(self.parse(ParseModes.RAW, num_lines=1)[0])
    id_str, topo_id_str, mol_id_str = line.split()
    assert id_str == 'ID', ('Identifier line should start with "ID", got %s.' %
                            line)
    topo_id = int(topo_id_str)
    # Special casing for SMU1. Fun.
    if smu_utils_lib.special_case_bt_id_from_dat_id(
        topo_id, self._molecule.bond_topo[-1].smiles):
      topo_id = smu_utils_lib.special_case_bt_id_from_dat_id(
          topo_id, self._molecule.bond_topo[-1].smiles)

    self._molecule.bond_topo[-1].topo_id = topo_id
    self._molecule.mol_id = (topo_id * 1000 + int(mol_id_str))

  def parse_cluster_info(self, num_lines):
    """Stores a string describing the compute cluster used for computations."""
    cluster_info = self.parse(ParseModes.RAW, num_lines=num_lines)
    self._molecule.prop.compute_cluster_info = '\n'.join(cluster_info) + '\n'

  def parse_stage1_timings(self):
    """Parses recorded timings for different computation steps.

    In the stage 1 files, the labels are implicit and only the first two ever
    have valid values so we'll just pull those out.
    """
    line = self.parse(ParseModes.RAW, num_lines=1)[0]
    values = str(line).strip().split()
    if values[0] != 'TIMINGS':
      raise ValueError('Bad timing line: {}'.format(line))
    if len(values) != 11:
      raise ValueError('Timing line has {} components, not 11: {}'.format(
          len(values), line))
    for v in values[3:]:
      if v != '-1':
        raise ValueError(
            'Expected all trailing timing to be -1, got {}'.format(v))

    calculation_statistics = self._molecule.prop.calculation_statistics
    calculation_statistics.add(computing_location='Geo', timings=values[1])
    calculation_statistics.add(computing_location='Force', timings=values[2])

  def parse_stage2_timings(self):
    """Parses recorded timings for different computation steps."""
    section = self.parse(ParseModes.RAW, num_lines=2)
    labels = str(section[0]).strip().split()
    values = str(section[1]).strip().split()[1:]
    assert len(labels) == len(
        values), 'Length mismatch between values %s and %s labels.' % (values,
                                                                       labels)
    calculation_statistics = self._molecule.prop.calculation_statistics
    for label, value in zip(labels, values):
      calculation_statistics.add()
      calculation_statistics[-1].computing_location = label
      calculation_statistics[-1].timings = value

  def parse_bonds(self):
    """Ignores BOND section is redudant given the topology information."""
    # Some entries do not have BOND sections, which is fine.
    if not self._next_line_startswith('BOND'):
      return
    self.parse(ParseModes.BYLABEL, label='BOND')

  def parse_gradient_norms(self):
    """Parses initial and optimized geometry energies and gradient norms."""
    section = self.parse(ParseModes.RAW, num_lines=2)
    assert str(section[0]).startswith('E_ini/G_norm') and str(
        section[1]).startswith(
            'E_opt/G_norm'
        ), 'Unable to parse section for gradient norm: %s.' % section
    items = str(section[0]).split()
    self._molecule.ini_geo.add()
    self._molecule.ini_geo[0].energy.val = float(items[1])
    self._molecule.ini_geo[0].gnorm.val = float(items[2])
    items = str(section[1]).split()
    self._molecule.opt_geo.energy.val = float(items[1])
    self._molecule.opt_geo.gnorm.val = float(items[2])

  def parse_coordinates(self, label, num_atoms):
    """Parses a section defining a molecule's geometry in Cartesian coordinates.

    Args:
      label: One of 'Initial Coords' and 'Optimized Coords' to determine the
        type of coordinates.
      num_atoms: Number of atoms in the molecule (with one matching line of atom
        coordinates).
    """
    if not self._next_line_startswith(label):
      return
    coordinate_section = self.parse(ParseModes.RAW, num_lines=num_atoms)
    assert label in VALID_COORDINATE_LABELS, 'Unknown label %s.' % label
    molecule = self._molecule
    if label == 'Initial Coords':
      if not molecule.ini_geo:
        molecule.ini_geo.add()
      geometry = molecule.ini_geo[0]
    else:
      geometry = molecule.opt_geo
    for line in coordinate_section:
      label1, label2, unused_atomic_number, x, y, z = str(line).strip().split()
      assert '%s %s' % (
          label1, label2
      ) == label, 'Found line with incorrect label "%s %s". Expected %s.' % (
          label1, label2, label)
      geometry.atompos.add()
      geometry.atompos[-1].x = float(x)
      geometry.atompos[-1].y = float(y)
      geometry.atompos[-1].z = float(z)

  def parse_rotational_constants(self):
    """Parses rotational constants vector (MHz)."""
    if not self._next_line_startswith('Rotational constants'):
      return
    constants = self.parse(ParseModes.RAW, num_lines=1)[0]
    values = str(constants).strip().split()[-3:]
    self._molecule.opt_geo.brot.val.extend(float(x) for x in values)

  def parse_symmetry_used(self):
    """Parses whether or not symmetry was used in the computation."""
    if not self._next_line_startswith('Symmetry used in calculation'):
      return
    symmetry = self.parse(ParseModes.RAW, num_lines=1)[0]
    self._molecule.prop.symmetry_used_in_calculation = str(
        symmetry).strip().split()[-1] != 'no'

  def parse_frequencies_and_intensities(self, num_atoms, header):
    """Parses a section with harmonic frequencies and intensities.

    There are 3 values per atom of each.

    Args:
      num_atoms: Number of atoms in the molecule.
      header: Whether to expect a 'Frequencies and intensities' (different in
        stage1 and stage2 files)
    """
    if header:
      if not self._next_line_startswith('Frequencies and intensities'):
        return
      # Skip the header line
      self.parse(ParseModes.SKIP, num_lines=1)
    else:
      # This is a bit of a hack. The frequencies and intensities can be missing
      # from a stage1 file. But the only way to know this is that these are the
      # last things expected in the entry.
      next_line = self._next_line()
      if not next_line or next_line.startswith(SEPARATOR_LINE):
        return

    section = self.parse(
        ParseModes.RAW, num_lines=math.ceil(3 * num_atoms / 10))
    section = [str(line).strip() for line in section]
    section = ' '.join(section).split()
    vib_freq = self._molecule.prop.vib_freq
    for value in section:
      vib_freq.val.append(float(value))

    section = self.parse(
        ParseModes.RAW, num_lines=math.ceil(3 * num_atoms / 10))
    section = [str(line).strip() for line in section]
    section = ' '.join(section).split()
    vib_intens = self._molecule.prop.vib_intens
    for value in section:
      vib_intens.val.append(float(value))

  def parse_gaussian_sanity_check(self):
    """Parses the gaussian sanity check section (present in SMU1-6)."""
    if not self._next_line_startswith('Gaussian sanity check'):
      return

    section = self.parse(ParseModes.RAW, num_lines=9)
    # An example line with two floats (most only have one)
    # Max. and Avg Dev. Frequencies       0.463700      0.144060
    # 012345678901234567890123456789012345678901234567890123456789
    for line, (prefix, fields) in zip(section[1:], GAUSSIAN_SANITY_CHECK_LINES):
      assert str(line).startswith(prefix)
      parts = line.split()
      if len(fields) == 1:
        setattr(self._molecule.prop.gaussian_sanity_check, fields[0],
                float(parts[-1]))
      elif len(fields) == 2:
        setattr(self._molecule.prop.gaussian_sanity_check, fields[0],
                float(parts[-2]))
        setattr(self._molecule.prop.gaussian_sanity_check, fields[1],
                float(parts[-1]))
      else:
        raise ValueError(f'Bad fields length {len(fields)}')

  def parse_vib_mode(self, num_atoms):
    """Parses a repeating section containing a number of normal modes.

    Args:
      num_atoms: Number of atoms in the molecule.

    Raises:
      ValueError: if line format is unexpected
    """
    if not self._next_line_startswith('Normal modes'):
      return

    # Skip the header line
    self.parse(ParseModes.SKIP, num_lines=1)

    properties = self._molecule.prop
    for _ in range(3 * num_atoms):
      if not self._next_line_startswith('Mode'):
        raise ValueError('Parsing vib_mode, expect Mode line, got: {}'.format(
            self._raw_contents[self.line_num]))

      self.parse(ParseModes.SKIP, num_lines=1)  # 'Mode   #i'

      section = self.parse(
          ParseModes.RAW, num_lines=math.ceil(3 * num_atoms / 10))
      section = [str(line).strip() for line in section]
      section = ' '.join(section).split()
      properties.vib_mode.add()
      for triplet in zip(section[::3], section[1::3], section[2::3]):
        properties.vib_mode[-1].disp.add()
        properties.vib_mode[-1].disp[-1].x = float(triplet[0])
        properties.vib_mode[-1].disp[-1].y = float(triplet[1])
        properties.vib_mode[-1].disp[-1].z = float(triplet[2])

  def parse_property_list(self):
    """Parses a section of properties stored as key-value pairs."""
    labels_and_values = self.parse(
        ParseModes.KEYVALUE,
        num_lines=50,
        allowed_keys=PROPERTIES_LABEL_FIELDS.keys())
    properties = self._molecule.prop
    for label in labels_and_values:
      if label in ['NIMAG', 'NUM_OPT']:
        setattr(properties, PROPERTIES_LABEL_FIELDS[label],
                int(labels_and_values[label]))
      elif label == 'NUCREP':
        value = float(labels_and_values[label])
        self._molecule.opt_geo.enuc.val = value
      else:
        value = float(labels_and_values[label])
        getattr(properties, PROPERTIES_LABEL_FIELDS[label]).val = value

  def parse_diagnostics(self):
    """Parses D1 and T1 diagnostics."""
    properties = self._molecule.prop

    if self._next_line_startswith('D1DIAG'):
      line = self.parse(ParseModes.RAW, num_lines=1)[0]
      items = str(line).strip().split()
      properties.wf_diag_d1_2sp.val = float(items[2])

    if self._next_line_startswith('T1DIAG'):
      line = self.parse(ParseModes.RAW, num_lines=1)[0]
      items = str(line).strip().split()
      properties.wf_diag_t1_2sp.val = float(items[2])
      properties.wf_diag_t1_2sd.val = float(items[4])
      properties.wf_diag_t1_3psd.val = float(items[6])

  def parse_homo_lumo(self):
    """Parses HOMO and LUMO values (at different levels of theory).

    Raises:
      ValueError: for unknown level of theory
    """
    if not self._next_line_startswith('HOMO/LUMO'):
      return

    homo_lumo_data = self.parse(ParseModes.BYLABEL, label='HOMO/LUMO')
    properties = self._molecule.prop
    for line in homo_lumo_data:
      items = str(line).strip().split()
      if items[1] == 'PBE0/6-311Gd':
        properties.orb_ehomo_pbe0_6311gd.val = float(items[2])
        properties.orb_elumo_pbe0_6311gd.val = float(items[3])
      elif items[1] == 'PBE0/aug-pc-1':
        properties.orb_ehomo_pbe0_augpc1.val = float(items[2])
        properties.orb_elumo_pbe0_augpc1.val = float(items[3])
      elif items[1] == 'PBE0/6-31++Gdp':
        properties.orb_ehomo_pbe0_631ppgdp.val = float(items[2])
        properties.orb_elumo_pbe0_631ppgdp.val = float(items[3])
      elif items[1] == 'PBE0/aug-pcS-1':
        properties.orb_ehomo_pbe0_augpcs1.val = float(items[2])
        properties.orb_elumo_pbe0_augpcs1.val = float(items[3])
      elif items[1] == 'HF/6-31Gd':
        properties.orb_ehomo_hf_631gd.val = float(items[2])
        properties.orb_elumo_hf_631gd.val = float(items[3])
      elif items[1] == 'HF/TZVP':
        properties.orb_ehomo_hf_tzvp.val = float(items[2])
        properties.orb_elumo_hf_tzvp.val = float(items[3])
      elif items[1] == 'HF/3':
        properties.orb_ehomo_hf_3.val = float(items[2])
        properties.orb_elumo_hf_3.val = float(items[3])
      elif items[1] == 'HF/4':
        properties.orb_ehomo_hf_4.val = float(items[2])
        properties.orb_elumo_hf_4.val = float(items[3])
      elif items[1] == 'HF/CVTZ':
        properties.orb_ehomo_hf_cvtz.val = float(items[2])
        properties.orb_elumo_hf_cvtz.val = float(items[3])
      elif items[1] == 'B3LYP/6-31++Gdp':
        properties.orb_ehomo_b3lyp_631ppgdp.val = float(items[2])
        properties.orb_elumo_b3lyp_631ppgdp.val = float(items[3])
      elif items[1] == 'B3LYP/aug-pcS-1':
        properties.orb_ehomo_b3lyp_augpcs1.val = float(items[2])
        properties.orb_elumo_b3lyp_augpcs1.val = float(items[3])
      else:
        raise ValueError('Invalid level of theory: %s.' % items[1])

  def parse_atomic_block(self):
    """Parses block of properties beginning with AT2.

    Raises:
      ValueError: if encountering and unknown field type.
    """
    if not self._next_line_startswith('AT2_'):
      return
    section = self.parse(ParseModes.BYLABEL, label='AT2_')
    properties = self._molecule.prop
    for line in section:
      label, rest = str(line[:20]).strip(), line[20:]
      field_name, field_type = ATOMIC_LABEL_FIELDS[label]

      # Special case for AT2_T1mol. This same value is written in two places
      # in the .dat file. We verify that the value already there (if there is
      # one) is the same as what we have here.
      if label == 'AT2_T1mol' and properties.HasField('wf_diag_t1_2sd'):
        new_val = float(rest)
        if not np.isclose(new_val, properties.wf_diag_t1_2sd.val, atol=.00015):
          raise ValueError(
              'ATOMIC-2 block AT2_T1mol ({:f}) differs from current value ({:f})'
              .format(new_val, properties.wf_diag_t1_2sd.val))

      if field_type == Atomic2FieldTypes.STRING:
        getattr(properties, field_name).val = str(rest)
      elif field_type == Atomic2FieldTypes.SCALAR:
        getattr(properties, field_name).val = float(rest)
      elif field_type == Atomic2FieldTypes.TRIPLE:
        triple_vals = [float(v) for v in str(rest).split()]
        if len(triple_vals) != 3:
          raise ValueError(f'Unexpected vals in line {line}')
        getattr(properties, field_name).val = triple_vals[0]
        getattr(properties, field_name.replace('at2_std',
                                               'at2_um')).val = triple_vals[1]
        getattr(properties,
                field_name.replace('at2_std', 'at2_um') +
                '_unc').val = triple_vals[2]
      else:
        raise ValueError(
            'Atomic block unknown field types {}'.format(field_type))

  def parse_excitation_energies_and_oscillations(self):
    """Parses exitation energies and length rep. osc. strengths at CC2/TZVP."""
    if not self._next_line_startswith(EXCITATION_HEADER):
      return
    segment = self.parse(ParseModes.RAW, num_lines=6)
    for line in segment[1:]:
      items = str(line).strip().split()
      properties = self._molecule.prop
      properties.exc_ene_cc2_tzvp.val.append(float(items[-2]))
      properties.exc_os_cc2_tzvp.val.append(float(items[-1]))

  def parse_nmr_isotropic_shieldings(self):
    """Parses NMR isotropic shieldings (ppm) for different levels of theory.

    Raises:
      ValueError: if line could not be parsed.
    """
    properties = self._molecule.prop
    while self._next_line_startswith('NMR isotropic shieldings'):
      shieldings_data = self.parse(
          ParseModes.RAW,
          num_lines=(len(self._molecule.bond_topo[-1].atom) + 1))
      theory_basis = str(shieldings_data[0]).split()[-1]
      field = getattr(properties,
                      NMR_ISOTROPIC_SHIELDINGS_LABEL_FIELDS[theory_basis])
      for line in shieldings_data[1:]:
        line = str(line)  # for type checking
        # Example line: '    1    6    141.2736   +/-    0.0064'
        # with offsets:  01234567890123456789012345678901234567
        value_str, pm_str, prec_str = line[10:25], line[25:28], line[28:]
        # An error check that the line appears to be formatted correctly.
        if pm_str != '+/-':
          raise ValueError('Could not parse nmr line: "{}"'.format(line))
        field.val.append(float(value_str))
        field.prec.append(float(prec_str))

  def parse_partial_charges(self):
    """Parses partial charges (e) for different levels of theory."""
    properties = self._molecule.prop
    while self._next_line_startswith('Partial charges'):
      partial_charges_data = self.parse(
          ParseModes.RAW,
          num_lines=(len(self._molecule.bond_topo[-1].atom) + 1))
      theory_basis = str(partial_charges_data[0]).strip().split()[-1]
      field = getattr(properties, PARTIAL_CHARGES_LABEL_FIELDS[theory_basis])
      for line in partial_charges_data[1:]:
        items = str(line).strip().split()
        field.val.append(float(items[2]))
        field.prec.append(float(items[-1]))

  def parse_polarizability(self):
    """Parses dipole-dipole polarizability."""
    if not self._next_line_startswith('Polarizability (au)'):
      return
    properties = self._molecule.prop
    header = self.parse(ParseModes.RAW, num_lines=1)  # Polarizability (au).
    items = str(header).strip().split()
    rank2_data = self.parse(ParseModes.KEYVALUE, num_lines=6)
    if items[-1].startswith('PBE0'):
      for label in RANK2_ENCODING_ORDER:
        setattr(properties.elec_pol_pbe0_augpc1, label,
                float(rank2_data[label]))
    elif items[-1].startswith('HF'):
      for label in RANK2_ENCODING_ORDER:
        setattr(properties.dipole_dipole_polarizability_hf_6_31gd, label,
                float(rank2_data[label]))

  def parse_multipole_moments(self):
    """Parses Di-, Quadru-, and Octopole moments in (au)."""
    properties = self._molecule.prop
    # PBE0 section.
    if self._next_line_startswith('Dipole moment (au):     PBE0/aug-pc-1'):
      self.parse(ParseModes.SKIP, num_lines=1)  # Dipole moment (au).
      dipole_data = self.parse(ParseModes.KEYVALUE, num_lines=3)
      properties.elec_dip_pbe0_augpc1.x = float(dipole_data['x'])
      properties.elec_dip_pbe0_augpc1.y = float(dipole_data['y'])
      properties.elec_dip_pbe0_augpc1.z = float(dipole_data['z'])
      self.parse(ParseModes.SKIP, num_lines=1)  # Quadrupole moment (au).
      quadrupole_data = self.parse(ParseModes.KEYVALUE, num_lines=6)
      for label in RANK2_ENCODING_ORDER:
        setattr(properties.elec_qua_pbe0_augpc1, label,
                float(quadrupole_data[label]))
      self.parse(ParseModes.SKIP, num_lines=1)  # Octopole moment (au).
      octopole_data = self.parse(ParseModes.KEYVALUE, num_lines=10)
      if '**********' in dict(octopole_data).values():
        raise SmuOverfullFloatFieldError()
      for label in RANK3_ENCODING_ORDER:
        setattr(properties.elec_oct_pbe0_augpc1, label,
                float(octopole_data[label]))
    # Hartree-Fock section.
    if self._next_line_startswith('Dipole moment (au):     HF/6-31Gd'):
      self.parse(ParseModes.SKIP, num_lines=1)  # Dipole moment (au).
      dipole_data = self.parse(ParseModes.KEYVALUE, num_lines=3)
      properties.elec_dip_hf_631gd.x = float(dipole_data['x'])
      properties.elec_dip_hf_631gd.y = float(dipole_data['y'])
      properties.elec_dip_hf_631gd.z = float(dipole_data['z'])
      self.parse(ParseModes.SKIP, num_lines=1)  # Quadrupole moment (au).
      quadrupole_data = self.parse(ParseModes.KEYVALUE, num_lines=6)
      for label in RANK2_ENCODING_ORDER:
        setattr(properties.elec_qua_hf_631gd, label,
                float(quadrupole_data[label]))
      self.parse(ParseModes.SKIP, num_lines=1)  # Octopole moment (au).
      octopole_data = self.parse(ParseModes.KEYVALUE, num_lines=10)
      if '**********' in dict(octopole_data).values():
        raise SmuOverfullFloatFieldError()
      for label in RANK3_ENCODING_ORDER:
        setattr(properties.elec_oct_hf_631gd, label,
                float(octopole_data[label]))

  def parse_stage1_to_proto(self):
    """Read _raw_contents and parses the various sections.

    This parses the "stage1" files which are just the geometry optimization
    before dedupping.

    This only reads one molecule from _raw_contents. To read multiple, you have
    to update _raw_contents between calls.

    Returns:
      dataset_pb2.Molecule or an Exception
    """
    self.parse(ParseModes.INITIALIZE)
    try:
      self._molecule = dataset_pb2.Molecule()
      self.parse(ParseModes.SKIP, num_lines=1)  # Separator.
      num_atoms = self.parse_stage1_header()
      self.parse_bond_topology()
      self.parse_identifier()
      self.parse_cluster_info(num_lines=4)
      self.parse_stage1_timings()
      self.parse_gradient_norms()
      self.parse_coordinates('Initial Coords', num_atoms)
      self.parse_coordinates('Optimized Coords', num_atoms)
      self.parse_frequencies_and_intensities(num_atoms, header=False)

      # Somewhere along the lines in the regeneration process (maybe just for
      # debugging), we add an extra blank line. We'll just skip it here and
      # ignore blank lines at the end.
      self.parse(ParseModes.SKIP_BLANK_LINES)
    except (SmuKnownError, ValueError, IndexError, KeyError,
            AssertionError) as exc:
      exc.mol_id = self._molecule.mol_id
      logging.info('Got exception during molecule %d: %s\n'
                   'traceback: %s', exc.mol_id, str(exc),
                   traceback.format_exc())
      return exc

    return self._molecule

  def parse_stage2_to_proto(self):
    """Read _raw_contents and parses the various sections.

    This parses the "stage2" files which are the complete ones from the end of
    the pipeline.

    This only reads one molecule from _raw_contents. To read multiple, you have
    to update _raw_contents between calls.

    Returns:
      dataset_pb2.Molecule with a single molecule, or an Exception
    """
    self.parse(ParseModes.INITIALIZE)
    try:
      self._molecule = dataset_pb2.Molecule()
      self.parse(ParseModes.SKIP, num_lines=1)  # Separator.
      num_atoms = self.parse_stage2_header()
      self.parse_database()
      self.parse_error_codes()
      self.parse_bond_topology()
      self.parse_identifier()
      self.parse_cluster_info(num_lines=8)
      self.parse_stage2_timings()  # Timings per step.
      self.parse_bonds()
      self.parse_gradient_norms()
      self.parse_coordinates('Initial Coords', num_atoms)
      self.parse_coordinates('Optimized Coords', num_atoms)
      self.parse_rotational_constants()
      self.parse_symmetry_used()
      # 'Frequencies and intensities'
      self.parse_frequencies_and_intensities(num_atoms, header=True)
      self.parse_gaussian_sanity_check()
      self.parse_vib_mode(num_atoms)
      self.parse_property_list()  # Key-value pairs: Energies, frequencies,...
      self.parse_diagnostics()
      self.parse_atomic_block()
      self.parse_homo_lumo()
      self.parse_excitation_energies_and_oscillations()
      self.parse_nmr_isotropic_shieldings()
      self.parse_partial_charges()
      self.parse_polarizability()
      self.parse_multipole_moments()
      # Somewhere along the lines in the regeneration process (maybe just for
      # debugging), we add an extra blank line. We'll just skip it here and
      # ignore blank lines at the end.
      self.parse(ParseModes.SKIP_BLANK_LINES)

    except (SmuKnownError, ValueError, IndexError, KeyError,
            AssertionError) as exc:
      exc.mol_id = self._molecule.mol_id
      logging.info('Got exception during molecule %d: %s\n'
                   'traceback: %s', exc.mol_id, str(exc),
                   traceback.format_exc())
      return exc

    return self._molecule

  def _process(self, parse_fn):
    line_generator = self._input_generator()
    while self._read_next_chunk(line_generator):
      yield parse_fn(), self._raw_contents

  def process_stage1(self):
    """Execute a pass through a SMU file's contents.

    The contents should be stage1 files from the initial geometry optimization
    and filtering.

    Yields:
      dataset_pb2.Entry (with one molecule each) or an Exception encountered
      during parsing, list of raw input lines
    """
    yield from self._process(self.parse_stage1_to_proto)

  def process_stage2(self):
    """Execute a pass through a SMU file's contents.

    The contents should be stage2 files from the end of the pipeline.

    Yielded line numbers are indices into self.raw_contents.

    Yields:
      dataset_pb2.Entry (with one molecule each) or an Exception encountered
      during parsing, list of raw input lines
    """
    yield from self._process(self.parse_stage2_to_proto)
