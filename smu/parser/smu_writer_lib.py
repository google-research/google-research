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
"""Writes Small Molecule Universe (SMU) files in custom Uni Basel format.

Used to write SMU entries from a protocol buffer to a SMU .dat file in Basel
format.
"""

import array
import copy
import re

from smu import dataset_pb2
from smu.parser import smu_parser_lib
from smu.parser import smu_utils_lib


class DatFormatMismatchError(Exception):
  pass


class RegeneratedLinesError(DatFormatMismatchError):

  def __init__(self, missing_lines, excess_lines):
    super().__init__()
    self.missing_lines = missing_lines
    self.excess_lines = excess_lines

  def __str__(self):
    return 'missing_lines:\n{}\nexcess_lines:\n{}\n'.format(
        '\n'.join(self.missing_lines), '\n'.join(self.excess_lines))


class LineOrderError(DatFormatMismatchError):

  def __init__(self, index, original_line, regenerated_line):
    super().__init__()
    self.index = index
    self.original_line = original_line
    self.regenerated_line = regenerated_line

  def __str__(self):
    return 'At {}, original:\n{}\ngenerated:\n{}\n'.format(
        self.index, self.original_line, self.regenerated_line)


# So this is pretty exciting. Fortran uses NaN and Infinity instead of nan
# and inf in its output, so if we want to match, we have to use those. This
# is very hacky solution to that because it relies on after the fact string
# replacements but it was easier to get the spacing right this way.
# We'll only use this in places where we know one of these could show up
class _FortranFloat(float):

  def __format__(self, format_spec):
    return (super(_FortranFloat, self).__format__(format_spec)  # pytype: disable=attribute-error
            .replace('nan', 'NaN').replace('     -inf', '-Infinity').replace(
                '     inf', 'Infinity'))


def get_long_molecule_name(molecule):
  return '{}.{}'.format(
    smu_utils_lib.get_composition(molecule.bond_topologies[0]),
    get_long_molecule_id(molecule.molecule_id))


def get_long_molecule_id(molecule_id):
  return '{:06d}.{:03d}'.format(molecule_id // 1000, molecule_id % 1000)


class SmuWriter:
  """A class to gather a SMU protocol buffer into a Basel-formatted string."""

  def __init__(self, annotate):
    """Initializes SMU7Writer.

    Args:
      annotate: bool, whether to provide annotations of the source proto fields
    """
    self.annotate = annotate

  def _molecule_index_string(self, molecule):
    if molecule.original_molecule_index == -1:
      return '*****'
    else:
      return str(molecule.original_molecule_index).rjust(5)

  def get_stage1_header(self, molecule):
    """Returns formatted header (separator and first line).

    This is for the stage1 format, which just contains the results of geometry
    optimization

    Args:
      molecule: dataset_pb2.Molecule.

    Returns:
      A multiline string representation of the header.
    """
    num_atoms = len(molecule.bond_topologies[0].atoms)
    result = smu_parser_lib.SEPARATOR_LINE + '\n'
    if self.annotate:
      result += ('# From original_molecule_index, topology, bond_topology_id, '
                 'error_{nstat1, nstatc, nstatt, frequences} molecule_id\n')
    errors = molecule.properties.errors
    result += '{:5s}{:5d}{:5d}{:5d}{:5d}{:5d}     {:s}\n'.format(
        self._molecule_index_string(molecule), errors.error_nstat1,
        errors.error_nstatc, errors.error_nstatt, errors.error_frequencies,
        num_atoms, smu_utils_lib.get_original_label(molecule))
    return result

  def get_stage2_header(self, molecule):
    """Returns formatted header (separator and first line).

    This is for the stage2 format which is at the end of the pipeline.

    Args:
      molecule: dataset_pb2.Molecule.

    Returns:
      A multiline string representation of the header.
    """
    num_atoms = len(molecule.bond_topologies[0].atoms)
    result = smu_parser_lib.SEPARATOR_LINE + '\n'
    if self.annotate:
      result += ('# From original_molecule_index, topology, '
                 'bond_topology_id, molecule_id\n')
    result += '{:s}{:5d}     {:s}\n'.format(
        self._molecule_index_string(molecule), num_atoms,
        smu_utils_lib.get_original_label(molecule))
    return result

  def get_database(self, molecule):
    """Returns the line indicating which database this molecule goes to.

    Args:
      molecule: A Molecule protocol buffer message.

    Returns:
      String
    """
    if not molecule.properties.HasField('errors'):
      # Standard database has the errors message filtered, so we assume this is
      # standard
      val = dataset_pb2.STANDARD
    elif molecule.properties.errors.which_database != dataset_pb2.UNSPECIFIED:
      val = molecule.properties.errors.which_database
    else:
      # The deprecated location
      val = molecule.which_database_deprecated
    if val == dataset_pb2.STANDARD:
      return 'Database   standard\n'
    elif (val == dataset_pb2.COMPLETE or val == dataset_pb2.UNSPECIFIED):
      return 'Database   complete\n'
    raise ValueError('Bad which_database: {}'.format(val))

  def get_error_codes(self, properties):
    """Returns a section of error/warning codes (as defined by Uni Basel).

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation
    """
    result = ''
    if self.annotate:
      result += '# From errors\n'

    result += 'Status     {:4d}\n'.format(properties.errors.status)
    result += 'Warn_T1    {:4d}{:4d}\n'.format(properties.errors.warn_t1,
                                               properties.errors.warn_t1_excess)
    result += 'Warn_BSE   {:4d}{:4d}\n'.format(
        properties.errors.warn_bse_b5_b6, properties.errors.warn_bse_cccsd_b5)
    result += 'Warn_EXC   {:4d}{:4d}{:4d}\n'.format(
        properties.errors.warn_exc_lowest_excitation,
        properties.errors.warn_exc_smallest_oscillator,
        properties.errors.warn_exc_largest_oscillator)
    result += 'Warn_VIB   {:4d}{:4d}\n'.format(
        properties.errors.warn_vib_linearity,
        properties.errors.warn_vib_imaginary)
    result += 'Warn_NEG   {:4d}\n'.format(properties.errors.warn_num_neg)

    return result

  def get_adjacency_code_and_hydrogens(self, topology):
    """Returns adjacency code and number of hydrogens bonded to heavy atoms.

    Args:
      topology: A BondTopology protocol buffer message.

    Returns:
      A multiline string representation of adjacency code and hydrogen numbers.
    """
    adjacency_matrix = smu_utils_lib.compute_adjacency_matrix(topology)
    side_length = len(adjacency_matrix)
    result = ''
    if self.annotate:
      result += '# From topology\n'
    result += '     '
    for i in range(0, side_length):
      for j in range(i + 1, side_length):
        result += str(adjacency_matrix[i][j])
    result += '\n     '
    num_bonded_hydrogens = smu_utils_lib.compute_bonded_hydrogens(
        topology, adjacency_matrix)
    return result + ''.join(str(item) for item in num_bonded_hydrogens) + '\n'

  def get_ids(self, molecule, stage, bt_idx):
    """Returns lines with identifiers.

    This include the smiles string, the file, and the ID line.
    We meed to know the stage because the SMU1 special cases are handled
    differently in the two stages.

    Args:
      molecule: dataset_pb2.Molecule
      stage: 'stage1' or 'stage2'
      bt_idx: bond topology index

    Returns:
      A multiline string representation of id lines.
    """
    result = ''
    if self.annotate:
      result += '# From smiles or properties.smiles_openbabel\n'
    if molecule.properties.HasField('smiles_openbabel'):
      result += molecule.properties.smiles_openbabel + '\n'
    else:
      result += molecule.bond_topologies[bt_idx].smiles + '\n'
    if self.annotate:
      result += '# From topology\n'
    result += smu_utils_lib.get_composition(
        molecule.bond_topologies[bt_idx]) + '\n'
    if self.annotate:
      result += '# From bond_topology_id, molecule_id\n'
    bond_topology_id = molecule.bond_topologies[bt_idx].bond_topology_id
    # Special case SMU1. Fun.
    if smu_utils_lib.special_case_dat_id_from_bt_id(bond_topology_id):
      if stage == 'stage1':
        bond_topology_id = 0
      elif stage == 'stage2':
        bond_topology_id = smu_utils_lib.special_case_dat_id_from_bt_id(
            bond_topology_id)
      else:
        raise ValueError(f'Unknown stage {stage}')
    result += 'ID{:8d}{:8d}\n'.format(bond_topology_id,
                                      molecule.molecule_id % 1000)
    return result

  def get_system(self, properties):
    """Returns information about cluster on which computations were performed.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation.
    """
    result = ''
    if self.annotate:
      result += '# From compute_cluster_info\n'
    result += properties.compute_cluster_info
    return result

  def get_stage1_timings(self, properties):
    """Returns recorded timings for different computation steps.

    This is for the stage1 format, which just contains the results of geometry
    optimization

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of timings for different computations.
    """
    if len(properties.calculation_statistics) == 0:
      return ''
    result = 'TIMINGS'

    def get_stat(s):
      for statistic in properties.calculation_statistics:
        if statistic.computing_location == s:
          return statistic.timings

    result += '{:>6s}{:>6s}'.format(get_stat('Geo'), get_stat('Force'))
    result += '    -1' * 8
    result += '\n'
    return result

  def get_stage2_timings(self, properties):
    """Returns recorded timings for different computation steps.

    This is for the stage2 format which is at the end of the pipeline.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of timings for different computations.
    """
    if len(properties.calculation_statistics) == 0:
      return ''
    labels = '       '
    values = 'TIMINGS'
    for statistic in properties.calculation_statistics:
      labels += statistic.computing_location.rjust(6)
      values += statistic.timings.rjust(6)
    labels = labels.replace('Force', ' Force').replace(' CC', 'CC').replace(
        'Polar', '  Polar').replace('  IP', 'IP')
    result = ''
    if self.annotate:
      result += '# From calculation_statistics\n'
    result += labels
    result += '\n'
    result += values
    result += '\n'
    return result

  def get_bonds(self, topology, properties):
    """Returns a bond section with atom pairs and bond types.

    Args:
      topology: A BondTopology protocol buffer message.
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of bond atom pairs and bond types.
    """
    if properties.errors.status >= 4:
      return ''
    adjacency_matrix = smu_utils_lib.compute_adjacency_matrix(topology)
    bonds = []
    for bond in topology.bonds:
      # Bond type is a six-digit integer ABCDEF, where A and B are the atomic
      # numbers of the two atoms forming the bond, C is the bond order, D and E
      # are the free valencies on the two atoms, and F encodes the charge.
      atom_types = [topology.atoms[bond.atom_a], topology.atoms[bond.atom_b]]
      bond_type = '%d%d' % (
          smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_types[0]],
          smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_types[1]])
      # Hydrogen atoms are not in the adjacency matrix.
      bond_order = 1 if '1' in bond_type else adjacency_matrix[bond.atom_a][
          bond.atom_b]
      charge = 0
      for i in range(2):
        if atom_types[i] == dataset_pb2.BondTopology.ATOM_NPOS:
          charge += 1
        elif atom_types[i] == dataset_pb2.BondTopology.ATOM_ONEG:
          charge -= 1
      # A total charge of -1 is encoded by integer value 3.
      if charge == -1:
        charge = 3
      free_valencies = (smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS[atom_types[0]] -
                        bond_order,
                        smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS[atom_types[1]] -
                        bond_order)
      # TODO(kohlhoff): Bonds between nitrogen atoms have their free valencies
      # stored in descending order. Identify the root cause.
      if bond_type == '77' and free_valencies[1] > free_valencies[0]:
        free_valencies = (free_valencies[1], free_valencies[0])
      bond_type += '%d%d%d%d' % (bond_order, free_valencies[0],
                                 free_valencies[1], charge)
      bonds.append('BOND%s%s     TYPE%s\n' %
                   (str(bond.atom_a + 1).rjust(5),
                    str(bond.atom_b + 1).rjust(5), bond_type.rjust(8)))
    result = ''
    if self.annotate:
      result += '# From bond_topology\n'
    result += ''.join(bonds)
    return result

  _DEPRECATED_ENERGY_FIELDS = [[
      'E_ini/G_norm', 'initial_geometry_energy_deprecated',
      'initial_geometry_gradient_norm_deprecated'
  ],
                               [
                                   'E_opt/G_norm',
                                   'optimized_geometry_energy_deprecated',
                                   'optimized_geometry_gradient_norm_deprecated'
                               ]]

  def get_gradient_norms(self, molecule, spacer):
    """Returns initial and optimized geometry energies and gradient norms.

    Args:
      molecule: dataset_pb2.Molecule
      spacer: spacer after label (differs between stage1 and stage2)

    Returns:
      A multiline string representation of geometry energies and gradient norms.
    """
    result = ''
    if molecule.optimized_geometry.HasField('energy'):
      for label, geometry in [('E_ini/G_norm', molecule.initial_geometries[0]),
                              ('E_opt/G_norm', molecule.optimized_geometry)]:
        if self.annotate:
          result += '# From energy, gnorm\n'
        result += '{}{}{:11.6f}{:12.6f}\n'.format(label, spacer,
                                                  geometry.energy.value,
                                                  geometry.gnorm.value)
    elif molecule.properties.HasField('optimized_geometry_energy_deprecated'):
      for label, field_energy, field_norm in self._DEPRECATED_ENERGY_FIELDS:
        if self.annotate:
          result += '# From %s, %s\n' % (field_energy, field_norm)
        result += '{}{}{:11.6f}{:12.6f}\n'.format(
            label, spacer,
            getattr(molecule.properties, field_energy).value,
            getattr(molecule.properties, field_norm).value)
    else:
      raise ValueError('All molecules should have energies')
    return result

  def get_coordinates(self, topology, molecule):
    """Returns a section with a molecule's initial and optimized geometries.

    Args:
      topology: A BondTopology protocol buffer message.
      molecule: A Molecule protocol buffer message.

    Returns:
      A multiline string representation of geometries in Cartesian coordinates.
    """
    coordinates = ''
    if (molecule.initial_geometries and
        molecule.initial_geometries[0].atom_positions):
      if self.annotate:
        coordinates += '# From initial_geometry.atom_positions\n'
      for i, atom in enumerate(topology.atoms):
        positions = molecule.initial_geometries[0].atom_positions[i]

        coordinates += 'Initial Coords%s%s%s%s\n' % (
            str(smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom]).rjust(8),
            '{:f}'.format(positions.x).rjust(12), '{:f}'.format(
                positions.y).rjust(12), '{:f}'.format(positions.z).rjust(12))
    if (molecule.HasField('optimized_geometry') and
        molecule.optimized_geometry.atom_positions):
      if self.annotate:
        coordinates += '# From optimized_geometry.atom_positions\n'
      for i, atom in enumerate(topology.atoms):
        positions = molecule.optimized_geometry.atom_positions[i]
        coordinates += 'Optimized Coords%s%s%s%s\n' % (
            str(smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom]).rjust(6),
            '{:f}'.format(positions.x).rjust(12), '{:f}'.format(
                positions.y).rjust(12), '{:f}'.format(positions.z).rjust(12))
    return coordinates

  def get_rotational_constants(self, molecule):
    """Returns rotational constants vector (MHz).

    Args:
      molecule: dataset_pb2.Molecule

    Returns:
      A string representation of the rotational constants vector.
    """
    result = ''
    if molecule.optimized_geometry.HasField('rotcon'):
      # result += '# From optimized_geometry.rotcon\n'
      vals = molecule.optimized_geometry.rotcon.value
    elif molecule.properties.HasField('rotational_constants_deprecated'):
      # result += '# From rotational_constants_deprecated\n'
      constants = molecule.properties.rotational_constants_deprecated
      vals = (constants.x, constants.y, constants.z)
    else:
      return ''
    if self.annotate:
      result += '# From rotational_constants_deprecated\n'
    result += (
        'Rotational constants (MHz)  {:-20.3f}{:-20.3f}{:-20.3f}\n'.format(
            vals[0], vals[1], vals[2]))
    return result

  def get_symmetry_used(self, properties):
    """Returns whether symmetry was used in the computations.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A string defining whether symmetry was used.
    """
    if not properties.HasField('symmetry_used_in_calculation'):
      return ''
    result = ''
    if self.annotate:
      result += '# From symmetry_used_in_calculation\n'
    symmetry_used = properties.symmetry_used_in_calculation
    result += 'Symmetry used in calculation   ' + ('yes\n' if symmetry_used else
                                                   ' no\n')
    return result

  def get_frequencies_and_intensities(self, properties, header):
    """Returns harmonic frequencies and intensities.

    Args:
      properties: A Properties protocol buffer message.
      header: bool, whether to print a header line

    Returns:
      A multiline string representation of harmonic frequencies and intensities.
    """
    if len(properties.harmonic_frequencies.value) == 0:  # pylint: disable=g-explicit-length-test
      return ''
    result = ''
    if header:
      result += 'Frequencies and intensities\n'
    if self.annotate:
      result += '# From harmonic_frequencies, '
      result += 'magnitude/harmonic intensity/normal mode order\n'
    frequencies = properties.harmonic_frequencies.value
    for i in range(0, len(frequencies), 10):
      result += ''.join(
          '{:7.1f}'.format(value).rjust(7) for value in frequencies[i:i + 10])
      result += '\n'
    if self.annotate:
      result += '# From harmonic_intensities\n'
    intensities = properties.harmonic_intensities.value
    for i in range(0, len(intensities), 10):
      result += ''.join(
          '{:7.1f}'.format(value).rjust(7) for value in intensities[i:i + 10])
      result += '\n'
    return result

  def get_gaussian_sanity_check(self, properties):
    """Returns gaussian sanity check section.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation
    """
    if not properties.HasField('gaussian_sanity_check'):
      return ''

    output_lines = ['Gaussian sanity check for FREQ/ELSTAT calculation\n']
    for prefix, fields in smu_parser_lib.GAUSSIAN_SANITY_CHECK_LINES:
      if self.annotate:
        output_lines.append('# From ' + ','.join(fields) + '\n')
      if len(fields) == 1:
        output_lines.append('{:30s}{:14.6f}\n'.format(
            prefix, getattr(properties.gaussian_sanity_check, fields[0])))
      elif len(fields) == 2:
        output_lines.append('{:30s}{:14.6f}{:14.6f}\n'.format(
            prefix, getattr(properties.gaussian_sanity_check, fields[0]),
            getattr(properties.gaussian_sanity_check, fields[1])))
      else:
        raise ValueError('Bad fields {fields} in _GAUSSIAN_SANITY_CHECK_LINES')

    return ''.join(output_lines)

  def get_normal_modes(self, properties):
    """Returns a repeated section containing a number of normal modes.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of the normal modes.
    """
    if len(properties.normal_modes) == 0:  # pylint: disable=g-explicit-length-test
      return ''
    result = 'Normal modes\n'
    if self.annotate:
      result += '# From normal_modes\n'
    for i, normal_modes in enumerate(properties.normal_modes):
      result += 'Mode%s\n' % str(i + 1).rjust(4)
      displacements = []
      for displacement in normal_modes.displacements:
        displacements += [displacement.x, displacement.y, displacement.z]
      for j in range(0, len(displacements), 10):
        result += ''.join('{:8.4f}'.format(value).rjust(8)
                          for value in displacements[j:j + 10])
        result += '\n'
    return result

  def get_properties(self, molecule):
    """Returns a variety of properties, in particular single point energies.

    Args:
      molecule: dataset_pb2.Molecule

    Returns:
      A multiline string representation of the labeled properties.
    """
    properties = molecule.properties
    float_line = '{:21s}{:-12.6f}\n'.format
    int_line = '{:21s}{:-5d}\n'.format
    result = ''
    for label, field in smu_parser_lib.PROPERTIES_LABEL_FIELDS.items():
      if label in ['NIMAG', 'NUM_OPT']:
        if not properties.HasField(field):
          continue
        if self.annotate:
          result += '# From %s\n' % field
        result += int_line(label, getattr(properties, field))

      elif label == 'NUCREP':
        value = None
        if molecule.optimized_geometry.HasField('enuc'):
          if self.annotate:
            result += '# From optimized_geometry.enuc\n'
          value = molecule.optimized_geometry.enuc.value
        elif properties.HasField('nuclear_repulsion_energy_deprecated'):
          if self.annotate:
            result += '# From nuclear_repulsion_energy_deprecated\n'
          value = properties.nuclear_repulsion_energy_deprecated.value
        if value is None:
          continue
        result += float_line(label, _FortranFloat(value))

      elif label == 'ZPE_unscaled':
        # This is just a special case because the number of significant digts is
        # different.
        if not properties.HasField(field):
          continue
        if self.annotate:
          result += '# From zpe_unscaled\n'
        result += 'ZPE_unscaled {:-16.2f}\n'.format(
            properties.zpe_unscaled.value)

      else:
        if not properties.HasField(field):
          continue
        if self.annotate:
          result += '# From %s\n' % field
        result += float_line(label,
                             _FortranFloat(getattr(properties, field).value))

    return result

  _T1_DIAGNOSTICS_FIELDS = [
      'diagnostics_t1_ccsd_2sp', 'diagnostics_t1_ccsd_2sd',
      'diagnostics_t1_ccsd_3psd'
  ]

  def get_diagnostics(self, properties):
    """Returns D1 diagnostics.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A string representation of the D1 and T1 diagnostics.
    """
    result = ''

    if properties.HasField('diagnostics_d1_ccsd_2sp'):
      if self.annotate:
        result += '# From diagnostics_d1_ccsd_2sp\n'
      result += ('D1DIAG    D1(CCSD/2sp) {:10.6f}\n'.format(
          properties.diagnostics_d1_ccsd_2sp.value))

    if properties.HasField(self._T1_DIAGNOSTICS_FIELDS[0]):
      if self.annotate:
        result += '# From %s\n' % ', '.join(self._T1_DIAGNOSTICS_FIELDS)
      result += (
          'T1DIAG    T1(CCSD/2sp) %s  T1(CCSD/2sd) %s  T1(CCSD/3Psd)%s\n' %
          tuple('{:.6f}'.format(getattr(properties, field).value).rjust(10)
                for field in self._T1_DIAGNOSTICS_FIELDS))

    return result

  def get_atomic_block(self, properties):
    """Returns block of ATOMIC2 properties.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A string representation of the ATOMIC2 related properties.
    """
    result = ''
    for label, (field,
                field_type) in smu_parser_lib.ATOMIC_LABEL_FIELDS.items():
      if not properties.HasField(field):
        continue
      if field_type == smu_parser_lib.Atomic2FieldTypes.STRING:
        if self.annotate:
          result += '# From %s\n' % field
        result += '{:20s}{:s}\n'.format(label, getattr(properties, field).value)
      elif field_type == smu_parser_lib.Atomic2FieldTypes.SCALAR:
        if self.annotate:
          result += '# From %s\n' % field
        # Different significant digits for some fields. Fun.
        if '_ENE_' in label:
          result += '{:16s}{:-17.6f}\n'.format(
              label, _FortranFloat(getattr(properties, field).value))
        else:
          result += '{:16s}{:-15.4f}\n'.format(
              label, _FortranFloat(getattr(properties, field).value))
      elif field_type == smu_parser_lib.Atomic2FieldTypes.TRIPLE:
        if self.annotate:
          result += '# From %s{,_um,_um_ci}\n' % field
        result += '{:17s}{:-12.2f}{:-12.2f}{:-12.2f}\n'.format(
            label, _FortranFloat(getattr(properties, field).value),
            _FortranFloat(getattr(properties, field + '_um').value),
            _FortranFloat(getattr(properties, field + '_um_ci').value))
      else:
        raise ValueError(
            'Atomic block unknown field types {}'.format(field_type))

    return result

  _HOMO_LUMO_LABEL_FIELDS = [
      ['PBE0/6-311Gd', 'pbe0_6_311gd'],
      ['PBE0/aug-pc-1', 'pbe0_aug_pc_1'],
      ['HF/6-31Gd', 'hf_6_31gd'],
      ['B3LYP/6-31++Gdp', 'b3lyp_6_31ppgdp'],
      ['B3LYP/aug-pcS-1', 'b3lyp_aug_pcs_1'],
      ['PBE0/6-31++Gdp', 'pbe0_6_31ppgdp'],
      ['PBE0/aug-pcS-1', 'pbe0_aug_pcs_1'],
      ['HF/TZVP', 'hf_tzvp'],
      ['HF/3', 'hf_3'],
      ['HF/4', 'hf_4'],
      ['HF/CVTZ', 'hf_cvtz'],
  ]

  def get_homo_lumo(self, properties):
    """Returns HOMO and LUMO values (at different levels of theory).

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of the HOMO/LUMO.
    """
    result = ''
    for label, field_stem in self._HOMO_LUMO_LABEL_FIELDS:
      homo_field = 'homo_' + field_stem
      lumo_field = 'lumo_' + field_stem
      if (not properties.HasField(homo_field) or
          not properties.HasField(lumo_field)):
        continue
      if self.annotate:
        result += '# From %s, %s\n' % (homo_field, lumo_field)
      result += 'HOMO/LUMO  %s%s%s\n' % (label.ljust(15), '{:.5f}'.format(
          getattr(properties, homo_field).value).rjust(11), '{:.5f}'.format(
              getattr(properties, lumo_field).value).rjust(11))

    return result

  def get_excitation_energies_and_oscillations(self, properties):
    """Returns excitation energies and length rep.

    osc.

    strengths at CC2/TZVP.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of the energies and oscillations.
    """
    if not properties.HasField('excitation_energies_cc2'):
      return ''
    result = smu_parser_lib.EXCITATION_HEADER + '\n'
    if self.annotate:
      result += ('# From excitation_energies_cc2, '
                 'excitation_oscillator_strengths_cc2\n')
    if len(properties.excitation_energies_cc2.value) != len(
        properties.excitation_oscillator_strengths_cc2.value):
      raise ValueError(
          'Unequal lengths for excitation energies (%d) and oscillations (%d)' %
          (len(properties.excitation_energies_cc2.value),
           len(properties.excitation_oscillator_strengths_cc2.value)))
    for i, (energy, oscillator_strength) in enumerate(
        zip(properties.excitation_energies_cc2.value,
            properties.excitation_oscillator_strengths_cc2.value)):
      result += '%s%s%s\n' % (str(i + 1).rjust(5),
                              '{:.5f}'.format(energy).rjust(18),
                              '{:.5f}'.format(oscillator_strength).rjust(16))
    return result

  def get_nmr_isotropic_shieldings(self, topology, properties):
    """Returns NMR isotropic shieldings (ppm) for different levels of theory.

    Args:
      topology: A BondTopology protocol buffer message.
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of the NMR isotropic shieldings.
    """
    result = ''
    for label, field in (
        smu_parser_lib.NMR_ISOTROPIC_SHIELDINGS_LABEL_FIELDS.items()):
      if not properties.HasField(field):
        continue
      if self.annotate:
        result += '# From %s\n' % field
      result += 'NMR isotropic shieldings (ppm): %s\n' % label
      for i, atom in enumerate(topology.atoms):
        result += '%s%s%s   +/-%s\n' % (
            str(i + 1).rjust(5),
            str(smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom]).rjust(5),
            '{:12.4f}'.format(getattr(properties, field).values[i]),
            '{:10.4f}'.format(getattr(properties, field).precision[i]))

    return result

  def get_partial_charges(self, topology, properties):
    """Returns formatted partial charges for different levels of theory.

    Args:
      topology: A BondTopology protocol buffer message.
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of the partial charges.
    """
    result = ''
    for label, field in smu_parser_lib.PARTIAL_CHARGES_LABEL_FIELDS.items():
      if not properties.HasField(field):
        continue
      result += 'Partial charges (e): %s\n' % label
      if self.annotate:
        result += '# From %s\n' % field
      for i, atom in enumerate(topology.atoms):
        result += '%s%s%s   +/-%s\n' % (
            str(i + 1).rjust(5),
            str(smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom]).rjust(5),
            '{:12.4f}'.format(getattr(properties, field).values[i]),
            '{:10.4f}'.format(getattr(properties, field).precision[i]))

    return result

  def format_for_tensors(self, label, val):
    return '   %s%s\n' % (label, '{:.5f}'.format(val).rjust(14))

  def get_rank2(self, prop):
    """Returns the output for a Rank2MolecularProperty.

    Args:
      prop: Rank2MolecularProperty

    Returns:
      string
    """
    out = ''
    if prop.matrix_values_deprecated:
      for label, val in zip(smu_parser_lib.RANK2_ENCODING_ORDER,
                            prop.matrix_values_deprecated):
        out += self.format_for_tensors(' ' + label, val)
    else:
      for label in smu_parser_lib.RANK2_ENCODING_ORDER:
        out += self.format_for_tensors(' ' + label, getattr(prop, label))
    return out

  def get_rank3(self, prop):
    """Returns the output for a Rank3MolecularProperty.

    Args:
      prop: Rank3MolecularProperty

    Returns:
      string
    """
    out = ''
    if prop.tensor_values_deprecated:
      for label, val in zip(smu_parser_lib.RANK3_ENCODING_ORDER,
                            prop.tensor_values_deprecated):
        out += self.format_for_tensors(label, val)
    else:
      for label in smu_parser_lib.RANK3_ENCODING_ORDER:
        out += self.format_for_tensors(label, getattr(prop, label))
    return out

  def get_polarizability(self, properties):
    """Returns dipole-dipole polarizability.

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of dipole-dipole polarizability.
    """
    if not properties.HasField('dipole_dipole_polarizability_pbe0_aug_pc_1'):
      return ''
    result = 'Polarizability (au):    PBE0/aug-pc-1\n'
    if self.annotate:
      result += '# From dipole_dipole_polarizability_pbe0_aug_pc_1\n'
    result += self.get_rank2(
        properties.dipole_dipole_polarizability_pbe0_aug_pc_1)
    return result

  def get_multipole_moments(self, properties):
    """Returns formatted Di-, Quadru-, and Octopole moments in (au).

    Args:
      properties: A Properties protocol buffer message.

    Returns:
      A multiline string representation of the multipole moments.
    """

    result = ''

    if properties.HasField('dipole_moment_pbe0_aug_pc_1'):
      result += 'Dipole moment (au):     PBE0/aug-pc-1\n'
      if self.annotate:
        result += '# From dipole_moment_pbe0_aug_pc_1\n'
      result += self.format_for_tensors(
          '  x', properties.dipole_moment_pbe0_aug_pc_1.x)
      result += self.format_for_tensors(
          '  y', properties.dipole_moment_pbe0_aug_pc_1.y)
      result += self.format_for_tensors(
          '  z', properties.dipole_moment_pbe0_aug_pc_1.z)

    if properties.HasField('quadrupole_moment_pbe0_aug_pc_1'):
      result += 'Quadrupole moment (au): PBE0/aug-pc-1\n'
      if self.annotate:
        result += '# From quadrupole_moment_pbe0_aug_pc_1\n'
      result += self.get_rank2(properties.quadrupole_moment_pbe0_aug_pc_1)

    if properties.HasField('octopole_moment_pbe0_aug_pc_1'):
      result += 'Octopole moment (au):   PBE0/aug-pc-1\n'
      if self.annotate:
        result += '# From octopole_moment_pbe0_aug_pc_1\n'
      result += self.get_rank3(properties.octopole_moment_pbe0_aug_pc_1)

    if properties.HasField('dipole_moment_hf_6_31gd'):
      result += 'Dipole moment (au):     HF/6-31Gd\n'
      if self.annotate:
        result += '# From dipole_moment_hf\n'
      result += self.format_for_tensors('  x',
                                        properties.dipole_moment_hf_6_31gd.x)
      result += self.format_for_tensors('  y',
                                        properties.dipole_moment_hf_6_31gd.y)
      result += self.format_for_tensors('  z',
                                        properties.dipole_moment_hf_6_31gd.z)

    if properties.HasField('quadrupole_moment_hf_6_31gd'):
      result += 'Quadrupole moment (au): HF/6-31Gd\n'
      if self.annotate:
        result += '# From quadrupole_moment_hf_6_31gd\n'
      result += self.get_rank2(properties.quadrupole_moment_hf_6_31gd)

    if properties.HasField('octopole_moment_hf_6_31gd'):
      result += 'Octopole moment (au):   HF/6-31Gd\n'
      if self.annotate:
        result += '# From octopole_moment_hf_6_31gd\n'
      result += self.get_rank3(properties.octopole_moment_hf_6_31gd)

    return result

  def process_stage1_proto(self, molecule):
    """Return the contents of molecule as a string in SMU7 stage1 file format.

    This is for the stage1 format, which just contains the results of geometry
    optimization

    Args:
      molecule: dataset_pb2.Molecule

    Returns:
      A string representation of the protocol buffer in Uni Basel's file format.
    """
    contents = []

    properties = molecule.properties
    bt_idx = smu_utils_lib.get_starting_bond_topology_index(molecule)

    contents.append(self.get_stage1_header(molecule))
    contents.append(
        self.get_adjacency_code_and_hydrogens(molecule.bond_topologies[bt_idx]))
    contents.append(self.get_ids(molecule, 'stage1', bt_idx))
    contents.append(self.get_system(properties))
    contents.append(self.get_stage1_timings(properties))
    contents.append(self.get_gradient_norms(molecule, spacer=' '))
    contents.append(
        self.get_coordinates(molecule.bond_topologies[bt_idx], molecule))
    contents.append(
        self.get_frequencies_and_intensities(properties, header=False))

    return ''.join(contents)

  def process_stage2_proto(self, molecule):
    """Return the contents of molecule as a string in SMU7 stage2 file format.

    This is for the stage2 format which is at the end of the pipeline.

    Args:
      molecule: dataset_pb2.Molecule

    Returns:
      A string representation of the protocol buffer in Uni Basel's file format.
    """
    contents = []

    properties = molecule.properties
    bt_idx = smu_utils_lib.get_starting_bond_topology_index(molecule)

    contents.append(self.get_stage2_header(molecule))
    contents.append(self.get_database(molecule))
    contents.append(self.get_error_codes(properties))
    contents.append(
        self.get_adjacency_code_and_hydrogens(molecule.bond_topologies[bt_idx]))
    contents.append(self.get_ids(molecule, 'stage2', bt_idx))
    contents.append(self.get_system(properties))
    contents.append(self.get_stage2_timings(properties))
    contents.append(
        self.get_bonds(molecule.bond_topologies[bt_idx], properties))
    contents.append(self.get_gradient_norms(molecule, spacer='         '))
    contents.append(
        self.get_coordinates(molecule.bond_topologies[bt_idx], molecule))
    contents.append(self.get_rotational_constants(molecule))
    contents.append(self.get_symmetry_used(properties))
    contents.append(
        self.get_frequencies_and_intensities(properties, header=True))
    contents.append(self.get_gaussian_sanity_check(properties))
    contents.append(self.get_normal_modes(properties))
    contents.append(self.get_properties(molecule))
    contents.append(self.get_diagnostics(properties))
    contents.append(self.get_atomic_block(properties))
    contents.append(self.get_homo_lumo(properties))
    contents.append(self.get_excitation_energies_and_oscillations(properties))
    contents.append(
        self.get_nmr_isotropic_shieldings(molecule.bond_topologies[bt_idx],
                                          properties))
    contents.append(
        self.get_partial_charges(molecule.bond_topologies[bt_idx], properties))
    contents.append(self.get_polarizability(properties))
    contents.append(self.get_multipole_moments(properties))

    return ''.join(contents)


class Atomic2InputWriter:
  """From molecule, produces the input file for the (fortran) atomic2 code."""

  def __init__(self):
    pass

  def get_filename_for_atomic2_input(self, molecule, bond_topology_idx):
    """Returns the expected filename for an atomic input.

    bond_topology_idx can be None (for the starting topology)
    """
    if bond_topology_idx is not None:
      return '{}.{}.{:03d}.inp'.format(
          smu_utils_lib.get_composition(
              molecule.bond_topologies[bond_topology_idx]),
          get_long_molecule_id(molecule.molecule_id),
          bond_topology_idx)
    else:
      return '{}.inp'.format(
          get_long_molecule_name(molecule))

  def get_mol_block(self, molecule, bond_topology_idx):
    """Returns the MOL file block with atoms and bonds.

    Args:
      molecule: dataset_pb2.Molecule
      bond_topology_idx: Bond topology index.

    Returns:
      list of strings
    """
    contents = []
    contents.append('\n')
    contents.append('{:3d}{:3d}  0  0  0  0  0  0  0  0999 V2000\n'.format(
        len(molecule.bond_topologies[bond_topology_idx].atoms),
        len(molecule.bond_topologies[bond_topology_idx].bonds)))
    for atom_type, coords in zip(
        molecule.bond_topologies[bond_topology_idx].atoms,
        molecule.optimized_geometry.atom_positions):
      contents.append(
          '{:10.4f}{:10.4f}{:10.4f} {:s}   0  0  0  0  0  0  0  0  0  0  0  0\n'
          .format(
              smu_utils_lib.bohr_to_angstroms(coords.x),
              smu_utils_lib.bohr_to_angstroms(coords.y),
              smu_utils_lib.bohr_to_angstroms(coords.z),
              smu_utils_lib.ATOM_TYPE_TO_RDKIT[atom_type][0]))
    for bond in molecule.bond_topologies[bond_topology_idx].bonds:
      contents.append('{:3d}{:3d}{:3d}  0\n'.format(bond.atom_a + 1,
                                                    bond.atom_b + 1,
                                                    bond.bond_type))

    return contents

  def get_energies(self, molecule):
    """Returns the $energies block.

    Args:
      molecule: dataset_pb2.Molecule

    Returns:
      list of strings
    """
    contents = []
    contents.append('$energies\n')
    contents.append('#              HF              MP2          '
                    'CCSD         CCSD(T)        T1 diag\n')

    def format_field(field_name):
      return '{:15.7f}'.format(getattr(molecule.properties, field_name).value)

    contents.append('{:7s}'.format('3') +
                    format_field('single_point_energy_hf_3') +
                    format_field('single_point_energy_mp2_3') + '\n')
    contents.append('{:7s}'.format('4') +
                    format_field('single_point_energy_hf_4') +
                    format_field('single_point_energy_mp2_4') + '\n')
    contents.append('{:7s}'.format('2sp') +
                    format_field('single_point_energy_hf_2sp') +
                    format_field('single_point_energy_mp2_2sp') +
                    format_field('single_point_energy_ccsd_2sp') +
                    format_field('single_point_energy_ccsd_t_2sp') + '\n')
    contents.append('{:7s}'.format('2sd') +
                    format_field('single_point_energy_hf_2sd') +
                    format_field('single_point_energy_mp2_2sd') +
                    format_field('single_point_energy_ccsd_2sd') +
                    format_field('single_point_energy_ccsd_t_2sd') +
                    format_field('diagnostics_t1_ccsd_2sd') + '\n')
    contents.append('{:7s}'.format('3Psd') +
                    format_field('single_point_energy_hf_3psd') +
                    format_field('single_point_energy_mp2_3psd') +
                    format_field('single_point_energy_ccsd_3psd') + '\n')
    contents.append('{:7s}'.format('C3') +
                    format_field('single_point_energy_hf_cvtz') +
                    format_field('single_point_energy_mp2ful_cvtz') + '\n')
    contents.append('{:7s}'.format('(34)') +
                    format_field('single_point_energy_hf_34') +
                    format_field('single_point_energy_mp2_34') + '\n')

    return contents

  def get_frequencies(self, molecule):
    """Returns the $frequencies block.

    Note that the only non-zero frequencies are shown. Generally, each
    molecule will have 6 zero frequencies for the euclidean degrees of freedom
    but some will only have 5. Any other number is considered an error.

    Args:
      molecule: dataset_pb2.Molecule

    Returns:
      list of strings

    Raises:
      ValueError: if number of zero frequencies is other than 5 or 6
    """
    contents = []

    trimmed_frequencies = [
        v for v in molecule.properties.harmonic_frequencies.value if v != 0.0
    ]

    contents.append('$frequencies{:5d}{:5d}{:5d}\n'.format(
        len(trimmed_frequencies), 0, 0))
    line = ''
    for i, freq in enumerate(trimmed_frequencies):
      line += '{:8.2f}'.format(freq)
      if i % 10 == 9:
        contents.append(line + '\n')
        line = ''
    if line:
      contents.append(line + '\n')
    return contents

  def process(self, molecule, bond_topology_idx):
    """Creates the atomic input file for molecule."""
    if (molecule.properties.errors.status < 0 or
        molecule.properties.errors.status > 3 or
        # While we should check all the fields, this is conveinient shortcut.
        not molecule.properties.HasField('single_point_energy_hf_3') or
        not molecule.properties.HasField('single_point_energy_mp2_3')):
      raise ValueError(
          f'Molecule {molecule.molecule_id} is lacking required info '
          'for generating atomic2 input. Maybe you need to query the complete DB?'
      )

    contents = []
    contents.append('SMU {}, RDKIT {}, bt {}({}/{}), geom opt\n'.format(
        molecule.molecule_id,
        molecule.bond_topologies[bond_topology_idx].smiles,
        molecule.bond_topologies[bond_topology_idx].bond_topology_id,
        bond_topology_idx + 1, len(molecule.bond_topologies)))
    contents.append(smu_utils_lib.get_original_label(molecule) + '\n')

    contents.extend(self.get_mol_block(molecule, bond_topology_idx))
    contents.extend(self.get_energies(molecule))
    contents.extend(self.get_frequencies(molecule))
    contents.append('$end\n')

    return ''.join(contents)


class CleanTextWriter:
  """From molecule, produces a clean text format intended for human consumption."""

  def __init__(self):
    pass

  def _fw_line(self, vals):
    """Create a fixed width line.

    Args:
      vals: sequence of tuples (position, string)

    Returns
      Newline terminated line
    """
    largest_val = max(vals, key=lambda v: v[0] + len(v[1]))
    out = array.array('u', ' ' * (largest_val[0] + len(largest_val[1])))
    out.append('\n')
    for pos, val in vals:
      out[pos:pos+len(val)] = array.array('u', val)
    return out.tounicode()

  def _align_dec_point(self, pos, val):
    return (pos - val.index('.'), val)

  def _compact_adj_matrix(self, adjacency_matrix):
    out = []
    side_length = len(adjacency_matrix)
    for i in range(0, side_length - 1):
      out.append(''.join(str(adjacency_matrix[i][j]) for j in range(i + 1, side_length)))
    return '.'.join(out)

  def _heavy_atom_list(self, topology):
    return ''.join([smu_utils_lib.ATOM_TYPE_TO_RDKIT[a][0]
                    for a in topology.atoms
                    if a != dataset_pb2.BondTopology.ATOM_H])

  def _atom_generator(self, topology):
    for atom_idx, atom in enumerate(topology.atoms):
      yield atom_idx, [(20, f'{atom_idx+1:2d}'),
                       (23, smu_utils_lib.ATOM_TYPE_TO_RDKIT[atom][0])]

  def _fate_string(self, molecule):
    # Probably a temporary hack. I'm adjusting our fate outputs to match what this format
    # expects rather than changin the fate values quite yet because that seems up in the air.
    fate = molecule.properties.errors.fate
    status = molecule.properties.errors.status

    if fate == dataset_pb2.Properties.FATE_DUPLICATE_SAME_TOPOLOGY:
      return 'DUPLICATE_SAME_TOPOLOGY'
    elif fate == dataset_pb2.Properties.FATE_DUPLICATE_DIFFERENT_TOPOLOGY:
      return 'DUPLICATE_DIFFERENT_TOPOLOGY'
    elif fate == dataset_pb2.Properties.FATE_GEOMETRY_OPTIMIZATION_PROBLEM:
      return 'FAILURE_GEO_OPT'
    elif fate == dataset_pb2.Properties.FATE_DISASSOCIATED:
      return 'FAILURE_TOPOLOGY_CHECK'
    elif fate == dataset_pb2.Properties.FATE_DISCARDED_OTHER:
      return 'FAILURE_STAGE2'
    elif fate == dataset_pb2.Properties.FATE_NO_CALCULATION_RESULTS:
      return 'FAILURE_NO_RESULTS'
    elif fate == dataset_pb2.Properties.FATE_CALCULATION_WITH_SERIOUS_ERROR:
      return 'ERROR_SERIOUS'
    elif fate == dataset_pb2.Properties.FATE_CALCULATION_WITH_MAJOR_ERROR:
      return 'ERROR_MAJOR'
    elif fate == dataset_pb2.Properties.FATE_CALCULATION_WITH_MODERATE_ERROR:
      return 'ERROR_MODERATE'
    elif fate == dataset_pb2.Properties.FATE_CALCULATION_WITH_WARNING_SERIOUS:
      if status > 0:
        return 'SUCCESS_NEUTRAL_WARNING_SERIOUS'
      return 'SUCCESS_ALL_WARNING_SERIOUS'
    elif fate == dataset_pb2.Properties.FATE_CALCULATION_WITH_WARNING_VIBRATIONAL:
      if status > 0:
        return 'SUCCESS_NEUTRAL_WARNING_VIBRATIONAL'
      return 'SUCCESS_ALL_WARNING_VIBRATIONAL'
    elif fate == dataset_pb2.Properties.FATE_SUCCESS:
      if status > 0:
        return 'SUCCESS_NEUTRAL'
      return 'SUCCESS_ALL'
    else:
      raise ValueError(f'Unhandled fate {fate}')


  def get_mol_id_block(self, molecule, long_name):
    out = []
    out.append('#\n')
    out.append('#mol_id    \n')
    out.append(self._fw_line([(1, 'mol_id'),
                              (31, get_long_molecule_id(molecule.molecule_id)),
                              (84, long_name),
                              ]))

    return out

  def get_mol_spec_block(self, molecule, long_name):
    out = []
    out.append('#\n')
    out.append('#mol_spec  \n')
    topology = molecule.bond_topologies[smu_utils_lib.get_starting_bond_topology_index(molecule)]
    base_vals = [(1, 'mol_spec'), (84, long_name)]
    out.append(self._fw_line(base_vals +
                             [(17, 'label'),
                              (31, long_name),
                              ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'topo_id'),
                              (31, str(molecule.molecule_id // 1000)),
                              ]))
    if molecule.properties.HasField('smiles_openbabel'):
      out.append(self._fw_line(base_vals +
                               [(17, 'smiles_obabel'),
                                (31, molecule.properties.smiles_openbabel),
                                ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'x_atoms'),
                              (31, self._heavy_atom_list(topology)),
                              ]))
    adjacency_matrix = smu_utils_lib.compute_adjacency_matrix(topology)
    num_bonded_hydrogens = smu_utils_lib.compute_bonded_hydrogens(
        topology, adjacency_matrix)
    out.append(self._fw_line(base_vals +
                             [(17, 'h_atoms'),
                              (31, ''.join(str(h) for h in num_bonded_hydrogens)),
                              ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'x_atpair_mat'),
                              (31, self._compact_adj_matrix(adjacency_matrix)),
                              ]))

    return out

  def get_calc_block(self, molecule, long_name):
    out = []
    if not molecule.properties.HasField('errors'):
      return out
    out.append('#\n')
    out.append('#calc      \n')
    base_vals = [(1, 'calc'), (84, long_name)]
    out.append(self._fw_line(base_vals +
                             [(17, 'status'),
                              (31, f'{molecule.properties.errors.status:3d}'),
                              ]))
    error_level = smu_utils_lib.molecule_calculation_error_level(molecule)
    if error_level <= 2:

      out.append(self._fw_line(base_vals +
                               [(17, 'warn_level'),
                                (31, '  ' + 'ABC'[error_level]),
                                ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'fate'),
                              (31, self._fate_string(molecule)),
                              ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'database'),
                              (31, 'standard'
                               if molecule.properties.errors.which_database == dataset_pb2.STANDARD
                               else 'complete'),
                              ]))

    out.append('#\n')
    out.append(self._fw_line([(0, '#'),
                              (1, 'calc'),
                              (17, 'warn 1 :'),
                              (31, 't1'),
                              (44, 'delta_t1'),
                              (57, 'bse_b6'),
                              (70, '{:13s}'.format('bse_eccsd')),
                              ]))
    out.append(self._fw_line([(0, '#'),
                              (1, 'calc'),
                              (17, 'warn 2 :'),
                              (31, 'exc_ene'),
                              (44, 'exc_osmin'),
                              (57, '{:26s}'.format('exc_osmax')),
                              ]))
    out.append(self._fw_line([(0, '#'),
                              (1, 'calc'),
                              (17, 'warn 3 :'),
                              (31, 'vib_linear'),
                              (44, 'vib_imag'),
                              (57, '{:26s}'.format('bsr_neg')),
                              ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'warn 1'),
                              (31, str(molecule.properties.errors.warn_t1)),
                              (44, str(molecule.properties.errors.warn_t1_excess)),
                              (57, str(molecule.properties.errors.warn_bse_b5_b6)),
                              (70, str(molecule.properties.errors.warn_bse_cccsd_b5)),
                              ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'warn 2'),
                              (31, str(molecule.properties.errors.warn_exc_lowest_excitation)),
                              (44, str(molecule.properties.errors.warn_exc_smallest_oscillator)),
                              (57, str(molecule.properties.errors.warn_exc_largest_oscillator)),
                              ]))
    out.append(self._fw_line(base_vals +
                             [(17, 'warn 3'),
                              (31, str(molecule.properties.errors.warn_vib_linearity)),
                              (44, str(molecule.properties.errors.warn_vib_imaginary)),
                              (57, str(molecule.properties.errors.warn_num_neg)),
                              ]))


    return out

  def get_duplicates_block(self, molecule, long_name):
    out = []
    out.append('#\n')
    out.append('#duplicate_found            \n')
    if len(molecule.duplicate_of) == 0:
      out.append(self._fw_line([(1, 'duplicate_found'),
                                (31, 'none'),
                                (84, long_name)]))
    else:
      for dup_id in sorted(molecule.duplicate_of):
        out.append(self._fw_line([(1, 'duplicate_found'),
                                  (31, get_long_molecule_id(dup_id)),
                                  (84, long_name)]))

    out.append('#\n')
    out.append('#duplicate_of               \n')
    if molecule.duplicated_by:
      dup_string = get_long_molecule_id(molecule.duplicated_by)
    else:
      dup_string = 'none'
    out.append(self._fw_line([(1, 'duplicate_of'),
                              (31, dup_string),
                              (84, long_name)]))

    return out

  def get_bond_topologies_block(self, molecule, long_name):
    out = []
    for bt_idx, bt in enumerate(molecule.bond_topologies):
      base_vals = [(1, 'bond_topo'),
                   (11, f'{bt_idx+1:2d}')]

      out.append('#\n')
      out.append(self._fw_line(base_vals +
                               [(0, '#'),
                                (14, 'of'),
                                (17, f'{len(molecule.bond_topologies):2d}')]))
      base_vals.append((84, long_name))
      out.append(self._fw_line(base_vals +
                               [(17, 'topo_id'),
                                (31, f'{bt.bond_topology_id:<d}')]))
      info = (('i' if bt.source & dataset_pb2.BondTopology.SOURCE_ITC else '.') +
              ('c' if bt.source & dataset_pb2.BondTopology.SOURCE_CSD else '.') +
              ('m' if bt.source & dataset_pb2.BondTopology.SOURCE_MLCR else '.') +
              ('u' if bt.source & dataset_pb2.BondTopology.SOURCE_CUSTOM else '.') +
              ('S' if bt.source & dataset_pb2.BondTopology.SOURCE_STARTING else '.'))
      out.append(self._fw_line(base_vals +
                               [(17, 'info'),
                                (31, info)]))
      out.append(self._fw_line(base_vals +
                               [(17, 'smiles_rdkit'),
                                (31, bt.smiles)]))
      out.append(self._fw_line(base_vals +
                               [(17, 'x_atoms'),
                                (31, self._heavy_atom_list(bt))]))
      adjacency_matrix = smu_utils_lib.compute_adjacency_matrix(bt)
      num_bonded_hydrogens = smu_utils_lib.compute_bonded_hydrogens(
        bt, adjacency_matrix)
      out.append(self._fw_line(base_vals +
                               [(17, 'h_atoms'),
                                (31, ''.join(str(h) for h in num_bonded_hydrogens)),
                                ]))
      out.append(self._fw_line(
        base_vals +
        [(17, 'x_atpair_mat'),
         (31,
          ('Cm:' if bt.source & dataset_pb2.BondTopology.SOURCE_STARTING else 'Rm:') +
          self._compact_adj_matrix(adjacency_matrix)),
         ]))

    return out

  def get_geometries_block(self, molecule, long_name):
    out = []

    def write_geometry(prefix, geom):
      out.append('#\n')
      out.append(self._fw_line([(0, '#'),
                                (1, prefix),
                                (9, 'atompos'),
                                (43, 'x'),
                                (55, 'y'),
                                (67, 'z'),
                                (84, '(au)'),
                                ]))
      base_vals = [(1, prefix), (9, 'atompos'), (84, long_name)]
      for atom_idx, atom_vals in self._atom_generator(molecule.bond_topologies[0]):
        out.append(self._fw_line(
          base_vals + atom_vals +
          [self._align_dec_point(37, f'{geom.atom_positions[atom_idx].x:.6f}'),
           self._align_dec_point(49, f'{geom.atom_positions[atom_idx].y:.6f}'),
           self._align_dec_point(61, f'{geom.atom_positions[atom_idx].z:.6f}'),
           ]))

      out.append('#\n')
      out.append(self._fw_line([(0, '#'),
                                (1, prefix),
                                (84, '(au; brot: MHz)' if geom.HasField('rotcon') else '(au)'),
                                ]))
      base_vals = [(1, prefix), (84, long_name)]
      if geom.HasField('energy'):
        out.append(self._fw_line(base_vals +
                                 [(9, 'energy'),
                                  self._align_dec_point(49, f'{geom.energy.value:.6f}'),
                                  ]))
      if geom.HasField('gnorm'):
        out.append(self._fw_line(base_vals +
                                 [(9, 'gnorm'),
                                  self._align_dec_point(49, f'{geom.gnorm.value:.6f}'),
                                  ]))
      if geom.HasField('enuc'):
        out.append(self._fw_line(base_vals +
                                 [(9, 'enuc'),
                                  self._align_dec_point(49, f'{geom.enuc.value:.4f}'),
                                  ]))
      if geom.HasField('rotcon'):
        for idx, val in enumerate(geom.rotcon.value):
          if val < 10_000:
            val_str = f'{val:.3f}'
          elif val < 100_000:
            val_str = f'{val:.2f}'
          elif val < 1_000_000:
            val_str = f'{val:.1f}'
          else:
            val_str = f'{val:.0f}' + '.'
          out.append(self._fw_line(base_vals +
                                   [(9, 'brot'),
                                    (21, str(idx + 1)),
                                    self._align_dec_point(49, val_str),
                                  ]))

    write_geometry('ini_geo', molecule.initial_geometries[0])
    if molecule.HasField('optimized_geometry'):
      write_geometry('opt_geo', molecule.optimized_geometry)

    return out

  def get_vib_block(self, molecule, long_name):
    out = []

    if molecule.properties.HasField('zpe_unscaled'):
      out.append('#\n')
      out.append(self._fw_line([(0, '#vib zpe'),
                                (84, '(unscaled, kcal/mol)'),
                                ]))
      out.append(self._fw_line([
        (1, 'vib zpe'),
        (self._align_dec_point(37, f'{molecule.properties.zpe_unscaled.value:.2f}')),
        (84, long_name),
      ]))

    if molecule.properties.HasField('harmonic_frequencies'):
      # Note that we assume if you have frequencies, you have intensities.
      out.append('#\n')
      out.append(self._fw_line([(0, '#vib freq'),
                                (35, 'freq'),
                                (45, 'intens'),
                                (84, '(cm-1, km/mol)'),
                                ]))
      for idx, (freq, intens) in enumerate(zip(molecule.properties.harmonic_frequencies.value,
                                               molecule.properties.harmonic_intensities.value)):
        out.append(self._fw_line([
          (1, 'vib freq'),
          (10, f'{idx+1:>2d}'),
          (self._align_dec_point(37, f'{freq:.1f}')),
          (self._align_dec_point(49, f'{intens:.1f}')),
          (84, long_name),
        ]))

    if len(molecule.properties.normal_modes):
      for mode_idx, mode in enumerate(molecule.properties.normal_modes):
        out.append('#\n')
        freq = molecule.properties.harmonic_frequencies.value[mode_idx]
        out.append(self._fw_line([(0, '#vib mode'),
                                  (10, f'{mode_idx+1:>2d}'),
                                  (41, 'x'),
                                  (53, 'y'),
                                  (65, 'z'),
                                  (84, f'(f={freq:8.1f} cm-1)'),
                                  ]))
        for atom_idx, atom_vals in self._atom_generator(molecule.bond_topologies[0]):
          disp = mode.displacements[atom_idx]
          out.append(self._fw_line(atom_vals +
                                   [(1, 'vib mode'),
                                    (10, f'{mode_idx+1:>2d}'),
                                    (self._align_dec_point(37, f'{disp.x:.4f}')),
                                    (self._align_dec_point(49, f'{disp.y:.4f}')),
                                    (self._align_dec_point(61, f'{disp.z:.4f}')),
                                    (84, long_name),
                                    ]))

    return out

  _SPE_CHECK_PAIRS = [
    ('tmol', 'single_point_energy_pbe0_6_311gd'),
    ('mrcc', 'single_point_energy_pbe0_6_311gd_mrcc'),
    ('orca', 'single_point_energy_pbe0_6_311gd_orca'),
  ]
  _SPE_CATION_PAIRS = [
    ('tmol', 'single_point_energy_pbe0_6_311gd_cat'),
    ('mrcc', 'single_point_energy_pbe0_6_311gd_cat_mrcc'),
    ('orca', 'single_point_energy_pbe0_6_311gd_cat_orca'),
  ]
  _SPE_STD_PAIRS = [
    ('hf_2sp', 'single_point_energy_hf_2sp'),
    ('hf_2sd', 'single_point_energy_hf_2sd'),
    ('hf_3psd', 'single_point_energy_hf_3psd'),
    ('hf_3', 'single_point_energy_hf_3'),
    ('hf_4', 'single_point_energy_hf_4'),
    ('hf_34', 'single_point_energy_hf_34'),
    ('hf_631gd', 'single_point_energy_hf_6_31gd'),
    ('hf_tzvp', 'single_point_energy_hf_tzvp'),
    ('hf_cvtz', 'single_point_energy_hf_cvtz'),
    ('mp2_2sp', 'single_point_energy_mp2_2sp'),
    ('mp2_2sd', 'single_point_energy_mp2_2sd'),
    ('mp2_3psd', 'single_point_energy_mp2_3psd'),
    ('mp2_3', 'single_point_energy_mp2_3'),
    ('mp2_4', 'single_point_energy_mp2_4'),
    ('mp2_34', 'single_point_energy_mp2_34'),
    ('mp2_tzvp', 'single_point_energy_mp2_tzvp'),
    ('mp2full_cvtz', 'single_point_energy_mp2ful_cvtz'),
    ('cc2_tzvp', 'single_point_energy_cc2_tzvp'),
    ('ccsd_2sp', 'single_point_energy_ccsd_2sp'),
    ('ccsd_2sd', 'single_point_energy_ccsd_2sd'),
    ('ccsd_3psd', 'single_point_energy_ccsd_3psd'),
    ('ccsd_t_2sp', 'single_point_energy_ccsd_t_2sp'),
    ('ccsd_t_2sd', 'single_point_energy_ccsd_t_2sd'),
    ('b3lyp_631ppgdp', 'single_point_energy_b3lyp_6_31ppgdp'),
    ('b3lyp_augpcs1', 'single_point_energy_b3lyp_aug_pcs_1'),
    ('pbe0_631ppgdp', 'single_point_energy_pbe0_6_31ppgdp'),
    ('pbe0_augpc1', 'single_point_energy_pbe0_aug_pc_1'),
    ('pbe0_augpcs1', 'single_point_energy_pbe0_aug_pcs_1'),
    ('pbe0d3_6311gd', 'single_point_energy_pbe0d3_6_311gd'),
  ]
  _SPE_COMP_PAIRS = [
    ('b5', 'single_point_energy_atomic_b5'),
    ('b6', 'single_point_energy_atomic_b6'),
    ('eccsd', 'single_point_energy_eccsd'),
  ]

  def get_spe_block(self, molecule, long_name):
    out = []

    def process_pairs(prefix, header_vals, pairs):
      this_out = []
      for short_name, field_name in pairs:
        if not molecule.properties.HasField(field_name):
          continue
        val = getattr(molecule.properties, field_name).value
        this_out.append(self._fw_line([(1, prefix),
                                       (17, short_name),
                                       self._align_dec_point(37, f'{val:.6f}'),
                                       (84, long_name),
                                       ]))
      if this_out:
        out.append('#\n')
        out.append(self._fw_line(header_vals))
        out.extend(this_out)

    process_pairs('spe check',
                  [(0, '#spe check'),
                   (33, 'pbe0_6311gd'),
                   (84, '(au)'),
                   ],
                  self._SPE_CHECK_PAIRS)
    process_pairs('spe cation',
                  [(0, '#spe cation'),
                   (33, 'pbe0_6311gd'),
                   (84, '(au)'),
                   ],
                  self._SPE_CATION_PAIRS)
    process_pairs('spe std',
                  [(0, '#spe std'),
                   (84, '(au)'),
                   ],
                  self._SPE_STD_PAIRS)
    process_pairs('spe comp',
                  [(0, '#spe comp'),
                   (84, '(au)'),
                   ],
                  self._SPE_COMP_PAIRS)

    return out

  _DIAGNOSTICS_PAIRS = [
    ('d1_2sp', 'diagnostics_d1_ccsd_2sp'),
    ('t1_2sp', 'diagnostics_t1_ccsd_2sp'),
    ('t1_2sd', 'diagnostics_t1_ccsd_2sd'),
    ('t1_3psd', 'diagnostics_t1_ccsd_3psd'),
    ]
  def get_diagnostics_block(self, molecule, long_name):
    out = []

    for short_name, field_name in self._DIAGNOSTICS_PAIRS:
      if not molecule.properties.HasField(field_name):
        continue
      val = getattr(molecule.properties, field_name).value
      out.append(self._fw_line([(1, 'wf_diag'),
                                (17, short_name),
                                self._align_dec_point(37, f'{val:.4f}'),
                                (84, long_name),
                                ]))

    if out:
      return ['#\n', '#wf_diag   \n'] + out
    else:
      return out;

  _BSR_SPLIT_RE = re.compile(r'\s\+\s')
  _BSR_POSITIONS = [51, 66]

  def _num_leading_digits(self, val):
    cnt = 0
    for c in val:
      if not c.isdigit():
        return cnt
      cnt += 1
    return cnt

  def _get_bsr_lines(self, base_vals, bsr_val):
    if not bsr_val:
      return []

    out_vals = []

    # We'll add all the components afer the 0th first (which creates all the lines)
    # then go back and add the 0th term to the first line. It's less special
    # casing in this loop to do it this way.
    components = [re.sub(r'\s+', ' ', s.strip()) for s in self._BSR_SPLIT_RE.split(bsr_val)]
    for comp_idx, comp in enumerate(components[1:]):
      if comp_idx % 2 == 0:
        out_vals.append(copy.copy(base_vals))
      pos = self._BSR_POSITIONS[comp_idx % 2]
      num_digits = self._num_leading_digits(comp)
      if num_digits == 0 or num_digits == 2:
        offset = 2
      elif num_digits == 1:
        offset = 3
      else:
        raise ValueError(f'bsr component {comp} has unexpected number of leading digits')
      out_vals[-1].extend([(pos, '+'),
                           (pos + offset, comp)])

    # Some bsr strings are a single term.
    if not out_vals:
      out_vals.append(copy.copy(base_vals))

    if components[0] == 'molecule':
      out_vals[0].append((41, 'molecule'))
    else:
      out_vals[0].append((39, components[0]))

    return [self._fw_line(vals) for vals in out_vals]

  def get_atomic2_gen_block(self, molecule, long_name):
    out = []

    base_vals = [(1, 'at2_gen'), (84, long_name)]
    out.extend(self._get_bsr_lines(base_vals + [(17, 'bsr_left')],
                                   molecule.properties.bond_separation_reaction_left.value))
    out.extend(self._get_bsr_lines(base_vals + [(17, 'bsr_right')],
                                   molecule.properties.bond_separation_reaction_right.value))

    if molecule.properties.HasField('diagnostics_t1_ccsd_2sd'):
      val = molecule.properties.diagnostics_t1_ccsd_2sd.value
      out.append(self._fw_line(base_vals +
                               [(17, 't1'),
                                self._align_dec_point(37, f'{val:.4f}'),
                                ]))

    if molecule.properties.HasField('diagnostics_t1_ccsd_2sp_excess'):
      val = molecule.properties.diagnostics_t1_ccsd_2sp_excess.value
      out.append(self._fw_line(base_vals +
                               [(17, 't1exc'),
                                self._align_dec_point(37, f'{val:.4f}'),
                                ]))

    if out:
      return ['#\n', '#at2_gen         \n'] + out
    else:
      return out;

  _ATOMIC2_PAIRS = [
    ('ereac', 'bond_separation_energy'),
    ('eae', 'atomization_energy_excluding_zpe'),
    ('ea0', 'atomization_energy_including_zpe'),
    ('hf0', 'enthalpy_of_formation_0k'),
    ('hf298', 'enthalpy_of_formation_298k'),
    ]
  def get_atomic2_um_block(self, molecule, long_name):
    out = []

    header_vals = [(0, '#at2_um'),
                   (37, 'val'),
                   (49, 'unc'),
                   (84, '(kcal/mol)'),
                   ]
    base_vals = [(1, "at2_um"), (84, long_name)]
    if molecule.properties.HasField('zpe_atomic_um'):
      out.append('#\n')
      out.append(self._fw_line(header_vals))
      out.append(self._fw_line(
        base_vals +
        [(9, 'zpe'),
         self._align_dec_point(37, f'{molecule.properties.zpe_atomic_um.value:.2f}'),
         self._align_dec_point(49, f'{molecule.properties.zpe_atomic_um_ci.value:.2f}'),
         ]))

    # This might be a terrible idea, but since there are so many fields with regular
    # naming, I'm going to construct the field name from pieces
    for method in ['b5', 'b6', 'eccsd']:
      this_out = []
      for short_name, field_prefix in self._ATOMIC2_PAIRS:
        field = field_prefix + '_'
        # HACK until I do field rnames
        if method != 'eccsd':
          field += 'atomic_'
        field += method

        if not molecule.properties.HasField(field + '_um'):
          continue

        this_out.append(self._fw_line(base_vals +
          [(9, short_name),
           (17, method),
           self._align_dec_point(37, '{:.2f}'.format(
             getattr(molecule.properties, field + '_um').value)),
           self._align_dec_point(49, '{:.2f}'.format(
             getattr(molecule.properties, field + '_um_ci').value)),
           ]))

      if this_out:
        out.append('#\n')
        out.append(self._fw_line(header_vals + [(17, method)]))
        out.extend(this_out)

    return out

  def get_atomic2_std_block(self, molecule, long_name):
    out = []

    header_vals = [(0, '#at2_std'),
                   (84, '(kcal/mol)'),
                   ]
    base_vals = [(1, "at2_std"),
                 (84, long_name)]
    if molecule.properties.HasField('zpe_atomic'):
      out.append('#\n')
      out.append(self._fw_line(header_vals))
      out.append(self._fw_line(
        base_vals +
        [(9, 'zpe'),
         self._align_dec_point(37, f'{molecule.properties.zpe_atomic.value:.2f}'),
         ]))

    # This might be a terrible idea, but since there are so many fields with regular
    # naming, I'm going to construct the field name from pieces
    for method in ['b5', 'b6', 'eccsd']:
      this_out = []
      for short_name, field_prefix in self._ATOMIC2_PAIRS:
        field = field_prefix + '_'
        # HACK until I do field rnames
        if method != 'eccsd':
          field += 'atomic_'
        field += method

        if not molecule.properties.HasField(field):
          continue

        this_out.append(self._fw_line(base_vals +
          [(9, short_name),
           (17, method),
           self._align_dec_point(37, '{:.2f}'.format(
             getattr(molecule.properties, field).value)),
           ]))

      if this_out:
        out.append('#\n')
        out.append(self._fw_line(header_vals + [(17, method)]))
        out.extend(this_out)

    return out

  _ORB_PAIRS = [
    ('hf_3', 'hf_3'),
    ('hf_4', 'hf_4'),
    ('hf_631gd', 'hf_6_31gd'),
    ('hf_cvtz', 'hf_cvtz'),
    ('hf_tzvp', 'hf_tzvp'),
    ('b3lyp_631ppgdp', 'b3lyp_6_31ppgdp'),
    ('b3lyp_augpcs1', 'b3lyp_aug_pcs_1'),
    ('pbe0_631ppgdp', 'pbe0_6_31ppgdp'),
    ('pbe0_6311gd', 'pbe0_6_311gd'),
    ('pbe0_augpc1', 'pbe0_aug_pc_1'),
    ('pbe0_augpcs1', 'pbe0_aug_pcs_1'),
    ]

  def get_orb_block(self, molecule, long_name):
    out = []

    for short_name, partial_field in self._ORB_PAIRS:
      homo_field = 'homo_' + partial_field
      lumo_field = 'lumo_' + partial_field
      if not molecule.properties.HasField(homo_field):
        continue
      out.append(self._fw_line([
        (1, 'orb'),
        (17, short_name),
        self._align_dec_point(37, '{:.5f}'.format(getattr(molecule.properties, homo_field).value)),
        self._align_dec_point(49, '{:.5f}'.format(getattr(molecule.properties, lumo_field).value)),
        (84, long_name),
        ]))

    if not out:
      return []

    return ['#\n', self._fw_line([(0, '#orb'),
                                  (38, 'ehomo'),
                                  (50, 'elumo'),
                                  (84, '(au)'),
                                  ])] + out

  def get_exc_block(self, molecule, long_name):
    out = []

    if not molecule.properties.HasField('excitation_energies_cc2'):
      return []

    out.append('#\n')
    out.append(self._fw_line([(0, '#exc'),
                              (40, 'ene'),
                              (53, 'os'),
                              (84, '(au)'),
                              ]))

    for idx, (exc, os) in enumerate(zip(
        molecule.properties.excitation_energies_cc2.value,
        molecule.properties.excitation_oscillator_strengths_cc2.value)):
      out.append(self._fw_line([(1, 'exc'),
                                (21, str(idx+1)),
                                self._align_dec_point(37, f'{exc:.5f}'),
                                self._align_dec_point(49, f'{os:.5f}'),
                                (84, long_name),
                                ]))

    return out

  _NMR_FIELDS = [
    ('b3lyp_', '631ppgdp', 'nmr_isotropic_shielding_b3lyp_6_31ppgdp'),
    ('b3lyp_', 'augpcs1', 'nmr_isotropic_shielding_b3lyp_aug_pcs_1'),
    ('pbe0_', '631ppgdp', 'nmr_isotropic_shielding_pbe0_6_31ppgdp'),
    ('pbe0_', 'augpcs1', 'nmr_isotropic_shielding_pbe0_aug_pcs_1'),
    ]
  _DEC_POINT_POSITIONS = [37, 49, 61, 73]

  def get_nmr_block(self, molecule, long_name):
    # This block is annoyingly differnt than the others because
    # * the header row is two lines long
    # * Values are not in a fixed place, but migrate based on what is available.
    # So in this one, we will build up the vals and make all teh strings at the ends
    lines_vals = [[] for _ in range(len(molecule.bond_topologies[0].atoms) + 2)]
    num_values_present = 0
    for line0_str, line1_str, field in self._NMR_FIELDS:
      if not molecule.properties.HasField(field):
        continue;
      pos = self._DEC_POINT_POSITIONS[num_values_present]
      lines_vals[0].append((pos - 5, f'{line0_str:>8s}'))
      lines_vals[1].append((pos - 5, f'{line1_str:>8s}'))
      for idx, val in enumerate(getattr(molecule.properties, field).values, start=2):
        lines_vals[idx].append(self._align_dec_point(pos, f'{val:.2f}'))
      num_values_present += 1

    if not num_values_present:
      return []

    lines_vals[0].extend([(0, '#nmr'), (84, '(ppm)')])
    lines_vals[1].append((0, '#nmr'))
    for atom_idx, atom_vals in self._atom_generator(molecule.bond_topologies[0]):
      lines_vals[atom_idx + 2].extend(atom_vals)
      lines_vals[atom_idx + 2].extend([(1, 'nmr'), (84, long_name)])

    return ['#\n'] + [self._fw_line(vals) for vals in lines_vals]

  _CHARGE_PAIRS = [
    ('esp', 'esp_fit'),
    ('mul', 'mulliken'),
    ('loe', 'loewdin'),
    ('nat', 'natural_nbo'),
    ]
  def get_charge_block(self, molecule, long_name):
    out = []

    # TODO: probalby can remove the dup here when I rename field
    for method_idx, (method, method_for_field) in enumerate(
        [('pbe0_augpc1', 'pbe0_aug_pc_1'),
         ('hf_631gd', 'hf_6_31gd')]):
      lines_vals = [[] for _ in range(len(molecule.bond_topologies[0].atoms) + 1)]
      num_values_present = 0

      for pos, (short_name, partial_field) in zip(self._DEC_POINT_POSITIONS,
                                                  self._CHARGE_PAIRS):
        field = f'partial_charges_{partial_field}_{method_for_field}'
        if not molecule.properties.HasField(field):
          continue

        lines_vals[0].append((pos + 2, short_name))
        for idx, val in enumerate(getattr(molecule.properties, field).values, start=1):
          lines_vals[idx].append(self._align_dec_point(pos, f'{val:.4f}'))
        num_values_present += 1

      if not num_values_present:
        continue

      lines_vals[0].extend([(0, '#chg'),
                            (5, str(method_idx + 1)),
                            (7, ':'),
                            (17, method),
                            (84, '(au)'),
                            ])
      for atom_idx, atom_vals in self._atom_generator(molecule.bond_topologies[0]):
        lines_vals[atom_idx + 1].extend(atom_vals)
        lines_vals[atom_idx + 1].extend([(1, 'chg'),
                                         (5, str(method_idx + 1)),
                                         (84, long_name),
                                         ])
      out.append('#\n')
      out.extend(self._fw_line(vals) for vals in lines_vals)

    return out

  def get_elec_block(self, molecule, long_name):
    out = []

    if molecule.properties.HasField('dipole_dipole_polarizability_pbe0_aug_pc_1'):
      out.append('#\n')
      out.append(self._fw_line([(0, '#elec pol'),
                                (32, 'pbe0_augpc1'),
                                (84, '(au)'),
                                ]))
      for comp in smu_parser_lib.RANK2_ENCODING_ORDER:
        out.append(self._fw_line([
          (1, 'elec pol'),
          (21, f'{comp:>3s}'),
          self._align_dec_point(37, '{:.3f}'.format(getattr(
            molecule.properties.dipole_dipole_polarizability_pbe0_aug_pc_1, comp))),
          (84, long_name),
        ]))

    def write_two_col(short_name, pbe0_field, hf_field, components):
      pbe0_val = None
      hf_val = None
      if molecule.properties.HasField(pbe0_field):
        pbe0_val = getattr(molecule.properties, pbe0_field)
      if molecule.properties.HasField(hf_field):
        hf_val = getattr(molecule.properties, hf_field)
      if pbe0_val or hf_val:
        out.append('#\n')
        header_vals = [(0, '#elec'),
                       (6, short_name),
                       (84, '(au)'),
                       ]
        if pbe0_val:
          header_vals.append((32, 'pbe0_augpc1'))
        if hf_val:
          header_vals.append((59, 'hf_631gd'))
        out.append(self._fw_line(header_vals))

        for comp in components:
          line_vals = [(1, 'elec'),
                       (6, short_name),
                       (21, f'{comp:>3s}'),
                       (84, long_name),
                       ]
          if pbe0_val:
            line_vals.append(self._align_dec_point(37, '{:.3f}'.format(
              getattr(pbe0_val, comp))))
          if hf_val:
            line_vals.append(self._align_dec_point(61, '{:.3f}'.format(
              getattr(hf_val, comp))))

          out.append(self._fw_line(line_vals))

    write_two_col('dip',
                  'dipole_moment_pbe0_aug_pc_1',
                  'dipole_moment_hf_6_31gd',
                  ['x', 'y', 'z'])
    write_two_col('qua',
                  'quadrupole_moment_pbe0_aug_pc_1',
                  'quadrupole_moment_hf_6_31gd',
                  smu_parser_lib.RANK2_ENCODING_ORDER)
    write_two_col('oct',
                  'octopole_moment_pbe0_aug_pc_1',
                  'octopole_moment_hf_6_31gd',
                  smu_parser_lib.RANK3_ENCODING_ORDER)

    return out

  def process(self, molecule):
    long_name = get_long_molecule_name(molecule)
    contents = ['#===============================================================================\n']
    contents.extend(self.get_mol_id_block(molecule, long_name))
    contents.extend(self.get_mol_spec_block(molecule, long_name))
    contents.extend(self.get_calc_block(molecule, long_name))
    contents.extend(self.get_duplicates_block(molecule, long_name))
    contents.extend(self.get_bond_topologies_block(molecule, long_name))
    contents.extend(self.get_geometries_block(molecule, long_name))
    contents.extend(self.get_vib_block(molecule, long_name))
    contents.extend(self.get_spe_block(molecule, long_name))
    contents.extend(self.get_diagnostics_block(molecule, long_name))
    contents.extend(self.get_atomic2_gen_block(molecule, long_name))
    contents.extend(self.get_atomic2_um_block(molecule, long_name))
    contents.extend(self.get_atomic2_std_block(molecule, long_name))
    contents.extend(self.get_orb_block(molecule, long_name))
    contents.extend(self.get_exc_block(molecule, long_name))
    contents.extend(self.get_nmr_block(molecule, long_name))
    contents.extend(self.get_charge_block(molecule, long_name))
    contents.extend(self.get_elec_block(molecule, long_name))

    return ''.join(contents)


NEGATIVE_ZERO_RE = re.compile(r'-(0\.0+)\b')


def check_dat_formats_match(original, generated):
  """Checks whether a regenerated .dat format matches the original.

  There are several known cases where output can be non-meaningfully different.

  Args:
    original: list of lines of a .dat file
    generated: list of lines of regenerated .dat file

  Raises:
    WrongNumberOfLinesError: when lengths don't match
    RegeneratedLinesError: when regenerated files contains lines not in
      original
    LineOrderError: when regenerated lines are the same, but in the wrong order
  """

  def normalize_lines(lines):
    # The original file has several coordinates stored as -0.0, which creates
    # mismatches when compared with the corresponding 0.0 in generated files.
    lines = [NEGATIVE_ZERO_RE.sub(r' \1', s).rstrip() for s in lines]
    # This code removes any blank lines at the very end.
    cnt = 0
    while not lines[-(cnt + 1)]:
      cnt += 1
    if cnt == 0:
      return lines
    return lines[:-cnt]

  fixed_original = normalize_lines(original)
  fixed_generated = normalize_lines(generated)

  # Check if the modified lines match the original.
  missing_lines = set(fixed_original) - set(fixed_generated)
  excess_lines = set(fixed_generated) - set(fixed_original)
  if missing_lines or excess_lines:
    raise RegeneratedLinesError(missing_lines, excess_lines)

  # Now check the order of the lines generated
  for idx, (orig, regen) in enumerate(zip(fixed_original, fixed_generated)):
    if orig != regen:
      raise LineOrderError(idx, orig, regen)
