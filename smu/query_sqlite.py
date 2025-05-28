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
"""Queries the SMU sqlite database.

Command line interface to extract molecules from thq SMU database.
"""
import contextlib
import enum
import os.path
import random
import sys

from absl import app
from absl import flags
from absl import logging
from google.protobuf import text_format
from rdkit import Chem

from smu import dataset_pb2
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom
from smu.geometry import topology_molecule
from smu.parser import smu_utils_lib
from smu.parser import smu_writer_lib


class OutputFormat(enum.Enum):
  """Output format enum."""
  PBTXT = 1
  SDF_OPT = 2
  SDF_INIT = 3
  SDF_INIT_OPT = 4
  ATOMIC2_INPUT = 5
  DAT = 6
  CLEAN_TEXT = 7


flags.DEFINE_string(
    'input_sqlite', None,
    'Path of sqlite file to read. Must be on the local filesystem.')
flags.DEFINE_string(
    'output_path', None,
    'Path to output file to write. If not specified, will write to stdout.')
flags.DEFINE_list('btids', [], 'List of bond topology ids to query')
flags.DEFINE_list('mids', [], 'List of molecule ids to query')
flags.DEFINE_list('smiles', [], 'List of smiles to query')
flags.DEFINE_list('stoichiometries', [], 'List of stoichiometries to query')
flags.DEFINE_string(
    'smarts', '',
    'SMARTS query to retrieve molomers with matching bond topology. '
    'Note that this is a single value, not a comma separated list')
flags.DEFINE_list(
    'topology_query_smiles', [],
    'List of smiles to query, where the valid bond lengths are '
    'given by --bond_lengths_csv and --bond_lengths. '
    'Will return all molecules where the given smiles is a '
    'valid decsription of that geometry given the bond lengths. '
    'If you are using the default bond lengths, you should just '
    'use --smiles as this method is much slower.')
flags.DEFINE_float('random_fraction', 0.0,
                   'Randomly return this fraction of DB.')
flags.DEFINE_enum_class('output_format', OutputFormat.PBTXT, OutputFormat,
                        'Format for the found SMU entries')
flags.DEFINE_enum_class(
    'which_topologies', smu_utils_lib.WhichTopologies.ALL,
    smu_utils_lib.WhichTopologies, 'This flag has double duty. '
    'For btids, smiles, and smarts queries, it specifies which topologies'
    'to match. For sdf and atomic2_input output formats, it specifies which bond '
    'topologies should be returned:\n '
    '"all" means all topologies,\n '
    '"starting" means the single topology used for the calculations,\n '
    '"itc" means all topologies detected with our original bond lengths,\n '
    '"mlcr" means all topologies using very permissive covalent radii\n '
    '(from Meng and Lewis), '
    '"csd" means all topologies using bond lengths from the '
    'Cambridge Structural Database')
flags.DEFINE_boolean(
    'redetect_topology', False,
    'Whether to rerun the topology detection on the molecules')
flags.DEFINE_string(
    'bond_lengths_csv', None,
    'File usually name <data>_bond_lengths.csv that contains the '
    'observed distribution of bond lengths.')
flags.DEFINE_string(
    'bond_lengths', None, 'Comma separated terms like form XYX:N-N '
    'where X is an atom type (CNOF*), Y is a bond type (-=#.~), '
    'and N is a possibly empty floating point number. ')

FLAGS = flags.FLAGS


class GeometryData:
  """Class GeometryData."""
  _singleton = None

  def __init__(self, bond_lengths_csv, bond_lengths_arg):
    if bond_lengths_csv is None:
      raise ValueError('--bond_lengths_csv required')
    logging.info('Loading bond_lengths')
    self.bond_lengths = (
        bond_length_distribution.AllAtomPairLengthDistributions())
    self.bond_lengths.add_from_sparse_dataframe_file(
        bond_lengths_csv,
        bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS,
        bond_length_distribution.STANDARD_SIG_DIGITS)
    logging.info('Done loading bond_lengths_csv')

    self.bond_lengths.add_from_string_spec(bond_lengths_arg)

  @classmethod
  def get_singleton(cls):
    if cls._singleton is None:
      cls._singleton = cls(FLAGS.bond_lengths_csv, FLAGS.bond_lengths)
    return cls._singleton


_SMARTS_BT_BATCH_SIZE = 20000


def smarts_query(db, smarts, which_topologies, outputter):
  """SMARTS query.

  Args:
    db:
    smarts:
    which_topologies:
    outputter:
  """
  if not smarts:
    return

  logging.info('Starting SMARTS query "%s"', smarts)
  bt_ids = list(db.find_topo_id_by_smarts(smarts))

  logging.info('SMARTS query "%s" produced bond topology %d results', smarts,
               len(bt_ids))

  if not bt_ids:
    return

  if len(bt_ids) > _SMARTS_BT_BATCH_SIZE:
    message = (
        f'WARNING: Smarts query "{smarts}" matched {len(bt_ids)} bond topologies. '
        'This may be very slow and produce the same molecule multiple times. '
        'Trying anyways...')
    logging.warning(message)
    print(message, file=sys.stderr, flush=True)

  count = 0
  for batch_idx in range(len(bt_ids) // _SMARTS_BT_BATCH_SIZE + 1):
    logging.info('Starting batch %d / %d', batch_idx,
                 len(bt_ids) // _SMARTS_BT_BATCH_SIZE + 1)
    for c in db.find_by_topo_id_list(
        bt_ids[batch_idx * _SMARTS_BT_BATCH_SIZE:(batch_idx + 1) *
               _SMARTS_BT_BATCH_SIZE], which_topologies):
      count += 1
      outputter.output(c)

  logging.info('SMARTS query produced %d molecules', count)


class PBTextOutputter:
  """Simple internal class to write entries to text protocol buffer."""

  def __init__(self, output_path):
    """Creates PBTextOutputter.

    Args:
      output_path: file path to write to
    """
    if output_path:
      self.outfile = open(output_path, 'w')
    else:
      self.outfile = sys.stdout
    print(
        '# proto-file: third_party/google_research/google_research/smu/dataset.proto',
        file=self.outfile)
    print('# proto-message: MultipleMolecules', file=self.outfile)

  def output(self, molecule):
    """Writes a molecule.

    Args:
      molecule: dataset_pb2.Molecule
    """
    # This is kind of a hack. We manually write the molecules { }
    # formatting expected by MultipleMolecules rather than actually using a
    # MultipleMolecules message.
    print('molecules {', file=self.outfile)
    self.outfile.write(
        text_format.MessageToString(
            molecule, use_short_repeated_primitives=True))
    print('}', file=self.outfile)

  def close(self):
    """Closes all resources."""
    self.outfile.close()


class SDFOutputter:
  """Simple internal class to write entries as multi molecule SDF files."""

  def __init__(self, output_path, init_geometry, opt_geometry,
               which_topologies):
    """Creates SDFOutputter.

    At least one of init_geometry and opt_geometry should be True

    Args:
      output_path: file path to write to
      init_geometry: bool, whether to write with ini_geo
      opt_geometry: bool, whether to write with opt_geo
      which_topologies: string, which topologies to return
    """
    self.init_geometry = init_geometry
    self.opt_geometry = opt_geometry
    self.which_topologies = which_topologies
    if output_path:
      self.writer = Chem.SDWriter(output_path)
    else:
      self.writer = Chem.SDWriter(sys.stdout)

  def output(self, molecule):
    """Writes a Molecule.

    Args:
      molecule: dataset_pb2.Molecule
    """
    for mol in smu_utils_lib.molecule_to_rdkit_molecules(
        molecule,
        include_initial_geometries=self.init_geometry,
        include_optimized_geometry=self.opt_geometry,
        which_topologies=self.which_topologies):
      self.writer.write(mol)

  def close(self):
    """Closes all resources."""
    self.writer.close()


class Atomic2InputOutputter:
  """Internal class to write output as the inputs to atomic code."""

  def __init__(self, output_path, which_topologies):
    """Creates Atomic2InputOutputter.

    Args:
      output_path: directory to write output files to
      which_topologies: enum used to select topologies
    """
    self.output_path = output_path
    if output_path and not os.path.isdir(self.output_path):
      raise ValueError(
          'ATOMIC-2 input requires directory as output path, got {}'.format(
              self.output_path))
    self.which_topologies = which_topologies
    self.atomic2_writer = smu_writer_lib.Atomic2InputWriter()

  def output(self, molecule):
    """Writes a molecule.

    Args:
      molecule:
    """
    for bt_idx, bt in smu_utils_lib.iterate_bond_topologies(
        molecule, self.which_topologies):
      if self.output_path is None:
        sys.stdout.write(self.atomic2_writer.process(molecule, bt_idx))
      else:
        with open(
            os.path.join(
                self.output_path,
                self.atomic2_writer.get_filename_for_atomic2_input(
                    molecule, bt_idx)), 'w') as f:
          f.write(self.atomic2_writer.process(molecule, bt_idx))
        if bt.info & dataset_pb2.BondTopology.SOURCE_STARTING:
          with open(
              os.path.join(
                  self.output_path,
                  self.atomic2_writer.get_filename_for_atomic2_input(
                      molecule, None)), 'w') as f:
            f.write(self.atomic2_writer.process(molecule, bt_idx))

  def close(self):
    """Closes all resources."""
    pass


class DatOutputter:
  """Internal class to write output as the original .dat format."""

  def __init__(self, output_path):
    """Creates DatOutputter.

    Args:
      output_path: file to write to
    """
    self.writer = smu_writer_lib.SmuWriter(annotate=False)
    if output_path:
      self.outfile = open(output_path, 'w')
    else:
      self.outfile = sys.stdout

  def output(self, molecule):
    """Writes a molecule.

    Args:
      molecule: dataset_pb2.Molecule
    """
    self.outfile.write(self.writer.process_stage2_proto(molecule))

  def close(self):
    """Closes all resources."""
    self.outfile.close()


class CleanTextOutputter:
  """Internal class to write output as the the clean, human readable text."""

  def __init__(self, output_path):
    """Creates CleanTextOutputter.

    Args:
      output_path: file to write to
    """
    self.writer = smu_writer_lib.CleanTextWriter()
    if output_path:
      self.outfile = open(output_path, 'w')
    else:
      self.outfile = sys.stdout

  def output(self, molecule):
    """Writes a molecule.

    Args:
      molecule: dataset_pb2.Molecule
    """
    self.outfile.write(self.writer.process(molecule))

  def close(self):
    """Closes all resources."""
    self.outfile.close()


class ReDetectTopologiesOutputter:
  """Reruns topology detection before handing to another outputter."""

  def __init__(self, outputter, db):
    self._wrapped_outputter = outputter
    self._geometry_data = GeometryData.get_singleton()
    self._matching_parameters = topology_molecule.MatchingParameters()
    self._db = db

  def output(self, molecule):
    """Writes a Molecule.

    Args:
      molecule: dataset_pb2.Molecule
    """
    matches = topology_from_geom.bond_topologies_from_geom(
        molecule,
        bond_lengths=self._geometry_data.bond_lengths,
        matching_parameters=self._matching_parameters)

    if not matches.bond_topology:
      logging.error('No bond topology matched for %s', molecule.mol_id)
    else:
      del molecule.bond_topo[:]
      molecule.bond_topo.extend(matches.bond_topology)
      for bt in molecule.bond_topo:
        bt.info = dataset_pb2.BondTopology.SOURCE_CUSTOM
        try:
          bt.topo_id = self._db.find_topo_id_for_smiles(bt.smiles)
        except KeyError:
          logging.error('Did not find bond topology id for smiles %s',
                        bt.smiles)

    self._wrapped_outputter.output(molecule)

  def close(self):
    """Closes all resources."""
    self._wrapped_outputter.close()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.get_absl_handler().use_absl_log_file()

  logging.info('Opening %s', FLAGS.input_sqlite)
  db = smu_sqlite.SMUSQLite(FLAGS.input_sqlite, 'r')
  if FLAGS.output_format == OutputFormat.PBTXT:
    outputter = PBTextOutputter(FLAGS.output_path)
  elif FLAGS.output_format == OutputFormat.SDF_INIT:
    outputter = SDFOutputter(
        FLAGS.output_path,
        init_geometry=True,
        opt_geometry=False,
        which_topologies=FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.SDF_OPT:
    outputter = SDFOutputter(
        FLAGS.output_path,
        init_geometry=False,
        opt_geometry=True,
        which_topologies=FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.SDF_INIT_OPT:
    outputter = SDFOutputter(
        FLAGS.output_path,
        init_geometry=True,
        opt_geometry=True,
        which_topologies=FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.ATOMIC2_INPUT:
    outputter = Atomic2InputOutputter(FLAGS.output_path, FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.DAT:
    outputter = DatOutputter(FLAGS.output_path)
  elif FLAGS.output_format == OutputFormat.CLEAN_TEXT:
    outputter = CleanTextOutputter(FLAGS.output_path)
  else:
    raise ValueError(f'Bad output format {FLAGS.output_format}')

  if FLAGS.redetect_topology:
    outputter = ReDetectTopologiesOutputter(outputter, db)

  with contextlib.closing(outputter):
    for mid in (int(x) for x in FLAGS.mids):
      molecule = db.find_by_mol_id(mid)
      outputter.output(molecule)

    for c in db.find_by_topo_id_list([int(x) for x in FLAGS.btids],
                                     FLAGS.which_topologies):
      outputter.output(c)

    for c in db.find_by_smiles_list(FLAGS.smiles, FLAGS.which_topologies):
      outputter.output(c)

    for stoich in FLAGS.stoichiometries:
      molecules = db.find_by_stoichiometry(stoich)
      for c in molecules:
        outputter.output(c)

    for smiles in FLAGS.topology_query_smiles:
      geometry_data = GeometryData.get_singleton()
      for c in db.find_by_topology(
          smiles, bond_lengths=geometry_data.bond_lengths):
        outputter.output(c)

    smarts_query(db, FLAGS.smarts, FLAGS.which_topologies, outputter)

    if FLAGS.random_fraction:
      for molecule in db:
        if random.random() < FLAGS.random_fraction:
          outputter.output(molecule)


if __name__ == '__main__':
  app.run(main)
