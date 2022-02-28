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
"""Queries the SMU sqlite database.

Command line interface to extract molecules from thq SMU database.
"""

import contextlib
import csv
import enum
import itertools
import os.path
import random
import sys

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from rdkit import Chem

from smu import dataset_pb2
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import smu_molecule
from smu.geometry import topology_from_geom
from smu.geometry import utilities
from smu.parser import smu_utils_lib
from smu.parser import smu_writer_lib


class OutputFormat(enum.Enum):
  pbtxt = 1
  sdf_opt = 2
  sdf_init = 3
  sdf_init_opt = 4
  atomic_input = 5
  dat = 6


flags.DEFINE_string(
    'input_sqlite', None,
    'Path of sqlite file to read. Must be on the local filesystem.')
flags.DEFINE_string(
    'output_path', None,
    'Path to output file to write. If not specified, will write to stdout.')
flags.DEFINE_list('btids', [], 'List of bond topology ids to query')
flags.DEFINE_list('cids', [], 'List of conformer ids to query')
flags.DEFINE_list('smiles', [], 'List of smiles to query')
flags.DEFINE_list('stoichiometries', [], 'List of stoichiometries to query')
flags.DEFINE_string('smarts', '',
                    'SMARTS query to retrieve confomers with matching bond topology. '
                    'Note that this is a single value, not a comma separated list')
flags.DEFINE_list(
    'topology_query_smiles', [],
    'List of smiles to query, where the valid bond lengths are '
    'given by --bond_lengths_csv and --bond_lengths. '
    'Will return all conformers where the given smiles is a '
    'valid decsription of that geometry given the bond lengths. '
    'If you are using the default bond lengths, you should just '
    'use --smiles as this method is much slower.')
flags.DEFINE_float('random_fraction', 0.0,
                   'Randomly return this fraction of DB.')
flags.DEFINE_enum_class('output_format', OutputFormat.pbtxt, OutputFormat,
                        'Format for the found SMU entries')
flags.DEFINE_enum(
  'which_topologies', 'all', ['all', 'best', 'starting'],
  'For sdf and atomic_input output formats, which bond '
  'topologies shoudl be returned? '
  '"all" means all topologies '
  '"best" means a single best topology '
  '"starting" means the single topology used for the calculations')
flags.DEFINE_boolean(
    'redetect_topology', False,
    'Whether to rerun the topology detection on the conformers')
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

def smarts_query(db, smarts, outputter):
  if not smarts:
    return

  logging.info('Starting SMARTS query "%s"', smarts)
  bt_ids = list(db.find_bond_topology_id_by_smarts(smarts))

  logging.info('SMARTS query "%s" produced bond topology %d results',
               smarts, len(bt_ids))

  if not bt_ids:
    return

  if len(bt_ids) > _SMARTS_BT_BATCH_SIZE:
    message = (
      f'WARNING: Smarts query "{smarts}" matched {len(bt_ids)} bond topologies. '
      'This may be very slow and produce the same conformer multiple times. '
      'Trying anyways...')
    logging.warning(message)
    print(message, file=sys.stderr, flush=True)

  count = 0
  for batch_idx in range(len(bt_ids) // _SMARTS_BT_BATCH_SIZE + 1):
    logging.info('Starting batch %d / %d',
                 batch_idx, len(bt_ids) // _SMARTS_BT_BATCH_SIZE + 1)
    for c in db.find_by_bond_topology_id_list(
        bt_ids[batch_idx * _SMARTS_BT_BATCH_SIZE
               :(batch_idx + 1) * _SMARTS_BT_BATCH_SIZE]):
      count += 1
      outputter.output(c)

  logging.info('SMARTS query produced %d conformers', count)


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
    print(
      '# proto-message: MultipleConformers',
      file=self.outfile)

  def output(self, conformer):
    """Writes a conformer.

    Args:
      conformer: dataset_pb2.Conformer
    """
    # This is kind of a hack. We manually write the conformers { }
    # formatting expected by MultipleConformers rather than actually using a
    # MultipleConformers message.
    print(
      'conformers {',
      file=self.outfile)
    self.outfile.write(str(conformer))
    print('}', file=self.outfile)

  def close(self):
    self.outfile.close()


class SDFOutputter:
  """Simple internal class to write entries as multi molecule SDF files."""

  def __init__(self, output_path, init_geometry, opt_geometry,
               which_topologies):
    """Creates SDFOutputter.

    At least one of init_geometry and opt_geometry should be True

    Args:
      output_path: file path to write to
      init_geometry: bool, whether to write with initial_geometries
      opt_geometry: bool, whether to write with optimized_geometry
      which_topologies: string, which topologies to return
    """
    self.init_geometry = init_geometry
    self.opt_geometry = opt_geometry
    self.which_topologies = which_topologies
    if output_path:
      self.writer = Chem.SDWriter(output_path)
    else:
      self.writer = Chem.SDWriter(sys.stdout)

  def output(self, conformer):
    """Writes a Conformer.

    Args:
      conformer: dataset_pb2.Conformer
    """
    for mol in smu_utils_lib.conformer_to_molecules(
        conformer,
        include_initial_geometries=self.init_geometry,
        include_optimized_geometry=self.opt_geometry,
        which_topologies=self.which_topologies):
      self.writer.write(mol)

  def close(self):
    self.writer.close()


class AtomicInputOutputter:
  """Internal class to write output as the inputs to atomic code."""

  def __init__(self, output_path):
    """Creates AtomicInputOutputter.

    Args:
      output_path: directory to write output files to
    """
    self.output_path = output_path
    if output_path and not os.path.isdir(self.output_path):
      raise ValueError(
          'Atomic input requires directory as output path, got {}'.format(
              self.output_path))
    self.atomic_writer = smu_writer_lib.AtomicInputWriter()

  def output(self, conformer):
    if self.output_path is None:
      sys.stdout.write(self.atomic_writer.process(conformer))
    else:
      with open(
          os.path.join(
              self.output_path,
              self.atomic_writer.get_filename_for_atomic_input(conformer)),
          'w') as f:
        f.write(self.atomic_writer.process(conformer))

  def close(self):
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

  def output(self, conformer):
    """Writes a conformer.

    Args:
      conformer: dataset_pb2.Conformer
    """
    self.outfile.write(self.writer.process_stage2_proto(conformer))

  def close(self):
    self.outfile.close()


class ReDetectTopologiesOutputter:
  """Reruns topology detection before handing to another outputter."""

  def __init__(self, outputter, db):
    self._wrapped_outputter = outputter
    self._geometry_data = GeometryData.get_singleton()
    self._matching_parameters = smu_molecule.MatchingParameters()
    self._db = db

  def output(self, conformer):
    """Writes a Conformer.

    Args:
      conformer: dataset_pb2.Conformer
    """
    matches = topology_from_geom.bond_topologies_from_geom(
        conformer,
        bond_lengths=self._geometry_data.bond_lengths,
        matching_parameters=self._matching_parameters)

    if not matches.bond_topology:
      logging.error('No bond topology matched for %s', conformer.conformer_id)
    else:
      del conformer.bond_topologies[:]
      conformer.bond_topologies.extend(matches.bond_topology)
      for bt in conformer.bond_topologies:
        try:
          bt.bond_topology_id = self._db.find_bond_topology_id_for_smiles(bt.smiles)
        except KeyError:
          logging.error('Did not find bond topology id for smiles %s',
                        bt.smiles)

    self._wrapped_outputter.output(conformer)

  def close(self):
    self._wrapped_outputter.close()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.get_absl_handler().use_absl_log_file()

  logging.info('Opening %s', FLAGS.input_sqlite)
  db = smu_sqlite.SMUSQLite(FLAGS.input_sqlite, 'r')
  if FLAGS.output_format == OutputFormat.pbtxt:
    outputter = PBTextOutputter(FLAGS.output_path)
  elif FLAGS.output_format == OutputFormat.sdf_init:
    outputter = SDFOutputter(
        FLAGS.output_path,
        init_geometry=True,
        opt_geometry=False,
        which_topologies=FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.sdf_opt:
    outputter = SDFOutputter(
        FLAGS.output_path,
        init_geometry=False,
        opt_geometry=True,
        which_topologies=FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.sdf_init_opt:
    outputter = SDFOutputter(
        FLAGS.output_path,
        init_geometry=True,
        opt_geometry=True,
        which_topologies=FLAGS.which_topologies)
  elif FLAGS.output_format == OutputFormat.atomic_input:
    outputter = AtomicInputOutputter(FLAGS.output_path)
  elif FLAGS.output_format == OutputFormat.dat:
    outputter = DatOutputter(FLAGS.output_path)
  else:
    raise ValueError(f'Bad output format {FLAGS.output_format}')

  if FLAGS.redetect_topology:
    outputter = ReDetectTopologiesOutputter(outputter, db)

  with contextlib.closing(outputter):
    for cid in (int(x) for x in FLAGS.cids):
      conformer = db.find_by_conformer_id(cid)
      outputter.output(conformer)

    for c in db.find_by_bond_topology_id_list([int(x) for x in FLAGS.btids]):
      outputter.output(c)

    for c in db.find_by_smiles_list(FLAGS.smiles):
      outputter.output(c)

    for stoich in FLAGS.stoichiometries:
      conformers = db.find_by_stoichiometry(stoich)
      for c in conformers:
        outputter.output(c)

    for smiles in FLAGS.topology_query_smiles:
      geometry_data = GeometryData.get_singleton()
      for c in db.find_by_topology(smiles,
                                   bond_lengths=geometry_data.bond_lengths):
        outputter.output(c)

    smarts_query(db, FLAGS.smarts, outputter)

    if FLAGS.random_fraction:
      for conformer in db:
        if random.random() < FLAGS.random_fraction:
          outputter.output(conformer)


if __name__ == '__main__':
  app.run(main)
