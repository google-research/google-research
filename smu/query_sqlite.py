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

"""Queries the SMU sqlite database.

Command line interface to extract molecules from thq SMU database.
"""

import contextlib
import csv
import enum
import os
import pandas as pd
import random
import sys
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from rdkit import Chem

from tensorflow.io import gfile
import tensorflow as tf
from smu import dataset_pb2
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import smu_molecule
from smu.geometry import topology_from_geom
from smu.parser import smu_utils_lib
from smu.parser import smu_writer_lib


class OutputFormat(enum.Enum):
  pbtxt = 1
  sdf_opt = 2
  sdf_init = 3
  sdf_init_opt = 4
  atomic_input = 5
  tfdata = 6


flags.DEFINE_string(
    'input_sqlite', None,
    'Path of sqlite file to read. Must be on the local filesystem.')
flags.DEFINE_string(
    'output_path', None,
    'Path to output file to write. If not specified, will write to stdout.')
flags.DEFINE_list('btids', [], 'List of bond topology ids to query')
flags.DEFINE_list('cids', [], 'List of conformer ids to query')
flags.DEFINE_list('smiles', [], 'List of smiles to query')
flags.DEFINE_float('random_fraction', 0.0,
                   'Randomly return this fraction of DB.')
flags.DEFINE_enum_class('output_format', OutputFormat.pbtxt, OutputFormat,
                        'Format for the found SMU entries')
flags.DEFINE_boolean('redetect_geometry', False,
                     'Whether to rerun the geometry detection on the conformers')
flags.DEFINE_string('bond_lengths_csv', None,
                    'File usually name <data>_bond_lengths.csv that contains the '
                    'observed distribution of bond lengths. '
                    'Only needed if --redetect_geometry')
flags.DEFINE_string('bond_topology_csv', None,
                    'File which contains the desription of all bond topologies '
                    'considered in SMU. Only needed if --redetect_geometry')

FLAGS = flags.FLAGS


class GeometryData:
  _singleton = None
  # These are copied from pipeline.py. Shoudl they be shared somehere?
  _BOND_LENGTHS_SIG_DIGITS = 3
  _BOND_LENGTHS_UNBONDED_RIGHT_TAIL_MASS = 0.9


  def __init__(self):
    if FLAGS.bond_lengths_csv is None:
      raise ValueError('--bond_lengths_csv required')
    logging.info('Loading bond_lengths')
    with gfile.GFile(FLAGS.bond_lengths_csv, 'r') as infile:
      df = pd.read_csv(infile, dtype={'length_str': str})
    self.bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
    self.bond_lengths.add_from_sparse_dataframe(
      df, self._BOND_LENGTHS_UNBONDED_RIGHT_TAIL_MASS, self._BOND_LENGTHS_SIG_DIGITS)
    logging.info('Done loading bond_lengths')

    if FLAGS.bond_topology_csv is None:
      raise ValueError('--bond_topology_csv required')
    logging.info('Loading bond topologies')
    self.smiles_id_dict = {}
    with gfile.GFile(FLAGS.bond_topology_csv, 'r') as infile:
      reader = csv.reader(iter(infile))
      next(reader)  # skip the header line
      for row in reader:
        bt_id, _, _, _, _, smiles = row
        self.smiles_id_dict[smiles] = int(bt_id)
    logging.info('Done loading bond topologies')

  @classmethod
  def get_singleton(cls):
    if cls._singleton is None:
      cls._singleton = cls()
    return cls._singleton


class PBTextOutputter:
  """Simple internal class to write entries to text protocol buffer."""

  def __init__(self, output_path):
    """Creates PBTextOutputter.

    Args:
      output_path: file path to write to
    """
    if output_path:
      self.outfile = gfile.GFile(output_path, 'w')
    else:
      self.outfile = sys.stdout

  def output(self, conformer):
    """Writes a conformer.

    Args:
      conformer: dataset_pb2.Conformer
    """
    self.outfile.write(str(conformer))

  def close(self):
    self.outfile.close()

class TfDataOutputter:
  """Writes output to TFDataRecord form"""

  def __init__(self, output_path: str):
    """Creates TfDataOutputter with output to `output_path`.

    Args:
      output_path:
    """
    self.output = tf.io.TFRecordWriter(path=output_path)

  def output(self, conformer):
    """Writes serialized `conformer`.

    Args:
    """
    self.output.write(conformer.SerializeToString())

  def close(self):
    self.output.close()

class SDFOutputter:
  """Simple internal class to write entries as multi molecule SDF files."""

  def __init__(self, output_path, init_geometry, opt_geometry):
    """Creates SDFOutputter.

    At least one of init_geometry and opt_geometry should be True

    Args:
      output_path: file path to write to
      init_geometry: bool, whether to write with initial_geometries
      opt_geometry: bool, whether to write with optimized_geometry
    """
    self.init_geometry = init_geometry
    self.opt_geometry = opt_geometry
    if output_path:
      self.writer = Chem.SDWriter(gfile.GFile(output_path, 'w'))
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
        include_all_bond_topologies=True):
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
    if output_path and not gfile.isdir(self.output_path):
      raise ValueError(
          'Atomic input requires directory as output path, got {}'.format(
              self.output_path))
    self.atomic_writer = smu_writer_lib.AtomicInputWriter()

  def output(self, conformer):
    if self.output_path is None:
      sys.stdout.write(self.atomic_writer.process(conformer))
    else:
      with gfile.GFile(
          os.path.join(
              self.output_path,
              self.atomic_writer.get_filename_for_atomic_input(conformer)),
          'w') as f:
        f.write(self.atomic_writer.process(conformer))

  def close(self):
    pass


class ReDetectTopologiesOutputter:
  """Reruns topology detection before handing to another outputter."""

  def __init__(self, outputter):
    self._wrapped_outputter = outputter
    self._geometry_data = GeometryData.get_singleton()
    self._matching_parameters = smu_molecule.MatchingParameters()
    self._matching_parameters.must_match_all_bonds = True
    self._matching_parameters.smiles_with_h = False
    self._matching_parameters.smiles_with_labels = False
    self._matching_parameters.neutral_forms_during_bond_matching = True
    self._matching_parameters.consider_not_bonded = True
    self._matching_parameters.ring_atom_count_cannot_decrease = False

  def output(self, conformer):
    matches = topology_from_geom.bond_topologies_from_geom(
      bond_lengths=self._geometry_data.bond_lengths,
      conformer_id=conformer.conformer_id,
      fate=conformer.fate,
      bond_topology=conformer.bond_topologies[0],
      geometry=conformer.optimized_geometry,
      matching_parameters=self._matching_parameters)

    if not matches.bond_topology:
      logging.error(f'Np bond topology matched for {conformer.conformer_id}')
    else:
      del conformer.bond_topologies[:]
      conformer.bond_topologies.extend(matches.bond_topology)
      for bt in conformer.bond_topologies:
        try:
          bt.bond_topology_id = self._geometry_data.smiles_id_dict[bt.smiles]
        except KeyError:
          logging.error(f'Did not find bond topology id for smiles {bt.smiles}')

    self._wrapped_outputter.output(conformer)

  def close(self):
    self._wrapped_outputter.close()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Opening %s', FLAGS.input_sqlite)
  db = smu_sqlite.SMUSQLite(FLAGS.input_sqlite, 'r')
  if FLAGS.output_format == OutputFormat.pbtxt:
    outputter = PBTextOutputter(FLAGS.output_path)
  elif FLAGS.output_format == OutputFormat.sdf_init:
    outputter = SDFOutputter(
        FLAGS.output_path, init_geometry=True, opt_geometry=False)
  elif FLAGS.output_format == OutputFormat.sdf_opt:
    outputter = SDFOutputter(
        FLAGS.output_path, init_geometry=False, opt_geometry=True)
  elif FLAGS.output_format == OutputFormat.sdf_init_opt:
    outputter = SDFOutputter(
        FLAGS.output_path, init_geometry=True, opt_geometry=True)
  elif FLAGS.output_format == OutputFormat.atomic_input:
    outputter = AtomicInputOutputter(FLAGS.output_path)
  elif FLAGS.output_format == OutputFormat.tfdata:
    outputter = TfDataOutputter(FLAGS.output_path)
  else:
    raise ValueError(f'Bad output format {FLAGS.output_format}')

  if FLAGS.redetect_geometry:
    outputter = ReDetectTopologiesOutputter(outputter)

  with contextlib.closing(outputter):
    for cid in (int(x) for x in FLAGS.cids):
      conformer = db.find_by_conformer_id(cid)
      outputter.output(conformer)
    for btid in (int(x) for x in FLAGS.btids):
      conformers = db.find_by_bond_topology_id(btid)
      if not conformers:
        raise KeyError(f'Bond topology {btid} not found')
      for c in conformers:
        outputter.output(c)
    for smiles in FLAGS.smiles:
      conformers = db.find_by_smiles(smiles)
      if not conformers:
        raise KeyError(f'SMILES {smiles} not found')
      for c in conformers:
        outputter.output(c)
    if FLAGS.random_fraction:
      for conformer in db:
        if conformer.fate == dataset_pb2.Conformer.FATE_SUCCESS and random.random(
        ) < FLAGS.random_fraction:
          outputter.output(conformer)


if __name__ == '__main__':
  app.run(main)
