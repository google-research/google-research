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

"""Generate the SQLite DB from TFRecord files."""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from tensorflow.io import gfile
from smu import dataset_pb2
from smu import smu_sqlite
from smu.parser import smu_utils_lib

flags.DEFINE_string('input_tfrecord', None, 'Glob of tfrecord files to read')
flags.DEFINE_string('output_sqlite', None, 'Path of sqlite file to generate')
flags.DEFINE_string(
    'bond_topology_csv', None,
    '(optional) Path of bond_topology.csv for smiles to btid mapping')
flags.DEFINE_boolean('mutate', False,
                     'Whether to modify the records with our last second changes')

FLAGS = flags.FLAGS


def mutate_conformer(encoded_molecule):
  """Make some small modifications to molecule.

  We made some last second (month?) changes to the records.
  Rather then rerunning the whole pipeline, we just hacked these
  changes into this step that creates the final database.
  Is that a pretty solution? No, but it's functional.
  """
  molecule = dataset_pb2.Molecule.FromString(encoded_molecule)

  # We change the fate categories, so we just recompute them.
  if molecule.prop.HasField('calc'):
    molecule.prop.calc.fate = smu_utils_lib.determine_fate(molecule)

  # We decided to remove the topology and geometry scores and sort the bond
  # topologies by a simple key instead.
  if len(molecule.bond_topo):
    new_bts = sorted(
      molecule.bond_topo, key=smu_utils_lib.bond_topology_sorting_key)
    del molecule.bond_topo[:]
    molecule.bond_topo.extend(new_bts)
  for bt in molecule.bond_topo:
    bt.ClearField('topology_score')
    bt.ClearField('geometry_score')

  # The duplciates_found field is in an arbitrary order, so we sort it
  if len(molecule.duplicate_found) > 1:
    new_dups = sorted(molecule.duplicate_found)
    del molecule.duplicate_found[:]
    molecule.duplicate_found.extend(new_dups)

  # We didn't do topology detection on a handful of topologies and left
  # The SOURCE_ITC and SOURCE_STARTING bits only set where it should really
  # be all the bits. So we just fix it here.
  # These are the mids for C N O F FF
  if molecule.mol_id in [899649001, 899650001, 899651001, 899652001, 1001]:
    assert(len(molecule.bond_topo) == 1)
    molecule.bond_topo[0].info = (
      dataset_pb2.BondTopology.SOURCE_STARTING |
      dataset_pb2.BondTopology.SOURCE_ITC |
      dataset_pb2.BondTopology.SOURCE_MLCR |
      dataset_pb2.BondTopology.SOURCE_CSD)

  return molecule.SerializeToString()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.get_absl_handler().use_absl_log_file()

  logging.info('Opening %s', FLAGS.output_sqlite)
  db = smu_sqlite.SMUSQLite(FLAGS.output_sqlite, 'c')

  if FLAGS.bond_topology_csv:
    logging.info('Starting smiles to btid inserts')
    smiles_id_dict = smu_utils_lib.smiles_id_dict_from_csv(
        open(FLAGS.bond_topology_csv))
    db.bulk_insert_smiles(smiles_id_dict.items())
    logging.info('Finished smiles to btid inserts')
  else:
    logging.info('Skipping smiles inserts')

  logging.info('Starting main inserts')
  dataset = tf.data.TFRecordDataset(gfile.glob(FLAGS.input_tfrecord))
  if FLAGS.mutate:
    db.bulk_insert((mutate_conformer(raw.numpy()) for raw in dataset),
                   batch_size=10000)
  else:
    db.bulk_insert((raw.numpy() for raw in dataset), batch_size=10000)

  logging.info('Starting vacuuming')
  db.vacuum()
  logging.info('Vacuuming finished')

if __name__ == '__main__':
  app.run(main)
