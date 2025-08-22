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
"""Generate the SQLite DB from TFRecord files."""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from tensorflow.io import gfile
from smu import dataset_pb2
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom
from smu.parser import smu_utils_lib

flags.DEFINE_string('input_tfrecord', None, 'Glob of tfrecord files to read')
flags.DEFINE_string('output_sqlite', None, 'Path of sqlite file to generate')
flags.DEFINE_string(
    'bond_topology_csv', None,
    '(optional) Path of bond_topology.csv for smiles to btid mapping')
flags.DEFINE_string(
    'bond_lengths_csv', None,
    'File usually name <data>_bond_lengths.csv that contains the '
    'observed distribution of bond lengths. Only needed with --mutate')
flags.DEFINE_boolean(
    'mutate', False,
    'Whether to modify the records with our last second changes')

FLAGS = flags.FLAGS

# see tools/find_oo_bonds.py for the source of this
_REDETECT_TOPOLOGY_LIST = [
    9906006,
    10909013,
    15809002,
    15833001,
    15833005,
    16145004,
    16256005,
    16313011,
    16390001,
    20457007,
    20457008,
    20488013,
    20687001,
    20819015,
    28131001,
    36449001,
    87574012,
    87858002,
    87858030,
    93650002,
    93650018,
    93657021,
    93657030,
    93728002,
    93886007,
    93886019,
    93886028,
    93886029,
    93886030,
    93890005,
    93890013,
    93890016,
    94005026,
    94005041,
    94104005,
    94136008,
    94136011,
    94136021,
    94136024,
    94136037,
    94136049,
    94136054,
    94186006,
    98115002,
    98115018,
    98115020,
    98115048,
    98129004,
    98186006,
    98186010,
    98186030,
    98186041,
    98186052,
    98186054,
    98186056,
    98409004,
    98409014,
    103777001,
    120442004,
    120446004,
    120446007,
    120613004,
    120902008,
    120902009,
    120904011,
    121033002,
    121134001,
    121202006,
    121334004,
    121334008,
    121334010,
    121337010,
    121637010,
    122045004,
    122078021,
    122139012,
    122174001,
    122184001,
    122498005,
    122498018,
    122520003,
    122520004,
    122823002,
    122823004,
    122823007,
    122827002,
    131445002,
    132670001,
    132670003,
    132670008,
    132687001,
    136661006,
    138355003,
    138355004,
    138865002,
    149032001,
    149032006,
    149110012,
    149110013,
    149206001,
    149206010,
    149206011,
    149225003,
    149225004,
    149225012,
    149237012,
    149237016,
    149237020,
    149245003,
    149720001,
    149735002,
    149735004,
    149820003,
    149824004,
    150081005,
    150081023,
    150081024,
    150367006,
    150492002,
    150716002,
    151198001,
    151198003,
    151198024,
    151206015,
    151206016,
    151348011,
    151402013,
    151528011,
    151528017,
    151552003,
    151552008,
    151552011,
    151586007,
    151586010,
    151586013,
    151593020,
    151618006,
    151660007,
    151660023,
    151691002,
    151691012,
    151868009,
    151868021,
    151868030,
    151868032,
    151932019,
    151932027,
    151938003,
    151960014,
    151975028,
    151977005,
    152012033,
    152012048,
    152145001,
    152145005,
    152406002,
    152469005,
    152473002,
    152488001,
    152488004,
    152488005,
    152488009,
    152488012,
    152488018,
    152488019,
    152497003,
    152497004,
    152497005,
    152501004,
    152518004,
    152530002,
    152530007,
    152548005,
    152548007,
    152635003,
    152879016,
    152879020,
    152887001,
    152897007,
    152897010,
    152940004,
    152940008,
    153400002,
    153635001,
    177244021,
    177341006,
    177341013,
    177341022,
    177528020,
    177535032,
    177535036,
    177563016,
    177563031,
    177577001,
    177577004,
    177628018,
    177628039,
    177652008,
    177652012,
    177652018,
    177652033,
    177652034,
    177652038,
    177698002,
    177698019,
    177698024,
    177698052,
    177716005,
    177725004,
    178390002,
    178398027,
    178405002,
    178419005,
    178550015,
    178550020,
    178564006,
    178564013,
    178564016,
    178564017,
    178564038,
    178588008,
    178602007,
    178690022,
    178690029,
    178690030,
    178690033,
    178690034,
    178690038,
    178720078,
    178729006,
    178993018,
    178993030,
    178993042,
    178993054,
    179051003,
    179059002,
    179059010,
    179059016,
    179131027,
    179180083,
    179756007,
    179855006,
    179855011,
    179867003,
    179867006,
    180114004,
    180229002,
    180229004,
    180229008,
    211395004,
    211567001,
    211803001,
    212925001,
    213392002,
    213511005,
    231297005,
    231297006,
    231367003,
    238230001,
    238745013,
    238893702,
    242108001,
    242108002,
    242795002,
    256351001,
    256362001,
    257077002,
    257774001,
    258192003,
    258543012,
    258671002,
    259383001,
    259383005,
    259576003,
    259710008,
    260962004,
    260988001,
    260988002,
    261574005,
    261574012,
    261880005,
    261898014,
    262437007,
    262466007,
    262823001,
    263329004,
    263497002,
    263792002,
    264202001,
    264202005,
    264771002,
    264874014,
    264874016,
    264961022,
    264961024,
    264961027,
    264961028,
    265141002,
    265141004,
    265270001,
    265270002,
    265317009,
    265441023,
    265469001,
    265469003,
    265472011,
    265733005,
    265910001,
    265910003,
    266052009,
    266291004,
    266637002,
    266637003,
    267188001,
    267188003,
    267188005,
    267275002,
    267354002,
    267354003,
    267354005,
    267354006,
    267354009,
    267354012,
    267354013,
    267354014,
    267354016,
    267354018,
    267681001,
    267681006,
    267681008,
    267681009,
    267681011,
    267681014,
    267681020,
    267688002,
    267688004,
    267688009,
    267688016,
    267688026,
    267688031,
    267688034,
    267688038,
    267688041,
    267688042,
    267688043,
    267688044,
    267699001,
    267699002,
    267699006,
    267699012,
    267745007,
    267789001,
    267789004,
    267789005,
    267789006,
    325924018,
    325947003,
    325947004,
    325956010,
    325997002,
    326186008,
    326186009,
    326207006,
    326207013,
    326429005,
    326442011,
    326589014,
    326849001,
    327399003,
    327651005,
    327651007,
    327651020,
    328976015,
    329223006,
    329729003,
    329771001,
    329862003,
    331393016,
    331393018,
    331412003,
    331424025,
    331701009,
    331896014,
    331912001,
    331912009,
    331912019,
    331912021,
    331927001,
    331927010,
    332215037,
    332215043,
    332383003,
    332431002,
    332431003,
    332970003,
    333045012,
    333045015,
    333045018,
    333056006,
    333150044,
    333150050,
    333205046,
    333214003,
    333589005,
    333635007,
    334214006,
    334233005,
    334233006,
    334241008,
    334414011,
    334422002,
    335281001,
    336530003,
    336561001,
    337111003,
    337111019,
    337119012,
    337119024,
    337119036,
    337119038,
    337119050,
    337124002,
    337124004,
    337139001,
    337139002,
    337139009,
    337139011,
    337172001,
    337172002,
    337172003,
    337193020,
    419905001,
    420870002,
    427364001,
    427824003,
    428124001,
    428406002,
    428741001,
    430183001,
    499265009,
    499343002,
    500103018,
    500103019,
    500127001,
    500127003,
    500663005,
    501538005,
    502016001,
    503250003,
    503548001,
    503548003,
    503548006,
    503560002,
    503916001,
    507535019,
    507706018,
    509371002,
    509480001,
    509887013,
    511144007,
    512025003,
    512025007,
    512072006,
    513083002,
    513485001,
    514752001,
    514898001,
    514898003,
    514898007,
    516624001,
    516624004,
    516624007,
    516624008,
    517715001,
    517778002,
    667874001,
    667907001,
    671945001,
]


def mutate_conformer(encoded_molecule, bond_lengths, smiles_id_dict):
  """Make some small modifications to molecule.

  We made some last second (months?) changes to the records.
  Rather then rerunning the whole pipeline, we just hacked these
  changes into this step that creates the final database.
  Is that a pretty solution? No, but it works.

  Args:
    encoded_molecule:
    bond_lengths:
    smiles_id_dict:

  Returns:

  """
  molecule = dataset_pb2.Molecule.FromString(encoded_molecule)

  # We changed the fate categories, so we just recompute them.
  if molecule.prop.HasField('calc'):
    molecule.prop.calc.fate = smu_utils_lib.determine_fate(molecule)

  # This is sad and ugly. Due to a bug in the CSD lengths, a few molecules did
  # not get the correct bond topologies assigned. So we just rerun detection
  # for this handful of topologies.
  if molecule.mol_id in _REDETECT_TOPOLOGY_LIST:
    old_bt_count = len(molecule.bond_topo)
    start_topo = molecule.bond_topo[
        smu_utils_lib.get_starting_bond_topology_index(molecule)]
    del molecule.bond_topo[:]
    molecule.bond_topo.append(start_topo)
    assert (topology_from_geom.standard_topology_sensing(
        molecule, bond_lengths, smiles_id_dict))
    new_bt_count = len(molecule.bond_topo)
    logging.info('Ran topology detection on %s, topo count %d to %d',
                 molecule.mol_id, old_bt_count, new_bt_count)

  # We decided to remove the topology and geometry scores and sort the bond
  # topologies by a simple key instead.
  if molecule.bond_topo:
    new_bts = sorted(
        molecule.bond_topo, key=smu_utils_lib.bond_topology_sorting_key)
    del molecule.bond_topo[:]
    molecule.bond_topo.extend(new_bts)
  for bt in molecule.bond_topo:
    bt.ClearField('topology_score')
    bt.ClearField('geometry_score')

  # The duplicates_found field is in an arbitrary order, so we sort it
  if len(molecule.duplicate_found) > 1:
    new_dups = sorted(molecule.duplicate_found)
    del molecule.duplicate_found[:]
    molecule.duplicate_found.extend(new_dups)

  # We didn't do topology detection on a handful of topologies and left
  # The SOURCE_DDT and SOURCE_STARTING bits only set where it should really
  # be all the bits. So we just fix it here.
  # These are the mids for C N O F FF O=O
  if molecule.mol_id in [
      899649001, 899650001, 899651001, 899652001, 1001, 4001
  ]:
    assert len(molecule.bond_topo) == 1
    molecule.bond_topo[0].info = (
        dataset_pb2.BondTopology.SOURCE_STARTING
        | dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_MLCR
        | dataset_pb2.BondTopology.SOURCE_CSD)

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
    bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
    bond_lengths.add_from_sparse_dataframe_file(
        FLAGS.bond_lengths_csv,
        bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS,
        bond_length_distribution.STANDARD_SIG_DIGITS)
    db.bulk_insert((mutate_conformer(raw.numpy(), bond_lengths, smiles_id_dict)
                    for raw in dataset),
                   batch_size=10000)
  else:
    db.bulk_insert((raw.numpy() for raw in dataset), batch_size=10000)

  logging.info('Starting vacuuming')
  db.vacuum()
  logging.info('Vacuuming finished')


if __name__ == '__main__':
  app.run(main)
