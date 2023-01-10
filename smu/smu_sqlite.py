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
"""Interface to a SQLite DB file for SMU data.

Provides a simpler interface than SQL to create and access the SMU data in an
SQLite database.

The majority of the data is stored as a blob, with just the bond topology id and
smiles string pulled out as fields.
"""

import datetime
import os
import sqlite3

from absl import logging
from rdkit import Chem

from smu import dataset_pb2
from smu.geometry import topology_from_geom
from smu.geometry import topology_molecule
from smu.parser import smu_utils_lib
import snappy

# The name of this table is a hold over before we did a big rename of
# Conformer to Molecule. The column that holds the protobuf is also called
# "conformer"
_MOLECULE_TABLE_NAME = 'conformer'
_BTID_TABLE_NAME = 'btid'
_SMILES_TABLE_NAME = 'smiles'


class ReadOnlyError(Exception):
  """Exception when trying to write to a read only DB."""
  pass


class SMUSQLite:
  """Provides an interface for SMU data to a SQLite DB file.

  The class hides away all the SQL fun with just Molecule protobuf visible in
  the interface.

  Internal details about the tables:
  There are 3 separate tables
  * molecule: Is the primary table which has columns
      * mid: integer molecule id (unique)
      * molecule: blob wire format proto of a molecule proto
  * btid: Used for lookups by bond topology id which has columns
      * btid: integer bond topology id (not unique)
      * mid: integer molecule id (not unique)
  * smiles: Used to map smiles to bond topology ids with columns
      * smiles: text canonical smiles string (unique)
      * btid: integer bond topology id
    Note that if multiple smiles strings are associated with the same bond
    toplogy id, the first one provided will be silently kept.
  """

  def __init__(self, filename, mode='r'):
    """Creates SMUSQLite.

    Args:
      filename: database file, must be on local filesystem
      mode: 'c' (create, deletes existing), 'w' (writable), 'r' (read only)

    Raises:
      FileNotFoundError: if 'r' and file does not exist
    """
    if mode == 'c':
      if os.path.exists(filename):
        os.remove(filename)
      self._read_only = False
      self._conn = sqlite3.connect(filename)
      self._maybe_init_db()
    elif mode == 'w':
      self._read_only = False
      self._conn = sqlite3.connect(filename)
      self._maybe_init_db()
    elif mode == 'r':
      if not os.path.exists(filename):
        raise FileNotFoundError(filename)
      self._conn = sqlite3.connect(filename)
      self._read_only = True
    else:
      raise ValueError('Mode must be c, r, or w')

    self._conn = sqlite3.connect(filename)

  def _maybe_init_db(self):
    """Create the table and indices if they do not exist."""
    make_table = (f'CREATE TABLE IF NOT EXISTS {_MOLECULE_TABLE_NAME} '
                  '(cid INTEGER PRIMARY KEY, '
                  'exp_stoich STRING, '
                  'conformer BLOB)')
    self._conn.execute(make_table)
    self._conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                       f'idx_cid ON {_MOLECULE_TABLE_NAME} (cid)')
    self._conn.execute(f'CREATE INDEX IF NOT EXISTS '
                       f'idx_exp_stoich ON {_MOLECULE_TABLE_NAME} '
                       '(exp_stoich)')
    self._conn.execute(f'CREATE TABLE IF NOT EXISTS {_BTID_TABLE_NAME} '
                       '(btid INTEGER, cid INTEGER)')
    self._conn.execute(f'CREATE INDEX IF NOT EXISTS '
                       f'idx_btid ON {_BTID_TABLE_NAME} (btid)')
    self._conn.execute(f'CREATE TABLE IF NOT EXISTS {_SMILES_TABLE_NAME} '
                       '(smiles TEXT, btid INTEGER)')
    self._conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                       f'idx_smiles ON {_SMILES_TABLE_NAME} (smiles)')
    self._conn.execute('PRAGMA synchronous = OFF')
    self._conn.execute('PRAGMA journal_mode = MEMORY')
    self._conn.commit()

  def bulk_insert(self, encoded_molecules, batch_size=10000, limit=None):
    """Inserts molecules into the database.

    Args:
      encoded_molecules: iterable for encoded dataset_pb2.Molecule
      batch_size: insert performance is greatly improved by putting multiple
        insert into one transaction. 10k was a reasonable default from some
        early exploration.
      limit: maximum number of records to insert

    Raises:
      ReadOnlyError: if mode is 'r'
      ValueError: If encoded_molecules is empty.
    """
    if self._read_only:
      raise ReadOnlyError()
    if not encoded_molecules:
      raise ValueError()

    insert_molecule = (f'INSERT INTO {_MOLECULE_TABLE_NAME} '
                       'VALUES (?, ?, ?)')
    insert_btid = f'INSERT INTO {_BTID_TABLE_NAME} VALUES (?, ?)'
    insert_smiles = (
        f'INSERT OR IGNORE INTO {_SMILES_TABLE_NAME} VALUES (?, ?) ')

    cur = self._conn.cursor()

    start_time = datetime.datetime.now()

    pending_molecule_args = []
    pending_btid_args = []
    pending_smiles_args = []

    def commit_pending():
      cur.executemany(insert_molecule, pending_molecule_args)
      cur.executemany(insert_btid, pending_btid_args)
      cur.executemany(insert_smiles, pending_smiles_args)
      pending_molecule_args.clear()
      pending_btid_args.clear()
      pending_smiles_args.clear()
      self._conn.commit()

    idx = None
    for idx, encoded_molecule in enumerate(encoded_molecules, 1):
      molecule = dataset_pb2.Molecule.FromString(encoded_molecule)
      expanded_stoich = (
          smu_utils_lib.expanded_stoichiometry_from_topology(
              molecule.bond_topo[0]))
      pending_molecule_args.append((molecule.mol_id, expanded_stoich,
                                    snappy.compress(encoded_molecule)))
      for bond_topology in molecule.bond_topo:
        pending_btid_args.append((bond_topology.topo_id, molecule.mol_id))
        pending_smiles_args.append(
            (bond_topology.smiles, bond_topology.topo_id))
      if batch_size and idx % batch_size == 0:
        commit_pending()
        elapsed = datetime.datetime.now() - start_time
        logging.info(
            'bulk_insert: committed at index %d, %f s total, %.6f s/record',
            idx, elapsed.total_seconds(),
            elapsed.total_seconds() / idx)

      if limit and idx >= limit:
        break

    # Commit a final time
    commit_pending()
    elapsed = datetime.datetime.now() - start_time
    logging.info('bulk_insert: Total records %d, %f s, %.6f s/record', idx,
                 elapsed.total_seconds(),
                 elapsed.total_seconds() / idx)

  def bulk_insert_smiles(self, smiles_btid_pairs, batch_size=10000):
    """Insert smiles to bond topology id mapping.

    Args:
      smiles_btid_pairs: iterable of pairs of (smiles, btid)
      batch_size: number of inserts per SQL call

    Raises:
      ReadOnlyError: if DB is read only
    """
    if self._read_only:
      raise ReadOnlyError()

    insert_smiles = (
        f'INSERT OR IGNORE INTO {_SMILES_TABLE_NAME} VALUES (?, ?) ')

    cur = self._conn.cursor()

    pending = []

    def commit_pending():
      cur.executemany(insert_smiles, pending)
      pending.clear()
      self._conn.commit()

    for idx, (smiles, btid) in enumerate(smiles_btid_pairs, 1):
      pending.append([smiles, btid])
      if batch_size and idx % batch_size == 0:
        commit_pending()

    # Commit a final time
    commit_pending()

  def vacuum(self):
    """Uses SQL VACUUM to clean up db.

    Raises:
      ReadOnlyError: if db is read only
    """
    if self._read_only:
      raise ReadOnlyError()
    cur = self._conn.cursor()
    cur.execute('VACUUM')
    self._conn.commit()

  def find_topo_id_for_smiles(self, smiles):
    """Finds the topo_id for the given smiles.

    Args:
      smiles: string to look up

    Returns:
      integer of topo_id

    Raises:
      KeyError: if smiles not found
    """
    cur = self._conn.cursor()
    select = f'SELECT btid FROM {_SMILES_TABLE_NAME} WHERE smiles = ?'
    cur.execute(select, (smiles,))
    result = cur.fetchall()

    if not result:
      raise KeyError(f'SMILES {smiles} not found')

    # Since it's a unique index, there should only be one result and it's a
    # tuple with one value.
    assert len(result) == 1
    assert len(result[0]) == 1
    return result[0][0]

  def get_smiles_id_dict(self):
    """Creates a dictionary of smiles to bond topology id.

    This is not the most efficient way to do this, but we will just pull
    everything out of the db and create a python dictionary.
    If we wanted to be smart, we could make a dictionary behaving object
    that does the queries as needed.

    Returns:
      Dict from SMILES string to bond topology id
    """
    cur = self._conn.cursor()
    select = f'SELECT smiles, btid FROM {_SMILES_TABLE_NAME}'
    cur.execute(select)

    return {smiles: bt_id for (smiles, bt_id) in cur}

  def find_by_mol_id(self, mid):
    """Finds the molecule associated with a molecule id.

    Args:
      mid: molecule id to look up.

    Returns:
      dataset_pb2.Molecule

    Raises:
      KeyError: if mid is not found
    """
    cur = self._conn.cursor()
    select = f'SELECT conformer FROM {_MOLECULE_TABLE_NAME} WHERE cid = ?'
    cur.execute(select, (mid,))
    result = cur.fetchall()

    if not result:
      raise KeyError(f'Molecule id {mid} not found')

    # Since it's a unique index, there should only be one result and it's a
    # tuple with one value.
    assert len(result) == 1
    assert len(result[0]) == 1
    return dataset_pb2.Molecule().FromString(snappy.uncompress(result[0][0]))

  def find_by_topo_id_list(self, btids, which_topologies):
    """Finds all the molecule associated with a bond topology id.

    Args:
      btids: list of bond topology id to look up.
      which_topologies: which topologies to match, see
        smu_utils_lib.WhichTopologies

    Yields:
      dataset_pb2.Molecule
    """
    cur = self._conn.cursor()
    # DISTINCT is because the same mid can have the same btid multiple times.
    select = (''.join([
        f'SELECT DISTINCT cid, conformer '
        f'FROM {_MOLECULE_TABLE_NAME} '
        f'INNER JOIN {_BTID_TABLE_NAME} USING(cid) '
        f'WHERE {_BTID_TABLE_NAME}.btid IN (', ','.join('?' for _ in btids), ')'
    ]))
    cur.execute(select, btids)
    for result in cur:
      molecule = dataset_pb2.Molecule().FromString(
          snappy.uncompress(result[1]))
      for _, bt in smu_utils_lib.iterate_bond_topologies(
          molecule, which_topologies):
        if bt.topo_id in btids:
          yield molecule
          break

  def find_by_smiles_list(self, smiles, which_topologies):
    """Finds all molecule associated with a given smiles string.

    Args:
      smiles: list of string
      which_topologies: which topologies to match, see
        smu_utils_lib.WhichTopologies

    Returns:
      iterable for dataset_pb2.Molecule
    """
    canon_smiles = [
        smu_utils_lib.compute_smiles_for_rdkit_molecule(
            Chem.MolFromSmiles(s, sanitize=False), include_hs=False)
        for s in smiles
    ]
    cur = self._conn.cursor()
    select = (''.join([
        f'SELECT btid FROM {_SMILES_TABLE_NAME} WHERE smiles IN (',
        ','.join('?' for _ in canon_smiles), ')'
    ]))
    cur.execute(select, canon_smiles)
    result = cur.fetchall()

    if not result:
      return []

    return self.find_by_topo_id_list([r[0] for r in result], which_topologies)

  def find_by_expanded_stoichiometry_list(self, exp_stoichs):
    """Finds all of the molecules with a stoichiometry.

    The expanded stoichiometry includes hydrogens as part of the atom type.
    See smu_utils_lib.expanded_stoichiometry_from_topology for a
    description.

    Args:
      exp_stoichs: list of string

    Returns:
      iterable of dataset_pb2.Molecule
    """
    cur = self._conn.cursor()
    select = (''.join([
        f'SELECT conformer '
        f'FROM {_MOLECULE_TABLE_NAME} '
        f'WHERE exp_stoich IN (', ','.join('?' for _ in exp_stoichs), ')'
    ]))
    cur.execute(select, exp_stoichs)
    return (dataset_pb2.Molecule().FromString(snappy.uncompress(result[0]))
            for result in cur)

  def find_by_stoichiometry(self, stoich):
    """Finds all molecules with a given stoichiometry.

    The stoichiometry is like "C6H12".

    Internally, the stoichiometry is converted a set of expanded stoichiometries
    and the query is done to find all of those.

    Args:
      stoich: stoichiometry string like "C6H12", case doesn't matter

    Returns:
      Iterable of type dataset_pb2.Molecule.
    """
    exp_stoichs = list(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry(stoich))
    return self.find_by_expanded_stoichiometry_list(exp_stoichs)

  def find_by_topology(
      self,
      smiles,
      bond_lengths,
      matching_parameters=topology_molecule.MatchingParameters()):
    """Find all molecules which have a detected bond topology.

    Note that this *redoes* the detection. If you want the default detected
    versions, you can just query by SMILES string. This is only useful if you
    adjust the distance thresholds for what a matching bond is.
    To adjust those, you probably want to use
    AllAtomPairLengthDistributions.add_from_string_spec

    Args:
      smiles: smiles string for the target bond topology
      bond_lengths: AllAtomPairLengthDistributions
      matching_parameters: controls the algorithm for matching topologies.
        Generally should not need to be modified.

    Yields:
      dataset_pb2.Molecule
    """
    query_bt = smu_utils_lib.rdkit_molecule_to_bond_topology(
        smu_utils_lib.smiles_to_rdkit_molecule(smiles))
    expanded_stoich = smu_utils_lib.expanded_stoichiometry_from_topology(
        query_bt)
    cnt_matched_molecule = 0
    cnt_molecule = 0
    logging.info('Starting query for %s with stoich %s', smiles,
                 expanded_stoich)
    for molecule in self.find_by_expanded_stoichiometry_list([expanded_stoich]):
      if not smu_utils_lib.molecule_eligible_for_topology_detection(molecule):
        continue
      cnt_molecule += 1
      matches = topology_from_geom.bond_topologies_from_geom(
          molecule,
          bond_lengths=bond_lengths,
          matching_parameters=matching_parameters)
      if smiles in [bt.smiles for bt in matches.bond_topology]:
        cnt_matched_molecule += 1
        del molecule.bond_topo[:]
        molecule.bond_topo.extend(matches.bond_topology)
        for bt in molecule.bond_topo:
          try:
            bt.info = dataset_pb2.BondTopology.SOURCE_CUSTOM
            bt.topo_id = self.find_topo_id_for_smiles(bt.smiles)
          except KeyError:
            logging.error('Did not find bond topology id for smiles %s',
                          bt.smiles)
        yield molecule
    logging.info('Topology query for %s matched %d / %d', smiles,
                 cnt_matched_molecule, cnt_molecule)

  def find_topo_id_by_smarts(self, smarts):
    """Find all bond topology ids that match a smarts pattern.

    Args:
      smarts: SMARTS string

    Yields:
      int, bond topology id

    Raises:
      ValueError: if smarts coudl not be parsed
    """
    pattern = Chem.MolFromSmarts(smarts)
    if not pattern:
      raise ValueError(f'Could not parse SMARTS {smarts}')

    for smiles, bt_id in self.smiles_iter():
      mol = smu_utils_lib.smiles_to_rdkit_molecule(smiles)
      # This is not the prettiest thing in the world. In order for ring markings
      # in the SMARTS to work, RingInfo has to be added. The simplest way to get
      # RingInfo set is to call this function. We didn't put this into the
      # smu_utils_lib just in case it messes something else up.
      Chem.GetSymmSSSR(mol)

      if mol.GetSubstructMatches(pattern):
        yield bt_id

  def smiles_iter(self):
    """Iterates through all (smiles, btid) pairs in the DB.

    Yields:
      (smiles, bt_id)
    """
    cur = self._conn.cursor()
    cur.execute('SELECT smiles, btid FROM smiles')
    yield from cur

  def __iter__(self):
    """Iterates through all dataset_pb2.Molecule in the DB."""
    select = f'SELECT conformer FROM {_MOLECULE_TABLE_NAME} ORDER BY rowid'
    cur = self._conn.cursor()
    cur.execute(select)
    return (dataset_pb2.Molecule().FromString(snappy.uncompress(result[0]))
            for result in cur)
