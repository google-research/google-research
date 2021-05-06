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

"""Interface to a SQLite DB file for SMU data.

Provides a simpler interface than SQL to create and access the SMU data in an
SQLite database.

The majority of the data is stored as a blob, with just the bond topology id and
smiles string pulled out as fields.
"""
import os

from absl import logging
import sqlite3

from smu import dataset_pb2

_CONFORMER_TABLE_NAME = 'conformer'
_BTID_TABLE_NAME = 'btid'
_SMILES_TABLE_NAME = 'smiles'


class ReadOnlyError(Exception):
  pass


class SMUSQLite:
  """Provides an interface for SMU data to a SQLite DB file.

  The class hides away all the SQL fun with just Conformer protobuf visible in
  the interface.

  Internal details about the tables:
  There are 3 separate tables
  * conformer: Is the primary table which has columns
      * cid: integer conformer id (unique)
      * conformer: blob wire format proto of a conformer proto
  * btid: Used for lookups by bond topology id which has columns
      * btid: integer bond topology id (not unique)
      * cid: integer conformer id (not unique)
  * smiles: Used to map smiles to bond topology ids with columns
      * smiles: text canonical smiles string (unique)
      * btid: integer bond topology id
    Note that if multiple smiles strings are associated with the same bond
    toplogy id, the first one provided will be silently kept.
  """

  def __init__(self, filename, mode):
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
    make_table = (f'CREATE TABLE IF NOT EXISTS {_CONFORMER_TABLE_NAME} '
                  '(cid INTEGER PRIMARY KEY, conformer BLOB)')
    self._conn.execute(make_table)
    self._conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                       f'idx_cid ON {_CONFORMER_TABLE_NAME} (cid)')
    self._conn.execute(f'CREATE TABLE IF NOT EXISTS {_BTID_TABLE_NAME} '
                       '(btid INTEGER, cid INTEGER)')
    self._conn.execute(f'CREATE INDEX IF NOT EXISTS '
                       f'idx_btid ON {_BTID_TABLE_NAME} (btid)')
    self._conn.execute(f'CREATE TABLE IF NOT EXISTS {_SMILES_TABLE_NAME} '
                       '(smiles TEXT, btid INTEGER)')
    self._conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS '
                       f'idx_smiles ON {_SMILES_TABLE_NAME} (smiles)')
    self._conn.commit()

  def bulk_insert(self, conformers, batch_size=10000):
    """Inserts conformers into the database.

    Args:
      conformers: iterable for dataset_pb2.Conformer
      batch_size: insert performance is greatly improved by putting multiple
        insert into one transaction. 10k was a reasonable default from some
        early exploration.

    Raises:
      ReadOnlyError: if mode is 'r'
    """
    if self._read_only:
      raise ReadOnlyError()

    insert_conformer = f'INSERT INTO {_CONFORMER_TABLE_NAME} VALUES (?, ?)'
    insert_btid = f'INSERT INTO {_BTID_TABLE_NAME} VALUES (?, ?)'
    insert_smiles = (f'INSERT INTO {_SMILES_TABLE_NAME} VALUES (?, ?) '
                     f'ON CONFLICT(smiles) DO NOTHING')

    cur = self._conn.cursor()

    for idx, conformer in enumerate(conformers, 1):
      cur.execute(insert_conformer,
                  (conformer.conformer_id, conformer.SerializeToString()))
      for bond_topology in conformer.bond_topologies:
        cur.execute(insert_btid, (bond_topology.bond_topology_id,
                                  conformer.conformer_id))
        cur.execute(insert_smiles,
                    (bond_topology.smiles,
                     bond_topology.bond_topology_id))
      if batch_size and idx % batch_size == 0:
        logging.info('bulk_insert: committing at index %d', idx)
        self._conn.commit()
    self._conn.commit()

  def find_by_conformer_id(self, cid):
    """Finds the conformer associated with a conformer id.

    Args:
      cid: conformer id to look up.

    Returns:
      dataset_pb2.Conformer

    Raises:
      KeyError: if cid is not found
    """
    cur = self._conn.cursor()
    select = f'SELECT conformer FROM {_CONFORMER_TABLE_NAME} WHERE cid = ?'
    cur.execute(select, (cid,))
    result = cur.fetchall()

    if not result:
      raise KeyError(f'Conformer id {cid} not found')

    # Since it's a unique index, there should only be one result and it's a
    # tuple with one value.
    assert len(result) == 1
    assert len(result[0]) == 1
    return dataset_pb2.Conformer().FromString(result[0][0])

  def find_by_bond_topology_id(self, btid):
    """Finds all the conformer associated with a bond topology id.

    Args:
      btid: bond topology id to look up.

    Returns:
      iterable of dataset_pb2.Conformer
    """
    cur = self._conn.cursor()
    select = (f'SELECT cid, conformer '
              f'FROM {_CONFORMER_TABLE_NAME} '
              f'INNER JOIN {_BTID_TABLE_NAME} USING(cid) '
              f'WHERE {_BTID_TABLE_NAME}.btid = ?')
    cur.execute(select, (btid,))
    return (dataset_pb2.Conformer().FromString(result[1]) for result in cur)

  def find_by_smiles(self, smiles):
    """Finds all conformer associated with a given smiles string.

    Args:
      smiles: string

    Returns:
      iterable for dataset_pb2.Conformer
    """
    # TODO(pfr): add canonicalization here
    cur = self._conn.cursor()
    select = f'SELECT btid FROM {_SMILES_TABLE_NAME} WHERE smiles = ?'
    cur.execute(select, (smiles,))
    result = cur.fetchall()

    if not result:
      return []

    # Since it's a unique index, there should only be one result and it's a
    # tuple with one value.
    assert len(result) == 1
    assert len(result[0]) == 1
    return self.find_by_bond_topology_id(result[0][0])

  def __iter__(self):
    """Iterates through all dataset_pb2.Conformer in the DB."""
    select = f'SELECT conformer FROM {_CONFORMER_TABLE_NAME} ORDER BY rowid'
    cur = self._conn.cursor()
    cur.execute(select)
    return (dataset_pb2.Conformer().FromString(result[0]) for result in cur)
