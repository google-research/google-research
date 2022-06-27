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

r"""Downloads and saves Pfam-A seed 34.0 to disk.

This script fetches Pfam-A seed, release 34.0 from harcoded URLs and parses the
data on-the-fly. As part of the preprocessing, sequences are clustered via
hiearchical single-linkage agglomerative clustering at multiple PID thresholds.
It saves the following info to disk, rooted at `FLAGS.outdir`:

+ `all` is a subdirectory containing all sequence-related info in TFRecord
  format. Each TFRecord entry corresponds to a sequence and has the following
  fields:
  + seq_key, fam_key, cla_key: unique int-valued key representing the sequence
    itself, and the family and clan it belongs to.
  + id, ac: strings representing the UniProtID and Pfam accession of the
    sequence.
  + start, end: ints representing the start and end positions of the domain in
    the whole protein.
  + ci_{pid_th}: an int representing the key of the cluster the sequence belongs
    to, for a finite set of PID threshold values.
  + seq_len: int, the length of the sequence (precomputed to speed-up filtering
    by length).
  + seq: a sequence of ints representing the sequence, encoded by a certain
    vocabulary.
  + ss: if present, a string representing the protein's secondary structure.

+ `family_metadata.csv` is a CSV file containing one row per Pfam family and the
  following fields:
  + fam_key: unique int-valued key representing the family.
  + id, ac: unique strings representing the family ID and accession.
  + cl, cl_key: unique string and int-value key representing the clan the family
    belongs to.
  + tp: string representing the type of Pfam entry.
  + match_cols: string indicating which columns in the MSA correspond to match
    states ('x') and which correspond to insert states ('-').
  + seq_cons: string containing the consensus sequence for the family.
  + ss_cons: string containing the consensus secondary structure for the family.
  + ref: string with reference annotation info.

+ `metadata.json` contains the following additional metadata fields:
  + 'vocab': serialized vocabulary used to encode the protein sequences.
  + 'pid_ths': the PID thresholds for which cluster labels are available.
  + 'family_metadata': absolute path to file `family_metadata.csv`.

The output of this script can be further preprocessed by `create_splits.py` to
generate iid/ood train/validation/test scripts.
"""

import gzip
import json
import os
import re
from typing import NamedTuple, Sequence, Tuple
import urllib.request

from absl import app
from absl import flags
from absl import logging

from Bio import AlignIO
from Bio import SeqIO

import numpy as np
from sklearn import cluster
import tensorflow as tf

from dedal import vocabulary


flags.DEFINE_string(
    'outdir', None,
    'Directory in which to save all preprocessed output files.')
flags.DEFINE_multi_integer(
    'pid_ths', [100],
    'Sequences with PIDs above these values will be clustered.')
flags.DEFINE_bool(
    'encode_seqs', True,
    'Whether to store protein sequences already encoded.')
flags.DEFINE_bool(
    'encode_ss', False,
    'Whether to store secondary structure sequences already encoded.')
flags.DEFINE_integer(
    'num_shards', 30,
    'Number of shards for TFRecords.')
flags.mark_flag_as_required('outdir')

FLAGS = flags.FLAGS


# URLs for database files (valid as of 04/04/2021)
URL_PFAM_SEED = 'http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam34.0/Pfam-A.seed.gz'
URL_PFAM_CLANS = 'http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam34.0/Pfam-A.clans.tsv.gz'


# Constant for char used to represent gaps
GAP = '-'
# Regex to extract sequence ID, start and end pos.
SEQID_PATTERN = re.compile('(\w*)/(\d*)-(\d*)')  # pylint: disable=anomalous-backslash-in-string
# Vocabulary used to encode protein sequences as lists integer-valued tokens.
VOCAB = vocabulary.alternative


### Auxiliary data structures and constants


class FamEntry(NamedTuple):
  """Auxiliary struct representing all info pertaining a family."""
  fam_key: str
  id: str
  ac: str
  cl: str
  cl_key: str
  tp: str
  match_cols: str
  seq_cons: str
  ss_cons: str
  ref: str

  @classmethod
  def to_csv_header(cls):
    """Returns CSV header corresponding to FamEntry fields."""
    return ','.join([s.upper() for s in cls._fields]) + '\n'

  def to_csv_line(self):
    """Returns CSV line (row) corresponding to a FamEntry object."""
    return ','.join(self) + '\n'


class SeqEntry(NamedTuple):
  """Auxiliary struct representing extra ID info pertaining a sequence."""
  seq_key: int
  id: str
  ac: str
  start: int
  end: int
  cla_key: int
  fam_key: int
  ci: Sequence[int]
  seq_len: int
  seq: str
  ss: str

  @staticmethod
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  @staticmethod
  def _str_feature(value, encoding='ascii'):
    value = bytes(value, encoding=encoding)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def value(self):
    """Serializes the entry for TFRecord storage."""
    feature = {}
    feature['seq_key'] = self._int64_feature([self.seq_key])
    feature['id'] = self._str_feature(self.id)
    feature['ac'] = self._str_feature(self.ac)
    feature['start'] = self._int64_feature([self.start])
    feature['end'] = self._int64_feature([self.end])
    feature['cla_key'] = self._int64_feature([self.cla_key])
    feature['fam_key'] = self._int64_feature([self.fam_key])
    for i, pid_th in enumerate(FLAGS.pid_ths):
      feature[f'ci_{pid_th}'] = self._int64_feature([int(self.ci[i])])
    feature['seq_len'] = self._int64_feature([self.seq_len])
    if FLAGS.encode_seqs:
      feature['seq'] = self._int64_feature(VOCAB.encode(self.seq.upper()))
    else:
      feature['seq'] = self._str_feature(self.seq.upper())
    if FLAGS.encode_ss:
      raise NotImplementedError('Not yet implemented.')
    else:
      feature['ss'] = self._str_feature(self.ss.upper())
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


### Auxiliary functions


def process_record(
    seq_key, fam_key, cla_key, rec):
  """Populates SeqEntry from sequence line in Stockholm format MSA."""
  m = SEQID_PATTERN.match(rec.id)
  seqid, start, end = m.group(1), int(m.group(2)), int(m.group(3))
  ac = rec.annotations['accession']
  seq = str(rec.seq)
  seq_len = len(seq) - seq.count(GAP)
  ss = rec.letter_annotations.get('secondary_structure', '')
  return SeqEntry(seq_key=seq_key, id=seqid, ac=ac, start=start, end=end,
                  cla_key=cla_key, fam_key=fam_key, ci=[], seq_len=seq_len,
                  seq=seq, ss=ss)


def compute_pairwise_distances(
    seq_table):
  """Computes 1 - PID for each pair of sequences in x."""
  # Builds a NumPy array representing the MSA.
  iupac_to_int = lambda entry: [ord(c) - ord(GAP) for c in entry.seq]
  x = np.array([iupac_to_int(entry) for entry in seq_table], dtype=np.uint8)
  # Computes sequence lengths, ignoring gaps.
  seq_lens = np.sum(x != 0, axis=1)
  assert np.all(
      seq_lens == np.fromiter((e.seq_len for e in seq_table), np.int64))
  # One-hot encodes sequences, representing gaps as an all-zeroes vector.
  x_oh = tf.one_hot(x, ord('Z') - ord(GAP) + 1)[:, :, 1:].numpy()
  # Computes number of matching characters for each pair of sequences.
  x_oh = x_oh.reshape((x_oh.shape[0], -1))
  n_matches = np.dot(x_oh, x_oh.T)
  # Computes normalized distance.
  dist_matrix = 1 - n_matches / np.minimum(seq_lens[:, None], seq_lens[None, :])
  return dist_matrix


def compute_cluster_labels(seq_table):
  """Applies single-link agglomerative clustering with different thresholds."""
  n_seq, n_th = len(seq_table), len(FLAGS.pid_ths)

  # Deals with singleton families.
  if n_seq == 1:
    return [seq_table[0]._replace(ci=np.asarray([0 for _ in FLAGS.pid_ths]))]

  # Computes pairwise PID.
  dist_matrix = compute_pairwise_distances(seq_table)
  # Computes cluster indicators for several PID thresholds. Along the way, sort
  # sequences in the family by cluster, ala radix sort (the lower the PID
  # threshold, the more significant the cluster "digit").
  cluster_indicators = []
  ind = np.arange(n_seq)
  for pid_th in sorted(FLAGS.pid_ths, reverse=True):
    distance_threshold = 1.0 - pid_th / 100
    agglomerative_clustering = cluster.AgglomerativeClustering(
        n_clusters=None, affinity='precomputed', linkage='single',
        distance_threshold=distance_threshold)
    cluster_indicators_i = agglomerative_clustering.fit_predict(dist_matrix)
    cluster_indicators.append(cluster_indicators_i)
    ind = ind[np.argsort(cluster_indicators_i[ind], kind='stable')]
  # Stack cluster_indicators as [n_seq, n_th] array, in ascending pid_th order.
  cluster_indicators = np.stack(cluster_indicators[::-1], axis=1)

  # Relabels clusters in increasing order of sequence index.
  def remove_duplicates_stable(x):
    """Removes duplicates in array, preserving order of entries."""
    x_unq, idx = np.unique(x, return_index=True)
    return x_unq[np.argsort(idx)]

  # Pre-applies reordering to clustering indicators.
  cluster_indicators = cluster_indicators[ind]
  for i in range(n_th):
    ci_map_i = np.argsort(remove_duplicates_stable(cluster_indicators[:, i]))
    cluster_indicators[:, i] = ci_map_i[cluster_indicators[:, i]]

  # Creates a new seq_table populated with the cluster indicators.
  key_offset = int(seq_table[0].seq_key)  # seq_table given sorted by seq_key.
  updated_seq_table = []
  for i, idx in enumerate(ind):
    cluster_indicators_i, entry = cluster_indicators[i], seq_table[idx]
    seq_key_i = key_offset + i
    updated_seq_table.append(
        entry._replace(seq_key=seq_key_i, ci=cluster_indicators_i))
  return updated_seq_table


def compute_match_cols(seq_table):
  msa = np.array([list(entry.seq) for entry in seq_table])
  match_cols = np.any(np.char.isupper(msa), axis=0)
  return ''.join([('x' if v else '-') for v in match_cols])


def seqs_to_disk(
    outdir, seq_tables):
  """Writes all info pertaining to sequences to disk as sharded TFRecords."""
  def gen():
    for seq_table in seq_tables:
      for entry in seq_table:
        yield entry.value()

  def reduce_func(key, ds):
    filename = tf.strings.join(
        [outdir, '/', tf.strings.as_string(key), '.tfrecords'])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(ds.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  ds = tf.data.Dataset.from_generator(
      gen, output_signature=tf.TensorSpec(shape=(), dtype=tf.string))
  ds = ds.enumerate()
  ds = ds.group_by_window(
      lambda i, _: i % FLAGS.num_shards, reduce_func, tf.int64.max)
  _ = list(ds)


def main(_):
  ### Downloads and parses the data.
  logging.info('Downloading and parsing clan info from %s...', URL_PFAM_CLANS)
  ac_to_cl = {}
  cl_to_cl_idx = {}
  with gzip.open(urllib.request.urlopen(URL_PFAM_CLANS)) as f:
    for line in f:
      line = line.decode().strip().split('\t')
      ac, cl = line[0], line[1]
      cl = cl if cl else ac  # Sets empty CL IDs to equal the Pfam accession ID.
      if ac not in ac_to_cl:
        ac_to_cl[ac] = cl
      else:
        raise ValueError(f'Duplicated entry for AC f{ac}.')
      if cl not in cl_to_cl_idx:
        cl_to_cl_idx[cl] = str(len(cl_to_cl_idx))
  logging.info('Got %s Pfam clans.', len(cl_to_cl_idx))

  logging.info('Downloading and parsing family info from %s...', URL_PFAM_SEED)
  fam_offset = 0  # Number of families seen so far.
  fam_table = []
  with gzip.open(
      urllib.request.urlopen(URL_PFAM_SEED), 'rt', encoding='ISO-8859-1') as f:
    for line in f:
      if line.startswith('#=GF ID'):  # Pfam entry ID.
        # Saves previous, now complete entry to fam_table.
        if fam_offset != 0:
          fam_table.append(FamEntry(**new_entry))
        new_entry = {
            'fam_key': str(fam_offset),
            'id': line.strip().split()[-1],
            'match_cols': '',  # To be filled-in later.
            'ss_cons': '',  # Optional.
            'ref': '',  # Optional.
        }
        fam_offset += 1
      elif line.startswith('#=GF AC'):  # Pfam accession.
        new_entry['ac'] = line.strip().split()[-1]
        # Clan of Pfam-A entry.
        new_entry['cl'] = ac_to_cl[new_entry['ac'].split('.')[0]]
        new_entry['cl_key'] = cl_to_cl_idx[new_entry['cl']]
      elif line.startswith('#=GF TP'):  # Type of entry.
        new_entry['tp'] = line.strip().split()[-1]
      elif line.startswith('#=GC seq_cons'):  # Consensus sequence.
        new_entry['seq_cons'] = line.strip().split()[-1]
      elif line.startswith('#=GC SS_cons'):  # Consensus secondary structure.
        new_entry['ss_cons'] = line.strip().split()[-1]
      elif line.startswith('#=GC RF'):  # Reference annotation.
        new_entry['ref'] = line.strip().split()[-1]
    # Saves last complete entry to fam_table.
    fam_table.append(FamEntry(**new_entry))
  logging.info('Parsed family info for %s Pfam families.', len(fam_table))

  logging.info('Downloading and parsing seq info from %s...', URL_PFAM_SEED)
  with gzip.open(
      urllib.request.urlopen(URL_PFAM_SEED), 'rt', encoding='ISO-8859-1') as f:
    msas = list(AlignIO.parse(f, 'stockholm'))
  logging.info('Retrieved %d MSAs in STOCKHOLM file.', len(msas))

  logging.info('Populating sequence-level fields... ')
  seq_offset = 0  # Number of sequences seen so far.
  seq_tables = []  # One per family.
  for fam_idx, msa in enumerate(msas):
    cla_key = int(fam_table[fam_idx].cl_key)
    seq_table = []
    for i, rec in enumerate(msa):
      seq_table.append(process_record(seq_offset + i, fam_idx, cla_key, rec))
    seq_table = compute_cluster_labels(seq_table)
    seq_offset += len(seq_table)
    seq_tables.append(seq_table)
  logging.info('Got %s sequences distributed along %s Pfam families.',
               sum(len(e) for e in seq_tables), len(seq_tables))

  logging.info('Computing insert col mask for each MSA...')
  updated_fam_table = []
  for fam_entry, seq_table in zip(fam_table, seq_tables):
    updated_fam_table.append(
        fam_entry._replace(match_cols=compute_match_cols(seq_table)))
  fam_table = updated_fam_table

  logging.info('Creating FASTA records from %s...', URL_PFAM_SEED)
  seq_records = []
  with gzip.open(
      urllib.request.urlopen(URL_PFAM_SEED), 'rt', encoding='ISO-8859-1') as f:
    for msa in AlignIO.parse(f, 'stockholm'):
      for rec in msa:
        rec.letter_annotations = {}
        rec.seq = rec.seq.ungap('-').upper()
        seq_records.append(rec)
  logging.info('Created FASTA records for %s sequences.', len(seq_records))

  ### Writes the data to CSV.
  pfam_seed_fams_path = os.path.join(FLAGS.outdir, 'family_metadata.csv')
  pfam_seed_seqs_path = os.path.join(FLAGS.outdir, 'all')
  pfam_seed_fa_path = os.path.join(FLAGS.outdir, 'pfam_seed.fa')
  metadata_path = os.path.join(FLAGS.outdir, 'metadata.json')
  tf.io.gfile.makedirs(FLAGS.outdir)
  tf.io.gfile.makedirs(pfam_seed_seqs_path)

  logging.info('Writing parsed family info to %s...', pfam_seed_fams_path)
  with tf.io.gfile.GFile(pfam_seed_fams_path, 'w') as f:
    f.write(FamEntry.to_csv_header())
    for entry in fam_table:
      f.write(entry.to_csv_line())

  logging.info('Writing parsed seq info to %s...', pfam_seed_seqs_path)
  seqs_to_disk(pfam_seed_seqs_path, seq_tables)

  logging.info('Writing FASTA records to %s...', pfam_seed_fa_path)
  with tf.io.gfile.GFile(os.path.join(pfam_seed_fa_path), 'w') as f:
    SeqIO.write(seq_records, f, 'fasta')

  logging.info('Writing metadata to %s...', metadata_path)
  metadata = {
      'family_metadata': pfam_seed_fams_path,
      'pid_ths': FLAGS.pid_ths,
      'vocab': VOCAB.get_config(),
  }
  with tf.io.gfile.GFile(metadata_path, 'w') as f:
    json.dump(metadata, f)

if __name__ == '__main__':
  app.run(main)
