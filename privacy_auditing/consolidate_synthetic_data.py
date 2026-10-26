# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Consolidates synthetic data from multiple experiment XIDs into one file.

Searches three locations for each XID (in priority order):
  1. Direct output:          {base}/{xid}/synthetic_data.bagz
  2. Per-work-unit output:   {base}/{xid}/{wid}/synthetic_data.bagz
  3. Per-work-unit shards:   {base}/{xid}/{wid}/shard_*.bagz

Work unit subdirectories are discovered automatically; the count need not be
specified. Output is {base}/{first_xid}/consolidated_synthetic_data.bagz.
"""

from collections.abc import Sequence
import itertools
import re

from absl import app
from absl import flags
from absl import logging
import bagz
from privacy_auditing import filesystem

_BASE_DIR = flags.DEFINE_string(
    'base_dir',
    None,
    'Base directory under which per-XID subdirectories are stored.',
    required=True,
)
_XIDS = flags.DEFINE_list(
    'xids',
    None,
    'Comma-separated experiment IDs to consolidate. Output is written under'
    ' the first XID.',
    required=True,
)
_TOTAL_RECORDS = flags.DEFINE_integer(
    'total_records',
    None,
    'If set, stop after this many records have been written. If not set, all'
    ' records are written.',
)


def _shard_key(path):
  m = re.search(r'shard_(\d+)', path)
  return int(m.group(1)) if m else 0


def _numeric_subdirs(directory):
  """Returns numerically-sorted names of integer-named subdirectories."""
  try:
    children = filesystem.listdir(directory)
  except OSError:
    return []
  return sorted((c for c in children if re.fullmatch(r'\d+', c)), key=int)


def _sources_for_xid(xid):
  """Returns ordered bagz paths for the given XID, or [] if none found."""
  xid_dir = f'{_BASE_DIR.value}/{xid}'

  # 1. Consolidated synthetic_data.bagz under the XID directory.
  if paths := filesystem.glob(f'{xid_dir}/synthetic_data.bagz'):
    logging.info('XID %s: %d direct consolidated shards.', xid, len(paths))
    return paths

  paths = []
  logging.info('XID %s: no direct consolidated shards.', xid)

  wid_dirs = _numeric_subdirs(xid_dir)

  # 2. Check for consolidated synthetic_data.bagz inside each work-unit
  #    directory.
  # 3. Check for per-work-unit shards if no consolidated output is found.
  wid_paths = []
  for w in wid_dirs:
    wid_path = f'{xid_dir}/{w}/synthetic_data.bagz'
    if filesystem.exists(wid_path):
      wid_paths.append(wid_path)
    else:
      wid_paths.extend(filesystem.glob(f'{xid_dir}/{w}/shard_*.bagz'))
  if wid_paths:
    logging.info(
        'XID %s: synthetic_data.bagz in %d work unit(s).', xid, len(wid_paths)
    )
    paths.extend(wid_paths)

  if len(paths) == 0:
    logging.warning('XID %s: no synthetic data found.', xid)

  return paths


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  xids = _XIDS.value
  output_path = f'{_BASE_DIR.value}/{xids[0]}/consolidated_synthetic_data.bagz'
  logging.info('Consolidating %d XID(s) → %s', len(xids), output_path)

  records = set()
  source_lists = filter(None, (_sources_for_xid(xid) for xid in xids))
  unique_sources = set(itertools.chain.from_iterable(source_lists))

  def stream_records(sources):
    for path in sorted(sources, key=_shard_key):
      logging.info('  %s', path)
      with bagz.BagReader(path) as reader:
        yield from reader

  limit = _TOTAL_RECORDS.value

  for record in stream_records(unique_sources):
    if limit is not None and len(records) >= limit:
      break
    records.add(record)

  if _TOTAL_RECORDS.value is not None and len(records) < _TOTAL_RECORDS.value:
    raise ValueError(
        f'Requested {_TOTAL_RECORDS.value} records, but only found'
        f' {len(records)}.'
    )

  with bagz.BagWriter(output_path) as writer:
    for record in records:
      writer.write(record)

  logging.info('Wrote %d records to %s.', len(records), output_path)


if __name__ == '__main__':
  app.run(main)
