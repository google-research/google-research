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

"""Beam pipeline for converting basel files to final output.

We get horrible fortran formatted text files from Basel. This pipeline
converts those into proto files, does all kinds of reprocessing and error
checking to produce the final outputs.
"""

import copy
import csv
import functools
import itertools
import logging as stdlogging
import numpy as np

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from tensorflow.io import gfile

from google.protobuf import json_format
from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom
from smu.geometry import smu_molecule
from smu.parser import smu_parser_lib
from smu.parser import smu_utils_lib
from smu.parser import smu_writer_lib

flags.DEFINE_string('input_stage1_dat_glob', None,
                    'Glob of stage1 dat files to read')
flags.DEFINE_string('input_stage2_dat_glob', None,
                    'Glob of stage2 dat files to read')
flags.DEFINE_string('input_bond_topology_csv', None,
                    'CSV file of bond topologies (see merge_bond_topologies)')
flags.DEFINE_string(
    'input_equivalent_glob', None,
    'Glob of files containing list of equivalent structure (usually '
    'list.equivalent_isomers.dat and list.equivalent_confomers.dat)')
flags.DEFINE_string('output_stem', None, 'Filestem for output files')
flags.DEFINE_integer('output_shards', 10,
                     'Number of output shards for our primary outputs')

FLAGS = flags.FLAGS

_METRICS_NAMESPACE = 'SMU'
_BOND_LENGTHS_SIG_DIGITS = 3
_BOND_LENGTHS_UNBONDED_MAX = 2.0
_BOND_LENGTHS_UNBONDED_RIGHT_TAIL_MASS = 0.9


def parse_equivalent_file(filename):
  """Parses the .dat of equivalent structure.

  The file is just pairs of entries where the first was kept over the second.
  Yields one entry per line keyed by the discarded conformer id.
  See merge_duplicate_information for how information is transferred to the kept
  conformer.

  Args:
    filename: string

  Yields:
    dataset_pb2.Conformer
  """
  with gfile.GFile(filename) as f:
    for line in f:
      kept_str, discard_str = line.split()
      _, _, kept_btid, kept_cid = smu_parser_lib.parse_long_identifier(kept_str)
      _, _, discard_btid, discard_cid = smu_parser_lib.parse_long_identifier(
          discard_str)
      # Convert to our conformer ids which include the btid
      kept_cid = kept_btid * 1000 + kept_cid
      discard_cid = discard_btid * 1000 + discard_cid

      yield dataset_pb2.Conformer(
          conformer_id=discard_cid, duplicated_by=kept_cid)


def parse_dat_file(filename, stage):
  """Beam pipeline component for reading dat files.

  Args:
    filename: filename to read
    stage: string 'stage1' or 'stage2'

  Yields:
    Pair of string (original dat), conformer
    conformer can be an Exception or a dataset_pb2.Conformer
  """
  smu_parser = smu_parser_lib.SmuParser(filename)
  if stage == 'stage1':
    process_fn = smu_parser.process_stage1
  else:
    process_fn = smu_parser.process_stage2
  for conformer, orig_dat_list in process_fn():
    orig_dat = '\n'.join(orig_dat_list) + '\n'

    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_dat_entry_read').inc()

    yield orig_dat, conformer


def partition_parse_success(input_tuple, num_partitions, stage):
  """Function to beam.Partition parsed inputs based on parse success.

  Args:
    input_tuple: pair of orig_contents, conformer (see parse_dat_file)
    num_partitions: (should always be 3)
    stage: string 'stage1' or 'stage2'

  Returns:
    int (0 for success, 1, for known error, 2 for unknown error)
  """
  assert num_partitions == 3
  _, conformer = input_tuple
  if not isinstance(conformer, Exception):
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_parse_success').inc()
    return 0  # Parse success
  else:
    if isinstance(conformer, smu_parser_lib.SmuKnownError):
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   stage + '_parse_known_error').inc()
      return 1  # Parse known error
    else:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   stage + '_parse_unknown_error').inc()
      return 2  # Parse unknown error


def regenerate_dat(input_tuple, stage):
  """Regenerates the original dat from conformer and compares it to original.

  Args:
    input_tuple: tuple of string (original contents), dataset_pb2.Conformer
    stage: string 'stage1' or 'stage2'

  Returns:
    original_dat, conformer, regenerated dat, int (0=mismatch, 1=match)
  """
  original_dat, conformer = input_tuple
  smu_writer = smu_writer_lib.SmuWriter(annotate=False)
  if stage == 'stage1':
    regen_dat = smu_writer.process_stage1_proto(conformer)
  else:
    regen_dat = smu_writer.process_stage2_proto(conformer)
  try:
    smu_writer_lib.check_dat_formats_match(original_dat.splitlines(),
                                           regen_dat.splitlines())
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_dat_format_matched').inc()
    return original_dat, conformer, regen_dat, 1
  except smu_writer_lib.DatFormatMismatchError:
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_dat_format_mismatched').inc()
    return original_dat, conformer, regen_dat, 0


def conformer_to_stat_values(conformer):
  """Beam transform to produce stats values for later aggregation.

  Each output will be a tuple of primary_key, secondary_key and these will be
  aggregated as counts.

  Args:
    conformer: dataset_pb2.Conformer

  Yields:
    primary_key, secondary_key
  """
  # Yield the values for all the relevant error fields.
  for field in ['status',
                'warn_t1',
                'warn_t1_excess',
                'warn_bse_b5_b6',
                'warn_bse_cccsd_b5',
                'warn_exc_lowest_excitation',
                'warn_exc_smallest_oscillator',
                'warn_exc_largest_oscillator',
                'warn_vib_linearity',
                'warn_vib_imaginary',
                'warn_num_neg',
                'error_nstat1',
                'error_nstatc',
                'error_nstatt',
                'error_frequencies']:
    yield 'errors.' + field, getattr(conformer.properties.errors, field)

  yield 'fate', dataset_pb2.Conformer.FateCategory.Name(conformer.fate)

  yield 'num_initial_geometries', len(conformer.initial_geometries)
  yield 'num_duplicates', len(conformer.duplicate_of)
  if not conformer.duplicated_by:
    yield 'num_topologies', len(conformer.bond_topologies)

  for field in smu_utils_lib.find_zero_values(conformer):
    yield 'zero_field', field

def bond_topology_summaries_from_csv(filename):
  """Beam DoFn for generating bare BondTopologySummary.

  Args:
    filename: csv file of bond topologies to read

  Yields:
    dataset_pb2.Entry
  """
  for bt in smu_utils_lib.generate_bond_topologies_from_csv(filename):
    summary = dataset_pb2.BondTopologySummary()
    summary.bond_topology.CopyFrom(bt)
    # Note that we leave all the counts as 0.
    yield bt.bond_topology_id, summary


class MergeConformersFn(beam.DoFn):
  """Merges conformers with the same id.

  Because of the stage1, stage2, and duplicate information, we can end up with
  multiple conformers with the same id. This merges them.
  """
  OUTPUT_TAG_MERGE_CONFLICT = 'conflict'

  def process(self, args):
    """"Merges conformers.

    Args:
      args: tuple of conformer_id(should match the id in all conformers) and
        conformers(iterable of dataset_pb2.Conformer)

    Yields:
      dataset_pb2.Conformer and tagged output (OUTPUT_TAG_MERGE_CONFLICT) with
      conflict output from smu_utils_lib.merge_conformer
    """
    conformer_id, conformers = args

    for c in conformers:
      if c.conformer_id != conformer_id:
        raise ValueError(
            f'In merged CID {conformer_id}, found CID {c.conformer_id} instead')

    # For signalling the first merging.
    sentinel = object()

    conflicts = []

    def _merge_two_conformers(conf0, conf1):
      if conf0 is sentinel:
        return conf1

      merged_conf, merge_conflict = smu_utils_lib.merge_conformer(conf0, conf1)
      if merge_conflict:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'conformer_merge_error').inc()
        conflicts.append(merge_conflict)
      return merged_conf

    beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'merged_conformers').inc()

    # Note that we convert the iterable to a list and do a deepcopy. We can't
    # modify the input and smu_utils_lib.merge_conformer wants to reserve the
    # right to modify either input so it's simplest to just copy it once right
    # off the bat.
    yield functools.reduce(_merge_two_conformers,
                           copy.deepcopy(list(conformers)), sentinel)

    for c in conflicts:
      yield beam.pvalue.TaggedOutput(
          MergeConformersFn.OUTPUT_TAG_MERGE_CONFLICT, c)


def extract_bond_lengths(conformer, dist_sig_digits, unbonded_max):
  """Yields quantized bond lengths.

  Args:
    conformer: dataset_pb2.Conformer
    dist_sig_digits: number of digits after decimal point to keep
    unbonded_max: maximum distance to report for unbonded pairs  output atom
      types are single charecters, sorted lexographically. bond_type is
      dataset_pb2.BondTopology.BondType dist_sig_digits is a string (to avoid
      vagaries of floating point compares)

  Yields:
    (atom type 1, atom type 2, bond type, quantized dist)
  """
  # These are considered "major" or worse errors
  if (conformer.properties.errors.status >= 8 or
      conformer.duplicated_by > 0):
    return

  bt = conformer.bond_topologies[0]
  format_str = '{:.%df}' % dist_sig_digits

  for atom_idx0, atom_idx1 in itertools.combinations(range(len(bt.atoms)), r=2):

    if (bt.atoms[atom_idx0] == dataset_pb2.BondTopology.ATOM_H or
        bt.atoms[atom_idx1] == dataset_pb2.BondTopology.ATOM_H):
      continue

    bond_type = dataset_pb2.BondTopology.BOND_UNDEFINED
    for bond in bt.bonds:
      if ((bond.atom_a == atom_idx0 and bond.atom_b == atom_idx1) or
          (bond.atom_a == atom_idx1 and bond.atom_b == atom_idx0)):
        bond_type = bond.bond_type
        break

    geom = conformer.optimized_geometry
    atom_pos0 = np.array([
        geom.atom_positions[atom_idx0].x, geom.atom_positions[atom_idx0].y,
        geom.atom_positions[atom_idx0].z
    ],
                         dtype=np.double)
    atom_pos1 = np.array([
        geom.atom_positions[atom_idx1].x, geom.atom_positions[atom_idx1].y,
        geom.atom_positions[atom_idx1].z
    ],
                         dtype=np.double)
    # The intention is the buckets are the left edge of an empricial CDF.
    dist = (
        np.floor(
            smu_utils_lib.bohr_to_angstroms(
                np.linalg.norm(atom_pos0 - atom_pos1)) * 10**dist_sig_digits) /
        10**dist_sig_digits)
    if (bond_type == dataset_pb2.BondTopology.BOND_UNDEFINED and
        dist > unbonded_max):
      continue

    atom_char0 = smu_utils_lib.ATOM_TYPE_TO_CHAR[bt.atoms[atom_idx0]]
    atom_char1 = smu_utils_lib.ATOM_TYPE_TO_CHAR[bt.atoms[atom_idx1]]
    if atom_char0 > atom_char1:
      atom_char0, atom_char1 = atom_char1, atom_char0

    yield atom_char0, atom_char1, bond_type, format_str.format(dist)


def write_bond_lengths(records, filename):
  """DoFn for writing the bond lengths.

  We write directly to filename because the entire pcollection
  should have been combined to a single entry.

  Args:
    records: records as expected by
      bond_length_distribution.sparse_dataframe_from_records
    filename: file to write to
  """
  with gfile.GFile(filename, 'w') as f:
    df = bond_length_distribution.sparse_dataframe_from_records(records)
    df.to_csv(f, index=False)


def smiles_to_id(bond_topology_filename):
  """DoFn for creating the smiles to id mapping.

  Reads the same merged_bond_topology file as bond_topology_summaries_from_csv
  and output. We could of course produce them both at the same time, but this
  is simpler.

  Args:
    bond_topology_filename: see FLAGS.input_bond_topology_csv

  Yields:
    smiles, bond_topology_id
  """
  with gfile.GFile(bond_topology_filename, 'r') as infile:
    reader = csv.reader(iter(infile))
    next(reader)  # skip the header line
    for row in reader:
      bt_id, _, _, _, _, smiles = row
      yield smiles, int(bt_id)


class UpdateConformerFn(beam.DoFn):
  """DoFn that performs several updates to fields in Conformer.

  * Updates the smiles string (with a tagged output to record the mismatches.
  * Adds Fate field
  * Adds additional bond topologies that match the geometry
  * various cleanup steps

  main output is dataset_pb2.Conformer
  smiles output is a tuple of
    conformer_id,
    SmilesCompareResult,
    original smiles,
    smiles_with_h,
    smiles_without_h
  """
  OUTPUT_TAG_SMILES_MISMATCH = 'tag_smiles'

  def setup(self):
    self._cached_bond_lengths = None

  def _compare_smiles(self, conformer):
    if len(conformer.bond_topologies) != 1:
      raise ValueError(
          'compare_smiles expects 1 bond topology; for CID {} got {}'.format(
              conformer.conformer_id, len(conformer.bond_topologies)))

    result, smiles_with_h, smiles_without_h = (
        smu_utils_lib.bond_topology_smiles_comparison(
            conformer.bond_topologies[0]))
    if result != smu_utils_lib.SmilesCompareResult.MATCH:
      yield beam.pvalue.TaggedOutput(
          UpdateConformerFn.OUTPUT_TAG_SMILES_MISMATCH,
          (conformer.conformer_id, result, conformer.bond_topologies[0].smiles,
           smiles_with_h, smiles_without_h))
      conformer.properties.smiles_openbabel = (
        conformer.bond_topologies[0].smiles)
      conformer.bond_topologies[0].smiles = smiles_without_h

  def _add_alternative_bond_topologies(self, conformer, smiles_id_dict):
    # Conformers with this high of a status code did not successfully
    # start stage2 and the geometries coudl be pretty much anything so
    # we don't bother putting these through.
    if conformer.properties.errors.status > 512:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'skipped_topology_matches').inc()
      return


    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 'attempted_topology_matches').inc()

    matching_parameters = smu_molecule.MatchingParameters()
    matching_parameters.must_match_all_bonds = False
    matching_parameters.smiles_with_h = False
    matching_parameters.smiles_with_labels = False

    matches = topology_from_geom.bond_topologies_from_geom(
      self._cached_bond_lengths,
      conformer.bond_topologies[0],
      conformer.optimized_geometry,
      matching_parameters)

    if not matches.bond_topology:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'no_topology_matches').inc()
      return

    del conformer.bond_topologies[:]
    conformer.bond_topologies.extend(matches.bond_topology)
    for bt in conformer.bond_topologies:
      try:
        bt.bond_topology_id = smiles_id_dict[bt.smiles]
      except KeyError:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'topology_match_smiles_failure').inc()

  def process(self, conformer, bond_length_records, smiles_id_dict):
    """Per conformer updates.

    Args:
      conformer: dataset_pb2.Conformer
      bond_length_records: tuples to go to bond_length_distribution.AllAtomPairLengthDistributions
      smiles_id_dict: dict from SMILES to bond topology id
    """
    # There is probably a better way to do this.
    # We get the side input with each call to process. We'll assume that it's
    # always the same input, so we set our cache value and never update it.
    # We only do this with bond_length_records because there is a reasonable
    # amount of processing in creating AllAtomPairLengthDistributions.
    # The smiles_id_dict is used directly.
    if not self._cached_bond_lengths:
      self._cached_bond_lengths = (
        bond_length_distribution.AllAtomPairLengthDistributions())
      self._cached_bond_lengths.add_from_sparse_dataframe(
        bond_length_distribution.sparse_dataframe_from_records(
          bond_length_records), _BOND_LENGTHS_UNBONDED_RIGHT_TAIL_MASS,
          _BOND_LENGTHS_SIG_DIGITS)

    conformer = copy.deepcopy(conformer)

    smu_utils_lib.clean_up_error_codes(conformer)
    smu_utils_lib.clean_up_sentinel_values(conformer)

    conformer.fate = smu_utils_lib.determine_fate(conformer)

    yield from self._compare_smiles(conformer)

    if (conformer.duplicated_by == 0 and
        conformer.properties.errors.status < 512):
      # The duplicate records do not need topology extraction and anything
      # with this high an error is pretty messed so, do we won't bother trying
      # to match the topolgy.
      self._add_alternative_bond_topologies(conformer, smiles_id_dict)

    yield conformer


def generate_keyed_conformers_for_duplicates(conformer):
  """Generates keyed conformers for duplicate merging.

  Every conformer yields itself keyed by its conformer_id
  Additonally, if duplicated_by is set, the conformer is yielded keyed by
  duplicated_by.

  Args:
    conformer: dataset_pb2.Conformer

  Yields:
    conformer_id, dataset_pb2.Conformer
  """
  yield conformer.conformer_id, conformer
  if conformer.duplicated_by > 0:
    yield conformer.duplicated_by, conformer


def merge_duplicate_information(conformer_id, conformers):
  """Merges duplicate information into one conformer.

  One entry in conformers should have the given conformer_id
  (call this the "main" conformer)
  Every other entry should have a duplicated_by set to conformer_id
  (call this an "other" conformer)

  The initial_geometry from other will copied to main.
  If the bond topology id is the same, this is trivial
  TODO(pfr, ianwatson): implement this copying with unequal bond topologies.

  Args:
    conformer_id: integer
    conformers: iterable of dataset_pb2.Conformer

  Returns:
    dataset_pb2.Conformer
  """
  matching_conformers = [
      c for c in conformers if c.conformer_id == conformer_id
  ]
  if len(matching_conformers) != 1:
    raise ValueError('Expected 1 conformers with id {}, got {}'.format(
        conformer_id, len(matching_conformers)))
  main_conformer = copy.deepcopy(matching_conformers[0])

  for conf in conformers:
    if conf.conformer_id == conformer_id:
      continue
    if conf.duplicated_by != conformer_id:
      raise ValueError(
          'Conformer {} should have duplicated_by {} but has {}'.format(
              conf.conformer_id, conformer_id, conf.duplicated_by))
    main_conformer.duplicate_of.append(conf.conformer_id)
    if conformer_id // 1000 == conf.conformer_id // 1000:
      # easy case! Bond topologies are the same, just copy over
      main_conformer.initial_geometries.append(conf.initial_geometries[0])
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'dup_same_topology').inc()
    else:
      # hard case. We have to figure out out to permute the atoms in the initial
      # geometry
      # TODO(pfr, ianwatson)
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'dup_diff_topology_unmatched').inc()
      pass

  return main_conformer


def to_keyed_bond_topology_summary(conformer):
  """Outputs BondTopologySummary for conformer.

  Args:
    conformer: dataset_pb2.Conformer

  Yields:
    bond topology id, BondTopologySummary
  """
  for summary in smu_utils_lib.conformer_to_bond_topology_summaries(conformer):
    yield summary.bond_topology.bond_topology_id, summary


def merge_bond_topology_summaries(summaries, field_names):
  """Merges BondToplogySummary protos.

  See CombineAndWriteBondTopologySummary for context.

  Args:
    summaries: iterable of BondTopologySummary
    field_names: list of field names to be aggregated

  Returns:
    BondTopologySummary
  """
  # For signalling the first merging.
  sentinel = object()

  def _merge_two_summaries(summary0, summary1):
    if summary0 is sentinel:
      # We'll just make one copy and use the sentinel to tell us when to do
      # that
      return copy.deepcopy(summary1)

    assert (summary0.bond_topology.bond_topology_id ==
            summary1.bond_topology.bond_topology_id)

    for name in field_names:
      setattr(summary0, name, getattr(summary0, name) + getattr(summary1, name))

    return summary0

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'merged_summaries').inc()

  return functools.reduce(_merge_two_summaries, summaries, sentinel)


def csv_format_bond_topology_summary(summary, field_names):
  """Formats BondToplogySummary protos as csv line.

  See CombineAndWriteBondTopologySummary for context.

  Args:
    summary: BondTopologySummary
    field_names: list of field names in the order for the csv

  Returns:
    BondTopologySummary
  """
  return ','.join([str(summary.bond_topology.bond_topology_id)] +
                  [str(getattr(summary, name)) for name in field_names])


class CombineAndWriteBondTopologySummary(beam.PTransform):
  """A composite transform for handling BondTopologySummary.

  The only reason we make this a composite transform is that multiple places
  need the list of count fields in BondTopologySummary, so we make it
  one time with a consistent ordering and use it in multiple places.
  """

  def expand(self, pcoll):
    field_names = []
    for field_descriptor in dataset_pb2.BondTopologySummary.DESCRIPTOR.fields:
      if field_descriptor.name.startswith('count_'):
        field_names.append(field_descriptor.name)

    return (pcoll
            | 'CombineByBTID' >> beam.CombinePerKey(
                merge_bond_topology_summaries, field_names=field_names)
            | 'DropBTID' >> beam.Values()
            | 'Reshuffle' >> beam.Reshuffle()
            | 'CSVFormat' >> beam.Map(
                csv_format_bond_topology_summary, field_names=field_names)
            | 'WriteCSV' >> beam.io.WriteToText(
                FLAGS.output_stem + '_bt_summary',
                header='bt_id,' + ','.join(field_names),
                num_shards=1,
                file_name_suffix='.csv'))


def make_complete_conformer(conformer):
  """Turns a Conformer into the complete form from the internal only.

  Args:
    conformer: dataset_pb2.Conformer

  Returns:
    dataset_pb2.Conformer
  """
  out = copy.deepcopy(conformer)
  smu_utils_lib.filter_conformer_by_availability(
      out, [dataset_pb2.STANDARD, dataset_pb2.COMPLETE])

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'complete_conformers').inc()

  return out


def make_standard_conformer(conformer):
  """Turns a Conformer into the standard form from the internal only.

  This must go through a FlatMap because some conformers are filtered.

  Args:
    conformer: dataset_pb2.Conformer

  Yields:
    at most one dataset_pb2.Conformer
  """
  out = copy.deepcopy(conformer)
  if not smu_utils_lib.conformer_to_standard(out):
    return

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'standard_conformers').inc()

  yield out


def key_to_string(key, value):
  return str(key), value


def csv_format(vals):
  return ','.join(str(v) for v in vals)


def conformer_to_json(conformer):
  return json_format.MessageToJson(
      conformer,
      preserving_proto_field_name=True,
      including_default_value_fields=True)


def dat_input_and_parsing_pipeline(root, stage):
  """Create multiple stages for parsing and validation .dat files.

  We read two types of .dat files, stage1 and stage2. The pipeline is similar
  between the two, so we use this function to cover those similar parts.

  Args:
    root: root of the pipeline
    stage: string that is either "stage1" or "stage2"

  Returns:
    PCollection of dataset_pb2.Conformer that are valid and matched
  """
  assert stage in ['stage1', 'stage2']

  label = stage.title()

  # Parse the files and split in collections based on whether the parsing worked
  if stage == 'stage1':
    input_files = gfile.glob(FLAGS.input_stage1_dat_glob)
  else:
    input_files = gfile.glob(FLAGS.input_stage2_dat_glob)
  parsed_inputs = (
      root
      | 'CreateInputs' + label >> beam.Create(input_files)
      | 'ReshuffleInput' + label >> beam.Reshuffle()
      | 'ParseDat' + label >> beam.FlatMap(parse_dat_file, stage))
  parsed_success, parsed_known_error, parsed_unknown_error = (
      parsed_inputs
      | 'PartitionParseError' + label >> beam.Partition(partition_parse_success,
                                                        3, stage))

  # For the parse errors, write out the original contents to files to be
  # examined later.
  _ = (
      parsed_known_error
      | 'ParsedKnownErrorReshuffle' + label >> beam.Reshuffle()
      | 'ExtractOriginalKnown' + label >>
      beam.MapTuple(lambda orig_dat, _: orig_dat)
      | 'WriteOriginalKnown' + label >> beam.io.WriteToText(
          FLAGS.output_stem + '_' + stage + '_original_known_error',
          num_shards=1,
          file_name_suffix='.dat'))
  _ = (
      parsed_unknown_error
      | 'ParsedUnknownErrorReshuffle' + label >> beam.Reshuffle()
      | 'ExtractOriginalUnknown' + label >>
      beam.MapTuple(lambda orig_dat, _: orig_dat)
      | 'WriteOriginalUnknown' + label >> beam.io.WriteToText(
          FLAGS.output_stem + '_' + stage + '_original_unknown_error',
          num_shards=1,
          file_name_suffix='.dat'))

  mismatched, matched = (
      parsed_success
      | 'RegenerateDat' + label >> beam.Map(regenerate_dat, stage)
      | 'PartitionByMatch' + label >> beam.Partition(lambda x, _: x[3], 2))

  # Write out the mismatched conformers, original and regenerated
  # Reshuffle before the forced write of a single shard
  reshuffled_mismatched = (
      mismatched
      | 'MismatchedReshuffle' + label >> beam.Reshuffle())
  _ = (
      reshuffled_mismatched
      | 'ExtractMismatchedOriginal' + label >> beam.Map(lambda x: x[0])
      | 'WriteMismatchedOriginal' + label >> beam.io.WriteToText(
          FLAGS.output_stem + '_' + stage + '_mismatched_original',
          num_shards=1,
          file_name_suffix='.dat'))
  _ = (
      reshuffled_mismatched
      | 'ExtractMismatchedRegen' + label >> beam.Map(lambda x: x[2])
      | 'WriteMismatchedRegen' + label >> beam.io.WriteToText(
          FLAGS.output_stem + '_' + stage + '_mismatched_regen',
          num_shards=1,
          file_name_suffix='.dat'))

  matched_conformers = (
      matched
      | 'ExtractMatchedConformer' + label >> beam.Map(lambda x: x[1]))

  return matched_conformers


def pipeline(root):
  """Beam pipeline.

  Args:
    root: the root of the pipeline.
  """
  stage1_matched_conformers = dat_input_and_parsing_pipeline(root, 'stage1')
  stage2_matched_conformers = dat_input_and_parsing_pipeline(root, 'stage2')

  # Create a collection of conformers with duplicate information
  equivalent_files = gfile.glob(FLAGS.input_equivalent_glob)
  equivalent_conformers = (
      root
      | 'CreateEquivInputs' >> beam.Create(equivalent_files)
      | 'ParseEquiv' >> beam.FlatMap(parse_equivalent_file))

  # Merge by bond_topology_id
  merged_results = (
      (stage1_matched_conformers, stage2_matched_conformers,
       equivalent_conformers)
      | 'FlattenAllConformers' >> beam.Flatten()
      | 'GroupByCID' >> beam.GroupBy(lambda c: c.conformer_id)
      | 'MergeConformers' >> beam.ParDo(MergeConformersFn()).with_outputs(
          MergeConformersFn.OUTPUT_TAG_MERGE_CONFLICT, main='conformers'))
  merged_conformers = merged_results['conformers']

  # Write out the merge conflicts
  _ = (
      merged_results[MergeConformersFn.OUTPUT_TAG_MERGE_CONFLICT]
      | 'ConflictsCSVFormat' >> beam.Map(csv_format)
      | 'ConflictsReshuffle' >> beam.Reshuffle()
      | 'WriteConflictsCSV' >> beam.io.WriteToText(
          FLAGS.output_stem + '_conflicts',
          header=csv_format(smu_utils_lib.MERGE_CONFLICT_FIELDS),
          num_shards=1,
          file_name_suffix='.csv'))

  # Get the bond length distributions
  bond_length_dists_pcoll = (
      merged_conformers
      | 'ExtractBondLengths' >> beam.FlatMap(
          extract_bond_lengths,
        dist_sig_digits=_BOND_LENGTHS_SIG_DIGITS,
        unbonded_max=_BOND_LENGTHS_UNBONDED_MAX)
      | 'CountBondLengths' >> beam.combiners.Count.PerElement()
      | 'ToListBondLengths' >> beam.combiners.ToList()
  )
  _ = (
    bond_length_dists_pcoll
    | 'WriteBondLengths' >> beam.ParDo(
      write_bond_lengths, filename=f'{FLAGS.output_stem}_bond_lengths.csv'))

  # Get the SMILES to id mapping needed for UpdateConformerFn
  smiles_id_pcoll = (
    root
    | 'BTInputForSmiles' >> beam.Create([FLAGS.input_bond_topology_csv])
    | 'GenerateSmilesToID' >> beam.FlatMap(smiles_to_id))
  smiles_id_dict = beam.pvalue.AsDict(smiles_id_pcoll)

  # Various per conformer processing
  update_results = (
      merged_conformers
      | 'UpdateConformers'
      >> beam.ParDo(UpdateConformerFn(),
                    beam.pvalue.AsSingleton(bond_length_dists_pcoll),
                    smiles_id_dict).with_outputs(
          UpdateConformerFn.OUTPUT_TAG_SMILES_MISMATCH, main='conformers'))
  updated_conformers = update_results['conformers']

  # Output SMILES mismatches
  _ = (
      update_results[UpdateConformerFn.OUTPUT_TAG_SMILES_MISMATCH]
      | 'ReshuffleSmilesOutput' >> beam.Reshuffle()
      | 'SmilesCSVFormat' >> beam.Map(csv_format)
      | 'WriteSmilesCSV' >> beam.io.WriteToText(
          FLAGS.output_stem + '_smiles_compare',
          header='conformer_id,compare,smiles_given,smiles_with_h,smiles_without_h',
          num_shards=1,
          file_name_suffix='.csv'))

  # Process duplicate information
  final_conformers = (
      updated_conformers
      | 'KeyedForDuplicates' >>
      beam.FlatMap(generate_keyed_conformers_for_duplicates)
      | 'DupGroupByKey' >> beam.GroupByKey()
      | 'MergeDupInfo' >> beam.MapTuple(merge_duplicate_information))

  # Pull the stats of various sorts write to a file
  _ = (
      final_conformers
      | 'ExtractStats' >> beam.FlatMap(conformer_to_stat_values)
      | 'CountStats' >> beam.combiners.Count.PerElement()
      | 'StatsCSVFormat' >> beam.MapTuple(lambda x, c: f'{x[0]},{x[1]},{c}')
      | 'WriteStatsCSV' >> beam.io.WriteToText(
          FLAGS.output_stem + '_stats',
          header='primary_key,secondary_key,count',
          num_shards=1,
          file_name_suffix='.csv'))

  # Generate the summary by bond topology.
  bare_bt_summaries = (
      root
      | 'BondTopologyInput' >> beam.Create([FLAGS.input_bond_topology_csv])
      | 'GenerateBareBTSummaries' >>
      beam.FlatMap(bond_topology_summaries_from_csv))
  real_bt_summaries = (
      final_conformers
      | 'GenerateBTSummaries' >> beam.FlatMap(to_keyed_bond_topology_summary))
  _ = ((bare_bt_summaries, real_bt_summaries)
       | 'FlattenAllBTSummaries' >> beam.Flatten()
       | 'FinishBTSummary' >> CombineAndWriteBondTopologySummary())

  # Make the filtered versions of the dataset
  complete_conformers = (
      final_conformers
      | 'MakeComplete' >> beam.Map(make_complete_conformer))

  standard_conformers = (
      final_conformers
      | 'MakeStandard' >> beam.FlatMap(make_standard_conformer))

  # Write the complete and standard conformers as binary protobuf in TFRecord.
  for id_str, collection in [['complete', complete_conformers],
                             ['standard', standard_conformers]]:
    _ = (
        collection
        | ('TFRecordReshuffle_' + id_str) >> beam.Reshuffle()
        | ('WriteTFRecord_' + id_str) >> beam.io.tfrecordio.WriteToTFRecord(
            f'{FLAGS.output_stem}_{id_str}_tfrecord',
            coder=beam.coders.ProtoCoder(dataset_pb2.Conformer),
            num_shards=FLAGS.output_shards))

  # Write the complete and standard conformers as JSON.
  # Bit of a hack here: the slowest part of the whole pipeline is writing out
  # the JSON for the complete conformers. So we just hard code a tripling of the
  # shards to get more parallelism.
  for id_str, collection, num_shards in [[
      'complete', complete_conformers, FLAGS.output_shards * 3
  ], ['standard', standard_conformers, FLAGS.output_shards]]:
    _ = (
        collection
        | ('JSONReshuffle_' + id_str) >> beam.Reshuffle()
        | ('ToJSON_' + id_str) >> beam.Map(conformer_to_json)
        | ('WriteJSON_' + id_str) >> beam.io.WriteToText(
            f'{FLAGS.output_stem}_{id_str}_json',
            compression_type='gzip',
            num_shards=num_shards,
            file_name_suffix='.json.gz'))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  stdlogging.getLogger().setLevel(stdlogging.INFO)
  logging.info('Pipeline Starts.')
  # If you have custom beam options, add them here.
  beam_options = None
  with beam.Pipeline(beam_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
