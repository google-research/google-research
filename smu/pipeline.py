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

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from tensorflow.io import gfile

from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom
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
    'list.equivalent_isomers.dat and list.equivalent_molomers.dat)')
flags.DEFINE_string('output_stem', None, 'Filestem for output files')
flags.DEFINE_integer('output_shards', 10,
                     'Number of output shards for our primary outputs')

FLAGS = flags.FLAGS

_METRICS_NAMESPACE = 'SMU'
_BOND_LENGTHS_UNBONDED_MAX = 2.0


def parse_equivalent_file(filename):
  """Parses the .dat of equivalent structure.

  The file is just pairs of entries where the first was kept over the second.
  Yields one entry per line keyed by the discarded molecule id.
  See merge_duplicate_information for how information is transferred to the kept
  molecule.

  Args:
    filename: string

  Yields:
    dataset_pb2.Molecule
  """
  with gfile.GFile(filename) as f:
    for line in f:
      kept_str, discard_str = line.split()
      _, _, kept_btid, kept_mid = smu_parser_lib.parse_long_identifier(kept_str)
      _, _, discard_btid, discard_mid = smu_parser_lib.parse_long_identifier(
          discard_str)
      # Convert to our molecule ids which include the btid
      kept_mid = kept_btid * 1000 + kept_mid
      discard_mid = discard_btid * 1000 + discard_mid

      yield dataset_pb2.Molecule(mol_id=discard_mid, duplicate_of=kept_mid)


def parse_dat_file(filename, stage):
  """Beam pipeline component for reading dat files.

  Args:
    filename: filename to read
    stage: string 'stage1' or 'stage2'

  Yields:
    Pair of string (original dat), molecule
    molecule can be an Exception or a dataset_pb2.Molecule
  """
  smu_parser = smu_parser_lib.SmuParser(filename)
  if stage == 'stage1':
    process_fn = smu_parser.process_stage1
  else:
    process_fn = smu_parser.process_stage2
  for molecule, orig_dat_list in process_fn():
    orig_dat = '\n'.join(orig_dat_list) + '\n'

    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_dat_entry_read').inc()

    yield orig_dat, molecule


def partition_parse_success(input_tuple, num_partitions, stage):
  """Function to beam.Partition parsed inputs based on parse success.

  Args:
    input_tuple: pair of orig_contents, molecule (see parse_dat_file)
    num_partitions: (should always be 3)
    stage: string 'stage1' or 'stage2'

  Returns:
    int (0 for success, 1, for known error, 2 for unknown error)
  """
  assert num_partitions == 3
  _, molecule = input_tuple
  if not isinstance(molecule, Exception):
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_parse_success').inc()
    return 0  # Parse success
  else:
    if isinstance(molecule, smu_parser_lib.SmuKnownError):
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   stage + '_parse_known_error').inc()
      return 1  # Parse known error
    else:
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   stage + '_parse_unknown_error').inc()
      return 2  # Parse unknown error


def regenerate_dat(input_tuple, stage):
  """Regenerates the original dat from molecule and compares it to original.

  Args:
    input_tuple: tuple of string (original contents), dataset_pb2.Molecule
    stage: string 'stage1' or 'stage2'

  Returns:
    original_dat, molecule, regenerated dat, int (0=mismatch, 1=match)
  """
  original_dat, molecule = input_tuple
  smu_writer = smu_writer_lib.SmuWriter(annotate=False)
  if stage == 'stage1':
    regen_dat = smu_writer.process_stage1_proto(molecule)
  else:
    regen_dat = smu_writer.process_stage2_proto(molecule)
  try:
    smu_writer_lib.check_dat_formats_match(original_dat.splitlines(),
                                           regen_dat.splitlines())
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_dat_format_matched').inc()
    return original_dat, molecule, regen_dat, 1
  except smu_writer_lib.DatFormatMismatchError:
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 stage + '_dat_format_mismatched').inc()
    return original_dat, molecule, regen_dat, 0


def molecule_to_stat_values(molecule):
  """Beam transform to produce stats values for later aggregation.

  Each output will be a tuple of primary_key, secondary_key and these will be
  aggregated as counts.

  Args:
    molecule: dataset_pb2.Molecule

  Yields:
    primary_key, secondary_key
  """
  # Yield the values for all the relevant error fields.
  for field in [
      'status', 'warn_t1', 'warn_delta_t1', 'warn_bse_b6', 'warn_bse_eccsd',
      'warn_exc_ene', 'warn_exc_osmin', 'warn_exc_osmax', 'warn_vib_linear',
      'warn_vib_imag', 'warn_bsr_neg', 'error_nstat1', 'error_nstatc',
      'error_nstatt', 'error_frequencies'
  ]:
    yield 'errors.' + field, getattr(molecule.prop.calc, field)

  yield 'fate', dataset_pb2.Properties.FateCategory.Name(
      molecule.prop.calc.fate)

  yield 'num_initial_geometries', len(
      [g for g in molecule.ini_geo if g.atompos])
  yield 'num_duplicates', len(molecule.duplicate_found)

  for field in smu_utils_lib.find_zero_values(molecule):
    yield 'zero_field', field

  if not molecule.duplicate_of and molecule.prop.calc.status < 512:
    yield 'num_topologies', len(molecule.bond_topo)

    yield 'num_topologies_itc', len([
        None for bt in molecule.bond_topo
        if bt.info & dataset_pb2.BondTopology.SOURCE_DDT
    ])
    yield 'num_topologies_mlcr', len([
        None for bt in molecule.bond_topo
        if bt.info & dataset_pb2.BondTopology.SOURCE_MLCR
    ])
    yield 'num_topologies_csd', len([
        None for bt in molecule.bond_topo
        if bt.info & dataset_pb2.BondTopology.SOURCE_CSD
    ])

    for bt in molecule.bond_topo:
      yield 'bt_source', bt.info


def bond_topology_summaries_from_csv(filename):
  """Beam DoFn for generating bare BondTopologySummary.

  Args:
    filename: csv file of bond topologies to read

  Yields:
    dataset_pb2.Entry
  """
  with gfile.GFile(filename, 'r') as infile:
    for bt in smu_utils_lib.generate_bond_topologies_from_csv(infile):
      summary = dataset_pb2.BondTopologySummary()
      summary.bond_topology.CopyFrom(bt)
      # Note that we leave all the counts as 0.
      yield bt.topo_id, summary


class MergeMoleculesFn(beam.DoFn):
  """Merges molecules with the same id.

  Because of the stage1, stage2, and duplicate information, we can end up with
  multiple molecules with the same id. This merges them.
  """
  OUTPUT_TAG_MERGE_CONFLICT = 'conflict'

  def process(self, args):
    """Merges molecules.

    Args:
      args: tuple of mol_id(should match the id in all molecules) and
        molecules(iterable of dataset_pb2.Molecule)

    Yields:
      dataset_pb2.Molecule and tagged output (OUTPUT_TAG_MERGE_CONFLICT) with
      conflict output from smu_utils_lib.merge_molecule

    Raises:
      ValueError: on inconsistent mol_id
    """
    mol_id, molecules = args

    for c in molecules:
      if c.mol_id != mol_id:
        raise ValueError(
            f'In merged CID {mol_id}, found CID {c.mol_id} instead')

    # For signalling the first merging.
    sentinel = object()

    conflicts = []

    def _merge_two_molecules(mol0, mol1):
      if mol0 is sentinel:
        return mol1

      merged_mol, merge_conflict = smu_utils_lib.merge_molecule(mol0, mol1)
      if merge_conflict:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'molecule_merge_error').inc()
        conflicts.append(merge_conflict)
      return merged_mol

    beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'merged_molecules').inc()

    # Note that we convert the iterable to a list and do a deepcopy. We can't
    # modify the input and smu_utils_lib.merge_molecule wants to reserve the
    # right to modify either input so it's simplest to just copy it once right
    # off the bat.
    yield functools.reduce(_merge_two_molecules, copy.deepcopy(list(molecules)),
                           sentinel)

    for c in conflicts:
      yield beam.pvalue.TaggedOutput(MergeMoleculesFn.OUTPUT_TAG_MERGE_CONFLICT,
                                     c)


def extract_bond_lengths(molecule, dist_sig_digits, unbonded_max):
  """Yields quantized bond lengths.

  Args:
    molecule: dataset_pb2.Molecule
    dist_sig_digits: number of digits after decimal point to keep
    unbonded_max: maximum distance to report for unbonded pairs  output atom
      types are single charecters, sorted lexographically. bond_type is
      dataset_pb2.BondTopology.BondType dist_sig_digits is a string (to avoid
      vagaries of floating point compares)

  Yields:
    (atom type 1, atom type 2, bond type, quantized dist)
  """
  # These are considered "major" or worse errors
  if (molecule.prop.calc.status >= 8 or molecule.duplicate_of > 0):
    return

  bt = molecule.bond_topo[0]
  format_str = '{:.%df}' % dist_sig_digits

  for atom_idx0, atom_idx1 in itertools.combinations(range(len(bt.atom)), r=2):

    if (bt.atom[atom_idx0] == dataset_pb2.BondTopology.ATOM_H or
        bt.atom[atom_idx1] == dataset_pb2.BondTopology.ATOM_H):
      continue

    # Hello huge hack. F-F creates problems for us because there is
    # exactly one molecule that has an F-F bond. We can't create an
    # empirical distribution out of 1 value. So we'll just drop that
    # one and let the FF molecule have no detected geometries.
    if (bt.atom[atom_idx0] == dataset_pb2.BondTopology.ATOM_F and
        bt.atom[atom_idx1] == dataset_pb2.BondTopology.ATOM_F):
      continue

    bond_type = smu_utils_lib.get_bond_type(bt, atom_idx0, atom_idx1)

    geom = molecule.opt_geo
    atom_pos0 = np.array([
        geom.atompos[atom_idx0].x, geom.atompos[atom_idx0].y,
        geom.atompos[atom_idx0].z
    ],
                         dtype=np.double)
    atom_pos1 = np.array([
        geom.atompos[atom_idx1].x, geom.atompos[atom_idx1].y,
        geom.atompos[atom_idx1].z
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

    atom_char0 = smu_utils_lib.ATOM_TYPE_TO_CHAR[bt.atom[atom_idx0]]
    atom_char1 = smu_utils_lib.ATOM_TYPE_TO_CHAR[bt.atom[atom_idx1]]
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
    smiles, topo_id
  """
  with gfile.GFile(bond_topology_filename, 'r') as infile:
    reader = csv.reader(iter(infile))
    next(reader)  # skip the header line
    for row in reader:
      bt_id, _, _, _, _, smiles = row
      yield smiles, int(bt_id)


def clean_up_molecule(molecule):
  """Miscellaneous clean up.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    copy of molecule with modifications
  """
  molecule = copy.deepcopy(molecule)

  smu_utils_lib.clean_up_error_codes(molecule)
  smu_utils_lib.clean_up_sentinel_values(molecule)

  return molecule


class UpdateMoleculeFn(beam.DoFn):
  """DoFn that performs several updates to fields in Molecule.

  * Updates the smiles string (with a tagged output to record the mismatches.
  * Adds Fate field
  * Adds additional bond topologies that match the geometry
  * various cleanup steps

  main output is dataset_pb2.Molecule
  smiles output is a tuple of
    mol_id,
    SmilesCompareResult,
    original smiles,
    smiles_with_h,
    smiles_without_h
  """
  OUTPUT_TAG_SMILES_MISMATCH = 'tag_smiles'

  def setup(self):
    self._cached_bond_lengths = None

  def _compare_smiles(self, molecule):
    if len(molecule.bond_topo) != 1:
      raise ValueError(
          'compare_smiles expects 1 bond topology; for CID {} got {}'.format(
              molecule.mol_id, len(molecule.bond_topo)))

    result, smiles_with_h, smiles_without_h = (
        smu_utils_lib.bond_topology_smiles_comparison(molecule.bond_topo[0]))
    if result != smu_utils_lib.SmilesCompareResult.MATCH:
      yield beam.pvalue.TaggedOutput(
          UpdateMoleculeFn.OUTPUT_TAG_SMILES_MISMATCH,
          (molecule.mol_id, result, molecule.bond_topo[0].smiles, smiles_with_h,
           smiles_without_h))
      molecule.prop.smiles_openbabel = (molecule.bond_topo[0].smiles)
      molecule.bond_topo[0].smiles = smiles_without_h

  def _add_alternative_bond_topologies(self, molecule, smiles_id_dict):
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 'attempted_topology_matches').inc()

    if not topology_from_geom.standard_topology_sensing(
        molecule, self._cached_bond_lengths, smiles_id_dict):
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'no_topology_matches').inc()

    for bt in molecule.bond_topo:
      if not bt.topo_id:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'topology_match_smiles_failure').inc()

  def process(self, molecule, bond_length_records, smiles_id_dict):
    """Per molecule updates.

    Args:
      molecule: dataset_pb2.Molecule
      bond_length_records: tuples to go to
        bond_length_distribution.AllAtomPairLengthDistributions
      smiles_id_dict: dict from SMILES to bond topology id

    Yields:
      Molecule.

    Raises:
      ValueError: if the bond_length_records are misformatted
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
      try:
        self._cached_bond_lengths.add_from_sparse_dataframe(
            bond_length_distribution.sparse_dataframe_from_records(
                bond_length_records),
            bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS,
            bond_length_distribution.STANDARD_SIG_DIGITS)
      except ValueError as err:
        raise ValueError(
            'Invalid sparse dataframe for molecule {0} org. ValueError: {1}'
            .format(str(molecule.mol_id), err)) from err

    molecule = copy.deepcopy(molecule)

    molecule.prop.calc.fate = smu_utils_lib.determine_fate(molecule)

    yield from self._compare_smiles(molecule)

    if smu_utils_lib.molecule_eligible_for_topology_detection(molecule):
      self._add_alternative_bond_topologies(molecule, smiles_id_dict)
    else:
      molecule.bond_topo[0].info = dataset_pb2.BondTopology.SOURCE_STARTING
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'skipped_topology_matches').inc()

    yield molecule


def generate_keyed_molecules_for_duplicates(molecule):
  """Generates keyed molecules for duplicate merging.

  Every molecule yields itself keyed by its mol_id
  Additonally, if duplicate_of is set, the molecule is yielded keyed by
  duplicate_of.

  Args:
    molecule: dataset_pb2.Molecule

  Yields:
    mol_id, dataset_pb2.Molecule
  """
  yield molecule.mol_id, molecule
  if molecule.duplicate_of > 0:
    yield molecule.duplicate_of, molecule


def merge_duplicate_information(mol_id, molecules):
  """Merges duplicate information into one molecule.

  One entry in molecules should have the given mol_id
  (call this the "main" molecule)
  Every other entry should have a duplicate_of set to mol_id
  (call this an "other" molecule)

  The initial_geometry from other will copied to main.
  If the bond topology id is the same, this is trivial
  TODO(pfr, ianwatson): implement this copying with unequal bond topologies.

  Args:
    mol_id: integer
    molecules: iterable of dataset_pb2.Molecule

  Returns:
    dataset_pb2.Molecule

  Raises:
    ValueError: if duplicate records not as expected
  """
  matching_molecules = [c for c in molecules if c.mol_id == mol_id]
  if len(matching_molecules) != 1:
    raise ValueError('Expected 1 molecules with id {}, got {}'.format(
        mol_id, len(matching_molecules)))
  main_molecule = copy.deepcopy(matching_molecules[0])

  for mol in molecules:
    if mol.mol_id == mol_id:
      continue
    if mol.duplicate_of != mol_id:
      raise ValueError(
          'Molecule {} should have duplicate_of {} but has {}'.format(
              mol.mol_id, mol_id, mol.duplicate_of))
    main_molecule.duplicate_found.append(mol.mol_id)
    if mol_id // 1000 == mol.mol_id // 1000:
      # easy case! Bond topologies are the same, just copy over
      main_molecule.ini_geo.append(mol.ini_geo[0])
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'dup_same_topology').inc()
    else:
      # hard case. We have to figure out out to permute the atoms in the initial
      # geometry
      # TODO(pfr, ianwatson)
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                   'dup_diff_topology_unmatched').inc()

  return main_molecule


def to_keyed_bond_topology_summary(molecule):
  """Outputs BondTopologySummary for molecule.

  Args:
    molecule: dataset_pb2.Molecule

  Yields:
    bond topology id, BondTopologySummary
  """
  for summary in smu_utils_lib.molecule_to_bond_topology_summaries(molecule):
    yield summary.bond_topology.topo_id, summary


def merge_bond_topology_summaries(summaries, field_names):
  """Merges BondTopologySummary protos.

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

    assert summary0.bond_topology.topo_id == summary1.bond_topology.topo_id

    for name in field_names:
      setattr(summary0, name, getattr(summary0, name) + getattr(summary1, name))

    return summary0

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'merged_summaries').inc()

  return functools.reduce(_merge_two_summaries, summaries, sentinel)


def csv_format_bond_topology_summary(summary, field_names):
  """Formats BondTopologySummary protos as csv line.

  See CombineAndWriteBondTopologySummary for context.

  Args:
    summary: BondTopologySummary
    field_names: list of field names in the order for the csv

  Returns:
    BondTopologySummary
  """
  return ','.join([str(summary.bond_topology.topo_id)] +
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


def make_complete_molecule(molecule):
  """Turns a Molecule into the complete form from the internal only.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    dataset_pb2.Molecule
  """
  out = copy.deepcopy(molecule)
  smu_utils_lib.filter_molecule_by_availability(
      out, [dataset_pb2.STANDARD, dataset_pb2.COMPLETE])

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'complete_molecules').inc()

  return out


def make_standard_molecule(molecule):
  """Turns a Molecule into the standard form from the internal only.

  This must go through a FlatMap because some molecules are filtered.

  Args:
    molecule: dataset_pb2.Molecule

  Yields:
    at most one dataset_pb2.Molecule
  """
  out = copy.deepcopy(molecule)
  if not smu_utils_lib.molecule_to_standard(out):
    return

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'standard_molecules').inc()

  yield out




def _csv_format(vals):
  return ','.join(str(v) for v in vals)


def dat_input_and_parsing_pipeline(root, stage):
  """Create multiple stages for parsing and validation .dat files.

  We read two types of .dat files, stage1 and stage2. The pipeline is similar
  between the two, so we use this function to cover those similar parts.

  Args:
    root: root of the pipeline
    stage: string that is either "stage1" or "stage2"

  Returns:
    PCollection of dataset_pb2.Molecule that are valid and matched
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

  # Write out the mismatched molecules, original and regenerated
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

  matched_molecules = (
      matched
      | 'ExtractMatchedMolecule' + label >> beam.Map(lambda x: x[1]))

  return matched_molecules


def pipeline(root):
  """Beam pipeline.

  Args:
    root: the root of the pipeline.
  """
  stage1_matched_molecules = dat_input_and_parsing_pipeline(root, 'stage1')
  stage2_matched_molecules = dat_input_and_parsing_pipeline(root, 'stage2')

  # Create a collection of molecules with duplicate information
  equivalent_files = gfile.glob(FLAGS.input_equivalent_glob)
  equivalent_molecules = (
      root
      | 'CreateEquivInputs' >> beam.Create(equivalent_files)
      | 'ParseEquiv' >> beam.FlatMap(parse_equivalent_file))

  # Merge by topo_id
  merged_results = (
      (stage1_matched_molecules, stage2_matched_molecules, equivalent_molecules)
      | 'FlattenAllMolecules' >> beam.Flatten()
      | 'GroupByCID' >> beam.GroupBy(lambda c: c.mol_id)
      | 'MergeMolecules' >> beam.ParDo(MergeMoleculesFn()).with_outputs(
          MergeMoleculesFn.OUTPUT_TAG_MERGE_CONFLICT, main='molecules'))
  merged_molecules = merged_results['molecules']

  # Write out the merge conflicts
  _ = (
      merged_results[MergeMoleculesFn.OUTPUT_TAG_MERGE_CONFLICT]
      | 'ConflictsCSVFormat' >> beam.Map(_csv_format)
      | 'ConflictsReshuffle' >> beam.Reshuffle()
      | 'WriteConflictsCSV' >> beam.io.WriteToText(
          FLAGS.output_stem + '_conflicts',
          header=_csv_format(smu_utils_lib.MERGE_CONFLICT_FIELDS),
          num_shards=1,
          file_name_suffix='.csv'))

  cleaned_molecules = (
      merged_molecules
      | 'CleanUpMolecules' >> beam.Map(clean_up_molecule))

  # Get the bond length distributions
  bond_length_dists_pcoll = (
      cleaned_molecules
      | 'ExtractBondLengths' >> beam.FlatMap(
          extract_bond_lengths,
          dist_sig_digits=bond_length_distribution.STANDARD_SIG_DIGITS,
          unbonded_max=_BOND_LENGTHS_UNBONDED_MAX)
      | 'CountBondLengths' >> beam.combiners.Count.PerElement()
      | 'ToListBondLengths' >> beam.combiners.ToList())
  _ = (
      bond_length_dists_pcoll
      | 'WriteBondLengths' >> beam.ParDo(
          write_bond_lengths, filename=f'{FLAGS.output_stem}_bond_lengths.csv'))

  # Get the SMILES to id mapping needed for UpdateMoleculeFn
  smiles_id_pcoll = (
      root
      | 'BTInputForSmiles' >> beam.Create([FLAGS.input_bond_topology_csv])
      | 'GenerateSmilesToID' >> beam.FlatMap(smiles_to_id))
  smiles_id_dict = beam.pvalue.AsDict(smiles_id_pcoll)

  # Various per molecule processing
  update_results = (
      cleaned_molecules
      | 'UpdateMolecules' >> beam.ParDo(
          UpdateMoleculeFn(), beam.pvalue.AsSingleton(bond_length_dists_pcoll),
          smiles_id_dict).with_outputs(
              UpdateMoleculeFn.OUTPUT_TAG_SMILES_MISMATCH, main='molecules'))
  updated_molecules = update_results['molecules']

  # Output SMILES mismatches
  _ = (
      update_results[UpdateMoleculeFn.OUTPUT_TAG_SMILES_MISMATCH]
      | 'ReshuffleSmilesOutput' >> beam.Reshuffle()
      | 'SmilesCSVFormat' >> beam.Map(_csv_format)
      | 'WriteSmilesCSV' >> beam.io.WriteToText(
          FLAGS.output_stem + '_smiles_compare',
          header='mol_id,compare,smiles_given,smiles_with_h,smiles_without_h',
          num_shards=1,
          file_name_suffix='.csv'))

  # Process duplicate information
  final_molecules = (
      updated_molecules
      | 'KeyedForDuplicates' >>
      beam.FlatMap(generate_keyed_molecules_for_duplicates)
      | 'DupGroupByKey' >> beam.GroupByKey()
      | 'MergeDupInfo' >> beam.MapTuple(merge_duplicate_information))

  # Pull the stats of various sorts write to a file
  _ = (
      final_molecules
      | 'ExtractStats' >> beam.FlatMap(molecule_to_stat_values)
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
      final_molecules
      | 'GenerateBTSummaries' >> beam.FlatMap(to_keyed_bond_topology_summary))
  _ = ((bare_bt_summaries, real_bt_summaries)
       | 'FlattenAllBTSummaries' >> beam.Flatten()
       | 'FinishBTSummary' >> CombineAndWriteBondTopologySummary())

  # Make the filtered versions of the dataset
  complete_molecules = (
      final_molecules
      | 'MakeComplete' >> beam.Map(make_complete_molecule))

  standard_molecules = (
      final_molecules
      | 'MakeStandard' >> beam.FlatMap(make_standard_molecule))

  # Write the complete and standard molecules as binary protobuf in TFRecord.
  for id_str, collection in [['complete', complete_molecules],
                             ['standard', standard_molecules]]:
    _ = (
        collection
        | ('TFRecordReshuffle_' + id_str) >> beam.Reshuffle()
        | ('WriteTFRecord_' + id_str) >> beam.io.tfrecordio.WriteToTFRecord(
            f'{FLAGS.output_stem}_{id_str}_tfrecord',
            coder=beam.coders.ProtoCoder(dataset_pb2.Molecule),
            num_shards=FLAGS.output_shards))



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
