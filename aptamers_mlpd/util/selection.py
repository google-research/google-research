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

# Lint as: python3
"""Utility functions for creating and using selection_pb2 protos.

See the proto file itself for detailed documentation on these fields.

Example usage:

  import selection

  rounds = {}
  rounds['init'] = selection.random(int(1e5))
  rounds['round1_input'] = selection.pcr('init', int(1e5))
  rounds['round1_output'] = selection.particle_display(
      'round1_output', int(1e5), int(2e5), {'target': 1.0})
  experiment = selection.create_experiment(rounds)
  print experiment

This outputs:

  rounds {
    key: "init"
    value {
      positive_reads {
        depth: 100000
      }
      method: RANDOM
    }
  }
  rounds {
    key: "round1_input"
    value {
      input: "init"
      positive_reads {
        depth: 100000
      }
      method: PCR
    }
  }
  rounds {
    key: "round1_output"
    value {
      input: "round1_output"
      positive_reads {
        depth: 100000
      }
      negative_reads {
        depth: 200000
      }
      method: PARTICLE_DISPLAY
      concentrations {
        key: "target"
        value: 1.0
      }
    }
  }
"""

import collections
import copy
import os


import numpy as np

from ..util import selection_pb2


class Error(Exception):
  """Exception class for errors in specifying selection experiments."""
  pass


def _make_round(input_round=None,
                positive_depth=None,
                negative_depth=None,
                positive_reads_field=None,
                negative_reads_field=None,
                method=selection_pb2.Round.INVALID,
                target_concentrations=None,
                background_concentrations=None):
  """Create a selection_pb.Round.

  Args:
    input_round: str, optional
    positive_depth: integer, optional
    negative_depth: integer, optional
    positive_reads_field: string, optional
    negative_reads_field: string, optional
    method: selection_pb2.Round.METHOD enum, optional
    target_concentrations: Dict[str, float], optional
    background_concentrations: Dict[str, float], optional

  Returns:
    selection_pb.Round object with the appropriate fields.

  Raises:
    ValueError: if either of the depth options is set and non-zero but the
      corresponding reads_field is not set.
  """
  round_proto = selection_pb2.Round(method=method)
  if input_round is not None:
    round_proto.input = input_round
  if positive_depth is not None:
    round_proto.positive_reads.depth = positive_depth
    if not positive_reads_field:
      raise ValueError('must supply positive_reads_field if positive_depth > 0')
  if negative_depth is not None:
    round_proto.negative_reads.depth = negative_depth
    if not negative_reads_field:
      raise ValueError('must supply positive_reads_field if positive_depth > 0')
  if positive_reads_field is not None:
    round_proto.positive_reads.name = positive_reads_field
  if negative_reads_field is not None:
    round_proto.negative_reads.name = negative_reads_field
  if target_concentrations is not None:
    round_proto.target_concentrations.update(target_concentrations)
  if background_concentrations is not None:
    round_proto.background_concentrations.update(background_concentrations)
  return round_proto


def random(depth=None, reads_field=None):
  """Create an selection_pb.Round correpsonding to an initial random library.

  Args:
    depth: optional integer number of aptamers to sequence after this round.
    reads_field: optional string indicating the field in the count table where
      corresponding reads are stored.

  Returns:
    selection_pb.Round object with method=RANDOM.
  """
  return _make_round(
      positive_depth=depth,
      positive_reads_field=reads_field,
      method=selection_pb2.Round.RANDOM)


def pcr(input_round, depth=None, reads_field=None):
  """Create a selection_pb.Round corresponding to PCR.

  Args:
    input_round: string naming the previous round to use as input to PCR.
    depth: optional integer number of aptamers to sequence after this round.
    reads_field: optional string indicating the field in the count table where
      corresponding reads are stored.

  Returns:
    selection_pb.Round object with method=PCR.
  """
  return _make_round(
      input_round,
      depth,
      positive_reads_field=reads_field,
      method=selection_pb2.Round.PCR)


def particle_display(input_round,
                     positive_depth=None,
                     negative_depth=None,
                     positive_reads_field=None,
                     negative_reads_field=None,
                     target_concentrations=None,
                     background_concentrations=None):
  """Create an PCR selection round.

  Args:
    input_round: string naming the previous round to use as input to PCR.
    positive_depth: optional integer number of postively selected aptamers to
      sequence after this round.
    negative_depth: optional integer number of negatively selected aptamers to
      sequence after this round.
    positive_reads_field: optional string indicating the field in the count
      table where positive reads are stored.
    negative_reads_field: optional string indicating the field in the count
      table where negative reads are stored.
    target_concentrations: optional Dict[str, float] giving concentrations of
      the target molecules.
    background_concentrations: optional Dict[str, float] giving concentrations
      of background molecules.

  Returns:
    selection_pb.Round object with method=PCR.
  """
  return _make_round(input_round, positive_depth, negative_depth,
                     positive_reads_field, negative_reads_field,
                     selection_pb2.Round.PARTICLE_DISPLAY,
                     target_concentrations, background_concentrations)


def selex(input_round,
          depth=None,
          reads_field=None,
          target_concentrations=None,
          background_concentrations=None):
  """Create a SELEX round.

  Args:
    input_round: string naming the previous round to use as input to SELEX.
    depth: optional integer number of aptamers to sequence after this round.
    reads_field: optional string indicating the field in the count table where
      corresponding reads are stored.
    target_concentrations: optional Dict[str, float] giving concentrations of
      the target molecules.
    background_concentrations: optional Dict[str, float] giving concentrations
      of background molecules.

  Returns:
    selection_pb.Round object with method=SELEX.
  """
  return _make_round(
      input_round,
      depth,
      positive_reads_field=reads_field,
      method=selection_pb2.Round.SELEX,
      target_concentrations=target_concentrations,
      background_concentrations=background_concentrations)


def _input_rounds(experiment_proto):
  input_rounds = set()
  for round_name, round_proto in experiment_proto.rounds.items():
    if not round_proto.input:
      input_rounds.add(round_name)
  return input_rounds


def _child_rounds(experiment_proto):
  children = collections.defaultdict(set)
  for round_name, round_proto in experiment_proto.rounds.items():
    if round_proto.input:
      children[round_proto.input].add(round_name)
  return children


def _is_selection_round(round_proto):
  return round_proto.method in [round_proto.SELEX, round_proto.PARTICLE_DISPLAY]


def _validate_rounds(experiment_proto, require_targets=True):
  """Verify that the rounds found in an Experiment proto are valid.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    require_targets: optional boolean indicating whether or not to require
      targets for relevant selection rounds.

  Returns:
    None.

  Raises:
    Error: If the experiment is not valid.
  """
  if require_targets and not all_target_and_background_names(experiment_proto):
    raise Error('no target or background concentrations set')

  input_rounds = _input_rounds(experiment_proto)
  children = _child_rounds(experiment_proto)

  reached = set(input_rounds)
  to_check = list(reached)
  while to_check:
    round_name = to_check.pop()
    round_proto = experiment_proto.rounds[round_name]
    if round_proto.input and round_proto.input not in experiment_proto.rounds:
      raise Error('round %r has invalid input' % round_name)
    if _is_selection_round(round_proto) and not round_proto.input:
      raise Error('selection round %r is missing required input' % round_name)
    if require_targets:
      if _is_selection_round(round_proto):
        targets = {k for k, v in round_proto.target_concentrations.items() if v}
        if not targets:
          raise Error('selection round %r has no target:' % round_name)
    child_rounds = children[round_name]
    to_check.extend(child_rounds)
    reached.update(child_rounds)

  not_reached = set(experiment_proto.rounds) - reached
  if not_reached:
    raise Error('some rounds reference invalid inputs: %r' % not_reached)


def validate_experiment(experiment_proto, require_targets=True):
  """Verify that an Experiment proto describes a valid aptamer experiment.

  Experiments that pass this test are suitable for running through our
  preprocessing pipeline and TensorFlow models.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    require_targets: optional boolean indicating whether or not to require
      targets for relevant selection rounds.

  Returns:
    None.

  Raises:
    Error: If the experiment is not valid.
  """
  if not experiment_proto.sequence_length:
    raise Error('sequence_length must be set')

  _validate_rounds(experiment_proto, require_targets=require_targets)


def _assign_measurement_ids(experiment_proto):
  """Assign sequential measurement IDs in an experiment.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    A new selection_pb2.Experiment with measurement IDs assigned.

  Raises:
    ValueError: If any measurements already have assigned IDs.
  """
  # box measurement_id in a list so we can assign to it in a closure
  # (if only we were use Python 3.4+ so we could use the nonlocal keyword...)
  measurement_id = [0]

  def _maybe_assign_next_measurement_id(measurement):
    if measurement.name:
      if measurement.measurement_id != 0:
        raise ValueError('measurement_id already set for measurement: %r'
                         % measurement)
      measurement_id[0] += 1
      measurement.measurement_id = measurement_id[0]

  input_rounds = _input_rounds(experiment_proto)
  children = _child_rounds(experiment_proto)

  experiment_proto = copy.deepcopy(experiment_proto)

  # BFS by sorted order per round. This ensures measurement IDs are
  # deterministic.
  to_assign = sorted(input_rounds)
  while to_assign:
    current_round_name = to_assign.pop()
    round_proto = experiment_proto.rounds[current_round_name]
    _maybe_assign_next_measurement_id(round_proto.positive_reads)
    _maybe_assign_next_measurement_id(round_proto.negative_reads)

    child_rounds = sorted(children[current_round_name])
    to_assign.extend(child_rounds)

  for binding_array in experiment_proto.binding_arrays:
    _maybe_assign_next_measurement_id(binding_array)

  return experiment_proto


def create_experiment(rounds,
                      binding_arrays=None,
                      assign_measurement_ids=True,
                      validate_rounds=True,
                      require_targets=True):
  """Create a selection_pb.Experiment from a dictionary of rounds.

  Args:
    rounds: Dict[str, selection_pb2.Round]. Dictionary keys name rounds.
    binding_arrays: optional List[selection_pb2.BindingArray] that are part of
      this experiment.
    assign_measurement_ids: optional boolean indicating whether or not to
      automatically assign measurement IDs to all SequencingReads.
    validate_rounds: optional boolean indicating whether or not to verify the
      integrity of the selection rounds in this experiment.
    require_targets: optional boolean indicating whether or not to require
      targets for relevant selection rounds. Only relevant is validate_rounds
      is True.

  Returns:
    selection_pb2.Experiment with the given rounds.

  Raises:
    ValueError: If any assign_measurement_ids
  """
  experiment_proto = selection_pb2.Experiment(
      rounds=rounds, binding_arrays=binding_arrays)
  if assign_measurement_ids:
    experiment_proto = _assign_measurement_ids(experiment_proto)
  if validate_rounds:
    _validate_rounds(experiment_proto, require_targets=require_targets)
  return experiment_proto


def get_template_sequence(experiment_proto):
  """Get the template_sequence from an experiment proto.

  If template_sequence is undefined, generates a sane default.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    String of length `experiment_proto.sequence_length'
  """
  template_sequence = experiment_proto.template_sequence
  if not template_sequence:
    template_sequence = 'N' * experiment_proto.sequence_length
  return template_sequence


def add_array_column(experiment_proto,
                     array_name,
                     measurement_id=None,
                     target_concentrations=None):
  """Add one binding array to an experiment_proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    array_name: The string name of the binding array.
    measurement_id: The optional integer measurement_id within the
      experiment_proto. If none, the current max + 1 will be used.
    target_concentrations: optional Dict[str, float] giving concentrations of
      the target molecules.

  Raises:
    Error: if the measurement_id is not unique within the experiment_proto.
  """
  # check the measurement id is unique
  if measurement_id:
    cur_ids, _ = measurement_ids_and_names(experiment_proto)
    if measurement_id in cur_ids:
      raise Error('The measurement_id %d cannot be used for array %s, it '
                  'already exists in the experiment_proto.' % (measurement_id,
                                                               array_name))
  else:
    measurement_id = max_measurement_id(experiment_proto) + 1

  experiment_proto.binding_arrays.add(
      name=array_name,
      measurement_id=measurement_id,
      target_concentrations=target_concentrations)


def round_from_count_name(name, experiment_proto):
  """Lookup the Round proto corresponding to a count table field.

  Args:
    name: string indicating the name of an output in the count table.
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    selection_pb2.Round describing a round with the given name in either its
    positive or negative reads.

  Raises:
    KeyError: If no matching round proto is found.
  """
  for round_proto in experiment_proto.rounds.values():
    if name in (round_proto.positive_reads.name,
                round_proto.negative_reads.name):
      return round_proto
  raise KeyError(name)


def reads_from_count_name(name, experiment_proto):
  """Lookup the SequencingReads proto corresponding to a count table field.

  Args:
    name: string indicating the name of an output in the count table.
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    selection_pb2.SequencingReads describing sequencing reads matching the given
    name.

  Raises:
    KeyError: If no matching count table field is found.
  """
  for round_proto in experiment_proto.rounds.values():
    if name == round_proto.positive_reads.name:
      return round_proto.positive_reads
    elif name == round_proto.negative_reads.name:
      return round_proto.negative_reads
  raise KeyError(name)


def max_measurement_id(experiment_proto):
  """Return the maximum id number in the experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
  Returns:
    integer representing the maximum measurement_id in the experiment_proto.
  """
  ids, _ = measurement_ids_and_names(experiment_proto)
  if ids:
    return max(ids)
  else:
    return -1


def measurement_ids_and_names(experiment_proto):
  """List all measurement IDs and names for an experiment.

  Results are sorted by measurement ID.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    measurement_ids: tuple of integer IDs for each measurement.
    column_names: tuple of string names correpsonding to each measurement ID.
  """
  pairs = []
  for round_proto in experiment_proto.rounds.values():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      if reads.name:
        pairs.append((reads.measurement_id, reads.name))
  for binding_array in experiment_proto.binding_arrays:
    if binding_array.name:
      pairs.append((binding_array.measurement_id, binding_array.name))
  if pairs:
    measurement_ids, column_names = list(zip(*sorted(pairs)))
    return measurement_ids, column_names
  else:
    return None, None


def parent_counts(experiment_proto):
  """Return a map from all counts to counts from their input round.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    Dict[str, str] mapping SequencingReads names to the read name for positive
    results from the previous. Reads without a parent count are omitted.
  """
  input_counts = {}
  for round_name, round_proto in experiment_proto.rounds.items():
    if round_proto.input:
      parent_round = experiment_proto.rounds[round_proto.input]
      input_counts[round_name] = parent_round.positive_reads.name
    else:
      input_counts[round_name] = None

  dependencies = {}
  for round_name, round_proto in experiment_proto.rounds.items():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      field = reads.name
      if field:
        parent_count = input_counts[round_name]
        if parent_count:
          dependencies[field] = parent_count
  return dependencies


def non_input_count_names(experiment_proto):
  """Return names of all counts of non-inputs from an Experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    List of strings in sorted order.
  """
  names = []
  for round_proto in experiment_proto.rounds.values():
    if round_proto.input:
      for reads in [round_proto.positive_reads, round_proto.negative_reads]:
        if reads.name:
          names.append(reads.name)
  return sorted(names)


def all_count_names(experiment_proto):
  """Return all count names from an Experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    List of strings in sorted order.
  """
  names = []
  for round_proto in experiment_proto.rounds.values():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      if reads.name:
        names.append(reads.name)
  return sorted(names)


def all_additional_output_names(experiment_proto):
  """Return all additional output names from an Experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    List of strings in sorted order.
  """
  names = [ao.name for ao in experiment_proto.additional_output]
  return sorted(names)


def binding_array_names(experiment_proto):
  """Return all binding array names from an Experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    List of strings in sorted order.
  """
  names = [array.name for array in experiment_proto.binding_arrays]
  return sorted(names)


def all_target_and_background_names(experiment_proto):
  """Determine names of all molecules for which to calculate affinities.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    List of strings giving names of target molecules followed by background
    molecules.
  """
  targets = set()
  backgrounds = set()

  for round_proto in experiment_proto.rounds.values():
    for k, v in round_proto.target_concentrations.items():
      # don't add targets with a concentration of exactly zero
      if v:
        targets.add(k)
    for k, v in round_proto.background_concentrations.items():
      if v:
        backgrounds.add(k)

  return sorted(targets) + [b for b in sorted(backgrounds) if b not in targets]


def _reads_fastq_paths(reads_proto):
  forward_paths = list(reads_proto.fastq_forward_path)
  reverse_paths = list(reads_proto.fastq_reverse_path)
  if len(forward_paths) > 1:
    raise NotImplementedError(
        'current code handles at most one FASTQ file (or pair) per round')
  if len(reverse_paths) > 1:
    raise NotImplementedError(
        'current code handles at most one FASTQ file (or pair) per round')
  return forward_paths + reverse_paths


def experiment_files_and_column_names(experiment_proto):
  """Extract files and column names from an Experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment proto describing the experimental
      configuration and results.

  Returns:
    is_paired: whether or not this sequencing was done paired
    files: list of string paths to FASTQ files with forward and reverse reads.
    column_names: list of string column names, one per every two files, in
      order. The lengths of the column_names is validated to be equal to
      the number of file pairs.

  Raises:
    NotImplementedError: if any reads are split across multiple forward or
      reverse FASTQ files.
    Error: If there are more than 2 files for the positive reads in a round.
      Or if the sequencing switches between paired and unpaired.
  """
  files = []
  column_names = []

  base_dir = experiment_proto.base_directory

  is_paired = None
  for name, round_proto in experiment_proto.rounds.items():
    positive_paths = _reads_fastq_paths(round_proto.positive_reads)
    negative_paths = _reads_fastq_paths(round_proto.negative_reads)

    if len(positive_paths) > 2:
      raise Error('There should be at most 2 files for one round but there '
                  'were more than 2 for %s' % (name))
    # check that the sequencing is consistent within the file
    if (is_paired is not None) and (is_paired != (len(positive_paths) == 2)):
      raise Error('The sequencing must be consistently paired or unpaired. '
                  'Previous round was %s but %s round is %s' %
                  (is_paired, name, len(positive_paths) == 2))
    is_paired = len(positive_paths) == 2
    column_names.append(round_proto.positive_reads.name)
    files.extend([os.path.join(base_dir, p) for p in positive_paths])

    if negative_paths:
      column_names.append(round_proto.negative_reads.name)
      files.extend([os.path.join(base_dir, p) for p in negative_paths])

  # TODO(mdimon): add this validation (probably in the validation function
  #   instead of here -- here it breaks unit tests)
  # validate that we have the right number of columns
  # if is_paired:
  #  if len(files) % 2 != 0:
  #    raise Error("FASTQ files must be in pairs but got %d files" %
  #                (len(files)))
  #  num_file_pairs = len(files) / 2
  # else:
  #  num_file_pairs = len(files)
  # if len(column_names) != num_file_pairs:
  #  raise Error("There are %d fastq file pairs, but %d column names given." %
  #              (num_file_pairs, len(column_names)))

  return is_paired, files, column_names


def compute_experiment_statistics(df, experiment_proto):
  """Compute statistics for all measurements in the given experiment.

  Args:
    df: pandas.DataFrame with columns corresponding to measurement names.
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    New selection_pb2.Experiment proto with read statitics filled in.
  """
  experiment_proto = copy.deepcopy(experiment_proto)

  for round_proto in experiment_proto.rounds.values():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      if reads.name:
        counts = df[reads.name]

        reads.statistics.total_depth = counts.sum()
        reads.statistics.num_uniques = (counts > 0).sum()
        reads.statistics.mean = counts.mean()
        # normalize by N, not N-1
        reads.statistics.std_dev = counts.std(ddof=0)

        log_counts = np.log(counts + 1)
        reads.statistics.mean_log_plus_one = log_counts.mean()
        reads.statistics.std_dev_log_plus_one = log_counts.std(ddof=0)

  return experiment_proto


def extract_measurement_statistic(experiment_proto, statistic):
  """Extract measurement statistics from a selection_pb2.Experiment.

  Args:
    experiment_proto: selection_pb2.Experiment proto with saved measurement
      statistics.
    statistic: string giving the name of the statistic to pull out of the proto.

  Returns:
    Dict[str, Number] giving statistics for each read.

  Raises:
    ValueError: if measurement statistics are missing.
  """
  output = {}
  for round_proto in experiment_proto.rounds.values():
    for reads in [round_proto.positive_reads, round_proto.negative_reads]:
      if reads.name:
        if not reads.HasField('statistics'):
          raise ValueError('missing statistics for reads: %r' % reads)
        output[reads.name] = getattr(reads.statistics, statistic)

  for feature in experiment_proto.additional_output:
    if feature.name:
      if not feature.HasField('statistics'):
        raise ValueError(
            'missing statistics for additional_output: %r' % feature)
      output[feature.name] = getattr(feature.statistics, statistic)
  return output
