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

"""Shared functions for preprocessing sequencing count data for TensorFlow.
"""
import copy
import hashlib
import os.path

import numpy
import pandas
from sklearn import cross_validation

from xxx import example_pb2
from xxx import feature_pb2

# Google internal
import text_format
import gfile
import logging
import sstable

from ..util import dna
from ..util import selection
from ..util import selection_pb2
from ..util import io_utils
from ..learning import config
from ..preprocess import utils


class Error(Exception):
  pass


def write_sstable(df,
                  path,
                  fwd_primer=None,
                  rev_primer=None):
  """Writes the dataframe out to path as tensorflow example protos.

  Args:
    df: the dataframe to write. Expected to have the sequence string as the
      'index' and to have a 'cluster' column. Each of the count_columns is
      expected to exist and have an integer count of instances of the sequence.
      Sequences are expected to have only valid bases (A, C, G, and T).
    path: the filepath for the sstable.
    fwd_primer: String indicating the forward primer.
    rev_primer: String indicating the reverse primer.
  Raises:
    Error: if a sequence string contains an illegal (non-ACGT) base.
  """

  # iterate through the dataframe getting each row as a dictionary
  # (used a dictionary instead of a tuple to access the count_columns by name)
  with sstable.SortingBuilder(path) as builder:
    for record in utils.iterdictionary(df):
      sequence = record["index"]

      # fail noisily here (bad sequences should have been removed earlier)
      if dna.has_invalid_bases(sequence):
        raise Error("Illegal base in record: %s" % (sequence))

      features = {}
      features["sequence"] = sequence
      for col_name in df.columns:
        # TODO(shoyer): consider filtering the columns we put into Example
        # protos before we try to write this table.
        features[col_name] = record[col_name]

      key = hashlib.sha1(sequence).hexdigest()
      proto = tensorflow_example(features)
      builder.Add(key, proto.SerializeToString())


def tensorflow_feature(value):
  """Create a TensorFlow feature from a scalar or array.

  Args:
    value: array_like value (coercible with numpy.array) to convert into a
      feature. Must have integer, float or bytes dtype.

  Returns:
    feature_pb2.Feature with the given value.
  """
  array = numpy.ravel(value)
  list_value = [v.item() for v in array.flat]
  if array.dtype.kind == "i":
    kwargs = dict(int64_list=feature_pb2.Int64List(value=list_value))
  elif array.dtype.kind == "f":
    kwargs = dict(float_list=feature_pb2.FloatList(value=list_value))
  elif array.dtype.kind == "S":
    kwargs = dict(bytes_list=feature_pb2.BytesList(value=list_value))
  else:
    raise TypeError("unable to handle dtype %s" % value.dtype)

  return feature_pb2.Feature(**kwargs)


def tensorflow_example(features_mapping):
  """Create a TensorFlow example from a mapping.

  Args:
    features_mapping: Dict[str, array_like] mapping from feature names to
      array_like values (coercible with numpy.array).

  Returns:
    example_pb2.Example combining all the given features.

  Raises:
    ValueError: If any values have some but not all null values.
  """
  features = {}
  for k, v in features_mapping.iteritems():
    fraction_null = numpy.mean(pandas.isnull(v))
    if fraction_null == 0:
      features[k] = tensorflow_feature(v)
    elif fraction_null != 1:
      # it's OK to skip all nulls, but not mixed nulls/non-nulls
      raise ValueError("feature %r has some but not all null values: %r" % (k,
                                                                            v))

  return example_pb2.Example(features=feature_pb2.Features(feature=features))


def update_experiment_read_counts(experiment_proto, count_df):
  """Update an selection_pb2.Experiment proto with read counts inplace."""
  for round_proto in experiment_proto.rounds.itervalues():
    for reads_proto in [round_proto.positive_reads, round_proto.negative_reads]:
      column_name = reads_proto.name
      if column_name in count_df.columns:
        reads_proto.depth = int(count_df[column_name].sum())


def _remove_fastq_paths(experiment_proto):
  """Remove FASTQ paths in an selection_pb2.Experiment proto."""
  for round_proto in experiment_proto.rounds.itervalues():
    del round_proto.positive_reads.fastq_forward_path[:]
    del round_proto.positive_reads.fastq_reverse_path[:]
    del round_proto.negative_reads.fastq_forward_path[:]
    del round_proto.negative_reads.fastq_reverse_path[:]
  experiment_proto.ClearField("base_directory")


def split_and_write(count_df, experiment_proto, save_dir):
  """Splits into folds, write each test fold to an sstable.

  Only the test folds are written out so the train table for any fold is
  all the other sstables. This limits re-writing the data multiple times.

  Args:
    count_df: dataframe where the index is the sequence and the rest of the
      columns are counts in each round of seletion.
    experiment_proto: selection_pb2.Experiment proto describing the experimental
      configuration and results.
    save_dir: string base name for the directory in which to save output.
  """
  experiment_proto = copy.deepcopy(experiment_proto)
  # These FASTQ files describe all the experiment data, not the split train /
  # test data.
  _remove_fastq_paths(experiment_proto)

  label_kfold = cross_validation.LabelKFold(count_df.cluster, n_folds=5)
  for i, (train, test) in enumerate(label_kfold):
    logging.info("Fold %d has %d train and %d test", i, len(train), len(test))

    test_counts = count_df.iloc[test]
    train_counts = count_df.iloc[train]

    for split_name, subcounts in [("test", test_counts), ("train",
                                                          train_counts)]:
      update_experiment_read_counts(experiment_proto, subcounts)
      path = os.path.join(save_dir,
                          "experiment_fold_%d_%s.pbtxt" % (i, split_name))
      with gfile.GFile(path, "w") as f:
        f.write(text_format.MessageToString(experiment_proto))

    # HDF5 can be quickly read and writen from Python
    logging.info("Saving count table as HDF5.")
    path = os.path.join(save_dir, "table_fold_%d.h5" % i)
    io_utils.write_dataframe_to_hdf5(test_counts, path)

    # we use the sstable of examples for TensorFlow
    logging.info("Saving SSTable of TensorFlow example protos.")
    path = os.path.join(save_dir, "examples_fold_%d.sstable" % i)
    write_sstable(test_counts, path,
                  experiment_proto.forward_primer,
                  experiment_proto.reverse_primer)


def filter_example_sstables(output_dir, input_dir, update_func):
  """Given a set of SSTables of TensorFlow examples, updates or filters them.

  Args:
    output_dir: directory where the updated/filtered sstables should be written.
    input_dir: directory containing the input sstables
    update_func: function to do updating. Takes a dataframe of original data,
      returns a dataframe with updated/filtered data. Returned dataframe should
      be valid input to the sequencing_counts.write_sstable function.
  """
  gfile.MakeDirs(output_dir)
  # The relevant parameters in the experiment proto (conditions, concentration,
  # sequence length) are identical for all folds and train/validation. Choose
  # one randomly as a template.
  experiment_path = os.path.join(input_dir,
                                 config.wetlab_experiment_train_pbtxt_path[0])
  with gfile.FastGFile(experiment_path) as f:
    experiment_proto = text_format.Parse(f.read(), selection_pb2.Experiment())
  names = selection.all_count_names(experiment_proto)

  summary_data = []
  for test_pbtxt, sstable_path, hdf5_path in zip(
      config.wetlab_experiment_val_pbtxt_path, config.example_sstable_paths,
      config.hdf5_paths):
    logging.info("Starting fold %s", test_pbtxt)
    fold_input_hdf5_path = os.path.join(input_dir, hdf5_path)
    fold_input_df = io_utils.read_dataframe_from_hdf5(fold_input_hdf5_path)

    fold_output_df = update_func(fold_input_df, names)
    logging.info("%s", fold_output_df.head(10))
    summary_data.append(fold_output_df.sum(axis=0))

    # Write out the data for this fold.
    fold_output_proto = copy.deepcopy(experiment_proto)
    update_experiment_read_counts(fold_output_proto, fold_output_df)
    fold_output_pbtxt_path = os.path.join(output_dir, test_pbtxt)
    with gfile.GFile(fold_output_pbtxt_path, "w") as f:
      f.write(text_format.MessageToString(fold_output_proto))

    fold_output_hdf5_path = os.path.join(output_dir, hdf5_path)
    io_utils.write_dataframe_to_hdf5(fold_output_df, fold_output_hdf5_path)

    fold_output_sstable_path = os.path.join(output_dir, sstable_path)
    write_sstable(fold_output_df, fold_output_sstable_path,
                  fold_output_proto.forward_primer,
                  fold_output_proto.reverse_primer)

  # Now write out the training experiment protos.
  for i, train_pbtxt in enumerate(config.wetlab_experiment_train_pbtxt_path):
    train_counts_df = pandas.concat(
        summary_data[:i] + summary_data[i + 1:], axis=1).T
    train_output_proto = copy.deepcopy(experiment_proto)
    update_experiment_read_counts(train_output_proto, train_counts_df)
    output_train_path = os.path.join(output_dir, train_pbtxt)
    with gfile.GFile(output_train_path, "w") as f:
      f.write(text_format.MessageToString(train_output_proto))
