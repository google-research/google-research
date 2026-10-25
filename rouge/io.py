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

"""Library for reading/writing input and score files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

from absl import logging
import six
from six.moves import zip
from six.moves import zip_longest



def compute_scores_and_write_to_csv(target_filepattern,
                                    prediction_filepattern,
                                    output_filename,
                                    scorer,
                                    aggregator,
                                    delimiter="\n"):
  """Runs aggregate score calculations and outputs results to a CSV file.

  Args:
    target_filepattern: Pattern for files containing target text.
    prediction_filepattern: Pattern for files containing prediction text.
    output_filename: Name of file to write results to.
    scorer: A BaseScorer object to compute scores.
    aggregator: An aggregator to aggregate scores. If None, outputs are
      per-example scores.
    delimiter: Record delimiter.
  """

  target_filenames = _glob(target_filepattern)
  prediction_filenames = _glob(prediction_filepattern)
  if (len(target_filenames) < 1 or
      len(target_filenames) != len(prediction_filenames)):
    raise ValueError("Must have equal and positive number of target and "
                     "prediction files. Found: %d target files (%s),"
                     " %d prediction files (%s)." %
                     (len(target_filenames), target_filepattern,
                      len(prediction_filenames), prediction_filepattern))

  scores = _compute_scores(target_filepattern, prediction_filepattern,
                           target_filenames, prediction_filenames, scorer,
                           delimiter)
  if aggregator:
    for score in scores:
      aggregator.add_scores(score)
    _write_aggregates_to_csv(output_filename, aggregator.aggregate())
  else:
    _write_scores_to_csv(output_filename, scores)


def _glob(filepattern):
  return glob.glob(filepattern)  # pylint: disable=unreachable


# Characters with special meaning in a glob pattern.
_GLOB_MAGIC = "*?["


def _literal_affixes(filepattern):
  """Returns the literal prefix and suffix around the wildcards of a pattern.

  For example, "/data/ref_*.txt" gives ("/data/ref_", ".txt").

  Args:
    filepattern: A glob pattern.

  Returns:
    A (prefix, suffix) tuple of literal strings.
  """
  first_magic = len(filepattern)
  for magic in _GLOB_MAGIC:
    index = filepattern.find(magic)
    if index != -1:
      first_magic = min(first_magic, index)

  last_magic = -1
  for magic in _GLOB_MAGIC:
    index = filepattern.rfind(magic)
    if index != -1:
      last_magic = max(last_magic, index)
  # A character class runs through its closing bracket.
  if "[" in filepattern and "]" in filepattern:
    last_magic = max(last_magic, filepattern.rfind("]"))

  prefix = filepattern[:first_magic]
  suffix = "" if last_magic == -1 else filepattern[last_magic + 1:]
  return prefix, suffix


def _pair_key(filepattern, filename):
  """Returns the part of `filename` that the wildcards of `filepattern` matched.

  This is the identity used to pair a target file with its prediction file.
  Keys are normalized so that zero-padded and unpadded ids compare equal, which
  means "ref_1.txt" and "out_01.txt" pair with each other.

  Args:
    filepattern: The glob pattern that produced `filename`.
    filename: A path matched by `filepattern`.

  Returns:
    A string key, or None if `filename` does not have the literals of
    `filepattern` around its wildcard section.
  """
  prefix, suffix = _literal_affixes(filepattern)
  if not filename.startswith(prefix) or not filename.endswith(suffix):
    return None
  if len(filename) < len(prefix) + len(suffix):
    return None
  key = filename[len(prefix):len(filename) - len(suffix)]
  if key.isdigit():
    # Treat "007" and "7" as the same example id.
    return key.lstrip("0") or "0"
  return key


def _pair_filenames(target_filepattern, prediction_filepattern,
                    target_filenames, prediction_filenames):
  """Pairs target and prediction files by the ids captured by their patterns.

  Files are matched on the text their patterns' wildcards matched, so a target
  and a prediction are only ever scored against each other when they refer to
  the same example. Every example must have exactly one file on each side.

  Args:
    target_filepattern: Pattern that produced `target_filenames`.
    prediction_filepattern: Pattern that produced `prediction_filenames`.
    target_filenames: List of target filenames.
    prediction_filenames: List of prediction filenames.

  Returns:
    A list of (target_filename, prediction_filename) pairs, ordered by target
    filename.

  Raises:
    ValueError: If an example has a file on only one side, a pattern captures
      no id, or two files capture the same id.
  """
  def _index(filepattern, filenames):
    """Maps id -> filename for one side."""
    index = {}
    for filename in filenames:
      key = _pair_key(filepattern, filename)
      if key is None:
        raise ValueError(
            "Cannot determine which example %s belongs to: pattern %r has "
            "literal sections that the filename does not contain." %
            (filename, filepattern))
      if key in index:
        raise ValueError(
            "Pattern %r captures id %r for both %s and %s, so files cannot be "
            "paired unambiguously." % (filepattern, key, index[key], filename))
      index[key] = filename
    return index

  target_index = _index(target_filepattern, target_filenames)
  prediction_index = _index(prediction_filepattern, prediction_filenames)

  unmatched_targets = sorted(set(target_index) - set(prediction_index))
  unmatched_predictions = sorted(set(prediction_index) - set(target_index))
  if unmatched_targets or unmatched_predictions:
    raise ValueError(
        "Target and prediction files do not describe the same examples, so "
        "they cannot be paired. Example ids only in targets: %s. Example ids "
        "only in predictions: %s. The id of a file is the part matched by the "
        "wildcards of its pattern (%r and %r). Narrow the patterns so that "
        "only the example id is matched, for example "
        "--target_filepattern='*.targets' and "
        "--prediction_filepattern='*.decodes'." %
        (unmatched_targets[:10], unmatched_predictions[:10],
         target_filepattern, prediction_filepattern))

  return [(target_index[key], prediction_index[key])
          for key in sorted(target_index, key=target_index.get)]


def _open(filepattern, mode="r"):
  return open(filepattern, mode)  # pylint: disable=unreachable


def _record_gen(filename, delimiter):
  """Opens file and yields records separated by delimiter."""
  with _open(filename) as f:
    records = f.read().split(six.ensure_str(delimiter))
  if records[-1]:
    # Need a final delimiter at end of file to be able to detect an empty last
    # record.
    logging.warn("Expected delimiter at end of file")
  else:
    records = records[:-1]
  for record in records:
    yield record


def _compute_scores(target_filepattern, prediction_filepattern,
                    target_filenames, prediction_filenames, scorer, delimiter):
  """Computes aggregates scores across the given target and prediction files.

  Target and prediction files are paired by the example id captured by their
  respective patterns, not by their position in a sorted listing, so a naming
  difference between the two sides cannot cause the wrong pairs to be scored.

  Args:
    target_filepattern: Pattern that produced `target_filenames`.
    prediction_filepattern: Pattern that produced `prediction_filenames`.
    target_filenames: List of filenames from which to read target lines.
    prediction_filenames: List of filenames from which to read prediction lines.
    scorer: A BaseScorer object to compute scores.
    delimiter: string delimiter between each record in input files

  Returns:
    A list of dicts mapping score_type to Score objects.
  Raises:
    ValueError: If invalid targets or predictions are provided.
  """

  scores = []
  for target_filename, prediction_filename in _pair_filenames(
      target_filepattern, prediction_filepattern, target_filenames,
      prediction_filenames):
    logging.info("Reading targets from %s.", target_filename)
    logging.info("Reading predictions from %s.", prediction_filename)
    targets = _record_gen(target_filename, delimiter)
    preds = _record_gen(prediction_filename, delimiter)
    for target_rec, prediction_rec in zip_longest(targets, preds):
      if target_rec is None or prediction_rec is None:
        raise ValueError("Must have equal number of lines across target and "
                         "prediction files. Mismatch between files: %s, %s." %
                         (target_filename, prediction_filename))
      scores.append(scorer.score(target_rec, prediction_rec))

  return scores


def _write_aggregates_to_csv(output_filename, aggregates):
  """Writes aggregate scores to an output CSV file.

  Output file is a comma separated where each line has the format:
    score_type-(P|R|F),low_ci,mean,high_ci

  P/R/F indicates whether the score is a precision, recall or f-measure.

  Args:
    output_filename: Name of file to write results to.
    aggregates: A dict mapping each score_type to a AggregateScore object.
  """

  logging.info("Writing results to %s.", output_filename)
  with _open(output_filename, "w") as output_file:
    output_file.write("score_type,low,mid,high\n")
    for score_type, aggregate in sorted(aggregates.items()):
      output_file.write("%s-R,%f,%f,%f\n" %
                        (score_type, aggregate.low.recall, aggregate.mid.recall,
                         aggregate.high.recall))
      output_file.write("%s-P,%f,%f,%f\n" %
                        (score_type, aggregate.low.precision,
                         aggregate.mid.precision, aggregate.high.precision))
      output_file.write("%s-F,%f,%f,%f\n" %
                        (score_type, aggregate.low.fmeasure,
                         aggregate.mid.fmeasure, aggregate.high.fmeasure))
  logging.info("Finished writing results.")


def _write_scores_to_csv(output_filename, scores):
  """Writes scores for each individual example to an output CSV file.

  Output file is a comma separated where each line has the format:
    id,score1,score2,score3,...

  The header row indicates the type of each score column.

  Args:
    output_filename: Name of file to write results to.
    scores: A list of dicts mapping each score_type to a Score object.
  """

  if len(scores) < 1:
    logging.warn("No scores to write")
    return
  rouge_types = sorted(scores[0].keys())

  logging.info("Writing results to %s.", output_filename)
  with _open(output_filename, "w") as out_file:
    out_file.write("id")
    for rouge_type in rouge_types:
      out_file.write(",{t}-P,{t}-R,{t}-F".format(t=rouge_type))
    out_file.write("\n")
    for i, result in enumerate(scores):
      out_file.write("%d" % i)
      for rouge_type in rouge_types:
        out_file.write(",%f,%f,%f" %
                       (result[rouge_type].precision, result[rouge_type].recall,
                        result[rouge_type].fmeasure))
      out_file.write("\n")
  logging.info("Finished writing results.")
