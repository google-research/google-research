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
import re

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

  # The two input patterns identify corresponding files through the same
  # wildcard value, not their unrelated lexical ordering.
  pairs = _pair_files_by_wildcard(
      target_filepattern, prediction_filepattern,
      target_filenames, prediction_filenames)
  scores = _compute_scores([target for target, _ in pairs],
                           [prediction for _, prediction in pairs],
                           scorer, delimiter)
  if aggregator:
    for score in scores:
      aggregator.add_scores(score)
    _write_aggregates_to_csv(output_filename, aggregator.aggregate())
  else:
    _write_scores_to_csv(output_filename, scores)


def _wildcard_pattern(filepattern):
  """Builds a regex to capture the wildcard portions of a glob pattern.

  A key may include multiple wildcard segments, including those in parent
  directories. Bracket and question-mark globs are also supported.
  """
  regex_parts = []
  index = 0
  while index < len(filepattern):
    character = filepattern[index]
    if character == "*":
      regex_parts.append("(.*?)")
    elif character == "?":
      regex_parts.append("(.)")
    elif character == "[":
      end = index + 1
      if end < len(filepattern) and filepattern[end] == "!":
        end += 1
      if end < len(filepattern) and filepattern[end] == "]":
        end += 1
      while end < len(filepattern) and filepattern[end] != "]":
        end += 1
      if end == len(filepattern):
        regex_parts.append(re.escape(character))
      else:
        contents = filepattern[index + 1:end].replace("\\\\", r"\\")
        if contents.startswith("!"):
          contents = "^" + contents[1:]
        elif contents.startswith("^"):
          contents = "\\" + contents
        regex_parts.append("([" + contents + "])")
        index = end
    else:
      regex_parts.append(re.escape(character))
    index += 1
  return re.compile("^" + "".join(regex_parts) + "$")


def _pair_files_by_wildcard(target_pattern, prediction_pattern,
                            target_filenames, prediction_filenames):
  """Pairs files using the portions matched by corresponding glob wildcards.

  Unlike independently sorting the two lists, this fails rather than silently
  evaluating unrelated documents if the naming schemes disagree.
  """
  if not glob.has_magic(target_pattern) or not glob.has_magic(prediction_pattern):
    # Preserve legacy behavior for two explicitly named single files.
    if len(target_filenames) == len(prediction_filenames) == 1:
      return [(target_filenames[0], prediction_filenames[0])]
    raise ValueError("Both file patterns must contain matching wildcards.")

  def key_to_file(filepattern, filenames):
    matcher = _wildcard_pattern(filepattern)
    mapping = {}
    for filename in filenames:
      match = matcher.fullmatch(filename)
      if match is None:
        raise ValueError("File %r does not match pattern %r." %
                         (filename, filepattern))
      identifier = match.groups()
      if identifier in mapping:
        raise ValueError("Duplicate wildcard identifier %r for %r and %r." %
                         (identifier, mapping[identifier], filename))
      mapping[identifier] = filename
    return mapping

  targets = key_to_file(target_pattern, target_filenames)
  predictions = key_to_file(prediction_pattern, prediction_filenames)
  if targets.keys() != predictions.keys():
    missing_predictions = sorted(set(targets) - set(predictions))
    missing_targets = sorted(set(predictions) - set(targets))
    raise ValueError(
        "Target and prediction files must have matching wildcard "
        "identifiers. No predictions for: %s; no targets for: %s." %
        (missing_predictions, missing_targets))

  return [(targets[key], predictions[key]) for key in sorted(targets)]


def _glob(filepattern):
  return glob.glob(filepattern)  # pylint: disable=unreachable


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


def _compute_scores(target_filenames, prediction_filenames, scorer, delimiter):
  """Computes aggregates scores across the given target and prediction files.

  Args:
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
  for target_filename, prediction_filename in zip(
      target_filenames, prediction_filenames):
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
