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

"""Official FRMT evaluation script."""
from __future__ import annotations

import collections
from collections.abc import Collection, Mapping, Sequence
import json
from typing import Any, Optional

from absl import app
from absl import flags
from absl import logging
import attrs
import bleurt.score as bleurt_lib
from etils import epath
import pandas

from frmt import evaluation

BleurtScorer = bleurt_lib.LengthBatchingBleurtScorer
Metrics = evaluation.Metrics

# ==============================================================================
# Flags
# ==============================================================================


PREDICTION_FILES = flags.DEFINE_list(
    'prediction_files',
    default=None,
    help=(
        'Path to the model prediction file. Should be in .txt or .tsv '
        'format. Each model output should be on its own line, aligned to the '
        'reference file. In the case of .tsv files, the English source '
        'should be in the first column, and the translation in the second; '
        'this will allow the program to check that the predictions are '
        'aligned to the references.'
    ),
    required=True,
)

DATASET_DIR = flags.DEFINE_string(
    'dataset_dir',
    default='./frmt/dataset',
    help='Path to the FRMT reference directory.',
)

SPLIT = flags.DEFINE_enum(
    'split',
    default='dev',
    enum_values=['dev', 'test'],
    help='Which data split (dev or test) to evaluate on.',
)

LANGUAGE = flags.DEFINE_enum(
    'language',
    default=None,
    enum_values=[
        'pt',
        'pt-BR',
        'pt-PT',
        'zh',
        'zh-CN',
        'zh-TW',
        'zh-TW_Simplified',
    ],
    help=(
        'Which language to evaluate on. If region code is unspecified, '
        'evaluates against regions and scripts for the provided language.'
    ),
    required=True,
)

BUCKET = flags.DEFINE_enum(
    'bucket',
    default=None,
    enum_values=['lexical', 'entity', 'random'],
    help='Which bucket to evaluate.',
    required=True,
)

EVAL_METRICS = flags.DEFINE_multi_enum_class(
    'metric',
    default='bleu',
    enum_class=evaluation.MetricType,
    help='Which evaluation metrics to compute. Case-insensitive.',
)

BLEURT_CHECKPOINT_DIR = flags.DEFINE_string(
    'bleurt_checkpoint_dir',
    default='./bleurt/checkpoints',
    help=(
        'Directory where BLEURT checkpoints are stored, with original '
        'checkpoint names.'
    ),
)

OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    default=None,
    help=(
        'Where to save the results--can be .txt, .csv, .tsv, or .json. If '
        'empty, results are printed to stdout.'
    ),
)

# ==============================================================================
# Data structures
# ==============================================================================


@attrs.frozen(eq=True, kw_only=True, order=True, slots=True)
class FilePair:
  """Container class for a prediction/reference file pair."""

  prediction_path: epath.Path = attrs.field(converter=epath.Path)
  reference_path: epath.Path = attrs.field(converter=epath.Path)
  bucket: str

  def as_dict(self):
    d = attrs.asdict(self)
    ordered_dict = collections.OrderedDict()
    for key in self.__slots__:  # pytype: disable=attribute-error
      if key not in d:  # E.g. '__weakref__'
        continue
      value = d[key]
      ordered_dict[key] = value.name if isinstance(value, epath.Path) else value
    return ordered_dict


# ==============================================================================
# Helper functions
# ==============================================================================


def _list_file_pairs(
    prediction_files,
    dataset_dir,
    *,
    split,
    bucket,
    language,
):
  """Gathers all the predictions/references we want to evaluate."""
  file_pairs = []
  dataset_path = epath.Path(dataset_dir)
  bucket_path = dataset_path / f'{bucket}_bucket'
  primary_language = language.split('-')[0]
  match_name = f'{primary_language}_{bucket}_{split}_en_{language}'
  for prediction_file in prediction_files:
    for reference_path in bucket_path.iterdir():
      if str(reference_path.name).startswith(match_name):
        file_pairs.append(
            FilePair(
                bucket=bucket,
                prediction_path=prediction_file,
                reference_path=reference_path,
            )
        )
  return file_pairs


def _read_tsv(file_path):
  """Reads a csv with two columns (source, translation) and no header."""
  translation_pairs = []
  with file_path.open() as f:
    # Note: the correct way to do this is with csv.DictReader, but some examples
    # have quote characters that confuse the csv parser. Since we know the
    # source never has its own tab or newline characters, basic Python string
    # manipulation is fine here, as long as the model doesn't predict tabs or
    # newlines.
    for line in f:
      line = line.strip()
      line = line.split('\t')
      if len(line) != 2:
        raise ValueError(
            f'Line {line} could not be parsed. You may need to manually '
            'replace tab or newline characters in the model output with '
            'spaces.'
        )
      source, translation = line
      translation_pairs.append(
          evaluation.TranslationPair(source=source, translation=translation)
      )
  return translation_pairs


def _read_txt(file_path):
  """Reads a txt file with translations (no source) on each line."""
  translation_pairs = []
  with file_path.open() as f:
    for line in f:
      translation_pairs.append(
          evaluation.TranslationPair(source=None, translation=line.strip())
      )
  return translation_pairs


def _read_predictions_and_references(
    file_pair,
):
  """Read in the predictions and references.

  Args:
    file_pair: The FilePair object containing the prediction and reference
      files.

  Returns:
    A tuple containing the model predictions and gold references (respectively)
      in the two files.
  """
  read_predictions_fn = {
      '.txt': _read_txt,
      '.tsv': _read_tsv,
  }.get(file_pair.prediction_path.suffix)
  if read_predictions_fn is None:
    raise ValueError(
        f'Predictions file `{file_pair.prediction_path}` has unsupported '
        f'suffix `{file_pair.prediction_path.suffix}`. Supported values '
        'are ".txt" and ".tsv".'
    )

  predictions = read_predictions_fn(file_pair.prediction_path)
  references = _read_tsv(file_pair.reference_path)
  if len(predictions) != len(references):
    prediction_sources = set(prediction.source for prediction in predictions)
    reference_sources = set(reference.source for reference in references)
    non_references = list(prediction_sources.difference(reference_sources))
    non_predictions = list(reference_sources.difference(prediction_sources))
    raise ValueError(
        f'{file_pair} has {len(predictions)} predictions but {len(references)} '
        'references (should be equal). Sample of 5 prediction sources not in '
        f'references: {non_references[:5]}. Sample of 5 reference sources not '
        f'in predictions: {non_predictions[:5]}.'
    )
  return predictions, references


def _records_to_string(records):
  """Creates a human-readable string representing a sequence of records."""
  parts = []
  for record in records:
    parts.append('\n'.join(f'{k}: {v}' for k, v in record.items()))
  return '\n\n'.join(parts) + '\n'


def _write_txt(
    output_path, records
):
  """Writes a collection of records to text."""
  output_path.write_text(_records_to_string(records))


def _write_json(
    output_path, records
):
  """Writes a collection of records to json."""
  output_path.write_text(json.dumps(records))


def _write_tsv(
    output_path,
    records,
):
  """Writes a collection of records to tsv."""
  df = pandas.DataFrame(records)
  output_path.write_text(df.to_csv(index=False, sep='\t'))


def _write_output(
    all_metrics,
    output_path,
):
  """Writes the output to file specified by the user."""
  records = []
  for file_pair, metrics in all_metrics.items():
    records.append(
        collections.OrderedDict(
            **file_pair.as_dict(),
            **metrics.as_dict(),
        )
    )

  if output_path is None:
    s = _records_to_string(records)
    logging.info(s)
    print(s)
    return

  write_metrics_fn = {
      '.txt': _write_txt,
      '.json': _write_json,
      '.tsv': _write_tsv,
  }.get(output_path.suffix)
  if write_metrics_fn is None:
    raise ValueError(
        f'Output path `{output_path}` has unsupported suffix '
        f'`{output_path.suffix}`. Supported values are ".txt", ".json", and '
        '".tsv".'
    )
  write_metrics_fn(output_path, records)


# ==============================================================================
# Main
# ==============================================================================


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  output_path = (
      epath.Path(OUTPUT_FILE.value) if OUTPUT_FILE.value is not None else None
  )
  if output_path is not None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

  if output_path is not None and output_path.suffix not in [
      '.txt',
      '.json',
      '.tsv',
  ]:
    raise ValueError(
        f'Output path `{output_path}` has unsupported suffix '
        f'`{output_path.suffix}`. Supported values are ".txt", ".json", and '
        '".tsv".'
    )

  # Enumerate all the prediction/reference pairs we want to evaluate.
  file_pairs = _list_file_pairs(
      PREDICTION_FILES.value,
      DATASET_DIR.value,
      split=SPLIT.value,
      bucket=BUCKET.value,
      language=LANGUAGE.value,
  )
  logging.info(
      'Running evaluation on the following prediction/reference pairs: %s',
      '\n'.join(map(str, file_pairs)),
  )

  # Run evaluation on all the input pairs.
  all_metrics: dict[FilePair, evaluation.Metrics] = collections.OrderedDict()
  if BLEURT_CHECKPOINT_DIR.value is not None:
    bleurt_scorer_cache = evaluation.BleurtScorerCache(
        BLEURT_CHECKPOINT_DIR.value
    )
  else:
    bleurt_scorer_cache = None
  for file_pair in file_pairs:
    predictions, references = _read_predictions_and_references(file_pair)
    all_metrics[file_pair] = evaluation.evaluate(
        predictions=predictions,
        references=references,
        eval_metrics=EVAL_METRICS.value,
        language=LANGUAGE.value,
        bleurt_scorer_cache=bleurt_scorer_cache,
    )

  _write_output(all_metrics=all_metrics, output_path=output_path)


if __name__ == '__main__':
  app.run(main)
