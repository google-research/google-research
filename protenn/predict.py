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

"""Functionally annotate a fasta file.

Write Pfam domain predictions as a TSV with columns
- sequence_name (string)
- predicted_label (string)
- start (int, 1-indexed, inclusive)
- end (int, 1-indexed, inclusive)
- label_description (string); a human-readable label description.
"""

import io
import json
import logging
import os
from typing import Dict, List, Optional

from absl import app
from absl import flags
from Bio.SeqIO import FastaIO
import pandas as pd
import tqdm

from protenn import inference_lib
from protenn import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF c++ logging set to ERROR
import tensorflow.compat.v1 as tf  # pylint: disable=g-import-not-at-top,g-bad-import-order


_logger = logging.getLogger('protenn')


_INPUT_FASTA_FILE_PATH_FLAG = flags.DEFINE_string(
    'i', None, 'Input fasta file path.'
)
_OUTPUT_WRITE_PATH_FLAG = flags.DEFINE_string(
    'o',
    '/dev/stdout',
    'Output write path. Default is to print to the terminal.',
)

_NUM_ENSEMBLE_ELEMENTS_FLAG = flags.DEFINE_integer(
    'num_ensemble_elements',
    1,
    'In order to run with more than one ensemble element, you will need to run '
    'install_models.py --install_ensemble=true. '
    'More ensemble elements takes more time, but tends to be more accurate. '
    'Run-time scales linearly with the number of ensemble elements. '
    'Maximum value of this flag is {}.'.format(
        utils.MAX_NUM_ENSEMBLE_ELS_FOR_INFERENCE
    ),
)
_MIN_DOMAIN_CALL_LENGTH_FLAG = flags.DEFINE_integer(
    'min_domain_call_length',
    20,
    "Don't consider any domain calls valid that are shorter than this length.",
)
_REPORTING_THRESHOLD_FLAG = flags.DEFINE_float(
    'reporting_threshold',
    0.025,
    'Number between 0 (exclusive) and 1 (inclusive). Predicted labels with '
    'confidence at least resporting_threshold will be included in the output.',
    lower_bound=1e-30,
    upper_bound=1.0,
)

_MODEL_CACHE_PATH_FLAG = flags.DEFINE_string(
    'model_cache_path',
    os.path.join(os.path.expanduser('~'), 'cached_models'),
    'Path from which to use downloaded models and metadata.',
)

# A list of inferrers that all have the same label set.
_InferrerEnsemble = List[inference_lib.Inferrer]


def _gcs_path_to_relative_unzipped_path(p):
  """Parses GCS path, to gets the last part, and removes .tar.gz."""
  return os.path.join(
      os.path.basename(os.path.normpath(p)).replace('.tar.gz', ''))


def _get_inferrer_paths(
    model_urls, model_cache_path
):
  """Convert list of model GCS urls to a list of locally cached paths."""
  return [
      os.path.join(model_cache_path, _gcs_path_to_relative_unzipped_path(p))
      for p in model_urls
  ]


def load_models(
    model_cache_path, num_ensemble_elements
):
  """Load models from cache path into inferrerLists.

  Args:
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models. switched from list of list of models to just list of model
    num_ensemble_elements: number of ensemble elements of each type to load.

  Returns:
    list_of_inferrers

  Raises:
    ValueError if the models were not found. The exception message describes
    that install_models.py needs to be rerun.
  """
  try:
    pfam_inferrer_paths = _get_inferrer_paths(
        utils.OSS_PFAM_ZIPPED_MODELS_URLS, model_cache_path
    )

    to_return = []
    for p in tqdm.tqdm(
        pfam_inferrer_paths[:num_ensemble_elements],
        desc='Loading models',
        position=0,
        leave=True,
        dynamic_ncols=True,
    ):
      to_return.append(inference_lib.Inferrer(p, use_tqdm=False))

    return to_return

  except tf.errors.NotFoundError as exc:
    err_msg = 'Unable to find cached models in {}.'.format(model_cache_path)
    if num_ensemble_elements > 1:
      err_msg += (
          ' Make sure you have installed the entire ensemble of models by '
          'running\n    install_models.py --install_ensemble '
          '--model_cache_path={}'.format(model_cache_path))
    else:
      err_msg += (
          ' Make sure you have installed the models by running\n    '
          'install_models.py --model_cache_path={}'.format(model_cache_path))
    err_msg += '\nThen try rerunning this script.'

    raise ValueError(err_msg) from exc


def _assert_fasta_parsable(input_text):
  with io.StringIO(initial_value=input_text) as f:
    fasta_itr = FastaIO.FastaIterator(f)
    end_iteration_sentinel = object()

    # Avoid parsing the entire FASTA contents by using `next`.
    # A malformed FASTA file will have no entries in its FastaIterator.
    # This is unfortunate (instead of it throwing an error).
    if next(fasta_itr, end_iteration_sentinel) is end_iteration_sentinel:
      raise ValueError('Failed to parse any input from fasta file. '
                       'Consider checking the formatting of your fasta file. '
                       'First bit of contents from the fasta file was\n'
                       '{}'.format(input_text.splitlines()[:3]))


def parse_input_to_text(input_fasta_path):
  """Parses input fasta file.

  Args:
    input_fasta_path: path to FASTA file.

  Returns:
    Contents of file as a string.

  Raises:
    ValueError if parsing the FASTA file gives no records.
  """
  _logger.info('Parsing input from %s', input_fasta_path)
  with tf.io.gfile.GFile(input_fasta_path, 'r') as input_file:
    input_text = input_file.read()

  _assert_fasta_parsable(input_text=input_text)
  return input_text


def input_text_to_df(input_text):
  """Converts fasta contents to a df with columns sequence_name and sequence."""
  with io.StringIO(initial_value=input_text) as f:
    fasta_records = list(FastaIO.FastaIterator(f))
    fasta_df = pd.DataFrame([(f.name, str(f.seq)) for f in fasta_records],
                            columns=['sequence_name', 'sequence'])

  return fasta_df


def perform_inference(
    input_df,
    models,
    model_cache_path,
    reporting_threshold,
    min_domain_call_length,
):
  """Perform inference for Pfam using given models.

  Args:
    input_df: pd.DataFrame with columns sequence_name (str) and sequence (str).
    models: list of Pfam inferrers
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
    min_domain_call_length: don't consider as valid any domain calls shorter
      than this length.

  Returns:
    df with columns sequence_name (str), predicted_label (str), start(int),
    end (int), description (str).
  """
  predictions = inference_lib.get_preds_at_or_above_threshold(
      input_df=input_df,
      inferrer_list=models,
      model_cache_path=model_cache_path,
      reporting_threshold=reporting_threshold,
      min_domain_call_length=min_domain_call_length,
  )

  print('\n')  # Because the tqdm bar is position 1, we need to print a newline.

  to_return_df = []
  for sequence_name, single_seq_predictions in zip(
      input_df.sequence_name, predictions
  ):
    for label, (start, end) in single_seq_predictions:
      to_return_df.append({
          'sequence_name': sequence_name,
          'predicted_label': label,
          'start': start,
          'end': end,
      })

  return pd.DataFrame(to_return_df)


def _sort_df_multiple_columns(df, key):
  """Sort df based on callable key.

  Args:
    df: pd.DataFrame.
    key: function from rows of df (namedtuples) to tuple. This is used in the
      builtin `sorted` method as the key.

  Returns:
    A sorted copy of df.
  """
  # Unpack into list to take advantage of builtin sorted function.
  # Note that pd.DataFrame.sort_values will not work because sort_values'
  # sorting function is applied to each column at a time, whereas we need to
  # consider multiple fields at once.
  df_rows_sorted = sorted(df.itertuples(index=False), key=key)
  return pd.DataFrame(df_rows_sorted, columns=df.columns)


def order_df_for_output(predictions_df):
  """Semantically group/sort predictions df for output.

  Sort order:
  Sort by query sequence name as they are in `predictions_df`.
  Sort by start index ascending.
  Given that, sort by description alphabetically.

  Args:
    predictions_df: df with columns sequence_name (str), predicted_label (str),
      start(int), end (int), description (str).

  Returns:
    df with columns sequence_name (str), predicted_label (str), start(int),
    end (int), description (str).
  """
  seq_name_to_original_order = {
      item: idx for idx, item in enumerate(predictions_df.sequence_name)
  }

  def _orderer_pfam(df_row):
    """See outer function doctsring."""
    return (
        seq_name_to_original_order[df_row.sequence_name],
        df_row.start,
        df_row.description,
    )

  pfam_df_sorted = _sort_df_multiple_columns(predictions_df, _orderer_pfam)
  return pfam_df_sorted


def format_df_for_output(
    predictions_df,
    *,
    model_cache_path = None,
    label_to_description = None,
):
  """Formats df for outputting.

  Args:
    predictions_df: df with columns sequence_name (str), predicted_label (str),
      start (int), end (int).
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    label_to_description: contents of label_descriptions.json.gz. Map from label
      to a human-readable description.

  Returns:
    df with columns sequence_name (str), predicted_label (str), start(int),
    end (int), description (str).
  """
  predictions_df = predictions_df.copy()

  if label_to_description is None:
    with tf.io.gfile.GFile(
        os.path.join(model_cache_path, 'accession_to_description_pfam_35.json')
    ) as f:
      label_to_description = json.loads(f.read())

  predictions_df['description'] = predictions_df.predicted_label.apply(
      label_to_description.__getitem__
  )

  return order_df_for_output(predictions_df)


def write_output(predictions_df, output_path):
  """Write predictions_df to tsv file."""
  _logger.info('Writing output to %s', output_path)
  with tf.io.gfile.GFile(output_path, 'w') as f:
    predictions_df.to_csv(f, sep='\t', index=False)


def run(
    input_text,
    models,
    reporting_threshold,
    label_to_description,
    model_cache_path,
    min_domain_call_length,
):
  """Runs inference and returns output as a pd.DataFrame.

  Args:
    input_text: contents of a fasta file.
    models: List of Pfam inferrers.
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
    label_to_description: contents of label_descriptions.json.gz. Map from label
      to a human-readable description.
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    min_domain_call_length: don't consider as valid any domain calls shorter
      than this length.

  Returns:
    df with columns sequence_name (str), predicted_label (str), start(int), end
    (int), description (str).
  """
  input_df = input_text_to_df(input_text)
  predictions_df = perform_inference(
      input_df=input_df,
      models=models,
      model_cache_path=model_cache_path,
      reporting_threshold=reporting_threshold,
      min_domain_call_length=min_domain_call_length,
  )

  predictions_df = format_df_for_output(
      predictions_df=predictions_df,
      label_to_description=label_to_description,
      model_cache_path=model_cache_path,
  )

  return predictions_df


def load_assets_and_run(
    input_fasta_path,
    output_path,
    num_ensemble_elements,
    model_cache_path,
    reporting_threshold,
    min_domain_call_length,
):
  """Loads models/metadata, runs inference, and writes output to tsv file.

  Args:
    input_fasta_path: path to FASTA file.
    output_path: path to which to write a tsv of inference results.
    num_ensemble_elements: Number of ensemble elements to load and perform
      inference with.
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
    min_domain_call_length: don't consider as valid any domain calls shorter
      than this length.
  """
  _logger.info('Running with %d ensemble elements', num_ensemble_elements)
  input_text = parse_input_to_text(input_fasta_path)

  models = load_models(model_cache_path, num_ensemble_elements)
  with open(
      os.path.join(model_cache_path, 'accession_to_description_pfam_35.json')
  ) as f:
    label_to_description = json.loads(f.read())

  predictions_df = run(
      input_text,
      models,
      reporting_threshold,
      label_to_description,
      model_cache_path=model_cache_path,
      min_domain_call_length=min_domain_call_length,
  )
  write_output(predictions_df, output_path)


def main(_):
  # TF logging is too noisy otherwise.
  tf.get_logger().setLevel(tf.logging.ERROR)

  load_assets_and_run(
      input_fasta_path=_INPUT_FASTA_FILE_PATH_FLAG.value,
      output_path=_OUTPUT_WRITE_PATH_FLAG.value,
      num_ensemble_elements=_NUM_ENSEMBLE_ELEMENTS_FLAG.value,
      model_cache_path=_MODEL_CACHE_PATH_FLAG.value,
      reporting_threshold=_REPORTING_THRESHOLD_FLAG.value,
      min_domain_call_length=_MIN_DOMAIN_CALL_LENGTH_FLAG.value,
  )


if __name__ == '__main__':
  _logger.info('Process started.')
  flags.mark_flags_as_required(['i'])

  app.run(main)
