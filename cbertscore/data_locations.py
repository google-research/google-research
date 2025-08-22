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

"""Utils for getting locations from disk.
"""

import csv
import re
from typing import Optional
from colabtools import sheets
import numpy as np
import pandas as pd
import tensorflow as tf

gfile = tf.io.gfile


DDW_CLINICIAN_RESPONSES_ = ''
INTERSPEECH_CLINICIAN_RESPONSES_ = ''
INTERSPEECH_SURVEY_TRIPLETS_ = ''
CANDIDATE_TRANSCRIPTS_FOR_SURVEY_ = ''
METEOR_SHEET_ = ''
METRICS_PKL_FILENAME_ = ''
EUPHONIA_DATA_CSV_ = ''



def edit_distance(target,
                  transcript,
                  c_ins = 1,
                  c_del = 1,
                  c_sub = 1):
  """Computes the Levenshtein distance between the target and the transcription strings.

  Args:
    target: Ground-truth after text normalization.
    transcript: ASR model output after text normalization.
    c_ins: Cost of inserting a word.
    c_del: Cost of deleting a word.
    c_sub: Cost of substituting a word.

  Returns:
    Edit distance between the two strings, where only (1) insertion, (2)
    deletion, (3) substitution are allowed.
  """
  reference = target.split()
  prediction = transcript.split()

  m = len(reference) + 1
  n = len(prediction) + 1

  # Initialization.
  d = np.zeros((m, n), dtype=int)

  for i in range(m):
    for j in range(n):
      if i == 0:
        d[0][j] = j
      elif j == 0:
        d[i][0] = i

  # Recursion.
  for i in range(1, m):
    for j in range(1, n):
      # If the words match, no cost.
      if reference[i - 1] == prediction[j - 1]:
        d[i][j] = d[i - 1][j - 1]
      else:
        substitution = d[i - 1][j - 1] + c_sub
        insertion = d[i][j - 1] + c_ins
        deletion = d[i - 1][j] + c_del
        d[i][j] = min(substitution, insertion, deletion)

  return d[len(reference)][len(prediction)]


def ddw_clinician_responses(
    spreadsheet_url = DDW_CLINICIAN_RESPONSES_):
  """Get responses from 2 clinicians from DDW."""
  spreadsheet_id = sheets.get_spreadsheet_id(spreadsheet_url)

  worksheets_df = sheets.get_worksheets(spreadsheet_id)
  assert len(worksheets_df) == 1
  worksheet_id = worksheets_df['Worksheet Id'].iloc[0]
  return sheets.get_cells(
      spreadsheet_id=spreadsheet_id,
      worksheet_id=worksheet_id,
      has_col_header=True)


def interspeech_clinician_responses(
    csv_filename = INTERSPEECH_CLINICIAN_RESPONSES_
    ):
  """Reads and parses clinician responses from Interspeech survey."""
  rows = []
  with gfile.Open(csv_filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      rows.append(row)

  data = {}
  for i in range(len(rows[0])):
    data[rows[1][i]] = [r[i] for r in rows[3:]]

  return pd.DataFrame(data=data)


def interspeech_clinician_responses_formatted(
    csv_filename = INTERSPEECH_CLINICIAN_RESPONSES_,
    minimum_number_of_responses_per_participant = 25,
    ):
  """Reads, parses, and formats clinician responses for Interspeech."""
  survey_df = interspeech_clinician_responses(csv_filename)

  question_columns = [x for x in survey_df.columns if x.startswith('Original')]
  assert len(question_columns) == 150

  if minimum_number_of_responses_per_participant:
    # Filter participants by minimum number of responses.
    indices_to_keep = []
    for i, r in survey_df[question_columns].iterrows():
      cur_count = sum([0 if response == '' else 1 for response in r])
      if cur_count >= minimum_number_of_responses_per_participant:
        indices_to_keep.append(i)
    survey_df = survey_df.loc[indices_to_keep]

  # `survey_df` is indexed by participant. Format to be indexed by question.
  cs, gts, t1s, t2s = [], [], [], []
  col_regx = 'Original sentence: \n\n" (.*?) " \n\n\n\n#1: (.*?) \n\n#2: (.*?)$'
  for c in question_columns:
    tt = re.match(col_regx, c)
    gt, t1, t2 = tt.group(1), tt.group(2), tt.group(3)

    ed1 = edit_distance(gt.strip().lower(), t1.strip().lower())
    ed2 = edit_distance(gt.strip().lower(), t2.strip().lower())
    if ed1 == 0 or ed1 > 3:
      print(f'Problem with t1: {ed1} {t1} {gt}')
      continue
    if ed2 == 0 or ed2 > 3:
      print(f'Problem with t2: {ed2} {t2} {gt}')
      continue

    cs.append(c)
    gts.append(gt)
    t1s.append(t1)
    t2s.append(t2)

  survey_results_df = pd.DataFrame(
      {'cs': cs, 'gts': gts, 't1s': t1s, 't2s': t2s})

  transcript_responses, number_of_responses, count_of_common_response = [], [], []  # pylint:disable=line-too-long
  for _, r in survey_results_df.iterrows():
    cur_responses = []
    for response in survey_df[r['cs']].values:
      if not response: continue
      match_obj = re.match('Transcript #([12]) is less useful.', response)
      if match_obj:
        transcript_response = int(match_obj.group(1))
      else:
        assert response == 'They are about the same.'
        transcript_response = -1
      cur_responses.append(transcript_response)
    transcript_responses.append(','.join([str(x) for x in cur_responses]))
    number_of_responses.append(len(cur_responses))
    _, counts = np.unique(cur_responses, return_counts=True)
    count_of_common_response.append(np.max(counts))
  survey_results_df['responses'] = transcript_responses
  survey_results_df['number_of_responses'] = number_of_responses
  survey_results_df['count_of_common_response'] = count_of_common_response

  return survey_results_df


def interspeech_responses_with_metrics(
    survey_csv_filename = INTERSPEECH_CLINICIAN_RESPONSES_,
    metrics_pickle_fn = METRICS_PKL_FILENAME_,
    ):
  """Interspeech clinician responses with metrics."""
  survey_results_df = interspeech_clinician_responses_formatted(
      survey_csv_filename)

  with gfile.Open(metrics_pickle_fn, 'rb') as f:
    metrics_df = pd.read_pickle(f)

  metric_cols = [c for c in metrics_df.columns if c not in ['gt', 't1', 't2']]
  metric_vals = {k: [] for k in metric_cols}

  print(survey_results_df.columns)
  for _, r in survey_results_df.iterrows():
    cur_gt, cur_t1, cur_t2 = r['gts'], r['t1s'].strip(), r['t2s'].strip()
    metric_row = metrics_df[metrics_df['gt'] == cur_gt]
    assert len(metric_row) == 1

    metric_t1 = metric_row['t1'].values[0].strip()
    metric_t2 = metric_row['t2'].values[0].strip()

    if metric_t1 != cur_t1:
      assert metric_t1 == cur_t2
      assert metric_t2 == cur_t1
      swapped = True
    else:
      assert metric_t1 == cur_t1
      assert metric_t2 == cur_t2, (metric_t2, cur_t2)
      swapped = False

    for metric_c in metric_cols:
      col_to_write_to = metric_c[:]
      if swapped:
        # 1 <-> 2
        if '1' in metric_c:
          assert metric_c.count('1') == 1
          assert '2' not in metric_c
          col_to_write_to = col_to_write_to.replace('1', '2')
        else:
          assert metric_c.count('2') == 1
          col_to_write_to = col_to_write_to.replace('2', '1')
      metric_vals[col_to_write_to].append(metric_row[metric_c].iloc[0])

  for metric_c in metric_cols:
    survey_results_df[metric_c] = metric_vals[metric_c]
  return survey_results_df


def interspeech_survey_triplets(
    spreadsheet_url = INTERSPEECH_SURVEY_TRIPLETS_):
  spreadsheet_id = sheets.get_spreadsheet_id(spreadsheet_url)

  worksheets_df = sheets.get_worksheets(spreadsheet_id)
  assert len(worksheets_df) == 1
  worksheet_id = worksheets_df['Worksheet Id'].iloc[0]
  return sheets.get_cells(
      spreadsheet_id=spreadsheet_id,
      worksheet_id=worksheet_id,
      has_col_header=True)

def candidate_transcripts_for_survey(
    candidate_transcript_filename = CANDIDATE_TRANSCRIPTS_FOR_SURVEY_,
    ground_truth_column = 'medical_dictation 0',
):
  """Reads and parses CSV of ASR transcripts of YT videos."""
  # NOTE: One of the columns is from the medical dictation model. This column
  # is special, since we are going to treat the output of this model as ground
  # truth.

  rows = []
  with gfile.Open(candidate_transcript_filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      rows.append(row)

  # One of the column names should be "Medical Dictation". That column
  # treated as ground truth. We hope to find sentences in the other columns that
  # roughly correspond to what we have as ground truth.
  assert ground_truth_column in rows[0]

  data = {}
  for i in range(len(rows[0])):
    data[rows[0][i]] = [r[i] for r in rows[1:]]

  return pd.DataFrame(data=data)


def get_meteor_values(spreadsheet_url = METEOR_SHEET_):
  """Gets precomputed METEOR values."""
  spreadsheet_id = sheets.get_spreadsheet_id(spreadsheet_url)
  worksheets_df = sheets.get_worksheets(spreadsheet_id)

  worksheets_df = worksheets_df[worksheets_df['Title'] == 'meteor_scores']

  assert len(worksheets_df) == 1

  worksheet_id = worksheets_df['Worksheet Id'].iloc[0]
  return sheets.get_cells(
      spreadsheet_id=spreadsheet_id,
      worksheet_id=worksheet_id,
      has_col_header=True)


def get_euphonia_assessment(
    filename = EUPHONIA_DATA_CSV_,
):
  """Read and parse euphonia data."""
  rows = []
  with gfile.Open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      rows.append(row)
  euphonia_df = pd.DataFrame(data=rows[1:], columns=rows[0])

  required_columns = [
      'ground_truth',
      'transcription_USM',
      'Passessment_USM',
      'transcription_Personalized',
      'Passessment_Personalized',
      'METEORscore_USM',
      'METEORscore_Personalized',
  ]
  for r in required_columns:
    assert r in euphonia_df.columns

  return euphonia_df[required_columns]


