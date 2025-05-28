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

"""Functions for calculating human-simulated scores."""

import numpy as np
import pandas as pd

import tqdm

from psyborgs import survey_bench_lib


SPID = (
    'item_preamble_id',
    'item_postamble_id',
    'response_scale_id',
    'response_choice_postamble_id',
    'model_id',
)


def logsumexp(x):
  c = x.max()
  return c + np.log(np.sum(np.exp(x - c)))


def normalize_logprobs(x):
  diff = x - logsumexp(x)
  probs = np.exp(diff)
  return probs / np.sum(probs)


# for generative mode


def normalize_response_logprobs(df):
  """Converts generated responses and their log-prob scores to normalized probabilities."""
  grouped = df.groupby(
      list(SPID) + ['item_id', 'model_output'], group_keys=False
  )

  # calculate the log probability sums for each unique combination of item_id,
  # model_output, and list(SPID)
  logprobs_sum = grouped['model_output_score'].sum()

  # normalize the log probabilities for each unique combination of item_id,
  # list(SPID)
  normalized_probs = (
      logprobs_sum.groupby(list(SPID) + ['item_id'], group_keys=False)
      .apply(normalize_logprobs)
      .values
  )

  # create a new dataframe with the normalized probabilities
  raw_response_scores_normalized_df = pd.DataFrame({
      'item_preamble_id': logprobs_sum.index.get_level_values(
          'item_preamble_id'
      ),
      'item_id': logprobs_sum.index.get_level_values('item_id'),
      'item_postamble_id': logprobs_sum.index.get_level_values(
          'item_postamble_id'
      ),
      'response_scale_id': logprobs_sum.index.get_level_values(
          'response_scale_id'
      ),
      'response_choice_postamble_id': logprobs_sum.index.get_level_values(
          'response_choice_postamble_id'
      ),
      'model_id': logprobs_sum.index.get_level_values('model_id'),
      # convert these `response_value`s to ints
      'response_value': logprobs_sum.index.get_level_values(
          'model_output'
      ).astype(int),
      'score': normalized_probs,
  })

  # return the resulting dataframe
  return raw_response_scores_normalized_df


# reshape response choice probability scores
def reshape_response_choice_probability_scores(
    raw_response_scores_df,
):
  """Reshapes raw data into columns of LLM scores for every response choice.

  Args:
    raw_response_scores_df: A DataFrame containing raw response scores in long
      format, with one unique prompt-continuation (containing a single response
      choice) and an LLM score for this unique combination for each row. Columns
      should include `item_preamble_id`, `item_id`, `item_postamble_id`,
      `response_scale_id`, `response_choice_postamble_id`, `model_id`, and
      `response_value`.

  Returns:
    A DataFrame containing raw response scores in wide format. Each row
      contains IDs representing a unique prompt-continuation specification and
      the LLM float scores for each response choice considered in the
      specification. Score columns are labeled in the format of `item_id` +
      `_` + `response_choice_value`. The column of the item `brsf1`'s
      response choice value of 1 would therefore be `brsf1_1`.
  """
  # create pivot table of response choice probabilities nested under item IDs
  df_raw_wide = raw_response_scores_df.pivot_table(
      index=list(SPID), columns=['item_id', 'response_value'], values=['score']
  )

  # collapse pivot table into flat column names representing item IDs paired
  # with response scale values
  df_raw_wide.columns = [
      f'{item_id}_{response_value}'
      for _, item_id, response_value in df_raw_wide.columns
  ]

  # reset index
  df_raw_wide = df_raw_wide.reset_index()

  return df_raw_wide


# determine human-simulated response values
def calculate_human_simulated_responses(
    raw_response_scores_df,
):
  """Selects the most likely response choices to simulate human responses.

  This function simulates human responses to individual survey measure items
    by 'selecting' response choices with the highest LLM probability score.
    The response value (an integer within the range of a given response
    scale) for the selected response choice is used for calculation of human-
    simulated scale scores.

    For instance, the LLM scores for item 'aa1' using a 5-point Likert scale
    might be .20, .40, .60, .80, and 1.00 for the response choices
    'strongly disagree', 'disagree', 'neither agree nor disagree', 'agree',
    and 'strongly agree', respectively. To simulate a human response to this
    item, we select 'strongly agree', the response choice with the highest
    LLM score. The corresponding integer value for 'strongly agree' on the
    5-point response scale is 5. Therefore, the simulated human response to
    item 'aa1' would be 5.

  Args:
    raw_response_scores_df: A DataFrame containing raw response scores in long
      format, with one unique prompt-continuation (containing a single response
      choice) and an LLM score for this unique combination for each row. Columns
      should include `item_preamble_id`, `item_id`, `item_postamble_id`,
      `response_scale_id`, `response_choice_postamble_id`, `model_id`, and
      `response_value`.

  Returns:
    A DataFrame containing prompt-continuation specification data and columns
      of human-simulated integer response values labeled by `item_id`.
  """
  # register `pandas.progress_apply` with tqdm
  tqdm.tqdm.pandas()

  print('Determining the most likely response choice per item. ')
  print('This could take a while! ... ')
  # retrieve rows with the most likely response choice
  df_item_responses = raw_response_scores_df.loc[
      raw_response_scores_df.groupby(list(SPID) + ['item_id'])['score'].idxmax()
  ].reset_index(drop=True)

  # reshape to wide
  df_simulated_item_responses_wide = (
      df_item_responses.pivot(
          index=list(SPID), columns=['item_id'], values='response_value'
      )
      .reset_index()
      .rename_axis(index=None, columns=None)
  )

  return df_simulated_item_responses_wide


# combined raw LLM scores and human-simulated choices into one DataFrame
def get_raw_and_simulated_responses(
    raw_response_scores_df, generative_mode = False
):
  """Returns combined DataFrame of raw LLM scores and simulated responses."""
  # if response data was created in generative mode, normalize model_output
  # log probabilities to probabilities.
  if generative_mode:
    raw_response_scores_df = normalize_response_logprobs(
        raw_response_scores_df
    )

  # reshape raw LLM response choice scores
  print('Reshaping raw LLM response choice scores... ')
  df_raw_reshaped = reshape_response_choice_probability_scores(
      raw_response_scores_df
  )

  # calculate and reshape human simulated item responses
  print('Calculating and reshaping human-simulated item responses... ')
  df_simulated_item_responses = calculate_human_simulated_responses(
      raw_response_scores_df
  )

  # combine the above into one DataFrame
  print(
      'Combining LLM scores and human-simulated responses into one'
      ' DataFrame... '
  )

  return df_simulated_item_responses.merge(df_raw_reshaped, how='inner')


# calculate session scale scores
def score_session(
    admin_session,
    raw_response_scores_df,
    verbose = False,
):
  """Calculates human-simulated scores from AdministrationSession results.

  This function treats each unique prompt-continuation specification as a
    simulated participant, indexed by a simulated participant ID (list(SPID)).
    Iterating through each multi-item scale in an AdministrationSession, it
    calculates a summary score for each list(SPID) by taking the average of all
    item response values (accounting for reverse-keyed items).

  Args:
    admin_session: An AdministrationSession.
    raw_response_scores_df: A DataFrame containing raw response scores in long
      format, with one unique prompt-continuation (containing a single response
      choice) and an LLM score for this unique combination for each row. Columns
      should include `item_preamble_id`, `item_id`, `item_postamble_id`,
      `response_scale_id`, `response_choice_postamble_id`, `model_id`, and
      `response_value`.
    verbose: A boolean. Prints simulated scores for debugging if True.

  Returns:
    A DataFrame of raw LLM scores, human-simulated response values, and human-
      simulated scale scores. Scale scores are labeled by `scale_id`.
  """
  measures = admin_session.measures
  scored_session_df = get_raw_and_simulated_responses(raw_response_scores_df)

  # for each scale, score simulated participants (list(SPID)s)
  for measure in measures.values():
    for scale_id, scale in measure.scales.items():
      # get scale scoring info
      item_ids = scale.item_ids
      reverse_keyed_item_ids = scale.reverse_keyed_item_ids
      response_scale_ids = scale.response_scale_ids
      scale_length = len(item_ids)

      # for each response scale type, score columnwise
      for response_scale_id in response_scale_ids:
        scale_point_range = len(
            admin_session.response_scales[response_scale_id].response_choices
        )

        # only work on rows that use the current response_scale
        df_response_scale_id_col = scored_session_df['response_scale_id']

        item_values = []

        for item_id in item_ids:
          original_values = scored_session_df[
              df_response_scale_id_col == response_scale_id
          ][item_id]

          # reverse key item value column if its item_id is in
          # reverse_keyed_item_ids; otherwise, keep the values the same
          if item_id in reverse_keyed_item_ids:
            processed_values = scale_point_range - original_values + 1
          else:
            processed_values = original_values

          item_values.append(processed_values)

        simulated_scale_scores = sum(item_values) / scale_length

        if verbose:
          print(
              'Simulated "'
              + response_scale_id
              + '" scale scores for '
              + scale_id
              + ': \n'
              + str(simulated_scale_scores)
              + '\n'
          )

        scored_session_df.loc[
            df_response_scale_id_col == response_scale_id, scale_id
        ] = simulated_scale_scores

  return scored_session_df
