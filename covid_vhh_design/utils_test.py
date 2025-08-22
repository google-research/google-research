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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import scipy.stats

from covid_vhh_design import utils


BASELINE = utils.BASELINE
ML = utils.ML


class UtilsTest(parameterized.TestCase):

  def assertFrameEqual(self,
                       expected_frame,
                       actual_frame,
                       sort_index=False,
                       sort_columns=False,
                       **kwargs):
    """Tests if two `pd.DataFrames` are equal."""
    if sort_index:
      expected_frame = expected_frame.sort_index(axis=0)
      actual_frame = actual_frame.sort_index(axis=0)
    if sort_columns:
      expected_frame = expected_frame.sort_index(axis=1)
      actual_frame = actual_frame.sort_index(axis=1)
    pd.testing.assert_frame_equal(expected_frame, actual_frame, **kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name='parent_seq',
          seq='ABCD',
          parent_seq='ABCD',
          is_invalid=False),
      dict(
          testcase_name='valid_mutation',
          seq='AXCD',
          parent_seq='ABCD',
          is_invalid=False),
      dict(
          testcase_name='invalid_mutations',
          seq='AXXD',
          parent_seq='ABCD',
          is_invalid=True),
      dict(
          testcase_name='with_deletion',
          seq='ABCD',
          parent_seq='ACD',
          is_invalid=True),
  )
  def test_sequence_mutates_valid_positions(self, seq, parent_seq, is_invalid):
    allowed_pos = (
        '0',
        '1',
        '2B',
    )
    pos_to_imgt = {0: '0', 1: '1', 2: '2A', 3: '2B'}
    self.assertEqual(
        utils.sequence_has_invalid_mutations(
            seq, parent_seq, pos_to_imgt, allowed_pos=allowed_pos), is_invalid)

  def test_get_source_annotations(self):
    round0_df = pd.DataFrame(
        dict(source_seq=['A', 'A', 'B'],
             source_std_group=['mbo', 'mbo', 'singles'])
    )
    round1_df = pd.DataFrame(
        dict(
            source_seq=['A', 'B', 'C'],
            source_std_group=['best_prior', 'baseline', 'mbo']))
    actual_annotations = utils.get_source_annotations({
        0: round0_df,
        1: round1_df
    }, column='source_std_group')
    expected_annotations = dict(A='mbo', B='singles', C='mbo')
    self.assertDictEqual(actual_annotations,
                         expected_annotations)

  def test_standardize_experimental_replicates(self):
    bindings = pd.DataFrame({
        'source_seq': ['A', 'A', 'B', 'B', 'A'],
        'replica': [0, 1, 0, 1, 0],
        'target_name': ['CoV2', 'CoV2', 'CoV2', 'CoV2', 'CoV1'],
        'value': [.5, .2, .8, np.inf, .3],
    })
    expected_df = pd.DataFrame({
        'source_seq': ['A', 'A', 'B', 'B', 'A'],
        'replica': [0, 1, 0, 1, 0],
        'target_name': ['CoV2', 'CoV2', 'CoV2', 'CoV2', 'CoV1'],
        'value': [-1., 0., 1, np.inf, 0.]
    })
    self.assertFrameEqual(
        utils.standardize_experimental_replicates(bindings), expected_df)

  def test_standardize_by_parent(self):
    cov1_parent_values = [0., .2, .1, np.inf]
    cov1_median = np.median(cov1_parent_values)
    cov1_iqr = scipy.stats.iqr(cov1_parent_values)

    df_cov1 = pd.DataFrame(
        (
            # parent values replica 1
            (1, 0, 0,),
            (1, 0, .2),
            (1, 0, .1),
            (1, 0, np.inf),
            # non-parent values replica 1
            (1, 1, .1),
            (1, 3, .5),
        ),
        columns=['replica', 'source_num_mutations', 'value'])
    df_cov1['target_name'] = 'CoV1'
    expected_cov1_df = df_cov1.copy()
    expected_cov1_df['value'] = (expected_cov1_df['value'] -
                                 cov1_median) / cov1_iqr

    cov2_parent_values = [.5, .3, .7, .1]
    cov2_median = np.median(cov2_parent_values)
    cov2_iqr = scipy.stats.iqr(cov2_parent_values)
    df_cov2 = pd.DataFrame(
        (
            # parent values replica 1
            (1, 0, .5),
            (1, 0, .3),
            (1, 0, .7),
            (1, 0, .1),
            # non-parent values replica 1
            (1, 1, .5),
            (1, 3, np.inf),
            # parent values replica 2
            (2, 0, 0),
            # non-parent values replica 2
            (2, 1, .2)),
        columns=['replica', 'source_num_mutations', 'value'])
    df_cov2['target_name'] = 'CoV2'
    expected_cov2_df = df_cov2.copy()
    expected_cov2_df['value'] = (expected_cov2_df['value'] -
                                 cov2_median) / cov2_iqr
    expected_cov2_df.loc[expected_cov2_df['replica'] == 2, 'value'] = [0, .2]

    actual_df = utils.standardize_by_parent(
        pd.concat([df_cov1, df_cov2], ignore_index=True))

    expected_df = pd.concat(
        [expected_cov1_df, expected_cov2_df],
        ignore_index=True)

    self.assertFrameEqual(actual_df, expected_df)

  def test_filter_sequences_with_min_replicas(self):
    df = pd.DataFrame({
        'source_seq': ['A', 'A', 'A', 'B', 'C', 'C'],
        'value': [.2, .1, .2, .3, -1, .3],
    })
    df['target_name'] = 'CoV2'
    self.assertFrameEqual(utils.filter_sequences_with_min_replicas(df, 2),
                          df[df['source_seq'].isin(['A', 'C'])])

  def test_compute_pvalues_by_target_fails_on_multiple_targets(self):
    df = pd.DataFrame({
        'source_seq': ['A', 'A', 'B', 'B'],
        'target_name': ['CoV1', 'CoV2', 'CoV2', 'CoV2'],
        'value': [.2, .1, .2, .3],
        'source_num_mutations': [0, 1, 2, 2]
    })
    with self.assertRaisesRegex(ValueError, 'Cannot compute pvalues'):
      _ = utils._compute_pvalues_by_target(
          df,
          correction_method='fdr_by',
          alpha=0.05,
          min_replicas=2,
      )

  def test_compute_pvalues(self):
    df = pd.DataFrame({
        'source_seq': ['A', 'A', 'A', 'B'],
        'replica': [0, 1, 0, 0],
        'target_name': ['CoV1', 'CoV2', 'CoV2', 'CoV2'],
        'value': [.2, .1, .2, .3],
        'source_num_mutations': [0, 0, 0, 1.],
    })
    pvalues_df = utils.compute_pvalues(
        df,
        alpha=0.05,
        min_replicas=2,
    )
    self.assertIn('pvalue', pvalues_df.columns)
    self.assertIn('pvalue_corrected', pvalues_df.columns)

    # Only compute pvalues for ('A', 'CoV2') due to min_counts requirement.
    self.assertLen(pvalues_df, 1)
    self.assertEqual(pvalues_df['source_seq'].iloc[0], 'A')
    self.assertEqual(pvalues_df['target_name'].iloc[0], 'CoV2')

  def test_get_metadata(self):
    df = pd.DataFrame(dict(
        source_seq=['A', 'A', 'B', 'B'],
        target_name=['CoV1', 'CoV2', 'CoV2', 'CoV2'],
        source_num_mutations=[1, 1, 2, 2],
        replica=[1, 2, 1, 2],
    ))
    expected_df = df = pd.DataFrame(dict(
        source_seq=['A', 'A', 'B'],
        target_name=['CoV1', 'CoV2', 'CoV2'],
        source_num_mutations=[1, 1, 2],
    ))
    self.assertFrameEqual(utils.get_metadata(df), expected_df)

  def test_aggregate_over_rounds(self):
    round0_df = pd.DataFrame(
        (('A', 0, 0, 0), ('A', 0, 1, 2), ('B', 1, 0, 1), ('B', 1, 1, 2)),
        columns=['source_seq', 'source_num_mutations', 'replica', 'value'],
    )
    round0_df['target_name'] = 'CoV2'

    round1_df = pd.DataFrame(
        (('A', 0, 0, 1), ('C', 2, 0, -1), ('C', 2, 1, -3)),
        columns=['source_seq', 'source_num_mutations', 'replica', 'value'],
    )
    round1_df['target_name'] = 'CoV2'

    expected_values = (  # Medians of aggregated affinities per round.
        pd.concat([
            utils.aggregate_affinities(round0_df),
            utils.aggregate_affinities(round1_df),
        ])
        .groupby('source_seq')['value']
        .median()
    )

    actual_df = utils.aggregate_over_rounds({0: round0_df, 1: round1_df})
    expected_df = pd.DataFrame(
        (
            ('A', 0, 0, expected_values['A']),
            ('B', 1, 0, expected_values['B']),
            ('C', 2, 1, expected_values['C']),
        ),
        columns=['source_seq', 'source_num_mutations', 'round', 'value'],
    )
    self.assertFrameEqual(
        actual_df[[
            'source_seq',
            'source_num_mutations',
            'round',
            'value',
        ]],
        expected_df,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='groupby_category',
          groupby='source_category',
          thresholds=[-1., -1.5],
          measurement_col='value',
          expected_df=pd.DataFrame({
              'source_category': [BASELINE, ML, BASELINE, ML],
              'mean': [1., 2 / 3, 0., 2 / 3],
              'sum': [1, 2, 0, 2],
              'count': [1, 3, 1, 3],
              'threshold': [-1., -1., -1.5, -1.5],
          })),
      dict(
          testcase_name='groupby_category_and_num_mutations',
          groupby=['source_category', 'source_num_mutations'],
          thresholds=-1.5,
          measurement_col='value',
          expected_df=pd.DataFrame({
              'source_category': [BASELINE, ML, ML],
              'source_num_mutations': [2, 2, 3],
              'mean': [0., .5, 1.],
              'sum': [0, 1, 1],
              'count': [1, 2, 1],
              'threshold': [-1.5, -1.5, -1.5],
          })),
      dict(
          testcase_name='pvalue',
          groupby='source_category',
          thresholds=0.05,
          measurement_col='pvalue',
          expected_df=pd.DataFrame({
              'source_category': [BASELINE, ML],
              'mean': [0., 1 / 3],
              'sum': [0, 1],
              'count': [1, 3],
              'threshold': [.05, .05]
          })),
  )
  def test_compute_hit_rate(self, groupby, thresholds, measurement_col,
                            expected_df):
    df = pd.DataFrame({
        'source_num_mutations': [2, 2, 2, 3],
        'value': [-2, np.inf, -1, -3],
        'source_category': [ML, ML, BASELINE, ML],
        'pvalue': [.1, .5, .1, .01]
    })
    actual_df = utils.compute_hit_rate(
        df,
        groupby=groupby,
        thresholds=thresholds,
        measurement_col=measurement_col,
        lesser_than=True)
    self.assertFrameEqual(actual_df, expected_df, sort_index=True)

  @parameterized.named_parameters(
      dict(
          testcase_name='top2_by_category',
          num_top_seqs=2,
          category_col='category',
          expected_seqs=['B', 'C', 'E', 'F'],
      ),
      dict(
          testcase_name='top_2_by_num_mutations',
          num_top_seqs=2,
          category_col='num_mutations',
          expected_seqs=['A', 'D', 'C', 'F'],
      ),
      dict(
          testcase_name='top_1_by_category',
          num_top_seqs=1,
          category_col='category',
          expected_seqs=['C', 'F'],
      ),
  )
  def test_extract_best_sequences_by_category(
      self, num_top_seqs, category_col, expected_seqs
  ):
    df = pd.DataFrame([
        dict(source_seq='A', value=1., category='ML', num_mutations=1),
        dict(source_seq='B', value=.6, category='ML', num_mutations=3),
        dict(source_seq='C', value=.2, category='ML', num_mutations=3),
        dict(source_seq='D', value=.9, category='Baseline', num_mutations=1),
        dict(source_seq='E', value=.8, category='Baseline', num_mutations=3),
        dict(source_seq='F', value=.5, category='Baseline', num_mutations=3),
    ])
    self.assertCountEqual(
        utils.extract_best_sequences_by_category(
            df,
            num_top_seqs=num_top_seqs,
            value_col='value',
            category_col=category_col,
        ),
        expected_seqs,
    )

  def test_find_multihits(self):
    df = pd.DataFrame(
        (
            ('Parent', 0, 'CoV1', 0),
            ('A', 1, 'CoV2', .5),
            ('B', 1, 'CoV1', .1),
            ('B', 1, 'CoV2', -.5),
            ('C', 1, 'CoV1', -2),
            ('C', 1, 'CoV2', -3),
        ),
        columns=['source_seq', 'source_num_mutations', 'target_name', 'value'])
    expected_hits = pd.DataFrame((
        ('Parent', 0, 0),
        ('A', 0, 0),
        ('B', 0, 0,),
        ('C', 1, 1,),
    ), columns=['source_seq', 'CoV1', 'CoV2']).set_index(
        'source_seq').rename_axis('target_name', axis='columns')
    actual_df = utils.find_multihits(df, how='iqr')

    self.assertFrameEqual(actual_df, expected_hits,
                          sort_index=True, sort_columns=True)

  @parameterized.parameters(
      ('ARC', [], 0),
      ('ARC', ['ACC'], 1.0),
      ('ARC', ['ARR', 'ACR'], 1.5)
  )
  def test_avg_distance_to_set(self, seq, other_seqs, expected_distance):
    self.assertEqual(utils.avg_distance_to_set(seq, np.array(other_seqs)),
                     expected_distance)


if __name__ == '__main__':
  absltest.main()
