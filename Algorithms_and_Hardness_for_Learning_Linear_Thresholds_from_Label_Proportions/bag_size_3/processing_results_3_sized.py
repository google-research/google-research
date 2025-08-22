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

"""Processing Results for 3 sized bags experiments."""
import pandas as pd

df = pd.read_csv('large_margin_3_sized_results.csv')

df['pct_sat_bags'] = 100 * (df['satbag'] / df['N'])

df['pct_sat_bags_random_thresh'] = 100 * (df['satbag_random_thresh'] / df['N'])

df_summarized = df.groupby(['D', 'N']).agg(
    avg_pct_sat_bags=('pct_sat_bags', 'mean'),
    stdev_pct_sat_bags=('pct_sat_bags', 'std'),
    avg_pct_sat_bags_random_thresh=('pct_sat_bags_random_thresh', 'mean'),
    stdev_pct_sat_bags_random_thresh=('pct_sat_bags_random_thresh', 'std'),
    avg_frac_rank_A=('frac_rank_A', 'mean'),
    stdev_frac_rank_A=('frac_rank_A', 'std'),
    avg_test_sat_tildec=('test_sat_tildec', 'mean'),
    stdev_test_sat_tildec=('test_sat_tildec', 'std'),
    avg_test_sat_random_thresh=('test_sat_random_thresh', 'mean'),
    stddev_test_sat_random_thresh=('test_sat_random_thresh',
                                   'std')).reset_index()

df_summarized[
    'avg_gain'] = df['pct_sat_bags'] / df['pct_sat_bags_random_thresh']
print(df_summarized.to_string())

df_summarized = df_summarized.round({
    'avg_pct_sat_bags': 2,
    'stdev_pct_sat_bags': 3,
    'avg_pct_sat_bags_random_thresh': 2,
    'stdev_pct_sat_bags_random_thresh': 3,
    'avg_gain': 2,
    'avg_frac_rank_A': 2,
    'stdev_frac_rank_A': 3,
    'avg_test_sat_tildec': 2,
    'stdev_test_sat_tildec': 3,
    'avg_test_sat_random_thresh': 2,
    'stdev_test_sat_random_thresh': 3
})
with open('large_margin_summarized_3_sized.tex', 'w') as towrite:
  towrite.write(df_summarized.to_latex(index=False, escape=False))

df = pd.read_csv('small_margin_3_sized_results.csv')

df['pct_sat_bags'] = 100 * (df['satbag'] / df['N'])

df['pct_sat_bags_random_thresh'] = 100 * (df['satbag_random_thresh'] / df['N'])

df_summarized = df.groupby(['D', 'N']).agg(
    avg_pct_sat_bags=('pct_sat_bags', 'mean'),
    stdev_pct_sat_bags=('pct_sat_bags', 'std'),
    avg_pct_sat_bags_random_thresh=('pct_sat_bags_random_thresh', 'mean'),
    stdev_pct_sat_bags_random_thresh=('pct_sat_bags_random_thresh', 'std'),
    avg_frac_rank_A=('frac_rank_A', 'mean'),
    stdev_frac_rank_A=('frac_rank_A', 'std'),
    avg_test_sat_tildec=('test_sat_tildec', 'mean'),
    stdev_test_sat_tildec=('test_sat_tildec', 'std'),
    avg_test_sat_random_thresh=('test_sat_random_thresh', 'mean'),
    stddev_test_sat_random_thresh=('test_sat_random_thresh',
                                   'std')).reset_index()

df_summarized[
    'avg_gain'] = df['pct_sat_bags'] / df['pct_sat_bags_random_thresh']
print(df_summarized.to_string())

df_summarized = df_summarized.round({
    'avg_pct_sat_bags': 2,
    'stdev_pct_sat_bags': 3,
    'avg_pct_sat_bags_random_thresh': 2,
    'stdev_pct_sat_bags_random_thresh': 3,
    'avg_gain': 2,
    'avg_frac_rank_A': 2,
    'stdev_frac_rank_A': 3,
    'avg_test_sat_tildec': 2,
    'stdev_test_sat_tildec': 3,
    'avg_test_sat_random_thresh': 2,
    'stdev_test_sat_random_thresh': 3
})
with open('small_margin_summarized_3_sized.tex', 'w') as towrite:
  towrite.write(df_summarized.to_latex(index=False, escape=False))
