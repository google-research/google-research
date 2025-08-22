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

"""Make predictions using sktime lib as baselines, and evaluate."""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import argparse
import copy
import datetime
import multiprocessing as mp
import os
import time

from data_formatting.datasets import get_favorita_data
from data_formatting.datasets import get_m3_data
from data_formatting.datasets import get_m3_df
from lib.evaluator import Evaluator
from lib.naive_scaling_baseline import NaiveScalingBaseline
from main import get_full_test_metrics
import numpy as np
import pandas as pd
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.transformations.series.detrend import Deseasonalizer
import torch
from tqdm import tqdm
from utils import get_summary
import wandb


def proc_print(msg):
  print(f'pid: {os.getpid()},\t{msg}')


def tprint(msg):
  now = datetime.datetime.now()
  proc_print(f'[{now}]\t{msg}')


def evaluate(
    dataset_factory,
    test_preds,
    naive_model,
    lead_time,
    scale01,
    test_t_min,
    target_service_level,
    unit_holding_cost,
    unit_stockout_cost,
    unit_var_o_cost,
    valid_t_start=None,
    test_t_start=None,
    target_dims=(0,),
    parallel=False,
    num_workers=0,
    device=torch.device('cpu'),
):
  """Evaluate predictions.

  Args:
    dataset_factory: factory for datasets
    test_preds: predicted values
    naive_model: baseline model
    lead_time: number of time points it takes for inventory to come in
    scale01: whether predictions are scaled between 0 and 1
    test_t_min: first timepoint for evaluation
    target_service_level: service level to use for safety stock calculation
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    valid_t_start: start of validation time period
    test_t_start: start of test time period
    target_dims: dimensions corresponding to target
    parallel:  whether to evaluate in parallel
    num_workers: number of workers for parallel dataloading
    device: device to perform computations on

  Returns:
    aggregate and per-series metrics
  """
  evaluator = Evaluator(0, scale01, device, target_dims=list(target_dims))

  scale_by_naive_model = False
  quantile_loss = None
  use_wandb = False
  test_metrics, expanded_test_metrics = get_full_test_metrics(
      dataset_factory,
      test_preds.unsqueeze(-1),
      num_workers,
      parallel,
      device,
      test_t_min,
      valid_t_start,
      test_t_start,
      evaluator,
      target_service_level,
      lead_time,
      unit_holding_cost,
      unit_stockout_cost,
      unit_var_o_cost,
      naive_model,
      scale_by_naive_model,
      quantile_loss,
      use_wandb,
  )
  return test_metrics, expanded_test_metrics


def get_favorita_df(impute0=True, N=None, test_t_max=None):
  """Retrieves Favorita dataset in a format amenable to sktime lib.

  Args:
    impute0: whether to impute with 0's
    N: number of series to limit to. if None, no limit
    test_t_max: maximum time point to evaluate

  Returns:

  """
  df = pd.read_csv('../data/favorita/favorita_df.csv').set_index('traj_id')
  if impute0:
    df = df.fillna(0)

  if N is not None:
    df = df[:N]

  df['N'] = df.shape[1]
  Ns = df['N']
  if test_t_max is not None:
    Ns = df['N'].clip(upper=test_t_max)
  series = df.drop('N', axis=1)
  series.columns = list(range(1, len(series.columns) + 1))

  if test_t_max is not None:
    series = series.iloc[:, :test_t_max]

  df = pd.concat([Ns, series], axis=1)
  print('Length of Favorita dataset: ', len(df))
  return df


class DeseasonalizedForecaster:
  """Wrapper around sktime forecasters that first de-seasonalizes the data."""

  def __init__(self, fc, sp=1, model='additive'):
    self.fc = fc
    self.deseasonalizer = Deseasonalizer(sp=sp, model=model)

  def fit(self, y):
    self.deseasonalizer.fit(y)
    y_new = self.deseasonalizer.transform(y)
    self.fc.fit(y_new)

  def reset(self):
    self.fc.reset()

  def predict(self, horizon):
    preds = self.fc.predict(horizon)
    preds = self.deseasonalizer.inverse_transform(preds)
    return preds

  def __str__(self):
    return 'Deseasonalized' + str(self.fc)


def predict_series(
    fc, row, series_len, first_cutoff, lead_time, test_t_max, test_t_min, i
):
  """Rolling forward in time, make predictions for the series.

  Args:
    fc: forecaster
    row: series
    series_len: length of series
    first_cutoff: first time point to make predictions at
    lead_time: time points it takes for inventory to come in
    test_t_max: maximum time point to evaluate
    test_t_min: minimum time point to evaluate
    i: index of series

  Returns:
    predictions for the series (numpy array)
  """
  print('predicting for series: ', i)
  y = row[list(range(1, series_len + 1))]
  y.index = y.index.astype(int)
  series_preds = np.ones((test_t_max - test_t_min, lead_time))
  for cutoff in range(first_cutoff, series_len):
    t = cutoff + 1  # current timepoint
    tr_ids = list(range(1, t))
    te_ids = list(range(t, min(t + lead_time, series_len + 1)))
    horizon = ForecastingHorizon(np.array(te_ids), is_relative=False)
    y_tr = y[y.index.isin(tr_ids)]

    # fit and make predictions
    fc.reset()
    fc.fit(y_tr)

    preds = fc.predict(horizon)
    series_preds[cutoff - first_cutoff, : len(preds)] = preds
  print('finished predicting series: ', i)
  return series_preds


def predict_roll_forward(
    fc,
    fc_name,
    df,
    folder,
    dataset_name,
    start,
    idx_range,
    test_t_max,
    test_t_min,
    lead_time,
    first_cutoff,
    pool=None,
):
  """Make predictions for entire dataframe, rolling forward in time.

  Saves predictions in folder. If previously saved, simply re-loads predictions.

  Args:
    fc: forecaster
    fc_name: forecaster name
    df: dataframe containing (univariate) data to make predictions for
    folder: folder to save predictions
    dataset_name: name of dataset
    start: starting unix time (to calculate runtime)
    idx_range: range of indices of series to consider
    test_t_max: maximum timepoint to evaluate
    test_t_min: minimum timepoint to evaluate
    lead_time: number of timepoints it takes for inventory to come in
    first_cutoff: first time point to make predictions for
    pool: multiprocessing pool, if available (else None)

  Returns:
    predictions for entire dataframe, across all time
  """
  fpath = os.path.join(folder, f'{dataset_name}_{fc_name}_N{len(df)}.npy')
  print(fpath)
  if os.path.exists(fpath):
    test_preds = np.load(fpath)
    print('loaded predictions from: ', fpath)
  else:
    print(fc_name, ' time: ', time.time() - start)
    print(idx_range)
    all_series_preds = []
    for i in tqdm(range(idx_range[0], idx_range[1])):
      row = df.iloc[i, :]
      series_len = int(row['N'])
      if pool is not None:
        all_series_preds.append((
            i,
            pool.apply_async(
                predict_series,
                [
                    copy.deepcopy(fc),
                    row,
                    series_len,
                    first_cutoff,
                    lead_time,
                    test_t_max,
                    test_t_min,
                    i,
                ],
            ),
        ))
      else:
        all_series_preds.append((
            i,
            predict_series(
                fc,
                row,
                series_len,
                first_cutoff,
                lead_time,
                test_t_max,
                test_t_min,
                i,
            ),
        ))

    if pool is not None:
      for i, series_preds in all_series_preds:
        while not series_preds.ready():
          print('waiting for series: ', i)
          time.sleep(100)

    test_preds = (
        np.ones(
            (idx_range[1] - idx_range[0], test_t_max - test_t_min, lead_time)
        )
        * 1e18
    )
    for i, series_preds in all_series_preds:
      if pool is not None:
        series_preds = series_preds.get()
      test_preds[i - idx_range[0], :] = series_preds
    np.save(fpath, test_preds, allow_pickle=False)
  print('finished predicting roll-forward: ', time.time() - start)
  return test_preds


def get_sktime_predictions(
    df,
    dataset_name,
    forecasters,
    idx_range,
    test_t_max,
    test_t_min,
    lead_time,
    first_cutoff,
    folder,
    parallel=False,
):
  """Get predictions for list of forecasters of interest.

  Args:
    df: dataset
    dataset_name: name of dataset
    forecasters: list of forecasters of interest
    idx_range: range of indices to evaluate
    test_t_max: maximum timepoint to evaluate
    test_t_min: minimum timepoint to evaluate
    lead_time: amount of timepoints it takes for inventory to come in
    first_cutoff: first timepoint to make predictions for
    folder: folder to save predictions
    parallel: whether to make predictions in parallel

  Returns:
    dictionary mapping forecasters to predictions
  """
  start = time.time()
  pool = None
  if parallel:
    num_proc = int(mp.cpu_count())
    pool = mp.Pool(num_proc)
    print('Number of processors: ', num_proc)

  fc_to_preds = {}
  for fc_name, fc in forecasters.items():
    test_preds = predict_roll_forward(
        fc,
        fc_name,
        df,
        folder,
        dataset_name,
        start,
        idx_range,
        test_t_max,
        test_t_min,
        lead_time,
        first_cutoff,
        pool,
    )
    fc_to_preds[fc_name] = test_preds

  if parallel:
    pool.close()
    pool.join()
  return fc_to_preds


def main():
  mp.set_start_method('spawn')
  parser = argparse.ArgumentParser()
  parser.add_argument('--parallel', action='store_true')
  parser.add_argument('--dataset', choices=['m3', 'favorita'])
  parser.add_argument(
      '--forecasters',
      choices=[
          'NaiveForecaster',
          'ExponentialSmoothing',
          'ThetaForecaster',
          'ARIMA',
          'DeseasonalizedThetaForecaster',
      ],
      action='append',
  )
  parser.add_argument('--N', type=int, default=None)
  parser.add_argument('--preds_only', action='store_true')
  parser.add_argument('--num_workers', type=int, default=0)

  args = parser.parse_args()

  wandb_log = True
  parallel = args.parallel
  num_workers = args.num_workers
  dataset_name = args.dataset

  if dataset_name == 'm3':
    data_fpath = '../data/m3/m3_industry_monthly_shuffled.csv'
    df = get_m3_df(N=args.N, csv_fpath=data_fpath, idx_range=None)
    Ns = df['N']
    series = df.drop('N', axis=1)
    series.columns = series.columns.astype(int)
    df = pd.concat([Ns, series], axis=1)

    idx_range = (20, len(df))

    test_t_min = 36
    test_t_max = 144
    valid_t_start = 72
    test_t_start = 108

    forecasting_horizon = 12
    input_window_size = 24

    lead_time = 6
    scale01 = False
    target_service_level = 0.95
    N = args.N
    periodicity = 12
    first_cutoff = test_t_min

    dataset_factory = get_m3_data(
        forecasting_horizon=forecasting_horizon,
        minmax_scaling=scale01,
        train_prop=None,
        val_prop=None,
        batch_size=None,
        input_window_size=input_window_size,
        csv_fpath=data_fpath,
        default_nan_value=1e15,
        rolling_evaluation=True,
        idx_range=idx_range,
        N=N,
    )

    unit_costs = [
        (1, 1, 1e-06),
        (1, 1, 1e-05),
        (1, 2, 1e-06),
        (1, 2, 1e-05),
        (1, 10, 1e-06),
        (1, 10, 1e-05),
        (2, 1, 1e-06),
        (2, 1, 1e-05),
        (2, 2, 1e-06),
        (2, 2, 1e-05),
        (2, 10, 1e-06),
        (2, 10, 1e-05),
        (10, 1, 1e-06),
        (10, 1, 1e-05),
        (10, 2, 1e-06),
        (10, 2, 1e-05),
        (10, 10, 1e-06),
        (10, 10, 1e-05),
    ]
  else:
    assert dataset_name == 'favorita'
    test_t_max = 396
    valid_t_start = 334
    test_t_start = 364
    test_t_min = 180
    forecasting_horizon = 30
    input_window_size = 90
    first_cutoff = test_t_min

    df = get_favorita_df(impute0=True, N=args.N, test_t_max=test_t_max)
    idx_range = (0, len(df))

    lead_time = 7
    scale01 = False
    N = args.N
    target_service_level = 0.95
    periodicity = 7
    dataset_factory = get_favorita_data(
        forecasting_horizon=forecasting_horizon,
        minmax_scaling=scale01,
        input_window_size=input_window_size,
        data_fpath='../data/favorita/favorita_tensor_full.npy',
        default_nan_value=1e15,
        rolling_evaluation=True,
        N=N,
        test_t_max=test_t_max,
    )

    unit_costs = [
        (1, 1, 1e-02),
        (1, 1, 1e-03),
        (1, 2, 1e-02),
        (1, 2, 1e-03),
        (1, 10, 1e-02),
        (1, 10, 1e-03),
        (2, 1, 1e-02),
        (2, 1, 1e-03),
        (2, 2, 1e-02),
        (2, 2, 1e-03),
        (2, 10, 1e-02),
        (2, 10, 1e-03),
        (10, 1, 1e-02),
        (10, 1, 1e-03),
        (10, 2, 1e-02),
        (10, 2, 1e-03),
        (10, 10, 1e-02),
        (10, 10, 1e-03),
    ]

  all_forecasters = {
      'NaiveForecaster': NaiveForecaster(sp=periodicity),
      'ExponentialSmoothing': ExponentialSmoothing(
          trend='add', seasonal='add', sp=periodicity
      ),
      'DeseasonalizedThetaForecaster': DeseasonalizedForecaster(
          ThetaForecaster(deseasonalize=False), sp=periodicity
      ),
      'ARIMA': ARIMA(),
  }

  forecasters = {k: all_forecasters[k] for k in args.forecasters}
  folder = 'sktime_predictions_seasonal/'
  if not os.path.exists(folder):
    os.makedirs(folder)
  fc_to_preds = get_sktime_predictions(
      df,
      dataset_name,
      forecasters,
      idx_range,
      test_t_max,
      test_t_min,
      lead_time,
      first_cutoff,
      folder,
      parallel=parallel,
  )

  if args.preds_only:
    return

  tprint('Making evaluations...')
  start = time.time()
  naive_model = NaiveScalingBaseline(
      forecasting_horizon=lead_time, init_alpha=1.0, periodicity=12, frozen=True
  )
  for fc_name, test_preds in fc_to_preds.items():
    test_preds = torch.from_numpy(test_preds)
    for unit_cost in unit_costs:
      print(unit_cost)
      unit_holding_cost, unit_stockout_cost, unit_var_o_cost = unit_cost
      test_metrics, expanded_test_metrics = evaluate(
          dataset_factory,
          test_preds,
          naive_model,
          lead_time,
          scale01,
          test_t_min,
          target_service_level,
          unit_holding_cost,
          unit_stockout_cost,
          unit_var_o_cost,
          valid_t_start=valid_t_start,
          test_t_start=test_t_start,
          parallel=parallel,
          num_workers=num_workers,
      )
      print('getting summary...')
      test_results = get_summary(
          test_metrics=test_metrics,
          model_name=fc_name,
          optimization_obj='None',
          max_steps='None',
          start=start,
          unit_holding_cost=unit_holding_cost,
          unit_stockout_cost=unit_stockout_cost,
          unit_var_o_cost=unit_var_o_cost,
          valid_t_start=valid_t_start,
          learned_alpha=None,
          quantile_loss=None,
          naive_model=naive_model,
          use_wandb=False,
          expanded_test_metrics=expanded_test_metrics,
          idx_range=None,
      )

      summary = test_results['summary']
      now = datetime.datetime.now()
      now = now.strftime('%m-%d-%Y-%H:%M:%S')
      tags = ['sktime']
      tag_str = ''.join(tags)
      if wandb_log:
        wandb.init(
            name=f'{tag_str}_{now}_{fc_name}_summary',
            project='sktime-seasonal-summaries',
            reinit=True,
            tags=tags,
            config={
                'model_name': fc_name,
                'unit_holding_cost': unit_holding_cost,
                'unit_stockout_cost': unit_stockout_cost,
                'unit_var_o_cost': unit_var_o_cost,
                'dataset_name': args.dataset,
            },
        )
        wandb.log(
            {
                'combined_test_perfs': wandb.Table(
                    dataframe=pd.DataFrame([summary])
                )
            }
        )
        wandb.finish()


if __name__ == '__main__':
  main()
