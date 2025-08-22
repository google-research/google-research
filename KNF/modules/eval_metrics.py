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

"""Evaluation Metric functions."""
import numpy as np


def SMAPE(test_preds, test_tgts):
  """Metric for M4 dataset.

  Refer to
  https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py

  Args:
    test_preds: models' predictions with shape of
    (num_samples, forecasting horizon, ...)
    test_tgts: ground truth that has the same shape as test_preds.

  Returns:
    short, medium, long forecasting horizon prediction sMAPE.
  """
  smape = np.abs(test_preds - test_tgts) * 200 / (
      np.abs(test_preds) + np.abs(test_tgts))
  fh = test_preds.shape[1]
  # return short, medium, long forecasting horizon and total sMAPE
  return np.round(np.mean(smape[:, :fh // 3]),
                  3), np.round(np.mean(smape[:, fh // 3:fh // 3 * 2]),
                               3), np.round(np.mean(smape[:, -fh // 3:]),
                                            3), np.round(np.mean(smape), 3)


def WRMSE(
    test_preds,
    test_tgts,  # ground truth that has the same shape as test_preds
    weights=np.array([
        4.30406509320417, 6.779921907472252, 2.3978952727983707,
        4.406719247264253, 3.555348061489413, 1.3862943611198906,
        5.8944028342648505, 2.079441541679836, 1.0986122886681098,
        2.3978952727983707, 1.0986122886681098, 1.6094379124341005,
        2.079441541679836, 1.791759469228055
    ])  # Importance weights for 14 Cryptos
):
  """Metric for Cryptos return predictions.

  RMSE should be weighted by the importance of each stock
  Refer to https://www.kaggle.com/competitions/g-research-crypto-forecasting/
  data?select=asset_details.csv

  Args:
    test_preds: models' predictions with shape of (number of stocks,
    number of samples for each stock, forecasting horizons, 8 features)
    test_tgts: ground truth that has the same shape as test_preds.
    weights: Importance weights for 14 Cryptos

  Returns:
    short, medium, long forecasting horizon prediction Weighted RMSE.
  """

  weights = np.expand_dims(weights, axis=(1, 2, 3))

  fh = test_preds.shape[2]
  wrmse = ((test_preds - test_tgts) * weights)**2

  # only evaluate predictions based on the last feature
  # (15-min ahead residulized returns)
  # return short, medium, long forecasting horizon and total Weighted RMSE
  return (
      np.sqrt(np.mean(wrmse[Ellipsis, : fh // 3, -1])),
      np.sqrt(np.mean(wrmse[Ellipsis, fh // 3 : fh // 3 * 2, -1])),
      np.sqrt(np.mean(wrmse[Ellipsis, -fh // 3, -1])),
      np.sqrt(np.mean(wrmse[Ellipsis, -1])),
  )


def RMSE(
    test_preds,
    test_tgts
):
  """Regular RMSE metric for basketball player trajectory predictions.

  Args:
    test_preds: # models' predictions with shape of (number of trajectories,
    number of samples for traj, forecasting horizons, 2 velocity components)
    test_tgts: ground truth that has the same shape as test_preds.

  Returns:
    short, medium, long forecasting horizon prediction Weighted RMSE.
  """
  fh = test_preds.shape[2]
  mse = np.mean((test_preds - test_tgts)**2, axis=0)
  # return short, medium, long forecasting horizon and total RMSE
  return np.sqrt(np.mean(mse[:, :fh // 3])), np.sqrt(
      np.mean(mse[:, fh // 3:fh // 3 * 2])), np.sqrt(
          np.mean(mse[:, fh // 3 * 2:])), np.sqrt(np.mean(mse))
