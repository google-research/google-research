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

"""Three-submodel quantile GBDT architecture, training, and probabilistic inference.

Flowracle predicts 13 conditional quantiles (`tau in {0.01, 0.05, 0.10, ...,
0.99}`)
of log-transformed monthly Full Natural Flow (`log1p(FNF / 10^4 AF)`) across 25
California CDEC watersheds and 6 monthly lead times (`l in {1, ..., 6}`) by
combining three specialized XGBoost quantile-regression sub-models:

  1. **Anchor (`models_elev`, 44 features)**: Shallow, strongly regularized
     baseline (`max_depth=5`, `n_estimators=4000`, `reg_lambda=600`) operating
     on static watershed coordinates, regional accumulations, and 3D-IDW snow
     and runoff predictors while excluding snowpack ripeness and time-since-peak
     counters.
  2. **Core (`models_melt`, 51 features)**: Deep nonlinear snowmelt learner
     (`max_depth=9`, `n_estimators=7000`, `reg_lambda=1100`) incorporating
     regional snowpack ripeness and time-since-peak-SWE counters to resolve
     elevation-dependent melt dynamics.
  3. **Anomaly (`models_resid`, 47 features)**: Intermediate-depth residual
     learner (`max_depth=6`, `n_estimators=6000`, `reg_lambda=575`) predicting
     deviations relative to each watershed and target month's median historical
     log-climatology from climatology-centered predictor anomalies, with static
     station coordinates removed to prevent spatial overfitting.

Sub-model predictions are combined via issuance-month-specific out-of-fold (OOF)
inverse-RMSE skill weights (`ENSEMBLE_POWER = 4.2`) shrunk toward uniform `1/3`
weights (`SHRINKAGE_FACTOR = 0.18`), shifted in log-space by
`LOG_BIAS_CORRECTION_FACTOR * sigma2_resid` to prevent positive volumetric bias
upon `expm1` back-transformation, sorted across quantiles to enforce
non-crossing,
and linearly interpolated to an `M = 100` member ensemble.
"""

import gc
import os
import pickle
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from flowracle.features import AUGMENTATION_EPOCHS
from flowracle.features import batch_to_table
from flowracle.features import CLIM_PERCENTILE
from flowracle.features import ENSEMBLE_POWER
from flowracle.features import LEAD_REG_GROWTH
from flowracle.features import LEAD_RELAX_SLOPE
from flowracle.features import LOG_BIAS_CORRECTION_FACTOR
from flowracle.features import NUM_ENSEMBLE
from flowracle.features import OUTER_QUANTILE_RELAX_FACTOR
from flowracle.features import POS_TERMS
from flowracle.features import QUANTILES
from flowracle.features import SHRINKAGE_FACTOR
from flowracle.features import TARGET_SCALE
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb

DEFAULT_MODEL_FILENAME = "xgb_1634_v135_ensemble.pkl"
NUM_COLLECT_BATCHES = 850


def detect_xgb_device():
  """Detects whether a CUDA GPU is available for XGBoost, falling back to CPU."""
  env_device = os.environ.get("FLOWRACLE_XGB_DEVICE", "").strip()
  if env_device:
    return env_device
  if shutil.which("nvidia-smi") is not None:
    try:
      proc = subprocess.run(
          ["nvidia-smi", "-L"],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          timeout=3,
          check=False,
      )
      if proc.returncode == 0 and b"GPU " in proc.stdout:
        return "cuda"
    except OSError:
      pass
    except subprocess.SubprocessError:
      pass
  return "cpu"


def get_submodel_xgb_params(
    device = None,
    n_estimators_override = None,
):
  """Returns `(params_elev, params_melt, params_resid)` for the 3 sub-models."""
  resolved_device = device if device is not None else detect_xgb_device()
  base = dict(
      tree_method="hist",
      device=resolved_device,
      subsample=0.85,
      colsample_bytree=0.55,
      random_state=42,
  )
  params_elev = {
      **base,
      "n_estimators": n_estimators_override or 4000,
      "max_depth": 5,
      "learning_rate": 0.008,
      "reg_lambda": 600.0,
      "min_child_weight": 80,
  }
  params_melt = {
      **base,
      "n_estimators": n_estimators_override or 7000,
      "max_depth": 9,
      "learning_rate": 0.005,
      "reg_lambda": 1100.0,
      "min_child_weight": 135,
  }
  params_resid = {
      **base,
      "n_estimators": n_estimators_override or 6000,
      "max_depth": 6,
      "learning_rate": 0.005,
      "reg_lambda": 575.0,
      "min_child_weight": 95,
  }
  return params_elev, params_melt, params_resid


def compute_submodel_feature_masks(
    feature_names,
):
  """Computes boolean feature masks `(feat_clim_mask, elev_mask, melt_mask, resid_mask)`."""
  static_or_lead = {
      "lead_sin",
      "lead_cos",
      "is_amjj",
      "stn_lat",
      "stn_lon",
      "stn_elev",
  }
  feat_clim_mask = np.array([n not in static_or_lead for n in feature_names])

  melt_f = {n for n in feature_names if "time_since_peak" in n}
  melt_mask = np.array([
      not any(
          x in n for x in ["monthly_mjo", "mjo_sin", "mjo_cos", "l_nbr_fnf_avg"]
      )
      for n in feature_names
  ])
  elev_mask = np.array([
      n not in melt_f
      and not any(
          x in n for x in ["monthly_mjo", "mjo_sin", "mjo_cos", "_ripeness"]
      )
      for n in feature_names
  ])
  resid_mask = np.array([
      n not in melt_f
      and not any(x in n for x in ["stn_lat", "stn_lon", "stn_elev", "is_amjj"])
      for n in feature_names
  ])
  return feat_clim_mask, elev_mask, melt_mask, resid_mask


def build_monotone_constraints(
    selected_feature_names,
):
  """Returns `+1` / `0` monotone constraint tuple for a sub-model's features."""
  return tuple(
      1 if any(term in name.lower() for term in POS_TERMS) else 0
      for name in selected_feature_names
  )


def _default_model_dir():
  return os.environ.get("FLOWRACLE_OUTPUT_DIR", "./model_output")


def fit_model(
    train_iter,
    prediction_batch_list,
    aux_ds,
    *,
    model_dir = None,
    num_collect_batches = NUM_COLLECT_BATCHES,
    augmentation_epochs = AUGMENTATION_EPOCHS,
    n_estimators_override = None,
    n_oof_splits = 4,
    device = None,
):
  """Fits the 3-submodel, 13-quantile Flowracle ensemble and writes checkpoint.

  Args:
    train_iter: Iterator yielding causal training batches.
    prediction_batch_list: Unused during training; accepted for API parity with
      standard `(train_iter, eval_list, aux_ds)` experiment runners.
    aux_ds: Static watershed and snow-sensor metadata dataset.
    model_dir: Directory where `xgb_1634_v135_ensemble.pkl` is saved.
    num_collect_batches: Maximum number of batches drawn from `train_iter`.
    augmentation_epochs: Number of stochastic jitter passes per training batch.
    n_estimators_override: Optional tree count override (e.g. for fast CPU
      tests).
    n_oof_splits: Number of K-Fold splits for out-of-fold skill & variance
      estimation (`4` in production).
    device: Optional XGBoost device (`'cuda'` or `'cpu'`).

  Returns:
    Serialized model dictionary containing trained boosters, climatology tables,
    issuance-month ensemble weights, OOF residual variance, and feature masks.
  """
  del prediction_batch_list  # Explicitly unused to guarantee zero test leakage.
  np.random.seed(42)
  out_dir = model_dir if model_dir is not None else _default_model_dir()
  os.makedirs(out_dir, exist_ok=True)

  train_iterator = iter(train_iter)
  if aux_ds is not None:
    s_count = len(aux_ds.station_id)
    coords_np = np.stack(
        [
            (aux_ds.latitude.values - 38.0) / 4.0,
            (aux_ds.longitude.values - 240.0) / 4.0,
            np.nan_to_num(aux_ds.elevation.values / 7500.0, 0.0),
        ],
        -1,
    ).astype(np.float32)
  else:
    import itertools  # pylint: disable=g-import-not-at-top
    from flowracle.data_pipeline import get_batch_predictors_ds  # pylint: disable=g-import-not-at-top
    from flowracle.data_pipeline import get_batch_target_ds  # pylint: disable=g-import-not-at-top

    try:
      first_batch = next(train_iterator)
    except StopIteration as exc:
      raise ValueError("`train_iter` yielded zero batches.") from exc
    preds_first = get_batch_predictors_ds(first_batch)
    if "past_fnf" in preds_first and "fnf" in preds_first["past_fnf"]:
      s_count = int(preds_first["past_fnf"]["fnf"].shape[-1])
    elif "target_fnf" in first_batch:
      s_count = int(
          get_batch_target_ds(first_batch)["target_fnf_fnf"].shape[-1]
      )
    else:
      s_count = 1
    coords_np = np.zeros((s_count, 3), dtype=np.float32)
    train_iterator = itertools.chain([first_batch], train_iterator)

  x_l, y_l, m_iss_l, t_mon_l, s_idx_l, l_idx_l = [], [], [], [], [], []
  f_names: Optional[List[str]] = None

  for _ in range(num_collect_batches):
    try:
      batch = next(train_iterator)
      for _ in range(augmentation_epochs):
        x_mat, y_vec, names, m_iss_batch = batch_to_table(
            batch, aux_ds, coords_np, jitter=True
        )
        f_names = names
        b_curr = len(m_iss_batch) // (6 * s_count)
        l_idx_b = np.repeat(np.arange(6), s_count * b_curr)
        s_idx_b = np.tile(np.repeat(np.arange(s_count), b_curr), 6)
        x_l.append(x_mat)
        y_l.append(y_vec)
        m_iss_l.append(m_iss_batch)
        t_mon_l.append((m_iss_batch + l_idx_b - 1) % 12 + 1)
        s_idx_l.append(s_idx_b)
        l_idx_l.append(l_idx_b)
      del batch
    except StopIteration:
      break
  gc.collect()

  if not x_l or f_names is None:
    raise ValueError("`train_iter` yielded zero batches.")

  x_tr = np.concatenate(x_l, 0)
  y_tr = np.concatenate(y_l, 0)
  m_iss_all = np.concatenate(m_iss_l, 0)
  t_mon_all = np.concatenate(t_mon_l, 0)
  s_idx_all = np.concatenate(s_idx_l, 0)
  l_idx_all = np.concatenate(l_idx_l, 0)
  del x_l, y_l, m_iss_l, t_mon_l, s_idx_l, l_idx_l
  gc.collect()

  # Station x target-month median climatology table for the Anomaly sub-model.
  clim_tab = np.zeros((s_count, 13), dtype=np.float32)
  feat_clim_tab = np.zeros((s_count, 13, x_tr.shape[1]), dtype=np.float32)
  for s in range(s_count):
    for m in range(1, 13):
      mask = (s_idx_all == s) & (t_mon_all == m)
      if mask.any():
        clim_tab[s, m] = np.percentile(y_tr[mask], CLIM_PERCENTILE)
        feat_clim_tab[s, m] = np.percentile(x_tr[mask], CLIM_PERCENTILE, axis=0)

  y_resid = y_tr - clim_tab[s_idx_all, t_mon_all]
  feat_mask, elev_mask, melt_mask, resid_mask = compute_submodel_feature_masks(
      f_names
  )
  x_tr_resid = x_tr.copy()
  x_tr_resid[:, feat_mask] -= feat_clim_tab[s_idx_all, t_mon_all][:, feat_mask]

  # Upweight April-July snowmelt target months (4:1 vs. non-AMJJ months).
  is_amjj_target = (t_mon_all >= 4) & (t_mon_all <= 7)
  w_amjj = np.where(is_amjj_target, 2.0, 0.5).astype(np.float32)
  sample_weights = (w_amjj / np.mean(w_amjj)).astype(np.float32)

  params_elev, params_melt, params_resid = get_submodel_xgb_params(
      device=device, n_estimators_override=n_estimators_override
  )
  f_names_arr = np.array(f_names)

  def train_q(
      x_data,
      y_data,
      params,
      mask,
      sw = None,
  ):
    mono = build_monotone_constraints(f_names_arr[mask].tolist())
    models: Dict[float, xgb.XGBRegressor] = {}
    for q in QUANTILES:
      q_params = params.copy()
      if q <= 0.05 or q >= 0.95:
        relax = (
            OUTER_QUANTILE_RELAX_FACTOR * (1.0 + LEAD_RELAX_SLOPE)
            if LEAD_RELAX_SLOPE > 0
            else OUTER_QUANTILE_RELAX_FACTOR
        )
        q_params["reg_lambda"] = q_params.get("reg_lambda", 500.0) * relax
      if LEAD_REG_GROWTH > 0:
        q_params["reg_lambda"] = q_params.get("reg_lambda", 500.0) * (
            1.0 + LEAD_REG_GROWTH * 0.5
        )
      models[q] = xgb.XGBRegressor(
          objective="reg:quantileerror",
          quantile_alpha=q,
          monotone_constraints=mono,
          **q_params,
      ).fit(x_data[:, mask], y_data, sample_weight=sw)
    return models

  # Out-of-fold K-Fold pass to estimate monthly sub-model weights and log-space
  # residual variance `sigma2_resid` for Jensen back-transformation smearing.
  kf = KFold(n_splits=n_oof_splits, shuffle=True, random_state=42)
  oof = np.zeros((len(y_tr), 3))
  mono_elev = build_monotone_constraints(f_names_arr[elev_mask].tolist())
  mono_melt = build_monotone_constraints(f_names_arr[melt_mask].tolist())
  mono_resid = build_monotone_constraints(f_names_arr[resid_mask].tolist())

  for tr_idx, val_idx in kf.split(x_tr):
    sw_tr = sample_weights[tr_idx]
    m1 = xgb.XGBRegressor(monotone_constraints=mono_elev, **params_elev).fit(
        x_tr[tr_idx][:, elev_mask], y_tr[tr_idx], sample_weight=sw_tr
    )
    m2 = xgb.XGBRegressor(monotone_constraints=mono_melt, **params_melt).fit(
        x_tr[tr_idx][:, melt_mask], y_tr[tr_idx], sample_weight=sw_tr
    )
    m3 = xgb.XGBRegressor(monotone_constraints=mono_resid, **params_resid).fit(
        x_tr_resid[tr_idx][:, resid_mask],
        y_resid[tr_idx],
        sample_weight=sw_tr,
    )

    oof[val_idx, 0] = m1.predict(x_tr[val_idx][:, elev_mask])
    oof[val_idx, 1] = m2.predict(x_tr[val_idx][:, melt_mask])
    oof[val_idx, 2] = (
        m3.predict(x_tr_resid[val_idx][:, resid_mask])
        + clim_tab[s_idx_all[val_idx], t_mon_all[val_idx]]
    )

  ens_weights: Dict[int, np.ndarray] = {}
  for m in range(1, 13):
    mask = m_iss_all == m
    if mask.any():
      rmse = np.sqrt(np.mean((oof[mask] - y_tr[mask, None]) ** 2, axis=0))
      inv_skill = 1.0 / (rmse**ENSEMBLE_POWER + 1e-9)
      w = inv_skill / np.sum(inv_skill)
      ens_weights[m] = (
          SHRINKAGE_FACTOR / 3.0 + (1.0 - SHRINKAGE_FACTOR) * w
      ).astype(np.float32)
    else:
      ens_weights[m] = np.array(
          [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32
      )

  oof_ens = np.zeros(len(y_tr), dtype=np.float32)
  for i, m in enumerate(m_iss_all):
    w = ens_weights.get(int(m), np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    oof_ens[i] = w[0] * oof[i, 0] + w[1] * oof[i, 1] + w[2] * oof[i, 2]
  sigma2_resid = float(np.mean((oof_ens - y_tr) ** 2))

  sigma2_lead = np.zeros(6, dtype=np.float32)
  for l_i in range(6):
    mask_l = l_idx_all == l_i
    sigma2_lead[l_i] = (
        float(np.mean((oof_ens[mask_l] - y_tr[mask_l]) ** 2))
        if mask_l.any()
        else sigma2_resid
    )

  model_data: Dict[str, Any] = {
      "models_elev": train_q(
          x_tr, y_tr, params_elev, elev_mask, sw=sample_weights
      ),
      "models_melt": train_q(
          x_tr, y_tr, params_melt, melt_mask, sw=sample_weights
      ),
      "models_resid": train_q(
          x_tr_resid, y_resid, params_resid, resid_mask, sw=sample_weights
      ),
      "clim_table": clim_tab,
      "sigma2_resid": sigma2_resid,
      "sigma2_lead": sigma2_lead,
      "ensemble_weights": ens_weights,
      "coords": coords_np,
      "names": f_names,
      "feat_clim_table": feat_clim_tab,
      "feat_mask": feat_mask,
      "masks": (elev_mask, melt_mask, resid_mask),
  }
  ckpt_path = os.path.join(out_dir, DEFAULT_MODEL_FILENAME)
  with open(ckpt_path, "wb") as f:
    pickle.dump(model_data, f)
  return model_data


def load_model(
    model_dir = None,
    filename = DEFAULT_MODEL_FILENAME,
):
  """Loads a serialized Flowracle ensemble dictionary from disk."""
  resolved_dir = model_dir if model_dir is not None else _default_model_dir()
  ckpt_path = (
      resolved_dir
      if os.path.isfile(resolved_dir)
      else os.path.join(resolved_dir, filename)
  )
  import warnings  # pylint: disable=g-import-not-at-top

  with open(ckpt_path, "rb") as f:
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message=".*No visible GPU is found.*")
      warnings.filterwarnings(
          "ignore", message=".*Device is changed from GPU to CPU.*"
      )
      warnings.filterwarnings("ignore", message=".*Changing updater from.*")
      data = pickle.load(f)  # pylint: disable=g-unsafe-pickle-load

  # Ensure checkpoints trained on CUDA run without device warnings on CPU hosts.
  if detect_xgb_device() == "cpu":
    for group_key in ("models_elev", "models_melt", "models_resid"):
      for booster in data.get(group_key, {}).values():
        try:
          booster.set_params(device="cpu")
        except Exception:  # pylint: disable=broad-except
          pass
  return data


def predict(
    prediction_batch_list,
    aux_ds = None,
    *,
    model = None,
    model_dir = None,
):
  """Generates 100-member probabilistic FNF forecasts in Acre-Feet.

  Args:
    prediction_batch_list: Sequence of causal evaluation batches.
    aux_ds: Static watershed and sensor metadata dataset.
    model: Pre-loaded model dictionary; if None, loads from `model_dir`.
    model_dir: Directory containing `xgb_1634_v135_ensemble.pkl` when `model` is
      None.

  Returns:
    Array of shape `(100, N_dates, 6, N_stations)` containing non-negative
    monthly Full Natural Flow ensemble forecasts in Acre-Feet (AF).
  """
  data = model if model is not None else load_model(model_dir=model_dir)
  coords_np = data["coords"]
  ens_qs = np.linspace(0.005, 0.995, NUM_ENSEMBLE)
  feat_mask = data["feat_mask"]
  feat_clim_tab = data["feat_clim_table"]
  elev_mask, melt_mask, resid_mask = data["masks"]
  sigma2_resid = data.get("sigma2_resid", 0.15)
  shift = float(LOG_BIAS_CORRECTION_FACTOR * sigma2_resid)

  if not prediction_batch_list:
    return np.empty((NUM_ENSEMBLE, 0, 6, coords_np.shape[0]), dtype=np.float64)

  n_leads = 6
  n_stations = coords_np.shape[0]
  x_blocks: List[np.ndarray] = []
  x_resid_blocks: List[np.ndarray] = []
  m_iss_blocks: List[np.ndarray] = []
  s_idx_blocks: List[np.ndarray] = []
  t_mon_blocks: List[np.ndarray] = []
  batch_sizes: List[int] = []

  for batch in prediction_batch_list:
    x_te, _, _, m_iss_arr = batch_to_table(
        batch, aux_ds, coords_np, jitter=False
    )
    batch_size = len(m_iss_arr) // (n_leads * n_stations)
    l_idx = np.repeat(np.arange(n_leads), n_stations * batch_size)
    s_idx = np.tile(np.repeat(np.arange(n_stations), batch_size), n_leads)
    t_mon = (m_iss_arr + l_idx - 1) % 12 + 1

    x_te_resid = x_te.copy()
    x_te_resid[:, feat_mask] -= feat_clim_tab[s_idx, t_mon][:, feat_mask]

    x_blocks.append(x_te)
    x_resid_blocks.append(x_te_resid)
    m_iss_blocks.append(m_iss_arr)
    s_idx_blocks.append(s_idx)
    t_mon_blocks.append(t_mon)
    batch_sizes.append(batch_size)

  x_all = np.concatenate(x_blocks, axis=0)
  x_resid_all = np.concatenate(x_resid_blocks, axis=0)
  m_iss_all = np.concatenate(m_iss_blocks, axis=0)
  s_idx_all = np.concatenate(s_idx_blocks, axis=0)
  t_mon_all = np.concatenate(t_mon_blocks, axis=0)

  p1 = np.column_stack(
      [data["models_elev"][q].predict(x_all[:, elev_mask]) for q in QUANTILES]
  )
  p2 = np.column_stack(
      [data["models_melt"][q].predict(x_all[:, melt_mask]) for q in QUANTILES]
  )
  p3 = (
      np.column_stack([
          data["models_resid"][q].predict(x_resid_all[:, resid_mask])
          for q in QUANTILES
      ])
      + data["clim_table"][s_idx_all, t_mon_all][:, None]
  )

  default_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
  weights_all = np.array(
      [data["ensemble_weights"].get(int(m), default_w) for m in m_iss_all]
  )
  q_out = (
      weights_all[:, 0:1] * p1
      + weights_all[:, 1:2] * p2
      + weights_all[:, 2:3] * p3
  )
  q_out_sorted = np.sort(np.maximum(0.0, q_out - shift), axis=1)
  ens_all = np.vstack(
      [np.interp(ens_qs, QUANTILES, row) for row in q_out_sorted]
  )

  all_preds: List[np.ndarray] = []
  row_offset = 0
  for batch_size in batch_sizes:
    n_rows = n_leads * n_stations * batch_size
    ens_chunk = ens_all[row_offset : row_offset + n_rows]
    vol = np.maximum(
        np.expm1(
            ens_chunk.reshape(n_leads, n_stations, batch_size, NUM_ENSEMBLE)
        )
        * TARGET_SCALE,
        0.0,
    ).transpose(3, 2, 0, 1)
    all_preds.append(vol)
    row_offset += n_rows

  return np.concatenate(all_preds, axis=1)


def fit_and_predict_fn(
    train_iter,
    eval_list,
    aux_ds = None,
    *,
    model_dir = None,
):
  """Fits Flowracle on `train_iter` and returns ensemble predictions on `eval_list`."""
  m_data = fit_model(train_iter, eval_list, aux_ds, model_dir=model_dir)
  return predict(eval_list, aux_ds, model=m_data)
