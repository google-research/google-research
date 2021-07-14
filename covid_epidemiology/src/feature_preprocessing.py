# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Helper functions for pre-processing model features."""
import collections
import datetime
import functools

import dataclasses
import numba
import numpy as np
import pandas as pd
from scipy.signal import windows
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from covid_epidemiology.src import constants


@dataclasses.dataclass
class FeaturePreprocessingConfig:
    # For time-series and static features
    imputation_strategy: str = "median"
    standardize: bool = True

    # For time-series features only
    ffill_features: bool = True
    bfill_features: bool = True


def preprocess_static_feature(
    static_feature_dict, imputation_strategy="median", standardize=False
):
    """Preprocessing for a dictionary of static features.

    Args:
      static_feature_dict: Dictionary of float values.
      imputation_strategy: "median" or "mean" or "most_frequent" imputation.
      standardize: If true, apply MinMax scaling.

    Returns:
      Preprocessing static features array and the fitted scaler.
    """
    if imputation_strategy not in ["mean", "median", "most_frequent", "constant"]:
        raise ValueError(
            "Given imputation strategy {} is not supported. "
            "Use value from ['mean', 'median', 'most_frequent', 'constant'] "
            "as an imputation strategy.".format(imputation_strategy)
        )

    location_keys = list(static_feature_dict.keys())

    static_values = np.empty([len(location_keys), 1])
    for idx, location_key in enumerate(location_keys):
        if static_feature_dict[location_key] is not None:
            static_values[idx] = static_feature_dict[location_key]
        else:
            static_values[idx] = np.nan

    if imputation_strategy:
        # Check whether all elements are NaNs.
        if np.all(np.isnan(static_values)):
            static_values = np.zeros_like(static_values)
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
            static_values = imputer.fit_transform(static_values)

    scaler = None
    if standardize:
        # Check if there are at least two elements with different values to avoid
        # issues.
        if np.unique(static_values).size < 2:
            static_values = np.zeros_like(static_values)
        else:
            scaler = preprocessing.MinMaxScaler()
            static_values = scaler.fit_transform(static_values)

    static_feature_dict = {}
    for ind, location_key in enumerate(location_keys):
        static_feature_dict[location_key] = static_values[ind, 0]

    return static_feature_dict, scaler


def cumulative_sum_ts_feature(
    ts_feature,
    initial_value=None,
):
    """Compute and return the cumulative sum of a feature.

    Args:
      ts_feature: Time-series feature in form of dictionary of lists.
      initial_value: If None no actions will be taken. Otherwise, the first value
        for each location will be set to this value if it is null.

    Returns:
      Array containing the cumulative sum of the time-series feature.
    """

    ts_feature_cumsum = {}
    for location, feature in ts_feature.items():
        df = pd.Series(feature)  # DataFrame of 1 column
        if initial_value is not None and np.isnan(df.iloc[0]):
            df.iloc[0] = initial_value
        df = df.cumsum(skipna=True)  # ignore NaNs when cumulating
        ts_feature_cumsum[location] = df.to_numpy().ravel()
    return ts_feature_cumsum


def preprocess_ts_feature(
    ts_feature,
    ffill_features=False,
    bfill_features=False,
    imputation_strategy="median",
    standardize=False,
    fitting_window=0,
    initial_value=None,
):
    """Preprocessing for a time series features array.

    Args:
      ts_feature: Time-series feature in form of dictionary of lists.
      ffill_features: If true, apply forward filling.
      bfill_features: If true, apply backwards filling.
      imputation_strategy: "median" or "mean" or "most_frequent" imputation.
      standardize: If true, apply MinMax scaling.
      fitting_window: Duration of the window to compute th statistics.
      initial_value: If None no actions will be taken. Otherwise, the first value
        for each location will be set to this value if it is null.

    Returns:
      Preprocessed time-series features array and the fitted scaler.
    """
    if imputation_strategy not in [
        "mean",
        "median",
        "most_frequent",
        "constant",
        "ffill-zero",
    ]:
        raise ValueError(
            "Given imputation strategy {} is not supported. "
            "Use value from ['mean', 'median', 'most_frequent', 'constant', 'ffill-zero'] "
            "as an imputation strategy.".format(imputation_strategy)
        )

    preprocessed_ts_feature = {}
    all_data_arrays = np.asarray([])
    train_data_arrays = np.asarray([])
    all_locations = ts_feature.keys()
    for location in all_locations:

        df = pd.DataFrame(ts_feature[location])
        if initial_value is not None:
            df.iloc[0, :] = df.iloc[0, :].fillna(value=initial_value)

        # Do forward filling in time.
        if ffill_features:
            df = df.fillna(method="ffill", axis=0)

        # Do backwards filling in time.
        if bfill_features:
            df = df.fillna(method="bfill", axis=0)

        data_array = df.to_numpy()

        all_data_arrays = np.append(all_data_arrays, data_array)
        train_data_arrays = np.append(train_data_arrays, data_array[:fitting_window])

    # If all the data is NaN, fill 0.
    if np.sum(np.isnan(all_data_arrays) == 0) == 0:
        all_data_arrays = np.nan_to_num(all_data_arrays, copy=False, nan=0)
        train_data_arrays = np.nan_to_num(train_data_arrays, copy=False, nan=0)

    if all_locations:
        # noinspection PyUnboundLocalVariable
        duration = np.size(data_array)

    train_data_arrays = np.expand_dims(train_data_arrays, 1)
    all_data_arrays = np.expand_dims(all_data_arrays, 1)

    # Impute the remaining NaNs.
    imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    imputer.fit(train_data_arrays)
    all_data_arrays = imputer.transform(all_data_arrays)

    # Standardize, considering temporal and spatial statistics.
    if standardize:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_data_arrays)
        all_data_arrays = scaler.transform(all_data_arrays)

    else:
        scaler = None

    for ind, location in enumerate(all_locations):
        # noinspection PyUnboundLocalVariable
        preprocessed_ts_feature[location] = all_data_arrays[
            ind * duration : (ind + 1) * duration, 0
        ]

    return preprocessed_ts_feature, scaler


def normalize_ts_feature(
    ts_feature, ts_normalize, epsilon=1.0, upper_limit=1e8, lower_limit=-1e8
):
    """Normalizes a time series features array with another array.

    Args:
      ts_feature: Time-series feature in form of dictionary of lists.
      ts_normalize: Time-series feature to normalize with.
      epsilon: A small value to prevent division by zero.
      upper_limit: Maximum value of the ratio.
      lower_limit: Minimum value of the ratio.

    Returns:
      Normalized time-series features array.
    """

    normalized_ts_feature = {}
    for location in ts_feature.keys():
        if ts_normalize[location] is None:
            normalized_ts_feature[location] = ts_feature[location] * np.nan
        else:
            normalized_ts_feature[location] = ts_feature[location] / (
                ts_normalize[location] + epsilon
            )
        normalized_ts_feature[location][
            normalized_ts_feature[location] > upper_limit
        ] = np.nan
        normalized_ts_feature[location][
            normalized_ts_feature[location] < lower_limit
        ] = np.nan

    return normalized_ts_feature


def normalize_static_feature(static_numerator, static_denomenator):
    """Normalizes a static covariate with another static covariate.

    Args:
      static_numerator: The numerator
      static_denomenator: The denominator

    Returns:
      The numerator divided by the denominator for all locations in the numerator.
      When the denominator is
    """
    normalized_feature = {}
    for location, value in static_numerator.items():
        normalized_feature[location] = divide_unless_none_or_nan(
            np.float32(value), np.float32(static_denomenator[location])
        )

    return normalized_feature


def divide_unless_none_or_nan(numerator, denominator):
    if numerator is None or denominator is None or np.isnan(denominator):
        return None
    else:
        return numerator / denominator


def construct_feature_ratios(features, local_window=7, epsilon=1e-8):
    """Construct features of max to mean ratios.

    Given a feature of cumulations, construct the max to mean ratio of all local
    windows.

    Args:
      features: cumulated features.
      local_window: size of the local window.
      epsilon: epsilon to prevent numerical error.

    Returns:
      normalized trend features in [0, 1].
    """
    rate = dict()
    for key in features.keys():
        rate[key] = np.zeros(features[key].shape)
        cul = features[key]
        rate[key][0] = 1.0 / local_window
        for i in range(1, len(features[key])):
            start_idx = max(0, i - local_window + 1)
            rate[key][i] = np.max(cul[start_idx : i + 1]) / (
                np.sum(cul[start_idx : i + 1]) + epsilon
            )
    return rate


def _assert_feature_lengths_for_location(
    ts_features,
    location,
    reference_key=constants.DEATH,
):
    expected_ts_length = len(ts_features[reference_key][location])
    for feature_name in ts_features:
        if len(ts_features[feature_name][location]) != expected_ts_length:
            raise ValueError(
                f"Location: {location} has different size missmatch for feature: "
                f"{feature_name}, was expected to match size for confirmed_gt"
            )


def get_all_valid_dates(ts_data):
    """Return a list of valid dates for the timeseries data."""
    last_date = ts_data[constants.DATE_COLUMN].max()
    start_date = pd.to_datetime(constants.STARTING_DATE)
    delta = last_date - start_date

    all_dates = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    print(
        f"Loading covariates for {len(all_dates)} days between {all_dates[0]} "
        f"and {all_dates[-1]}"
    )
    return all_dates


def ts_feature_df_to_nested_dict(
    ts_data,
    locations,
    all_dates,
    feature_name_map,
    location_col_name,
):
    """Convert a DataFrame to a dictionary of dictionaries.

    Args:
      ts_data: A DataFrame with the input features.
      locations: All the locations that we have data for.
      all_dates: All the dates that we have data for.
      feature_name_map: A dictionary of the feature aliases tp the `feature_name`
        in the DataFrame.
      location_col_name: The name of the column that has the location information.

    Returns:
      A mapping from the feature name to its value, the value of each feature
        is map from location to np.ndarray.
    """
    # 3 level dictionary representing feature_name, location and date_index.
    num_dates = len(all_dates)
    ts_features = collections.defaultdict(
        functools.partial(
            collections.defaultdict,
            functools.partial(np.zeros, shape=(num_dates,), dtype="float32"),
        )
    )

    dt_index = {
        pd.Timestamp(dt).to_pydatetime(): idx for idx, dt in enumerate(all_dates)
    }

    # We filter to locations later because we want to keep features that aren't in
    # all locations.
    ts_data = ts_data[ts_data[constants.DATE_COLUMN].isin(dt_index)].copy()
    # Map dates to the index and features to their alias
    ts_data["date_index"] = ts_data[constants.DATE_COLUMN].map(dt_index)
    # Create a series mapping the feature, date, and location to the value.
    multi_series = ts_data.set_index(
        [constants.FEATURE_NAME_COLUMN, "date_index", location_col_name]
    )[constants.FEATURE_VALUE_COLUMN].astype("float32")

    # Create a multi-index DataFrame where the columns are the locations.
    multi_df = (
        multi_series[~multi_series.index.duplicated(keep="last")]
        .unstack(location_col_name)
        .reindex(columns=locations)
    )
    all_date_index = list(dt_index.values())

    # Go through features and convert to a dictionary
    features_in_data = multi_df.index.unique(level=0)

    # calculate once outside the feature loop
    dow_for_dates = np.asarray([d.weekday() for d in all_dates], dtype=np.int32)

    for feature_alias, raw_feature_name in feature_name_map.items():
        # Treat DOW features specially since they are the same for all locations.
        if raw_feature_name in constants.DAY_OF_WEEK_FEATURES:
            output_array = _calculate_dow_feature(dow_for_dates, raw_feature_name)
            # Note that all locations reference the same array so modifying a DOW
            # feature (for whatever reason) would modify it for all locations.
            ts_features[feature_alias] = {
                loc_name: output_array for loc_name in locations
            }
            continue

        if raw_feature_name not in features_in_data:
            # For vaccine data, we skip this checker because in COUNTY, vaccine data
            # is only available after 2021/03/12.
            if "govex" in raw_feature_name:
                # TODO(nyoder): Clean this up after COUNTY vaccine features data checkin
                continue
            else:
                raise ValueError(f"The feature {raw_feature_name} was not in the data.")

        feat_df = multi_df.loc[raw_feature_name, :].reindex(all_date_index)
        ts_features[feature_alias] = {
            geo: series.values for geo, series in feat_df.items()
        }

    for location in locations:
        _assert_feature_lengths_for_location(ts_features, location)

    return ts_features


def static_covariate_value_or_none_for_location(
    location_data, location_id, location_id_column
):
    """Returns static feature value if exists.

    Args:
      location_data: Location data.
      location_id: ID of the location
      location_id_column: The name of the column where the ID is found.
    """

    feature_value = location_data.loc[
        location_data[location_id_column] == location_id, constants.FEATURE_VALUE_COLUMN
    ]

    if not feature_value.empty:
        return np.float32(feature_value.iloc[0])
    else:
        return None


def create_population_age_ranges(static_features):
    """Adds population age features to the static feature dictionary.

    Args:
      static_features: Mapping from feature names to feature values, where each
        value is a mapping from a location to a numeric value.
    """
    for range_name, range_groups in constants.COUNTY_POP_AGE_RANGES.items():
        static_features[range_name] = _subgroups_fraction(
            static_features,
            range_groups,
            constants.COUNTY_POPULATION,
        )


def _subgroups_fraction(static_features, sub_groups, total_feature_name, epsilon=1e-8):
    """Calculates the ratio of the sum of the groups and the total population.

    Args:
      static_features: Mapping from feature names to feature values, where each
        value is a mapping from a location to a numeric value.
      sub_groups: List of census age ranges to sum
      total_feature_name: The feature name for the total population.
      epsilon: A small numeric constant to avoid division by zero

    Returns:
      A mapping of the location to the percentage of the total population that is
      in that in the specified groups.
    """
    static_df = pd.DataFrame.from_dict(static_features)
    static_df = static_df.replace({None: np.nan})
    output_series = static_df[sub_groups].sum(axis=1) / (
        static_df[total_feature_name] + epsilon
    )
    return output_series.replace({np.nan: None}).to_dict()


def convert_static_features_to_constant_ts(
    static_features,
    static_scalers,
    ts_features,
    ts_scalers,
    features_to_convert,
    len_ts,
):
    """Changes static features to be time-series in-place.

    Note that the specified features are removed from `static_features` and
    created in `ts_features`. Nothing is returned as the dictionaries are updated
    in-place.

    Args:
      static_features: Mapping from feature names to feature values, where each
        value is a mapping from a location to a numeric value.
      static_scalers: Scalers for static features.
      ts_features: Mapping from feature names to feature values, where each value
        is a mapping from a location to an array of values across time.
      ts_scalers: Scalers for time series covariates.
      features_to_convert: The names of features to be converted.
      len_ts: The length of the output time-series.
    """
    for feature in features_to_convert:
        ts_features[feature] = _make_ts_from_static_feature(
            static_features[feature], len_ts
        )
        ts_scalers[feature] = static_scalers[feature]
        del static_features[feature]
        del static_scalers[feature]


def _make_ts_from_static_feature(static_feature, len_ts):
    """Makes a time series feature from a static feature.

    Note that this function does not remove the feature from the static_features.

    Args:
      static_feature: Mapping from locations to a numeric value.
      len_ts: The length of the output time series.

    Returns:
      A mapping from locations to constant arrays.
    """
    output_ts = {}
    for loc, loc_value in static_feature.items():
        if loc_value is None:
            output_ts[loc] = np.ones(len_ts) * np.nan
        else:
            output_ts[loc] = np.ones(len_ts) * loc_value

    return output_ts


def _calculate_dow_feature(
    day_of_week_array,
    dow_feature,
):
    """Add day of week and weekend to the features."""
    feature_to_dow_number = {
        constants.MONDAY: 0,
        constants.TUESDAY: 1,
        constants.WEDNESDAY: 2,
        constants.THURSDAY: 3,
        constants.FRIDAY: 4,
        constants.SATURDAY: 5,
        constants.SUNDAY: 6,
    }
    if dow_feature == constants.DAY_OF_WEEK:
        output = day_of_week_array
    elif dow_feature == constants.WEEKEND_DAY:
        output = day_of_week_array > 4
    elif dow_feature == constants.DOW_WINDOW:
        # Should help engage more of the GAM kernel at each timestep
        # Keep sym true to have the peak be unique (i.e. 1)
        window = windows.hann(7, sym=True)
        output = np.empty(day_of_week_array.shape, dtype=np.float32)
        for i in range(7):
            output[day_of_week_array == i] = window[i]
    elif dow_feature in feature_to_dow_number:
        output = day_of_week_array == feature_to_dow_number[dow_feature]
    else:
        raise ValueError(f"{dow_feature} is not a supported DOW feature.")

    return output.astype(np.int32)


@numba.njit(parallel=True, nogil=True, cache=True)
def smooth_period(timeseries, indicator, smooth_coef):
    """Smooth a single timeseries given a period."""
    timeseries = timeseries.T
    indicator = indicator.T
    n, t = timeseries.shape
    smoothed_single_timeseries = np.zeros((n, t), dtype=np.float32)
    for i in numba.prange(n):
        pre_idx = np.zeros(t, dtype=np.int32)
        post_idx = np.zeros(t, dtype=np.int32)
        pre = -1
        for j in range(t):
            pre_idx[j] = pre
            if indicator[i, j] == 1:
                pre = j

        post = t
        for j in range(t - 1, -1, -1):
            post_idx[j] = post
            if indicator[i, j] == 1:
                post = j

        for j in range(t):
            if pre_idx[j] == -1 or post_idx[j] == t or indicator[i, j] == 0:
                smoothed_single_timeseries[i, j] = timeseries[i, j]
            else:
                r = 1.0 * (j - pre_idx[j]) / (post_idx[j] - pre_idx[j])
                smoothed_single_timeseries[i, j] = (1.0 - smooth_coef) * timeseries[
                    i, j
                ] + smooth_coef * (
                    (1.0 - r) * timeseries[i, pre_idx[j]]
                    + r * timeseries[i, post_idx[j]]
                )
    smoothed_single_timeseries = smoothed_single_timeseries.T
    return smoothed_single_timeseries
