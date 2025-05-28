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

"""Script for running single experiment and saving its plot."""

import numpy as np
import sklearn.datasets
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

from dp_regression import experiment


# generate synthetic data
synthetic_x, synthetic_y = sklearn.datasets.make_regression(
    22000, n_features=10, n_informative=10, noise=10)
synthetic_x = np.column_stack(
    (synthetic_x, np.ones(synthetic_x.shape[0]))).astype(float)

# set dpsgd optimal hyperparameters
# these are tuned to be optimal for DPSGD on the Synthetic dataset
# corresponding DPSGD hyperparameter settings for other datasets  appear in the
# paper's Appendix
dpsgd_opt_params = {}
dpsgd_opt_params["num_epochs"] = 20
dpsgd_opt_params["clip_norm"] = 1
dpsgd_opt_params["learning_rate"] = 1
dpsgd_opt_params["batch_size"] = 128
dpsgd_opt_params["noise_multiplier"] = 1.4716796875

# set dpsgd suboptimal hyperparameters
# these are the 90th percentile R^2 hyperparameters for DPSGD on Synthetic
dpsgd_subopt_params = {}
dpsgd_subopt_params["num_epochs"] = 20
dpsgd_subopt_params["clip_norm"] = 1e-3
dpsgd_subopt_params["learning_rate"] = 0.1
dpsgd_subopt_params["batch_size"] = 64
dpsgd_subopt_params["noise_multiplier"] = 1.150390625

# set remaining parameters
eps = np.log(3)
delta = 1e-5
m_range = np.linspace(1000, 2000, 5)
num_trials = 3

# verify dpsgd hyperparameters
compute_dp_sgd_privacy(len(synthetic_x), dpsgd_opt_params["batch_size"],
                       dpsgd_opt_params["noise_multiplier"],
                       dpsgd_opt_params["num_epochs"], delta)
compute_dp_sgd_privacy(len(synthetic_x), dpsgd_subopt_params["batch_size"],
                       dpsgd_subopt_params["noise_multiplier"],
                       dpsgd_subopt_params["num_epochs"], delta)

# run r2 experiments and store r2 quantiles and times
results = experiment.run_trials(synthetic_x, synthetic_y, eps, delta, m_range,
                                dpsgd_opt_params, dpsgd_subopt_params,
                                num_trials)

# plot r2 experiments results
experiment.plot_r2(results[0], results[1], results[2], m_range, num_trials,
                   "synthetic_r2")
experiment.plot_time(results[3], m_range, num_trials, "synthetic_time", True)

# Instructions for preprocessing the other datasets appear below.

# california:https://www.kaggle.com/datasets/camnugent/california-housing-prices
# Save the file as california.csv and use
# housing = pd.read_csv("california.csv", skipinitialspace=False)
# housing.dropna(inplace = True)
# housing = housing.to_numpy()[:,:-1].astype(np.float32)
# housing_X = housing[:, :-1]
# housing_X = np.column_stack((housing_X, np.ones(housing_X.shape[0])))
# housing_Y = housing[:, -1]

# diamonds: https://www.kaggle.com/datasets/shivam2503/diamonds
# Save the file as diamonds.csv and use
# diamonds = pd.read_csv("diamonds.csv", skipinitialspace=False)
# diamonds.dropna(inplace = True)
# diamonds_Y = diamonds["price"].to_numpy()
# diamonds = diamonds.drop("price", axis=1)
# diamonds = diamonds.replace(
#     to_replace=["Fair", "Good", "Very Good", "Premium", "Ideal"],
#     value=[1, 2, 3, 4, 5])
# diamonds = diamonds.replace(
#     to_replace=["J", "I", "H", "G", "F", "E", "D"], value=[1, 2, 3, 4, 5, 6, 7])
# diamonds = diamonds.replace(
#     to_replace=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
#     value=[1, 2, 3, 4, 5, 6, 7, 8])
# diamonds = diamonds.drop("Unnamed: 0", axis=1)
# diamonds_X = diamonds.to_numpy()
# diamonds_X = np.column_stack(
#     (diamonds_X, np.ones(diamonds_X.shape[0]))).astype(float)

# traffic: https://data.ny.gov/Transportation/Weigh-In-Motion-Station-Vehicle-Traffic-Counts-201/gdpg-i86w
# Save the file as traffic.csv and use
# traffic = pd.read_csv("traffic.csv", skipinitialspace=False)
# traffic = traffic[["Class 1", "Class 2", "Class 3"]]
# traffic.dropna(inplace = True)
# traffic_Y = traffic["Class 2"].to_numpy()
# traffic_X = traffic[["Class 1", "Class 3"]].to_numpy()
# traffic_X = np.column_stack(
#     (traffic_X, np.ones(traffic_X.shape[0]))).astype(float)

# nba: https://www.kaggle.com/datasets/nathanlauga/nba-games?select=games.csv
# Save the file as nba.csv and use
# nba = pd.read_csv("nba.csv", skipinitialspace=False)
# nba = nba[[
#     "PTS_home", "FT_PCT_home", "FG3_PCT_home", "FG_PCT_home", "AST_home",
#     "REB_home"
# ]]
# nba.dropna(inplace = True)
# nba_Y = nba["PTS_home"].to_numpy()
# nba_X = nba[[
#     "FT_PCT_home", "FG3_PCT_home", "FG_PCT_home", "AST_home", "REB_home"
# ]].to_numpy()
# nba_X = np.column_stack((nba_X, np.ones(nba_X.shape[0]))).astype(float)

# beijing: https://www.kaggle.com/datasets/ruiqurm/lianjia?resource=download
# Save the file as beijing.csv and use
# beijing = pd.read_csv("beijing.csv", usecols=["DOM", "followers", "totalPrice",
#                                             "square", "kitchen", "buildingType",
#                                             "renovationCondition",
#                                             "buildingStructure", "ladderRatio",
#                                             "elevator", "fiveYearsProperty",
#                                             "subway", "district",
#                                             "communityAverage"])
# beijing.dropna(inplace = True)
# building_type = pd.get_dummies(beijing["buildingType"], prefix="buildingType")
# beijing = beijing.drop("buildingType",axis = 1)
# beijing = beijing.join(building_type)
# renovation_condition = pd.get_dummies(
#     beijing["renovationCondition"], prefix="renovationCondition")
# beijing = beijing.drop("renovationCondition",axis = 1)
# building_structure = pd.get_dummies(
#     beijing["buildingStructure"], prefix="buildingStructure")
# building_structure = pd.get_dummies(beijing["buildingStructure"], prefix="buildingStructure")
# beijing = beijing.drop("buildingStructure",axis = 1)
# beijing = beijing.join(building_structure)
# beijing_Y = beijing["totalPrice"].to_numpy()
# beijing = beijing.drop("totalPrice", axis=1)
# beijing_X = np.column_stack(
#     (beijing_X, np.ones(beijing_X.shape[0]))).astype(float)
# beijing_X = np.column_stack((beijing_X, np.ones(beijing_X.shape[0]))).astype(float)

# garbage: https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5
# Save the file as garbage.csv and use
# garbage = pd.read_csv("garbage.csv"", usecols=["BOROUGH", "PAPERTONSCOLLECTED",
#                                     "MGPTONSCOLLECTED", "REFUSETONSCOLLECTED"])
# garbage.dropna(inplace = True)
# garbage_Y = garbage["REFUSETONSCOLLECTED"].to_numpy().astype(np.float32)
# garbage = garbage.drop("REFUSETONSCOLLECTED", axis=1)
# borough = pd.get_dummies(garbage["BOROUGH"], prefix="BOROUGH")
# garbage = garbage.drop("BOROUGH", axis=1)
# garbage = garbage.join(borough)
# garbage_X = garbage.to_numpy().astype(np.float32)
# garbage_X = np.column_stack((garbage_X, np.ones(garbage_X.shape[0])))

# mlb: https://www.kaggle.com/datasets/samaxtech/mlb-games-data-from-retrosheet?select=game_log.csv
# Save the file as mlb.csv and use
# mlb = pd.read_csv("mlb.csv, usecols=["v_strikeouts", "v_walks", "v_pitchers_used",
#                                 "v_errors", "h_at_bats", "h_hits", "h_doubles",
#                                 "h_triples", "h_homeruns", "h_stolen_bases",
#                                 "h_score"])
# mlb.dropna(inplace = True)
# mlb_Y = mlb["h_score"].to_numpy().astype(np.float32)
# mlb = mlb.drop("h_score", axis=1)
# mlb_X = mlb.to_numpy()
# mlb_X = np.column_stack((mlb_X, np.ones(mlb_X.shape[0]))).astype(float)
# mlb_X.shape
