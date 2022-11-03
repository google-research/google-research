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

"""Intersectional fairness with many constraint."""

import random

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn import model_selection
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco


flags.DEFINE_boolean("constrained", True, "Perform constrained optimization?")
flags.DEFINE_float("dual_scale", 0.01, "Dual scale for gamma-updates.")
flags.DEFINE_float("epsilon", 0.01, "Slack.")
flags.DEFINE_integer("loops", 100000, "No. of loops.")
flags.DEFINE_integer("num_layers", 2,
                     "No. of hidden layers for multiplier model.")
flags.DEFINE_integer("num_nodes", 100,
                     "No. of hidden nodes for multiplier model.")

FLAGS = flags.FLAGS


def load_data():
  """Loads and returns data."""
  # List of column names in the dataset.
  column_names = ["state", "county", "community", "communityname", "fold",
                  "population", "householdsize", "racepctblack", "racePctWhite",
                  "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
                  "agePct16t24", "agePct65up", "numbUrban", "pctUrban",
                  "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc",
                  "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
                  "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap",
                  "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov",
                  "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
                  "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
                  "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
                  "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
                  "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
                  "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids",
                  "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig",
                  "PctImmigRecent", "PctImmigRec5", "PctImmigRec8",
                  "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
                  "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
                  "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
                  "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
                  "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR",
                  "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc",
                  "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt",
                  "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
                  "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
                  "RentHighQ", "MedRent", "MedRentPctHousInc",
                  "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters",
                  "NumStreet", "PctForeignBorn", "PctBornSameState",
                  "PctSameHouse85", "PctSameCity85", "PctSameState85",
                  "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
                  "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
                  "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol",
                  "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
                  "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
                  "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea",
                  "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",
                  "LemasPctPolicOnPatr", "LemasGangUnitDeploy",
                  "LemasPctOfficDrugUn", "PolicBudgPerPop",
                  "ViolentCrimesPerPop"]

  dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"

  # Read dataset from the UCI web repository and assign column names.
  data_df = pd.read_csv(dataset_url, sep=",", names=column_names,
                        na_values="?")

  # Make sure there are no missing values in the "ViolentCrimesPerPop" column.
  assert not data_df["ViolentCrimesPerPop"].isna().any()

  # Binarize the "ViolentCrimesPerPop" column and obtain labels.
  crime_rate_70_percentile = data_df["ViolentCrimesPerPop"].quantile(q=0.7)
  labels_df = (data_df["ViolentCrimesPerPop"] >= crime_rate_70_percentile)

  # Now that we have assigned binary labels,
  # we drop the "ViolentCrimesPerPop" column from the data frame.
  data_df.drop(columns="ViolentCrimesPerPop", inplace=True)

  # Group features.
  groups_df = pd.concat(
      [data_df["racepctblack"], data_df["racePctAsian"],
       data_df["racePctHisp"]], axis=1)

  # Drop categorical features.
  data_df.drop(
      columns=["state", "county", "community", "communityname", "fold"],
      inplace=True)

  # Handle missing features.
  feature_names = data_df.columns
  for feature_name in feature_names:
    missing_rows = data_df[feature_name].isna()
    if missing_rows.any():
      data_df[feature_name].fillna(0.0, inplace=True)  # Fill NaN with 0.
      missing_rows.rename(feature_name + "_is_missing", inplace=True)
      # Append boolean "is_missing" feature.
      data_df = data_df.join(missing_rows)

  labels = labels_df.values.astype(np.float32)
  groups = groups_df.values.astype(np.float32)
  features = data_df.values.astype(np.float32)

  # Set random seed so that the results are reproducible.
  np.random.seed(121212)

  # Train, vali and test indices.
  train_indices, test_indices = model_selection.train_test_split(
      range(features.shape[0]), test_size=0.25)
  train_indices, vali_indices = model_selection.train_test_split(
      train_indices, test_size=1./3.)

  # Train features, labels and protected groups.
  x_train = features[train_indices, :]
  y_train = labels[train_indices]
  z_train = groups[train_indices]

  # Vali features, labels and protected groups.
  x_vali = features[vali_indices, :]
  y_vali = labels[vali_indices]
  z_vali = groups[vali_indices]

  # Test features, labels and protected groups.
  x_test = features[test_indices, :]
  y_test = labels[test_indices]
  z_test = groups[test_indices]

  return (x_train, y_train, z_train, x_vali, y_vali, z_vali, x_test, y_test,
          z_test)


def error_rate(labels, predictions, groups=None):
  # Returns the error rate for given labels and predictions.
  if groups is not None:
    if np.sum(groups) == 0.0:
      return 0.0
    predictions = predictions[groups]
    labels = labels[groups]
  signed_labels = labels - 0.5
  return np.mean(signed_labels * predictions <= 0.0)


def group_membership_thresholds(
    group_feature_train, group_feature_vali, group_feature_test, thresholds):
  """Returns the group membership vectors on train, test and vali sets."""
  group_memberships_list_train_ = []
  group_memberships_list_vali_ = []
  group_memberships_list_test_ = []
  group_thresholds_list = []

  for t1 in thresholds[0]:
    for t2 in thresholds[1]:
      for t3 in thresholds[2]:
        group_membership_train = (group_feature_train[:, 0] > t1) & (
            group_feature_train[:, 1] > t2) & (group_feature_train[:, 2] > t3)
        group_membership_vali = (group_feature_vali[:, 0] > t1) & (
            group_feature_vali[:, 1] > t2) & (group_feature_vali[:, 2] > t3)
        group_membership_test = (group_feature_test[:, 0] > t1) & (
            group_feature_test[:, 1] > t2) & (group_feature_test[:, 2] > t3)
        if (np.mean(group_membership_train) <= 0.01) or (
            np.mean(group_membership_vali) <= 0.01) or (
                np.mean(group_membership_test) <= 0.01):
          # Only consider groups that are at least 1% in size.
          continue
        group_memberships_list_train_.append(group_membership_train)
        group_memberships_list_vali_.append(group_membership_vali)
        group_memberships_list_test_.append(group_membership_test)
        group_thresholds_list.append([t1, t2, t3])

  group_memberships_list_train_ = np.array(group_memberships_list_train_)
  group_memberships_list_vali_ = np.array(group_memberships_list_vali_)
  group_memberships_list_test_ = np.array(group_memberships_list_test_)
  group_thresholds_list = np.array(group_thresholds_list)

  return (group_memberships_list_train_, group_memberships_list_vali_,
          group_memberships_list_test_, group_thresholds_list)


def violation(
    labels, predictions, epsilon, group_memberships_list):
  # Returns violations across different group feature thresholds.
  viol_list = []
  overall_error = error_rate(labels, predictions)
  for kk in range(group_memberships_list.shape[0]):
    group_err = error_rate(
        labels, predictions, group_memberships_list[kk, :].reshape(-1,))
    viol_list += [group_err - overall_error - epsilon]
  return np.max(viol_list), viol_list


def evaluate(
    features, labels, model, epsilon, group_membership_list):
  # Evaluates and prints stats.
  predictions = model(features).numpy().reshape(-1,)
  print("Error %.3f" % error_rate(labels, predictions))
  _, viol_list = violation(labels, predictions, epsilon, group_membership_list)
  print("99p Violation %.3f" % np.quantile(viol_list, 0.99))
  print()


def create_model(dimension):
  # Creates linear Keras model with no hidden layers.
  layers = []
  layers.append(tf.keras.Input(shape=(dimension,)))
  layers.append(tf.keras.layers.Dense(1))
  model = tf.keras.Sequential(layers)
  return model


def create_multiplier_model(
    feature_dependent_multiplier=True, dim=1, hidden_layers=None):
  """Creates Lagrange multipler model with specified hidden layers."""
  if feature_dependent_multiplier:
    layers = []
    layers.append(tf.keras.Input(shape=dim))
    for num_nodes in hidden_layers:
      layers.append(tf.keras.layers.Dense(num_nodes, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, bias_initializer="ones"))

    # Keras model.
    multiplier_model = tf.keras.Sequential(layers)
    multiplier_weights = multiplier_model.trainable_weights
  else:
    common_multiplier = tf.Variable(1.0, name="common_multiplier")
    # Ignore feature input, and return common multiplier.
    multiplier_model = lambda x: common_multiplier
    multiplier_weights = [common_multiplier]
  return multiplier_model, multiplier_weights


def train_unconstrained(
    dataset, group_info, epsilon=0.01, loops=10000, skip_steps=400):
  """Train unconstrained classifier.

  Args:
    dataset: train, vali and test sets
    group_info: group memberships on train, vali and test sets and thresholds
    epsilon: constraint slack
    loops: number of gradient steps
    skip_steps: steps to skip before snapshotting metrics
  """
  tf.set_random_seed(121212)
  np.random.seed(212121)
  random.seed(333333)

  x_train, y_train, _, x_vali, y_vali, _, x_test, y_test, _ = dataset

  (group_memberships_list_train, group_memberships_list_vali,
   group_memberships_list_test, _) = group_info

  model = create_model(x_train.shape[-1])
  features_tensor = tf.constant(x_train)
  labels_tensor = tf.constant(y_train)

  predictions = lambda: model(features_tensor)
  predictions_vali = lambda: model(x_vali)
  predictions_test = lambda: model(x_test)

  context = tfco.rate_context(predictions, labels=lambda: labels_tensor)
  overall_error = tfco.error_rate(context, penalty_loss=tfco.HingeLoss())
  problem = tfco.RateMinimizationProblem(overall_error)

  loss_fn, update_ops_fn, _ = tfco.create_lagrangian_loss(problem)
  optimizer = tf.keras.optimizers.Adagrad(0.1)

  objectives_list = []
  objectives_list_test = []
  objectives_list_vali = []
  violations_list = []
  violations_list_test = []
  violations_list_vali = []
  model_weights = []

  for ii in range(loops):
    update_ops_fn()
    optimizer.minimize(loss_fn, var_list=model.trainable_weights)

    # Snapshot iterate once in 1000 loops.
    if ii % skip_steps == 0:
      pred = np.reshape(predictions(), (-1,))
      err = error_rate(y_train, pred)
      max_viol, viol_list = violation(
          y_train, pred, epsilon, group_memberships_list_train)

      pred_test = np.reshape(predictions_test(), (-1,))
      err_test = error_rate(y_test, pred_test)
      _, viol_list_test = violation(
          y_test, pred_test, epsilon, group_memberships_list_test)

      pred_vali = np.reshape(predictions_vali(), (-1,))
      err_vali = error_rate(y_vali, pred_vali)
      max_viol_vali, viol_list_vali = violation(
          y_vali, pred_vali, epsilon, group_memberships_list_vali)

      objectives_list.append(err)
      objectives_list_test.append(err_test)
      objectives_list_vali.append(err_vali)
      violations_list.append(viol_list)
      violations_list_test.append(viol_list_test)
      violations_list_vali.append(viol_list_vali)
      model_weights.append(model.get_weights())

      if ii % 1000 == 0:
        print("Epoch %d | Error = %.3f | Viol = %.3f | Viol_vali = %.3f" %
              (ii, err, max_viol, max_viol_vali), flush=True)

  # Best candidate index.
  best_ind = np.argmin(objectives_list)
  model.set_weights(model_weights[best_ind])

  print("Train:")
  evaluate(x_train, y_train, model, epsilon, group_memberships_list_train)
  print("\nVali:")
  evaluate(x_vali, y_vali, model, epsilon, group_memberships_list_vali)
  print("\nTest:")
  evaluate(x_test, y_test, model, epsilon, group_memberships_list_test)


def train_constrained(
    dataset, group_info, epsilon=0.01, learning_rate=0.1, dual_scale=5.0,
    loops=10000, feature_dependent_multiplier=True, hidden_layers=None,
    skip_steps=400):
  """Train constrained classifier wth Lagrangian model.

  Args:
    dataset: train, vali and test sets
    group_info: group memberships on train, vali and test sets and thresholds
    epsilon: constraint slack
    learning_rate: learning rate for theta
    dual_scale: learning rate for gamma = dual_scale * learning_rate
    loops: number of gradient steps
    feature_dependent_multiplier: should the multiplier model be feature
      dependent. If False, a common multipler is used for all constraints
    hidden_layers: list of hidden layer nodes to be used for multiplier model
    skip_steps: steps to skip before snapshotting metrics
  """
  tf.set_random_seed(121212)
  np.random.seed(212121)
  random.seed(333333)

  x_train, y_train, z_train, x_vali, y_vali, _, x_test, y_test, _ = dataset

  (group_memberships_list_train,
   group_memberships_list_vali,
   group_memberships_list_test,
   group_memberships_thresholds_train) = group_info

  # Models and group thresholds tensor.
  model = create_model(x_train.shape[-1])
  multiplier_model, multiplier_weights = create_multiplier_model(
      feature_dependent_multiplier=feature_dependent_multiplier,
      dim=3,
      hidden_layers=hidden_layers)
  group_thresholds = tf.Variable(np.ones(3) * 0.1, dtype=tf.float32)

  # Features, labels, predictions, multipliers.
  features_tensor = tf.constant(x_train)
  labels_tensor = tf.constant(y_train)
  features_tensor_vali = tf.constant(x_vali)

  predictions = lambda: model(features_tensor)
  predictions_vali = lambda: model(features_tensor_vali)
  predictions_test = lambda: model(x_test)
  def multiplier_values():
    return tf.abs(multiplier_model(tf.reshape(group_thresholds, shape=(1, -1))))

  # Lagrangian loss function.
  def lagrangian_loss():
    # Separate out objective, constraints and proxy constraints.
    objective = problem.objective()
    constraints = problem.constraints()
    proxy_constraints = problem.proxy_constraints()

    # Set-up custom Lagrangian loss.
    primal = objective
    multipliers = multiplier_values()
    primal += tf.stop_gradient(multipliers) * proxy_constraints
    dual = dual_scale * multipliers * tf.stop_gradient(constraints)
    return primal - dual

  # Objective.
  context = tfco.rate_context(
      predictions,
      labels=lambda: labels_tensor)
  overall_error = tfco.error_rate(context)

  # Slice and subset group predictions and labels.
  def group_membership():
    return (z_train[:, 0] > group_thresholds[0]) & (
        z_train[:, 1] > group_thresholds[1]) & (
            z_train[:, 2] > group_thresholds[2])

  def group_predictions():
    pred = predictions()
    groups = tf.reshape(group_membership(), (-1, 1))
    return pred[groups]

  def group_labels():
    groups = tf.reshape(group_membership(), (-1,))
    return labels_tensor[groups]

  # Constraint.
  group_context = tfco.rate_context(
      group_predictions,
      labels=group_labels)
  group_error = tfco.error_rate(group_context)
  constraints = [group_error <= overall_error + epsilon]

  # Set up constrained optimization problem and optimizer.
  problem = tfco.RateMinimizationProblem(overall_error, constraints)
  optimizer = tf.keras.optimizers.Adagrad(learning_rate)
  var_list = model.trainable_weights + multiplier_weights

  objectives_list = []
  objectives_list_test = []
  objectives_list_vali = []
  violations_list = []
  violations_list_test = []
  violations_list_vali = []
  model_weights = []

  # Training
  for ii in range(loops):
    # Sample a group membership at random.
    random_index = np.random.randint(
        group_memberships_thresholds_train.shape[0])
    group_thresholds.assign(group_memberships_thresholds_train[random_index, :])

    # Gradient op.
    problem.update_ops()
    optimizer.minimize(lagrangian_loss, var_list=var_list)

    # Snapshot iterate once in 1000 loops.
    if ii % skip_steps == 0:
      pred = np.reshape(predictions(), (-1,))
      err = error_rate(y_train, pred)
      max_viol, viol_list = violation(
          y_train, pred, epsilon, group_memberships_list_train)

      pred_test = np.reshape(predictions_test(), (-1,))
      err_test = error_rate(y_test, pred_test)
      _, viol_list_test = violation(
          y_test, pred_test, epsilon, group_memberships_list_test)

      pred_vali = np.reshape(predictions_vali(), (-1,))
      err_vali = error_rate(y_vali, pred_vali)
      max_viol_vali, viol_list_vali = violation(
          y_vali, pred_vali, epsilon, group_memberships_list_vali)

      objectives_list.append(err)
      objectives_list_test.append(err_test)
      objectives_list_vali.append(err_vali)
      violations_list.append(viol_list)
      violations_list_test.append(viol_list_test)
      violations_list_vali.append(viol_list_vali)
      model_weights.append(model.get_weights())

      if ii % 1000 == 0:
        print("Epoch %d | Error = %.3f | Viol = %.3f | Viol_vali = %.3f" %
              (ii, err, max_viol, max_viol_vali), flush=True)

  # Best candidate index.
  best_ind = tfco.find_best_candidate_index(
      np.array(objectives_list), np.array(violations_list),
      rank_objectives=False)
  model.set_weights(model_weights[best_ind])

  print("Train:")
  evaluate(x_train, y_train, model, epsilon, group_memberships_list_train)
  print("\nVali:")
  evaluate(x_vali, y_vali, model, epsilon, group_memberships_list_vali)
  print("\nTest:")
  evaluate(x_test, y_test, model, epsilon, group_memberships_list_test)


def main(argv):
  del argv

  tf.compat.v1.enable_eager_execution()

  # Load data.
  dataset = load_data()
  _, _, z_train, _, _, z_vali, _, _, z_test = dataset

  # Group Thresholds for 3 Groups
  group_threshold_range = []
  for jj in range(3):
    group_threshold_range.append([np.quantile(
        z_train[:, jj], kk) for kk in np.arange(0.05, 1.0, 0.1)])

  # Group memberships based on group thresholds.
  group_info = group_membership_thresholds(
      z_train, z_vali, z_test, group_threshold_range)

  if FLAGS.constrained:
    if FLAGS.num_layers < 0:
      train_constrained(
          dataset,
          group_info,
          feature_dependent_multiplier=False,
          epsilon=FLAGS.epsilon,
          dual_scale=FLAGS.dual_scale,
          loops=FLAGS.loops)
    else:
      train_constrained(
          dataset,
          group_info,
          feature_dependent_multiplier=True,
          hidden_layers=[FLAGS.num_nodes] * FLAGS.num_layers,
          epsilon=FLAGS.epsilon,
          dual_scale=FLAGS.dual_scale,
          loops=FLAGS.loops)
  else:
    train_unconstrained(
        dataset, group_info, epsilon=FLAGS.epsilon, loops=FLAGS.loops)


if __name__ == "__main__":
  app.run(main)
