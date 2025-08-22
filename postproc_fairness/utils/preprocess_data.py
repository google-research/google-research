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

"""Functions for preprocessing the datasets."""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow.keras.utils import FeatureSpace
from postproc_fairness.utils import load_adult
from postproc_fairness.utils import utils

COMPAS_DATA_PATH = "postproc_fairness/data/compas/compas-scores-two-years.csv"
HSLS_DATA_PATH = "postproc_fairness/data/hsls/hsls_knn_impute.pkl"

# Fix the same random seed for the train/test split as Alghamdi et al to obtain
# comparable results. The seed is selected by me such that a pretrained logreg
# and GBM model trained on the resulting training split gets roughly the same
# accuracy as what is reported in Alghamdi et al for the base model.
RANDOM_SEED_FOR_DATASET = {
    "hsls": 11,
    "compas": 5,
}


def preprocess_adult(sample=1.0, valid_ratio=0.2):
  """Preprocesses the Adult dataset."""
  train_and_valid_data = load_adult.get_uci_data(split="train", sample=sample)
  test_data = load_adult.get_uci_data(split="test", sample=sample)

  def strip_white_spaces(df):
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df

  train_and_valid_data = strip_white_spaces(train_and_valid_data)
  test_data = strip_white_spaces(test_data)

  # pylint: disable=line-too-long
  # Not using all the features available in the original dataset.
  # Selecting the same features as https://github.com/HsiangHsu/Fair-Projection/blob/main/baseline-methods/DataLoader.py#L19.
  # drop "education" and native-country (education already encoded in education-num)
  # pylint: enable=line-too-long
  drop_col = ["workclass", "occupation", "education", "native-country"]
  train_and_valid_data.drop(drop_col, inplace=True, axis=1)
  test_data.drop(drop_col, inplace=True, axis=1)
  # keep only entries marked as ``White'' or ``Black''
  ix = train_and_valid_data["race"].isin(["White", "Black"])
  train_and_valid_data = train_and_valid_data.loc[ix, :]
  ix = test_data["race"].isin(["White", "Black"])
  test_data = test_data.loc[ix, :]

  valid_data_size = int(len(train_and_valid_data) * valid_ratio)
  valid_data = train_and_valid_data.iloc[:valid_data_size]
  train_data = train_and_valid_data.iloc[valid_data_size:]

  feature_space = FeatureSpace(
      features={
          "age": FeatureSpace.float_normalized(),
          "education-num": FeatureSpace.float_normalized(),
          "marital-status": FeatureSpace.string_categorical(output_mode="int"),
          "relationship": FeatureSpace.string_categorical(output_mode="int"),
          "race": FeatureSpace.string_categorical(output_mode="int"),
          "sex": FeatureSpace.float(),
          "capital-gain": FeatureSpace.float_normalized(),
          "capital-loss": FeatureSpace.float_normalized(),
          "hours-per-week": FeatureSpace.float_normalized(),
          "target": FeatureSpace.float(),
      },
      output_mode="dict",
  )
  feature_space.adapt(utils.df_to_dataset(train_data))
  train_data_dict = feature_space(dict(train_data))
  valid_data_dict = feature_space(dict(valid_data))
  test_data_dict = feature_space(dict(test_data))

  categorical_features = ["marital-status", "relationship", "race"]
  for f in categorical_features:
    train_data_dict[f] = train_data_dict[f] - 1
    valid_data_dict[f] = valid_data_dict[f] - 1
    test_data_dict[f] = test_data_dict[f] - 1

  train_df = pd.DataFrame.from_dict(
      {k: v.numpy().reshape(-1) for k, v in train_data_dict.items()}
  )
  train_y = train_df.pop("target")
  valid_df = pd.DataFrame.from_dict(
      {k: v.numpy().reshape(-1) for k, v in valid_data_dict.items()}
  )
  valid_y = valid_df.pop("target")
  test_df = pd.DataFrame.from_dict(
      {k: v.numpy().reshape(-1) for k, v in test_data_dict.items()}
  )
  test_y = test_df.pop("target")
  return train_df, train_y, valid_df, valid_y, test_df, test_y


def preprocess_compas():
  """Preprocesses the Compas dataset."""
  df = pd.read_csv(COMPAS_DATA_PATH, index_col=0)

  # pylint: disable=line-too-long
  # Same preprocessing as https://github.com/HsiangHsu/Fair-Projection/blob/main/baseline-methods/DataLoader.py.
  # select features for analysis
  # pylint: enable=line-too-long
  df = df[[
      "age",
      "c_charge_degree",
      "race",
      "sex",
      "priors_count",
      "days_b_screening_arrest",
      "is_recid",
      "c_jail_in",
      "c_jail_out",
  ]]

  # drop missing/bad features (following ProPublica's analysis)
  # ix is the index of variables we want to keep.

  # Remove entries with inconsistent arrest information.
  ix = df["days_b_screening_arrest"] <= 30
  ix = (df["days_b_screening_arrest"] >= -30) & ix

  # remove entries entries where compas case could not be found.
  ix = (df["is_recid"] != -1) & ix

  # remove traffic offenses.
  ix = (df["c_charge_degree"] != "O") & ix

  # trim dataset
  df = df.loc[ix, :]

  # create new attribute "length of stay" with total jail time.
  df["length_of_stay"] = (
      pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])
  ).apply(lambda x: x.days)

  # drop 'c_jail_in' and 'c_jail_out'
  # drop columns that won't be used
  drop_col = ["c_jail_in", "c_jail_out", "days_b_screening_arrest"]
  df.drop(drop_col, inplace=True, axis=1)

  # keep only African-American and Caucasian
  df = df.loc[df["race"].isin(["African-American", "Caucasian"]), :]

  # binarize race
  # African-American: 0, Caucasian: 1
  df.loc[:, "race"] = df["race"].apply(lambda x: 1 if x == "Caucasian" else 0)

  # binarize gender
  # Female: 1, Male: 0
  df.loc[:, "sex"] = df["sex"].apply(lambda x: 1 if x == "Male" else 0)

  # rename columns 'sex' to 'gender'
  df.rename(index=str, columns={"sex": "gender"}, inplace=True)

  # binarize degree charged
  # Misd. = -1, Felony = 1
  df.loc[:, "c_charge_degree"] = df["c_charge_degree"].apply(
      lambda x: 1 if x == "F" else -1
  )

  # reset index
  df.reset_index(inplace=True, drop=True)

  df = df.rename({"is_recid": "target"}, axis=1)

  df["c_charge_degree"] = df["c_charge_degree"].astype(int)
  df["race"] = df["race"].astype(int)
  df["gender"] = df["gender"].astype(int)

  # pylint: disable=line-too-long
  print(
      "ATTENTION: We use no validation split for this dataset, to follow as"
      " closely as possible the methodology in"
      " https://github.com/HsiangHsu/Fair-Projection/blob/main/baseline-methods/DataLoader.py"
  )
  # pylint: enable=line-too-long
  train_data, test_data = model_selection.train_test_split(
      df, test_size=0.3, random_state=RANDOM_SEED_FOR_DATASET["compas"]
  )

  feature_space = FeatureSpace(
      features={
          "age": FeatureSpace.float_normalized(),
          "c_charge_degree": FeatureSpace.integer_categorical(
              output_mode="int"
          ),
          "race": (
              FeatureSpace.float()
          ),  # This is the sensitive feature; don't normalize
          "gender": FeatureSpace.integer_categorical(output_mode="int"),
          "priors_count": FeatureSpace.float_normalized(),
          "length_of_stay": FeatureSpace.float_normalized(),
          "target": FeatureSpace.float(),  # This is the target; don't normalize
      },
      output_mode="dict",
  )
  feature_space.adapt(utils.df_to_dataset(train_data))
  train_data_dict = feature_space(dict(train_data))
  test_data_dict = feature_space(dict(test_data))

  train_data_dict["gender"] = train_data_dict["gender"] - 1
  test_data_dict["gender"] = test_data_dict["gender"] - 1

  train_df = pd.DataFrame.from_dict(
      {k: v.numpy().reshape(-1) for k, v in train_data_dict.items()}
  )
  train_y = train_df.pop("target")
  test_df = pd.DataFrame.from_dict(
      {k: v.numpy().reshape(-1) for k, v in test_data_dict.items()}
  )
  test_y = test_df.pop("target")
  return train_df, train_y, test_df, test_y, test_df, test_y


def preprocess_hsls():
  """Preprocesses the HSLS dataset."""
  df = pd.read_pickle(HSLS_DATA_PATH)

  # pylint: disable=line-too-long
  # Same preprocessing as https://github.com/HsiangHsu/Fair-Projection/blob/main/baseline-methods/DataLoader.py.
  # pylint: enable=line-too-long
  ## Setting NaNs to out-of-range entries
  ## entries with values smaller than -7 are set as NaNs
  df[df <= -7] = np.nan

  ## Dropping all rows or columns with missing values
  ## this step significantly reduces the number of samples
  df = df.dropna()

  # Creating racebin & gradebin & sexbin variables
  # X1SEX: 1 -- Male, 2 -- Female, -9 -- NaN -> Preprocess it to: 0 -- Female,
  # 1 -- Male, drop NaN
  # X1RACE: 0 -- BHN, 1 -- WA
  df["gradebin"] = df["grade9thbin"]
  df["racebin"] = np.logical_or(
      ((df["studentrace"] * 7).astype(int) == 7).values,
      ((df["studentrace"] * 7).astype(int) == 1).values,
  ).astype(int)
  df["sexbin"] = df["studentgender"].astype(int)

  # Dropping race and 12th grade data just to focus on the 9th grade prediction
  df = df.drop(
      columns=["studentgender", "grade9thbin", "grade12thbin", "studentrace"]
  )

  ## Scaling ##
  scaler = preprocessing.MinMaxScaler()
  df = pd.DataFrame(
      scaler.fit_transform(df), columns=df.columns, index=df.index
  )

  df = df.rename({"gradebin": "target"}, axis=1)

  # pylint: disable=line-too-long
  print(
      "ATTENTION: We use no validation split for this dataset, to follow as"
      " closely as possible the methodology in"
      " https://github.com/HsiangHsu/Fair-Projection/blob/main/baseline-methods/DataLoader.py"
  )
  # pylint: enable=line-too-long
  train_data, test_data = model_selection.train_test_split(
      df, test_size=0.3, random_state=RANDOM_SEED_FOR_DATASET["hsls"]
  )

  train_data_dict = dict(train_data)
  test_data_dict = dict(test_data)

  train_df = pd.DataFrame.from_dict(
      {k: v.to_numpy().reshape(-1) for k, v in train_data_dict.items()}
  )
  train_y = train_df.pop("target")
  test_df = pd.DataFrame.from_dict(
      {k: v.to_numpy().reshape(-1) for k, v in test_data_dict.items()}
  )
  test_y = test_df.pop("target")

  return train_df, train_y, test_df, test_y, test_df, test_y
