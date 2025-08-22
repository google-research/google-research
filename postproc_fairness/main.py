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

"""Main logic for training models."""
import logging
import os
import time
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_model_remediation import min_diff
import postproc_fairness.models_lib as models
from postproc_fairness.utils import log_utils
from postproc_fairness.utils import preprocess_data
from postproc_fairness.utils import utils


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "loss",
    "erm",
    "Type of loss to use. Possible choices are: erm, mindiff, postproc_mindiff,"
    " eqodds_mindiff, eqodds_postproc_mindiff",
)
flags.DEFINE_enum(
    "dataset", "adult", ["adult", "compas", "hsls"], "Dataset name."
)
flags.DEFINE_float(
    "mmd_kernel_decay_length",
    0.1,
    "decay used for kernel"
    "applies only for min_diff_mmd_gauss and min_diff_mmd_lapl",
)
flags.DEFINE_string(
    "num_hidden_units_list",
    "64,",
    "csv list of hidden layer sizes in base classifier.",
)
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("epochs", 100, "Number of epochs.")
flags.DEFINE_float(
    "fraction_mindiff_data",
    1.0,
    "Fraction of mindiff data to use (i.e. data with sensitive attribute).",
)
flags.DEFINE_float("learning_rate", 0.01, "Adagrad learning rate.")
flags.DEFINE_integer(
    "min_eval_frequency", 1, "How often (epochs) to run evaluation."
)
flags.DEFINE_string(
    "model_dir",
    "outputs/",
    "Directory for output.",
)
flags.DEFINE_float(
    "mindiff_weight",
    1.0,
    "Strengh of the adversarial/min_diff head.Must be positive",
)
flags.DEFINE_string(
    "postproc_num_hidden_units_list",
    "64,",
    "csv list of hidden layer sizes in the postprocessing head.",
)
flags.DEFINE_enum(
    "postproc_regularizer_name",
    "None",
    ["None", "multiplier_l2", "kl"],
    "Name of the regularizer used with the mindiff loss for postprocessing.",
)
flags.DEFINE_float(
    "postproc_regularization_strength",
    1.0,
    "Regularization strength for postprocessing.",
)
flags.DEFINE_string(
    "pretrained_model_config_path",
    None,
    "Config file with location of pretrained model checkpoints.",
)
flags.DEFINE_string(
    "pretrained_outputs_path",
    None,
    "Path to file that contains outputs of a pretrained model on the entire"
    " dataset. This can be used as the base model for postprocessing.",
)
flags.DEFINE_bool(
    "use_half_training_data",
    False,
    "If set, use only half of the training data. Furthermore, if"
    " `FLAGS.same_data_for_base_and_postproc` is set, then the same half is"
    " used to train both the base model and the postprocessing multiplier."
    " Otherwise, one half is used to train h_base, and the other to train"
    " M(X).",
)
flags.DEFINE_bool(
    "same_data_for_base_and_postproc",
    True,
    "If set, use the same training data to train the base model and the"
    " postprocessing multiplier.",
)

tf.get_logger().setLevel(logging.INFO)


def main(_):
  curr_model_dir = os.path.join(FLAGS.model_dir, f"{int(time.time())}")

  if FLAGS.loss in ["postproc_mindiff", "eqodds_postproc_mindiff"]:
    if FLAGS.pretrained_model_config_path is None:
      assert FLAGS.pretrained_outputs_path is not None, (
          "Need to have at least one of `pretrained_model_config_path` and"
          " `pretrained_outputs_path` be set. Now they are both None."
      )
    if FLAGS.pretrained_outputs_path is None:
      assert FLAGS.pretrained_model_config_path is not None, (
          "Need to have at least one of `pretrained_model_config_path` and"
          " `pretrained_outputs_path` be set. Now they are both None."
      )

  if FLAGS.dataset == "adult":
    train_data_df, train_y, valid_df, valid_y, test_df, test_y = (
        preprocess_data.preprocess_adult()
    )
  elif FLAGS.dataset == "compas":
    train_data_df, train_y, valid_df, valid_y, test_df, test_y = (
        preprocess_data.preprocess_compas()
    )
  elif FLAGS.dataset == "hsls":
    train_data_df, train_y, valid_df, valid_y, test_df, test_y = (
        preprocess_data.preprocess_hsls()
    )
  else:
    raise RuntimeError("Unknown dataset name", FLAGS.dataset)
  sensitive_attribute = utils.get_sensitive_attribute_for_dataset(FLAGS.dataset)

  if FLAGS.use_half_training_data:
    assert FLAGS.loss in ["postproc_mindiff", "eqodds_postproc_mindiff"]
    half_len = len(train_data_df) // 2
    if FLAGS.same_data_for_base_and_postproc:
      print("Use THE SAME data for base and multiplier training.")
      train_data_df = train_data_df.copy(deep=True)[:half_len]
      train_y = train_y.copy(deep=True)[:half_len]
    else:
      print("Use DIFFERENT data for base and multiplier training.")
      train_data_df = train_data_df.copy(deep=True)[half_len:]
      train_y = train_y.copy(deep=True)[half_len:]

  if FLAGS.pretrained_outputs_path is not None:
    pretrained_outputs_df = pd.read_csv(FLAGS.pretrained_outputs_path)

    if FLAGS.dataset == "adult":
      pretrained_outputs_df.rename(columns={"income": "target"}, inplace=True)
    elif FLAGS.dataset == "compas":
      pretrained_outputs_df.rename(columns={"is_recid": "target"}, inplace=True)
    elif FLAGS.dataset == "hsls":
      pretrained_outputs_df.rename(columns={"gradebin": "target"}, inplace=True)

    # For the sake of not mixing up the covariates and the pretrained outputs,
    # use the newly loaded data.
    curr_outputs_df = pretrained_outputs_df[
        pretrained_outputs_df["split"] == "train"
    ].copy(deep=True)
    train_y = curr_outputs_df.pop("target")
    train_data_df = curr_outputs_df.drop(columns="split")

    curr_outputs_df = pretrained_outputs_df[
        pretrained_outputs_df["split"] == "valid"
    ].copy(deep=True)
    valid_y = curr_outputs_df.pop("target")
    valid_df = curr_outputs_df.drop(columns="split")

    curr_outputs_df = pretrained_outputs_df[
        pretrained_outputs_df["split"] == "test"
    ].copy(deep=True)
    test_y = curr_outputs_df.pop("target")
    test_df = curr_outputs_df.drop(columns="split")

  print("Data has the following columns: ", train_data_df.columns.values)

  train_ds = utils.df_to_dataset(
      train_data_df, targets=train_y, convert_to_float=False
  ).batch(FLAGS.batch_size)
  original_valid_ds = utils.df_to_dataset(
      valid_df, targets=valid_y, convert_to_float=False
  ).batch(FLAGS.batch_size)

  mindiff_data_idxs = None
  if FLAGS.loss in [
      "mindiff",
      "postproc_mindiff",
      "eqodds_mindiff",
      "eqodds_postproc_mindiff",
  ]:
    unprivileged_encoding = (
        train_data_df[sensitive_attribute].value_counts().argmin()
    )
    unprivileged_neg = train_data_df[
        (train_data_df[sensitive_attribute] == unprivileged_encoding)
        & (train_y == 0)
    ]
    unprivileged_neg_y = train_y[
        (
            (train_data_df[sensitive_attribute] == unprivileged_encoding)
            & (train_y == 0)
        )
    ]
    privileged_neg = train_data_df[
        (train_data_df[sensitive_attribute] != unprivileged_encoding)
        & (train_y == 0)
    ]
    privileged_neg_y = train_y[
        (train_data_df[sensitive_attribute] != unprivileged_encoding)
        & (train_y == 0)
    ]

    def subsample_data(x, y, fraction):
      idxs = np.random.choice(x.index, int(fraction * len(x)), replace=False)
      return x.loc[idxs], y.loc[idxs], idxs

    unprivileged_neg, unprivileged_neg_y, unprivileged_neg_idxs = (
        subsample_data(
            unprivileged_neg, unprivileged_neg_y, FLAGS.fraction_mindiff_data
        )
    )
    privileged_neg, privileged_neg_y, privileged_neg_idxs = subsample_data(
        privileged_neg, privileged_neg_y, FLAGS.fraction_mindiff_data
    )
    mindiff_data_idxs = np.concatenate(
        [unprivileged_neg_idxs, privileged_neg_idxs]
    )

    privileged_neg_ds = utils.df_to_dataset(
        privileged_neg, targets=privileged_neg_y, convert_to_float=False
    )
    unprivileged_neg_ds = utils.df_to_dataset(
        unprivileged_neg,
        targets=unprivileged_neg_y,
        convert_to_float=False,
    )
    print(
        f"{len(unprivileged_neg)} negative labeled unprivileged examples (i.e."
        f" {sensitive_attribute}={unprivileged_encoding})"
    )
    print(
        f"{len(privileged_neg)} negative labeled privileged examples (i.e."
        f" {sensitive_attribute}!={unprivileged_encoding})"
    )

    unprivileged_pos_ds, privileged_pos_ds, num_repeat_pos = None, None, None
    if FLAGS.loss in ["eqodds_mindiff", "eqodds_postproc_mindiff"]:
      unprivileged_pos = train_data_df[
          (train_data_df[sensitive_attribute] == unprivileged_encoding)
          & (train_y == 1)
      ]
      unprivileged_pos_y = train_y[
          (
              (train_data_df[sensitive_attribute] == unprivileged_encoding)
              & (train_y == 1)
          )
      ]
      privileged_pos = train_data_df[
          (train_data_df[sensitive_attribute] != unprivileged_encoding)
          & (train_y == 1)
      ]
      privileged_pos_y = train_y[
          (train_data_df[sensitive_attribute] != unprivileged_encoding)
          & (train_y == 1)
      ]
      unprivileged_pos, unprivileged_pos_y, unprivileged_pos_idxs = (
          subsample_data(
              unprivileged_pos, unprivileged_pos_y, FLAGS.fraction_mindiff_data
          )
      )
      privileged_pos, privileged_pos_y, privileged_pos_idxs = subsample_data(
          privileged_pos, privileged_pos_y, FLAGS.fraction_mindiff_data
      )

      privileged_pos_ds = utils.df_to_dataset(
          privileged_pos, targets=privileged_pos_y, convert_to_float=False
      )
      unprivileged_pos_ds = utils.df_to_dataset(
          unprivileged_pos,
          targets=unprivileged_pos_y,
          convert_to_float=False,
      )
      num_repeat_pos = (
          len(train_y) // min(len(unprivileged_pos), len(privileged_pos)) + 1
      )
      mindiff_data_idxs = np.concatenate([
          unprivileged_neg_idxs,
          privileged_neg_idxs,
          unprivileged_pos_idxs,
          privileged_pos_idxs,
      ])
      print(len(unprivileged_pos), "positive labeled unprivileged examples")
      print(len(privileged_pos), "positive labeled privileged examples")

    print(mindiff_data_idxs.shape, "mindiff data indexes shape")

    # Pack the data.
    num_repeat_neg = (
        len(train_y) // min(len(unprivileged_neg), len(privileged_neg)) + 1
    )

    if FLAGS.loss in ["eqodds_mindiff", "eqodds_postproc_mindiff"]:
      train_ds = min_diff.keras.utils.pack_min_diff_data(
          original_dataset=train_ds,
          min_diff_dataset=min_diff.keras.utils.build_min_diff_dataset(
              sensitive_group_dataset={
                  "neg": unprivileged_neg_ds.repeat(num_repeat_neg).batch(
                      FLAGS.batch_size, drop_remainder=True
                  ),
                  "pos": unprivileged_pos_ds.repeat(num_repeat_pos).batch(
                      FLAGS.batch_size, drop_remainder=True
                  ),
              },
              nonsensitive_group_dataset={
                  "neg": privileged_neg_ds.repeat(num_repeat_neg).batch(
                      FLAGS.batch_size, drop_remainder=True
                  ),
                  "pos": privileged_pos_ds.repeat(num_repeat_pos).batch(
                      FLAGS.batch_size, drop_remainder=True
                  ),
              },
          ),
      )
    else:
      train_ds = min_diff.keras.utils.pack_min_diff_data(
          original_dataset=train_ds,
          sensitive_group_dataset=unprivileged_neg_ds.repeat(
              num_repeat_neg
          ).batch(FLAGS.batch_size, drop_remainder=True),
          nonsensitive_group_dataset=privileged_neg_ds.repeat(
              num_repeat_neg
          ).batch(FLAGS.batch_size, drop_remainder=True),
      )
    print("Total dataset size is", sum([b[1].shape[0] for b in train_ds]))

  # BASELINE MODEL
  logger = log_utils.CustomLogger(curr_model_dir)
  log_metrics_callback = log_utils.LogMetricsCallback(
      logger=logger,
      train_df=train_data_df,
      train_y=train_y,
      valid_df=valid_df,
      valid_y=valid_y,
      test_df=test_df,
      mindiff_data_idxs=mindiff_data_idxs,
      test_y=test_y,
      sensitive_attribute=sensitive_attribute,
      log_every_n_epochs=FLAGS.min_eval_frequency,
  )
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=curr_model_dir, histogram_freq=1
  )
  callback_list = [log_metrics_callback, tensorboard_callback]

  if FLAGS.loss in ["postproc_mindiff", "eqodds_postproc_mindiff"]:
    # Load pretrained model weights, if no pretrained outputs are provided.
    if FLAGS.pretrained_model_config_path is not None:
      pretrained_model_path = models.get_pretrained_model_path(
          FLAGS.pretrained_model_config_path, FLAGS.num_hidden_units_list
      )
      base_model = models.load_model_from_saved_weights(
          base_model_path=pretrained_model_path,
          feature_name_list=train_data_df.columns.values,
          num_hidden_units_list=FLAGS.num_hidden_units_list,
          sensitive_attribute=sensitive_attribute,
      )
    else:
      base_model = None
    model = models.create_postproc_model(
        base_model=base_model,
        feature_name_list=train_data_df.columns.values,
        num_hidden_units_list=FLAGS.postproc_num_hidden_units_list,
        regularizer_name=FLAGS.postproc_regularizer_name,
        regularization_strength=FLAGS.postproc_regularization_strength,
    )
  else:
    model = models.create_base_mlp_model(
        feature_name_list=train_data_df.columns.values,
        num_hidden_units_list=FLAGS.num_hidden_units_list,
        sensitive_attribute=sensitive_attribute,
    )

  if FLAGS.loss in ["mindiff", "postproc_mindiff"]:
    if FLAGS.loss == "postproc_mindiff":
      assert (
          FLAGS.mindiff_weight == 1.0
      ), "Not allowed to change the mindiff weight for postprocessing."
    model = min_diff.keras.models.min_diff_model.MinDiffModel(
        original_model=model,
        loss=min_diff.losses.MMDLoss(
            min_diff.losses.GaussianKernel(
                kernel_length=FLAGS.mmd_kernel_decay_length
            )
        ),
        loss_weight=FLAGS.mindiff_weight,
    )
  elif FLAGS.loss in ["eqodds_mindiff", "eqodds_postproc_mindiff"]:
    model = min_diff.keras.models.min_diff_model.MinDiffModel(
        original_model=model,
        loss={
            "neg": min_diff.losses.MMDLoss(
                min_diff.losses.GaussianKernel(
                    kernel_length=FLAGS.mmd_kernel_decay_length
                )
            ),
            "pos": min_diff.losses.MMDLoss(
                min_diff.losses.GaussianKernel(
                    kernel_length=FLAGS.mmd_kernel_decay_length
                )
            ),
        },
        loss_weight=FLAGS.mindiff_weight,
    )

  optimizer = tf.keras.optimizers.Adagrad(learning_rate=FLAGS.learning_rate)
  loss = tf.keras.losses.BinaryCrossentropy()
  model.compile(
      optimizer=optimizer,
      loss=loss
      if FLAGS.loss not in ["postproc_mindiff", "eqodds_postproc_mindiff"]
      else None,
      metrics=utils.get_default_metrics(),
  )
  model.fit(
      train_ds,
      epochs=FLAGS.epochs,
      validation_data=original_valid_ds,
      callbacks=callback_list,
  )

  print(f"{FLAGS.loss} - Training eval")
  log_utils.print_eval(
      model,
      data_df=train_data_df,
      sensitive_attribute=sensitive_attribute,
      targets=train_y,
      batch_size=FLAGS.batch_size,
  )
  print(f"{FLAGS.loss} - Validation eval")
  log_utils.print_eval(
      model,
      data_df=valid_df,
      sensitive_attribute=sensitive_attribute,
      targets=valid_y,
      batch_size=FLAGS.batch_size,
  )
  print(f"{FLAGS.loss} - Test eval")
  log_utils.print_eval(
      model,
      data_df=test_df,
      sensitive_attribute=sensitive_attribute,
      targets=test_y,
      batch_size=FLAGS.batch_size,
  )

  model_location = os.path.join(
      curr_model_dir, f"model_weights_{FLAGS.loss}.weights.h5"
  )
  # Don't save models for post-processing.
  if FLAGS.loss == "mindiff":
    models.save_weights_to_file(
        model=model.original_model,
        filepath=model_location,
    )
  elif FLAGS.loss == "erm":
    models.save_weights_to_file(model, model_location)


if __name__ == "__main__":
  app.run(main)
