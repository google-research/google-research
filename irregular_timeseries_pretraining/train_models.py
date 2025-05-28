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

"""For list of strategies, runs pretraining and finetuning, and evaluates on validation set."""

import gc
import os
import sys
from modules import data_generators
from modules.experiment_helpers import build_model
from modules.experiment_helpers import classweighted_mortality_loss
from modules.experiment_helpers import ClearMemory
from modules.experiment_helpers import CustomCallbackSupervised
from modules.experiment_helpers import downsample_data
from modules.experiment_helpers import forecast_loss_V
from modules.experiment_helpers import load_data
from modules.experiment_helpers import masked_MSE_loss_SUM
from modules.experiment_helpers import strategy_dict_from_string
from modules.masking_utils import gen_fullreconstruction_labels
from modules.masking_utils import load_geometric_masks
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from strategy_config import args_dict_literaleval
from strategy_config import FT_args_to_save
from strategy_config import PT_args_to_save
from strategy_config import static_args_dict
from strategy_config import sweep_dict_list
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
data_name = sys.argv[2]
path_to_data = sys.argv[3]
start = sys.argv[4]
num_strategies_to_try = sys.argv[5]
savefolder = static_args_dict["results_dir"]


# Specify hyperparameter settings to try
start = int(start)
num_strategies_to_try = int(num_strategies_to_try)

ordering_path = os.path.join(
    static_args_dict["results_dir"], "STRATEGIES_LIST.txt"
)
strategies = np.loadtxt(ordering_path, dtype="str", delimiter="\t")
strategies_this_round = strategies[start : start + num_strategies_to_try]


#  Restricting GPU & CPU usage

os.environ["OMP_NUM_THREADS"] = "8"
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


#  Load & preprocess data

(
    fore_train_ip,
    fore_train_op,
    fore_valid_ip,
    fore_valid_op,
    train_ip,
    train_op,
    valid_ip,
    valid_op,
    test_ip,
    test_op,
    time_mean,
    time_stddev,
) = load_data(path_to_data, static_args_dict["raw_times"])

V = int(fore_valid_op.shape[1] / 2)
outV = int(fore_valid_op.shape[1] / 2)
D = fore_train_ip[0].shape[1]
fore_max_len = fore_train_ip[1].shape[1]
max_len = fore_train_ip[1].shape[1]


## Downsampling for "small" data experiments
downsampled_frac = static_args_dict["downsampled_frac"]
if downsampled_frac:
  (
      fore_train_ip,
      fore_train_op,
      fore_valid_ip,
      fore_valid_op,
      train_ip,
      train_op,
      valid_ip,
      valid_op,
      test_ip,
      test_op,
  ) = downsample_data(
      downsampled_frac,
      fore_train_ip,
      fore_train_op,
      fore_valid_ip,
      fore_valid_op,
      train_ip,
      train_op,
      valid_ip,
      valid_op,
      test_ip,
      test_op,
  )

  data_name_save = data_name + "-" + str(downsampled_frac)
else:
  data_name_save = data_name


class_weights = compute_class_weight(
    class_weight="balanced", classes=[0, 1], y=train_op
)

recon_out_train = gen_fullreconstruction_labels(fore_train_ip[1:], outpred="v")
recon_out_valid = gen_fullreconstruction_labels(fore_valid_ip[1:], outpred="v")


#  Iterate through strategies

print("Total options: ", len(list(ParameterGrid(sweep_dict_list))))
print("Unique options: ", len(strategies))


for strategy_string in strategies_this_round:
  #  Getting args_dict from string; fixing for saving

  args_dict = strategy_dict_from_string(strategy_string, args_dict_literaleval)

  pt_save_string_list = [v + "-" + str(args_dict[v]) for v in PT_args_to_save]
  pt_save_string = "~".join(pt_save_string_list)

  ft_save_string_list = [v + "-" + str(args_dict[v]) for v in FT_args_to_save]
  ft_save_string = "~".join(ft_save_string_list)

  #  PRETRAINING
  # Compute number of non-null elements for forecasting vs masking ->
  # reweighting by this ratio should give us a relatively even emphasis on
  # forecasting vs reconstruction losses
  balanced_weighting = np.array(
      [np.sum(fore_train_ip[3] != 0) / np.sum(fore_train_op[:, V:]), 1]
  )

  ## Creating results folders
  log_path = os.path.join(
      savefolder, data_name_save, "logs/{}.txt".format(pt_save_string)
  )
  batches_log_path = os.path.join(
      savefolder, data_name_save, "logs_batches/{}.txt".format(pt_save_string)
  )
  checkpoint_path = os.path.join(
      savefolder,
      data_name_save,
      "models/%s/weights.{epoch:02d}.h5" % pt_save_string,
  )
  checkpoint_dir = os.path.dirname(checkpoint_path)

  for path in [log_path, batches_log_path, checkpoint_path]:
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
      os.makedirs(dirname)

  #  generate data augmentations
  augs_list = []

  if args_dict["PT_task"] == (0, 0):
    pass
  else:
    if args_dict["aug_jitter_std"] > 0:
      augs_list.append(data_generators.aug_jitter(args_dict["aug_jitter_std"]))

    if args_dict["aug_maskrate"] > 0:
      # Specify what the masked values will be replaced with
      if args_dict["aug_maskval"] == "obvious":
        mask_vals = [
            static_args_dict["continuous_mask_values"],
            static_args_dict["continuous_mask_values"],
            V + 1,
        ]
      else:
        mask_vals = [0, 0, 0]

      # Augmentation function based on masking type
      if args_dict["aug_masksampling"] == "random":
        augs_list.append(
            data_generators.aug_random_masking(
                args_dict["aug_maskrate"],
                max_len,
                mask_vals,
                include_demos=True,
                include_padding=False,
                inmask=args_dict["aug_maskpart"],
            )
        )
      else:
        global_reg_mask, lm = load_geometric_masks(
            static_args_dict["geo_masks_dir"], args_dict["aug_maskrate"]
        )

        augs_list.append(
            data_generators.aug_geometric_masking(
                V,
                global_reg_mask,
                mask_vals,
                include_demos=True,
                include_padding=False,
                inmask=args_dict["aug_maskpart"],
                raw_times=static_args_dict["raw_times"],
                time_mean=time_mean,
                time_stddev=time_stddev,
            )
        )

  if augs_list > 0:
    get_aug = data_generators.final_aug_generator(augs_list)
    print(augs_list)
  else:

    def aug(x):
      return x

    get_aug = aug

  ## Data generator -- wrapper for dataset
  gen = data_generators.DataGenerator(
      fore_train_ip,
      get_aug,
      forecast=True,
      forecast_out=fore_train_op,
      reconstruct=True,
      reconstruct_out=recon_out_train,
      batch_size=static_args_dict["batch_size"],
      shuffle=True,
  )
  val_gen = data_generators.DataGenerator(
      fore_valid_ip,
      get_aug,
      forecast=True,
      forecast_out=fore_valid_op,
      reconstruct=True,
      reconstruct_out=recon_out_valid,
      batch_size=static_args_dict["batch_size"],
      shuffle=True,
  )

  # Load model
  model, PT_model = build_model(
      D,
      fore_max_len,
      V,
      static_args_dict["nn_d"],
      static_args_dict["nn_N"],
      static_args_dict["nn_he"],
      static_args_dict["dropout"],
      pretraining=True,
  )

  pretrained_model_path = os.path.join(checkpoint_dir, "weights.BEST.h5")

  # Check if the model was already pretrained - if not, run pretraining
  if os.path.exists(pretrained_model_path):
    PT_model.load_weights(pretrained_model_path)

  # Pretrain model
  else:
    lw = balanced_weighting * np.array(list(args_dict["PT_task"]))
    print("Loss weights:", lw, args_dict["PT_task"])

    PT_model.compile(
        optimizer=Adam(
            static_args_dict["lr"], clipnorm=static_args_dict["clipnorm"]
        ),
        loss=[forecast_loss_V(V), masked_MSE_loss_SUM(max_len)],
        loss_weights=list(lw),
    )

    csv_logger = tf.keras.callbacks.CSVLogger(log_path)
    es = EarlyStopping(
        monitor="val_loss",
        patience=static_args_dict["patience"],
        mode="min",
        restore_best_weights=True,
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        min_delta=1e-10,
        monitor="val_loss",
        verbose=1,
        save_weights_only=True,
    )
    try:
      PT_model.fit(
          x=gen,
          validation_data=val_gen,
          epochs=static_args_dict["max_pt_eps"],
          callbacks=[csv_logger, es, cp_callback, ClearMemory()],
      )
    except KeyboardInterrupt:
      PT_model.save_weights(
          os.path.join(checkpoint_dir, "weights.KEYBOARDINTERRUPTED.h5")
      )
      print(
          'Interrupted: Output saved to: "{}"'.format(
              os.path.join(checkpoint_dir, "weights.KEYBOARDINTERRUPTED.h5")
          )
      )

    PT_model.save_weights(os.path.join(checkpoint_dir, "weights.BEST.h5"))

  #  FINETUNING
  #  Update augmentations for finetuning
  if args_dict["FT_aug"] == "same":
    print("Keeping same augmentation strategy for finetuning")
  else:
    print("Resetting agumentation function to identity for finetuning")

    def no_aug(x):
      return x

    get_aug = no_aug

  #  Set paths for results
  model_savedir = os.path.join(
      savefolder,
      data_name_save,
      "finetuning-1val",
      "MODELS/{}/".format(ft_save_string),
  )
  tr_epochs_log_savedir = os.path.join(
      savefolder, data_name_save, "finetuning-1val", "SUPERVISED_LOGS_EPOCHS/"
  )
  tr_epochs_log_path = os.path.join(
      tr_epochs_log_savedir, "{}.csv".format(ft_save_string)
  )

  for folder in [model_savedir, tr_epochs_log_savedir]:
    if not os.path.isdir(folder):
      os.makedirs(folder)

  savepath = os.path.join(model_savedir, "model.h5")
  print(savepath)

  #  Skip finetuning if it's already been done
  if len(os.listdir(model_savedir)) >= 1:
    print("SUPERVISED ALREADY TRAINED")
    print("saved at: ", model_savedir)
    continue

  #  Load pretrained weights & finetune on training data

  with open(tr_epochs_log_path, "a") as f:
    f.write("{}\n".format(savepath))

  PT_model.load_weights(pretrained_model_path)
  print("loaded weights from:", pretrained_model_path)

  gen = data_generators.DataGenerator(
      train_ip,
      get_aug,
      forecast=True,
      forecast_out=train_op,
      reconstruct=False,
      batch_size=static_args_dict["batch_size"],
      shuffle=True,
  )
  val_gen = data_generators.DataGenerator(
      valid_ip,
      get_aug,
      forecast=True,
      forecast_out=valid_op,
      reconstruct=False,
      batch_size=static_args_dict["batch_size"],
      shuffle=True,
  )
  print(
      "Num train:",
      gen.num_samples,
      "Num valid:",
      val_gen.num_samples,
      "Num tr batches:",
      len(gen),
  )

  model.compile(
      loss=classweighted_mortality_loss(class_weights),
      optimizer=Adam(static_args_dict["lr"]),
  )

  print(
      "Trainable weights: ",
      np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]),
  )
  es = EarlyStopping(
      monitor="custom_metric",
      patience=static_args_dict["patience"],
      mode="max",
      restore_best_weights=True,
  )
  cus = CustomCallbackSupervised(
      training_data=(train_ip, train_op),
      validation_data=(valid_ip, valid_op),
      batch_size=static_args_dict["batch_size"],
      tr_epochs_log_path=tr_epochs_log_path,
  )
  his = model.fit(
      x=gen,
      epochs=1000,
      validation_data=val_gen,
      callbacks=[cus, es],
      verbose=0,
  ).history

  model.save_weights(savepath)

  gc.collect()
  K.clear_session()
f.close()
