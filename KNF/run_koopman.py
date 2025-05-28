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

"""Main script for training KNF."""
import os
import random
import time

from absl import app
from args import FLAGS
from modules.data_classes import CrytosDataset
from modules.data_classes import CustomDataset
from modules.data_classes import M4Dataset
from modules.data_classes import TrajDataset
from modules.eval_metrics import RMSE
from modules.eval_metrics import SMAPE
from modules.eval_metrics import WRMSE
from modules.models import Koopman
from modules.train_utils import eval_epoch_koopman
from modules.train_utils import get_lr
from modules.train_utils import train_epoch_koopman
import numpy as np
import torch
from torch import nn
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv):
  del argv

  random.seed(FLAGS.seed)  # python random generator
  np.random.seed(FLAGS.seed)  # numpy random generator

  torch.manual_seed(FLAGS.seed)
  torch.cuda.manual_seed_all(FLAGS.seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  input_dim = FLAGS.input_dim
  input_length = FLAGS.input_length
  latent_dim = FLAGS.latent_dim
  train_output_length = FLAGS.train_output_length
  num_steps = FLAGS.num_steps
  control_hidden_dim = FLAGS.control_hidden_dim
  control_num_layers = FLAGS.control_num_layers
  encoder_hidden_dim = FLAGS.hidden_dim
  decoder_hidden_dim = FLAGS.hidden_dim
  encoder_num_layers = FLAGS.num_layers
  decoder_num_layers = FLAGS.num_layers

  use_revin = FLAGS.use_revin
  use_instancenorm = FLAGS.use_instancenorm
  regularize_rank = FLAGS.regularize_rank
  add_global_operator = FLAGS.add_global_operator
  add_control = FLAGS.add_control

  output_dim = input_dim
  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs

  learning_rate = FLAGS.learning_rate

  freq = FLAGS.data_freq
  data_dir = FLAGS.data_dir
  direc = os.path.join(data_dir, "train.npy")
  direc_test = os.path.join(data_dir, "test.npy")

  data_dict = {
      "mini": CustomDataset,
      "M4": M4Dataset,
      "Cryptos": CrytosDataset,
      "Traj": TrajDataset
  }

  metric_dict = {
      "mini": SMAPE,
      "M4": SMAPE,
      "Cryptos": WRMSE,
      "Traj": RMSE
  }

  dataset = data_dict[FLAGS.dataset]
  eval_metric = metric_dict[FLAGS.dataset]

  train_set = dataset(
      input_length=input_length,
      output_length=train_output_length,
      freq=freq,
      direc=direc,
      mode="train",
      jumps=FLAGS.jumps)
  valid_set = dataset(
      input_length=input_length,
      output_length=train_output_length,
      freq=freq,
      direc=direc,
      mode="valid",
      jumps=FLAGS.jumps)
  test_set = dataset(
      input_length=input_length,
      output_length=FLAGS.test_output_length,
      freq=freq,
      direc=direc,
      direc_test=direc_test,
      mode="test")

  train_loader = data.DataLoader(
      train_set, batch_size=batch_size, shuffle=True, num_workers=1
  )
  valid_loader = data.DataLoader(
      valid_set, batch_size=batch_size, shuffle=True, num_workers=1
  )
  test_loader = data.DataLoader(
      test_set, batch_size=batch_size, shuffle=False, num_workers=1
  )

  model_name = (
      "Koopman_"
      + str(FLAGS.dataset)
      + "_seed{}_jumps{}_freq{}_poly{}_sin{}_exp{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_enchid{}_dechid{}_trm{}_conhid{}_enclys{}_declys{}_trmlys{}_conlys{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}"
      .format(
          FLAGS.seed,
          FLAGS.jumps,
          freq,
          FLAGS.num_poly,
          FLAGS.num_sins,
          FLAGS.num_exp,
          batch_size,
          learning_rate,
          FLAGS.decay_rate,
          input_dim,
          input_length,
          train_output_length,
          num_steps,
          encoder_hidden_dim,
          decoder_hidden_dim,
          FLAGS.transformer_dim,
          control_hidden_dim,
          encoder_num_layers,
          decoder_num_layers,
          FLAGS.transformer_num_layers,
          control_num_layers,
          latent_dim,
          use_revin,
          use_instancenorm,
          regularize_rank,
          add_global_operator,
          add_control,
      )
  )

  print(model_name)
  results_dir = FLAGS.dataset + "_results/"
  if os.path.exists(results_dir + model_name + ".pth"):
    model, last_epoch, learning_rate = torch.load(results_dir + model_name +
                                                  ".pth")
    print("Resume Training")
    print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
  else:
    last_epoch = 0
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    model = Koopman(
        # number of steps of historical observations encoded at every step
        input_dim=input_dim,
        # input length of ts
        input_length=input_length,
        # number of output features
        output_dim=output_dim,
        # number of prediction steps every forward pass
        num_steps=num_steps,
        # hidden dimension of encoder
        encoder_hidden_dim=encoder_hidden_dim,
        # hidden dimension of decoder
        decoder_hidden_dim=decoder_hidden_dim,
        # number of layers in the encoder
        encoder_num_layers=encoder_num_layers,
        # number of layers in the decoder
        decoder_num_layers=decoder_num_layers,
        # number of feature
        num_feats=FLAGS.num_feats,
        # dimension of finite koopman space
        latent_dim=latent_dim,
        # whether to learn a global operator shared across all time series
        add_global_operator=add_global_operator,
        # whether to use a feedback module
        add_control=add_control,
        # hidden dim in the control module
        control_hidden_dim=control_hidden_dim,
        # number of layers in the control module
        use_revin=use_revin,  # whether to use reversible normalization
        control_num_layers=control_num_layers,
        # whether to use instance normalization on hidden states
        use_instancenorm=use_instancenorm,
        # Regularize rank.
        regularize_rank=regularize_rank,
        # number of pairs of sine and cosine measurement functions
        num_sins=FLAGS.num_sins,
        # the highest order of polynomial functions
        num_poly=FLAGS.num_poly,
        # number of exponential functions
        num_exp=FLAGS.num_exp,
        # Number of the head the transformer encoder
        num_heads=FLAGS.num_heads,
        # hidden dimension of tranformer encoder
        transformer_dim=FLAGS.transformer_dim,
        # number of layers in the transformer encoder
        transformer_num_layers=FLAGS.transformer_num_layers,
        # dropout rate of MLP modules
        dropout_rate=FLAGS.dropout_rate
    ).to(device)

    print("New model")
  print("number of params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad))

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=1, gamma=FLAGS.decay_rate
  )  # stepwise learning rate decay
  loss_fun = nn.MSELoss()

  all_train_rmses, all_eval_rmses = [], []
  best_eval_rmse = 1e6

  for epoch in range(last_epoch, num_epochs):
    start_time = time.time()

    train_rmse = train_epoch_koopman(
        train_loader,
        model,
        loss_fun,
        optimizer,
        regularize_rank=regularize_rank)
    eval_rmse, _, _ = eval_epoch_koopman(
        valid_loader, model, loss_fun, regularize_rank=regularize_rank)

    if eval_rmse < best_eval_rmse:
      best_eval_rmse = eval_rmse
      best_model = model
      torch.save([best_model, epoch, get_lr(optimizer)],
                 results_dir + model_name + ".pth")

    all_train_rmses.append(train_rmse)
    all_eval_rmses.append(eval_rmse)

    if np.isnan(train_rmse) or np.isnan(eval_rmse):
      raise ValueError("The model generate NaN values")

    # train the model at least 60 epochs and do early stopping
    if epoch > FLAGS.min_epochs and np.mean(all_eval_rmses[-10:]) > np.mean(
        all_eval_rmses[-20:-10]):
      break

    epoch_time = time.time() - start_time
    scheduler.step()
    print("Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}"
          .format(epoch + 1, epoch_time / 60, train_rmse, eval_rmse))

  _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model,
                                                loss_fun)

  # Denormalize the predictions
  if FLAGS.dataset == "M4" or FLAGS.dataset == "mini":
    test_preds = (test_preds * test_set.ts_stds.reshape(
        -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)
    test_tgts = (test_tgts * test_set.ts_stds.reshape(
        -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)

  elif FLAGS.dataset == "Cryptos":
    test_preds = test_preds.reshape(
        14, -1, FLAGS.test_output_length,
        8)  # 14 stocks x num_samples x #steps x 8 features
    test_tgts = test_tgts.reshape(14, -1, FLAGS.test_output_length, 8)
    stds = np.expand_dims(test_set.ts_stds, axis=(1, 2))
    means = np.expand_dims(test_set.ts_means, axis=(1, 2))
    test_preds = test_preds * stds + means
    test_tgts = test_tgts * stds + means

  else:
    stds = np.expand_dims(test_set.ts_stds, axis=(1, 2))
    means = np.expand_dims(test_set.ts_means, axis=(1, 2))

    test_preds = test_preds.reshape(len(means), -1, FLAGS.test_output_length, 2)
    test_tgts = test_tgts.reshape(len(means), -1, FLAGS.test_output_length, 2)

    test_preds = test_preds * stds + means
    test_tgts = test_tgts * stds + means

  torch.save(
      {
          "test_preds": test_preds,
          "test_tgts": test_tgts,
          "eval_score": eval_metric(test_preds, test_tgts)
      }, results_dir + "test_" + model_name + ".pt")

  print(model_name)
  print(eval_metric(test_preds, test_tgts))


if __name__ == "__main__":
  app.run(main)
