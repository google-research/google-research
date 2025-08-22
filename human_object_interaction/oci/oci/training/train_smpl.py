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

"""Trainer for SMPL model."""

# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=g-importing-member
# pylint: disable=unused-argument
# pylint: disable=using-constant-test


from copy import deepcopy
import os
import os.path as osp
from typing import DefaultDict

from loguru import logger
import numpy as np
from oci.datasets import behave_data as behave_data_utils
from oci.model import batch_norm as batch_norm_utils
from oci.model import oci_smpl_model
from oci.utils import model_vis_utils
from oci.utils import tensor_utils
import smpl_utils
from tensorboardX import SummaryWriter
import torch


class OCISMPLTrainer:

  def __init__(self, opts):
    self.opts = opts

    if torch.cuda.is_available():
      device = torch.device("cuda:0")
    else:
      device = torch.device("cpu")
    self.device = device

  def define_model(self,):
    opts = self.opts
    device = self.device
    model = oci_smpl_model.OCISMPLModel(opts.model).to(device)
    self.model = model
    self.model.train()

    if False:
      self.smplh_models = {}

      (
          self.smplh_models["mesh_male"],
          self.smplh_models["model_male"],
      ) = smpl_utils.smplh_canonical(
          gender="male",
          smpl_model_path=self.smpl_native,
          use_pca=False,
          flat_hand_mean=True,
      )
      (
          self.smplh_models["mesh_female"],
          self.smplh_models["model_female"],
      ) = smpl_utils.smplh_canonical(
          gender="female",
          smpl_model_path=self.smpl_native,
          use_pca=False,
          flat_hand_mean=True,
      )

    if opts.train.num_pretrain_epochs > 0:
      self.load_network(
          self.model, "model", epoch_label=opts.train.num_pretrain_epochs)
    return

  def init_dataset(self,):
    opts = self.opts
    dataloader = behave_data_utils.behave_dataoader(
        opts.dataloader, shuffle=True)
    self.dataloader = dataloader

    val_opts = deepcopy(opts)
    val_opts.dataloader.sample_all_verts = True
    val_opts.dataloader.batch_size = 2
    val_opts.dataloader.shuffle = True
    self.val_dataloader = behave_data_utils.behave_dataoader(
        val_opts.dataloader, shuffle=True)
    return

  def init_training(self,):
    opts = self.opts
    self.save_dir = f"./checkpoints/{opts.name}"
    self.define_model()
    self.init_dataset()
    self.init_logger()
    self.define_criterion()
    self.init_optimizer()

    os.makedirs(self.save_dir, exist_ok=True)
    opts_file = osp.join(self.save_dir, "opts.yaml")
    with open(opts_file, "w") as f:
      f.write(opts.dump())
    return

  def init_logger(self,):
    opts = self.opts
    log_dir = f"./tb_logs/{opts.name}"
    self.tensorboard_writer = SummaryWriter(log_dir)
    return

  def log_to_tensorboard(self, visuals):
    for key in visuals.keys():
      if key == "index":
        continue
      self.tensorboard_writer.add_image(
          f"val/{key}",
          to_tensorboard_img(visuals[key]),
          global_step=self.total_steps,
      )

    return

  def create_visuals(self, num_visuals=1):
    opts = self.opts
    val_dataloader = self.val_dataloader
    all_visuals = []
    val_scalars = DefaultDict(list)

    for bx, batch in enumerate(val_dataloader):
      bx_visuals = {}
      batch_index = bx
      data_inputs = {}
      data_inputs["seq_folder"] = batch["seq_folder"][batch_index]
      data_inputs["seq_frame_id"] = batch["frame_id"][batch_index].item()
      data_inputs["can2world"] = tensor_utils.tensor_to_numpy(
          batch["can2world"][batch_index])
      smpl_faces = batch["smpl_mesh_f"][batch_index]
      visualizer = model_vis_utils.VisualizeModelOutputs(
          seq_folder=osp.join(opts.dataloader.seq_folder,
                              data_inputs["seq_folder"]),
          vis_mesh=opts.visualization.viz_mesh,
          faces=smpl_faces.numpy(),
      )

      model_inputs, model_targets = self.set_input(batch)
      predictions = self.val_forward(model_inputs, model_targets)

      if opts.model.sequence_model:
        data_inputs["can_points_curr"] = model_inputs["points_input"][
            batch_index, -1].cpu()
      else:
        data_inputs["can_points_curr"] = model_inputs["points_input"][
            batch_index].cpu()

      data_inputs["can_points_futr"] = model_inputs["points_futr"][
          batch_index].cpu()

      predictions["points_futr"] = predictions["smpl_verts"][
          batch_index].data.cpu()

      visuals = visualizer.visualize_outputs(data_inputs, predictions)

      for key in visuals.keys():
        bx_visuals[key] = visuals[key]
      bx_visuals["index"] = self.total_steps
      for key in ["total_loss"]:
        val_scalars["total_loss"].append(predictions["total_loss"])
      all_visuals.append(bx_visuals)
      if (bx + 1) > num_visuals:
        break

    for key in val_scalars.keys():
      val_scalars[key] = np.mean(val_scalars[key])
    return all_visuals, val_scalars

  def val_forward(self, model_inputs, model_targets):
    predictions = self.model.forward(model_inputs)
    loss = self.model.compute_loss(model_targets)

    with torch.no_grad():
      weight_dict = self.weight_dict
      weighted_loss = {}
      for key in loss.keys():
        if key in weight_dict.keys():
          weighted_loss[key] = loss[key] * weight_dict[key]
        else:
          logger.info(f"{key} no weight available")

      total_loss = 0.0
      self.loss_factors = {}
      for key, l in weighted_loss.items():
        self.loss_factors[key] = l.mean()
        total_loss += self.loss_factors[key]

    predictions["total_loss"] = total_loss.item()
    return predictions

  def train(self):
    opts = self.opts
    dataloader = behave_data_utils.behave_dataoader(opts.dataloader)
    num_epochs = opts.train.num_epochs
    self.total_steps = total_steps = 0
    self.epoch_iter = 0
    for epoch in range(num_epochs):
      self.epoch_iter = 0
      self.epoch = epoch
      if epoch > 500:
        batch_norm_utils.turnNormOff(self.model)
      for bx, batch in enumerate(dataloader):
        model_inputs, model_targets = self.set_input(batch)
        predictions = self.forward(model_inputs, model_targets)
        self.backward()
        total_steps += 1
        self.total_steps = total_steps
        self.epoch_iter += 1

        if total_steps % opts.logging.log_freq == 0:
          self.print_scalars()
          scalars = self.get_current_scalars()
          self.log_tb_scalars(step=total_steps, train_scalars=scalars)

        if total_steps % opts.logging.val_log_freq == 0:
          visuals, val_scalars = self.create_visuals()
          self.log_to_tensorboard(visuals[0],)
          self.log_tb_scalars(step=self.total_steps, val_scalars=val_scalars)

        if total_steps % opts.logging.hist_freq == 0 and opts.logging.histogram:
          self.log_tb_histograms(
              step=self.total_steps,
              model_inputs=model_inputs,
              model_targets=model_targets,
              predictions=predictions,
          )
      self.epoch += 1
      if self.epoch % opts.logging.save_epoch_freq == 0:
        self.save_network(self.model, "model", self.epoch)

  def save_network(self, network, network_label, epoch_label, gpu_id=None):
    print("Saving Network to device ")
    save_filename = f"{network_label}_net_{epoch_label}.pth"
    save_path = os.path.join(self.save_dir, save_filename)
    if isinstance(network, torch.nn.DataParallel):
      torch.save(network.module.state_dict(), save_path)
    elif isinstance(network, torch.nn.parallel.DistributedDataParallel):
      torch.save(network.module.state_dict(), save_path)
    else:
      torch.save(network.state_dict(), save_path)
    # if gpu_id is not None and torch.cuda.is_available():
    #   network.cuda(device=gpu_id)
    return

  def load_network(self, network, network_label, epoch_label, network_dir=None):
    save_filename = f"{network_label}_net_{epoch_label}.pth"
    if network_dir is None:
      network_dir = self.save_dir
    save_path = os.path.join(network_dir, save_filename)
    print(f"Loading model : {save_path}")
    network.load_state_dict(torch.load(save_path, map_location="cpu"))
    return

  def print_scalars(self,):
    scalars = self.get_current_scalars()
    msg = f"epoch: {self.epoch}, total_steps : {self.total_steps}, "
    for key, value in scalars.items():
      msg += f"{key}  {value:.4f}, "
    logger.info(msg)
    return

  def log_tb_scalars(
      self,
      step,
      train_scalars=None,
      val_scalars=None,
  ):
    if train_scalars:
      for key, value in train_scalars.items():
        self.tensorboard_writer.add_scalar(
            tag=f"train/{key}", scalar_value=value, global_step=step)

    if val_scalars:
      for key, value in val_scalars.items():
        self.tensorboard_writer.add_scalar(
            tag=f"val/{key}", scalar_value=value, global_step=step)

    return

  def log_tb_histograms(self, step, model_inputs, model_targets, predictions):
    # breakpoint()
    for key in ["smpl_pose", "smpl_beta", "smpl_transl"]:
      self.tensorboard_writer.add_histogram(
          tag=f"train_pred/{key}", values=predictions[key], global_step=step)
    for key in ["futr_smpl_pose", "futr_smpl_beta", "futr_smpl_transl"]:
      self.tensorboard_writer.add_histogram(
          tag=f"train_gt/{key}", values=model_targets[key], global_step=step)
    return

  def set_input(self, batch):
    opts = self.opts
    model_inputs = {}
    model_inputs["points_input"] = batch["can_input_pts"].to(self.device)
    # model_inputs['points_curr'] = batch['can_points_curr'].to(self.device)
    model_inputs["points_futr"] = batch["can_points_futr"].to(self.device)
    model_inputs["points_can"] = batch["can_points_smpl"].to(self.device)
    model_inputs["smpl_pose"] = batch["smpl_pose"].to(self.device)
    model_inputs["smpl_beta"] = batch["smpl_beta"].to(self.device)
    model_inputs["smpl_transl"] = batch["smpl_transl"].to(self.device)
    model_inputs["gender"] = batch["gender"].to(self.device)

    if not opts.model.sequence_model:
      model_inputs["points_input"] = model_inputs["points_input"][:, 0]
      model_inputs["smpl_pose"] = model_inputs["smpl_pose"][:, 0]
      model_inputs["smpl_beta"] = model_inputs["smpl_beta"][:, 0]

    model_targets = {}
    model_targets["futr_smpl_pose"] = batch["futr_smpl_pose"].to(self.device)
    model_targets["futr_smpl_beta"] = batch["futr_smpl_beta"].to(self.device)
    model_targets["futr_smpl_transl"] = batch["futr_smpl_transl"].to(
        self.device)

    return model_inputs, model_targets

  def init_optimizer(self,):
    opts = self.opts
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=opts.optim.learning_rate,
        betas=(opts.optim.beta1, opts.optim.beta2),
    )
    len_dataloader = len(self.dataloader)
    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        self.optimizer,
        max_lr=opts.optim.learning_rate,
        epochs=opts.train.num_epochs,
        steps_per_epoch=len_dataloader,
        cycle_momentum=False,
        pct_start=0.005,
    )
    return

  def get_current_scalars(self,):
    lr = self.scheduler.get_last_lr()[0]
    loss_dict = {
        "total_loss": self.smoothed_total_loss,
        "lr": lr,
    }

    for k in self.smoothed_factor_losses.keys():
      loss_dict["loss_" + k] = self.smoothed_factor_losses[k]
    return loss_dict

  def backward(self,):
    opts = self.opts
    self.optimizer.zero_grad()
    self.total_loss.backward()
    self.optimizer.step()
    self.scheduler.step()
    return

  def forward(self, model_inputs, model_targets):
    opts = self.opts
    predictions = self.model.forward(model_inputs)
    loss = self.model.compute_loss(model_targets)
    weight_dict = self.weight_dict

    weighted_loss = {}
    for key in loss.keys():
      if key in weight_dict.keys():
        weighted_loss[key] = loss[key] * weight_dict[key]
      else:
        logger.info(f"{key} no weight available")

    total_loss = 0.0
    self.loss_factors = {}
    for key, l in weighted_loss.items():
      self.loss_factors[key] = l.mean()
      total_loss += self.loss_factors[key]

    self.total_loss = total_loss

    smooth_alpha = 0.0
    for k in self.smoothed_factor_losses.keys():
      if k in self.loss_factors.keys():
        self.smoothed_factor_losses[k] = (
            smooth_alpha * self.smoothed_factor_losses[k] +
            (1 - smooth_alpha) * self.loss_factors[k].item())

    self.smoothed_total_loss = (
        self.smoothed_total_loss * smooth_alpha +
        (1 - smooth_alpha) * self.total_loss.item())

    return predictions

  def define_criterion(self):
    opts = self.opts
    self.smoothed_total_loss = 0.0
    self.smoothed_factor_losses = {
        "points_3d": 0.0,
        "pose": 0.0,
        "trans": 0.0,
        "beta": 0.0,
    }
    self.weight_dict = {}
    self.weight_dict["pose"] = opts.loss.pose
    self.weight_dict["trans"] = opts.loss.trans
    self.weight_dict["beta"] = opts.loss.beta
    return


def to_tensorboard_img(img):
  img = np.moveaxis(img, -1, 0)
  return img
