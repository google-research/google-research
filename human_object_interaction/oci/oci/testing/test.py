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

"""A test of SMPL model behavior."""

import os
import os.path as osp
import sys

from loguru import logger
from oci.datasets import behave_data as behave_data_utils
from oci.html_utils import html_viz
from oci.model import batch_norm as batch_norm_utils
from oci.model import oci_model
from oci.model import oci_seq_model
from oci.utils import model_vis_utils
from oci.utils import tensor_utils
import torch

sys.path.append("/home/nileshkulkarni_google_com/Research/behave_code")

# pylint: disable=g-import-not-at-top
# pylint: enable=g-import-not-at-top
# pylint: disable=missing-function-docstring
# pylint: disable=g-builtin-op
# pylint: disable=too-many-format-args


class OCITester:
  """An OCI Tester."""

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
    if opts.model.sequence_model:
      model = oci_seq_model.OCIModel(opts.model).to(device)
    else:
      model = oci_model.OCIModel(opts.model).to(device)
    self.model = model
    self.load_network(self.model, "model", opts.test.num_epochs)
    model.eval()
    batch_norm_utils.turnNormOff(self.model)
    return

  def init_dataset(self,):
    opts = self.opts
    dataloader = behave_data_utils.behave_dataoader(
        opts.dataloader, shuffle=True)
    self.dataloader = dataloader
    return

  def init_testing(self,):
    opts = self.opts
    self.save_dir = f"./checkpoints/{opts.name}"
    self.define_model()
    self.init_dataset()
    self.init_html_vis()
    os.makedirs(self.save_dir, exist_ok=True)
    opts_file = osp.join(self.save_dir, "opts_test.yaml")
    with open(opts_file, "w") as f:
      f.write(opts.dump())
    return

  def init_html_vis(self,):
    opts = self.opts
    save_dir = "./visual_logs/{}".format(opts.name, "test")
    os.makedirs(save_dir, exist_ok=True)
    key_order = ["index", "curr_frame_1", "futr_frame_1", "pred_futr_frame_1"]
    self.html_viz = html_viz.HTMLVis(save_dir, key_order=key_order)
    return

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
    if gpu_id is not None and torch.cuda.is_available():
      network.cuda(device=gpu_id)
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
      msg += f"{key}  {value:.3f}, "
    logger.info(msg)
    return

  def set_input(self, batch):
    opts = self.opts
    model_inputs = {}
    model_inputs["points_input"] = batch["can_input_pts"].to(self.device)
    model_inputs["points_futr"] = batch["can_points_futr"].to(self.device)
    model_inputs["points_can"] = batch["can_points_smpl"].to(self.device)
    model_inputs["smpl_pose"] = batch["smpl_pose"].to(self.device)
    model_inputs["smpl_beta"] = batch["smpl_beta"].to(self.device)

    if not opts.model.sequence_model:
      model_inputs["points_input"] = model_inputs["points_input"][:, 0]
      model_inputs["smpl_pose"] = model_inputs["smpl_pose"][:, 0]
      model_inputs["smpl_beta"] = model_inputs["smpl_beta"][:, 0]
    self.model_inputs = model_inputs
    self.batch = batch
    return

  def create_visuals(self,):
    opts = self.opts
    batch_index = 0
    data_inputs = {}
    data_inputs["seq_folder"] = self.batch["seq_folder"][batch_index]
    data_inputs["seq_frame_id"] = self.batch["frame_id"][batch_index].item()
    data_inputs["can2world"] = tensor_utils.tensor_to_numpy(
        self.batch["can2world"][batch_index])

    if opts.model.sequence_model:
      data_inputs["can_points_curr"] = self.model_inputs["points_input"][
          batch_index, -1].cpu()
    else:
      data_inputs["can_points_curr"] = self.model_inputs["points_input"][
          batch_index].cpu()

    if opts.dataloader.relative_targets:
      data_inputs["can_points_futr"] = self.model.convert_relative_to_absolute(
          data_inputs["can_points_curr"],
          self.model_inputs["points_futr"])[batch_index]
    else:
      data_inputs["can_points_futr"] = self.model_inputs["points_futr"][
          batch_index].cpu()

    predictions = {}
    if opts.dataloader.relative_targets:
      predictions["points_futr"] = self.model.convert_relative_to_absolute(
          data_inputs["can_points_curr"],
          self.predictions["points_futr"])[batch_index]
    else:
      predictions["points_futr"] = self.predictions["points_futr"][
          batch_index].cpu()

    predictions["points_futr"] = tensor_utils.tensor_to_numpy(
        predictions["points_futr"])
    smpl_faces = self.batch["smpl_mesh_f"][batch_index]
    visualizer = model_vis_utils.VisualizeModelOutputs(
        osp.join(opts.dataloader.seq_folder, data_inputs["seq_folder"]),
        vis_mesh=opts.visualization.viz_mesh,
        faces=smpl_faces.numpy(),
    )
    visuals = visualizer.visualize_outputs(data_inputs, predictions)
    visuals["index"] = self.batch["frame_id"][batch_index].item()

    self.html_viz.add_results(visuals)
    return

  def get_predicted_points_futr(self,):
    opts = self.opts
    if opts.dataloader.relative_targets:
      points_futr = (
          self.predictions["points_futr"].cpu() + self.batch["can_points_curr"])
    else:
      points_futr = self.predictions["points_futr"]
    points_futr = tensor_utils.tensor_to_numpy(points_futr)
    return points_futr

  def get_current_scalars(self,):
    lr = self.scheduler.get_last_lr()[0]
    loss_dict = {
        "total_loss": self.smoothed_total_loss,
        "lr": lr,
    }

    for k in self.smoothed_factor_losses.keys():
      loss_dict["loss_" + k] = self.smoothed_factor_losses[k]
    return loss_dict

  def predict(self,):

    self.predictions = self.model.forward(self.model_inputs)
    self.loss = self.model.compute_loss(self.model_inputs)
    return

  def define_criterion(self):
    self.smoothed_total_loss = 0.0
    self.smoothed_factor_losses = {
        "points_3d": 0.0,
    }
    return

  def test(self):
    opts = self.opts
    dataloader = self.dataloader
    self.total_steps = total_steps = 0

    for _, batch in enumerate(dataloader):
      self.set_input(batch)
      self.predict()
      self.create_visuals()
      total_steps += 1
      self.total_steps = total_steps

      if total_steps >= opts.test.num_iters:
        break
    self.html_viz.write_html()
