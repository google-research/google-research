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

"""Tester for SMPL."""
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=redefined-outer-name
# pylint: disable=g-wrong-blank-lines
# pylint: disable=g-inconsistent-quotes
# pylint: disable=broad-except

import os
import os.path as osp
import sys

from loguru import logger
from oci.datasets import behave_data as behave_data_utils
from oci.html_utils import html_viz
from oci.model import batch_norm as batch_norm_utils
from oci.model import oci_smpl_model
from oci.utils import model_vis_utils
from oci.utils import motion_sequence
from oci.utils import tensor_utils
from psbody.mesh import Mesh
import torch

sys.path.append("/home/nileshkulkarni_google_com/Research/behave_code")


class OCITesterSMPL:

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
    save_dir = "./work_dir/htmls/{}/{}/".format(opts.name, "inference")
    os.makedirs(save_dir, exist_ok=True)
    key_order = ["index", "start_frame_id", "rend_video_path"]
    self.html_viz = HTMLVis(save_dir, key_order=key_order)
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
    self.batch = batch
    return model_inputs, model_targets

  def create_sequence_visuals(self, model_inputs):
    opts = self.opts
    batch_index = 0
    data_inputs = {}
    index = self.batch["index"][batch_index].item()
    data_inputs["seq_folder"] = self.batch["seq_folder"][batch_index]
    data_inputs["seq_frame_id"] = self.batch["frame_id"][batch_index].item()
    data_inputs["can2world"] = tensor_utils.tensor_to_numpy(
        self.batch["can2world"][batch_index])
    smpl_model_path = opts.dataloader.smpl.native_path
    motion_sequence_generator = motion_sequence.MotionSequenceGenerator(
        opts.visualization.motion_sequence,
        self.model,
        smpl_model_path=smpl_model_path,
    )
    (
        output_meshes,
        sequence_predictions,
    ) = motion_sequence_generator.generate_sequence(input_data=model_inputs)

    sequence_animator = model_vis_utils.SequenceVisualizer(
        seq_folder=osp.join(opts.dataloader.seq_folder,
                            data_inputs["seq_folder"]),
        start_frame_id=data_inputs["seq_frame_id"] - 2,
        can2WorldRT=data_inputs["can2world"],
    )
    save_dir = osp.join(f"./work_dir/results/{opts.name}/{opts.mode}/{index}")
    os.makedirs(save_dir, exist_ok=True)
    obj_mesh = Mesh(
        self.batch["obj_mesh_v"][batch_index].numpy(),
        self.batch["obj_mesh_f"][batch_index].numpy(),
    )
    num_sequences = len(output_meshes)
    try:
      rend_video_path = sequence_animator.forward(
          save_dir=save_dir,
          mesh_sequences=output_meshes,
          object_meshes=[obj_mesh] * num_sequences,
      )
    except Exception as _:
      logger.info(f"Failed to render video for {index}")
      rend_video_path = "./not_avaliable"

    visuals = {}
    visuals["rend_video_path"] = rend_video_path
    visuals["index"] = index
    visuals["start_frame_id"] = data_inputs["seq_frame_id"]
    return visuals

  def create_next_visuals(self,):
    opts = self.opts
    batch_index = 0
    data_inputs = {}
    # index = self.batch["index"][batch_index].item()
    data_inputs["seq_folder"] = self.batch["seq_folder"][batch_index]
    data_inputs["seq_frame_id"] = self.batch["frame_id"][batch_index].item()
    data_inputs["can2world"] = tensor_utils.tensor_to_numpy(
        self.batch["can2world"][batch_index])
    smpl_model_path = opts.dataloader.smpl.native_path
    motion_sequence_generator = motion_sequence.MotionSequenceGenerator(
        opts.visualization.motion_sequence,
        self.model,
        smpl_model_path=smpl_model_path,
    )
    # (
    #     output_meshes,
    #     sequence_predictions,
    # ) = motion_sequence_generator.generate_sequence(input_data=model_inputs)

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
    opts = self.opts

    # self.predictions = self.model.forward(self.model_inputs)
    # self.loss = self.model.compute_loss(self.model_inputs)
    return

  def define_criterion(self):
    opts = self.opts
    self.smoothed_total_loss = 0.0
    self.smoothed_factor_losses = {
        "points_3d": 0.0,
    }
    return

  def test(self):
    opts = self.opts
    dataloader = self.dataloader
    self.total_steps = total_steps = 0
    all_visuals = []
    for bx, batch in enumerate(dataloader):
      model_inputs, model_targets = self.set_input(batch)
      self.predict()
      visuals = self.create_sequence_visuals(model_inputs)
      all_visuals.append(visuals)
      total_steps += 1
      self.total_steps = total_steps
      self.html_viz.add_results(visuals)
      if total_steps >= opts.test.num_iters:
        break
    self.html_viz.write_html()


class HTMLVis(html_viz.HTMLVisBase):

  @staticmethod
  def write_results_block(fp, keyorder, results, save_dir):
    fp.write("<table>\n")
    fp.write("<tr>")
    for keyname in keyorder:
      fp.write(f"<th> {keyname} </th>")
    fp.write("</tr>")

    for result in results:
      fp.write("<tr>\n")
      index = result["index"]
      index_save_dir = osp.join(save_dir, "media", f"{index}")

      for key in keyorder:
        if key == "index":
          fp.write(f"<td> {result[key]} </td>")
        elif key == "start_frame_id":
          fp.write(f"<td> {result[key]} </td>")
        elif key == "rend_video_path":
          keypath = osp.relpath(result[key], save_dir)
          fp.write(
              f'<td><video height="480" controls> <source src="{keypath}" type="video/mp4" /> </video></td>'
          )
      fp.write("</tr>\n")
    fp.write("</table>\n")
    return
