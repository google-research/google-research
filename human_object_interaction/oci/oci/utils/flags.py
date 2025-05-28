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

"""Flags for training and testing.
"""

# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=unused-argument
# pylint: disable=g-no-space-after-comment


from fvcore.common.config import CfgNode

_C = CfgNode()

_C.dataloader = CfgNode()
_C.dataloader.single_sequence = False
_C.dataloader.smpl_fit_name = "fit02"
_C.dataloader.obj_fit_name = "fit01"
_C.dataloader.seq_folder = "/mnt/data/Research/data/BEHAVE/sequences/"
_C.dataloader.seq_filter = "chairwood_sit"
_C.dataloader.seq_start = 0
_C.dataloader.seq_end = -1
_C.dataloader.num_input_frames = 1
_C.dataloader.num_workers = 16
_C.dataloader.batch_size = 8
_C.dataloader.datadir = "/home/nileshkulkarni_google_com/Research/data/BEHAVE/"
_C.dataloader.smpl = CfgNode()
_C.dataloader.smpl.native_path = (
    "/mnt/data/Research/object_interactions/smplx/clean_models2/")
_C.dataloader.num_sampled_points = 5000
_C.dataloader.sample_all_verts = False
_C.dataloader.shuffle = True
# _C.dataloader.relative_targets = True
_C.train = CfgNode()
_C.train.num_epochs = 1000
_C.train.num_pretrain_epochs = -1

_C.test = CfgNode()
_C.test.num_epochs = 1000
_C.test.num_iters = 10

_C.optim = CfgNode()
_C.optim.learning_rate = 5e-4
_C.optim.beta1 = 0.90
_C.optim.beta2 = 0.99

_C.smpl_style = True
_C.model = CfgNode()
_C.model.sequence_model = True
_C.model.mlp_activation = "leakyReLU"
_C.model.smpl_name = "smpl"

_C.model.pose_encoder = CfgNode()
_C.model.pose_encoder.num_neurons = 128
_C.model.pose_encoder.latentD = 128

_C.model.feature_backbone = CfgNode()
_C.model.feature_backbone.d_out = 128

_C.model.predictor_out_type = "3d"

_C.model.pose_seq_model = CfgNode()
_C.model.pose_seq_model.hidden_dim = 128
_C.model.pose_seq_model.num_layers = 4

_C.model.point_seq_model = CfgNode()
_C.model.point_seq_model.hidden_dim = 128
_C.model.point_seq_model.num_layers = 4

_C.model.relative_targets = False
_C.model.relative_inputs = False

_C.model.loss_type = "smooth_l1_loss"  # "mse" , "l1"

vibe_model = CfgNode()

vibe_model.temporal_type = "gru"

vibe_model.tgru = CfgNode()
vibe_model.tgru.n_layers = 2
vibe_model.tgru.add_linear = False
vibe_model.tgru.use_residual = False
vibe_model.tgru.bidirectional = False
vibe_model.tgru.hidden_size = 1024
vibe_model.smpl_model_path = _C.dataloader.smpl.native_path
vibe_model.relative_targets = False
_C.model.vibe_model = vibe_model

_C.optimizer = CfgNode()

_C.loss = CfgNode()

_C.loss.points_3d = 1.0
_C.loss.pose = 1.0
_C.loss.beta = 1.0
_C.loss.trans = 1.0

_C.logging = CfgNode()
_C.logging.log_freq = 20
_C.logging.save_epoch_freq = 100

_C.logging.val_log_freq = 100
_C.logging.histogram = False
_C.logging.hist_freq = 100

_C.visualization = CfgNode()
_C.visualization.viz_mesh = True

motion_sequence = _C.visualization.motion_sequence = CfgNode()
motion_sequence.num_future_frames = 10

_C.mode = "train"
_C.name = "test_exp"


def get_cfg_defaults():
  return _C.clone()
