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

"""Default config."""
import ml_collections


def get_config():
  """Default config for DDM with scalar latents."""

  config = ml_collections.ConfigDict()

  config.predict_z0 = False

  config.latent_dim = 8

  config.use_bases = False

  config.sample_save_dir = ''
  config.num_samples = 64

  config.num_sample_nodes = 80000
  config.num_verts = 30000  # Number of vertices in triangulation

  config.spec = ''
  config.spec_type = 'hks'
  config.spec_features = 16
  config.num_clust = 6

  # Flag whether to sample during training
  # Sampling can take up extra memory, so set to false if memory is an issue

  config.sample_during_training = False

  # DDM UNet parameters
  # Number of levels in unet, currently has no effect because levels are
  # hardcoded.
  config.num_levels = 2

  config.attn_neigh = 64

  #############################################################################
  # Key parameters
  #############################################################################

  ## Directory containing the weights for the pre-trained VAE
  # 8 dim latents, 10K, kl 1e-2
  config.trainer_checkpoint_dir = ''

  # Directory containing train/test tfrecords for models
  config.geom_record_path = ''

  config.obj_name = ''  # 3D model name

  config.obj_labels = ''  # .ply file with labeled faces

  config.inpaint_labels = ''  # .ply file with labeled faces

  config.k_conv = 8  # Number of neighbors in each convolution
  config.unet_features = 64  # Number of features at finest resolution in unet.
  config.mlp_layers = (
      1  # Num layers in Tangent MLP preceeding each convolution res block.
  )
  config.hdim = 8  # Number of hidden layers in convolution filters
  config.ddm_schedule = 'cos'  # Diffusion sampling timescale

  # Flag whether to train the model from scratch (True) or load a pre-trained
  # model ang generate samples (False)
  config.train = True
  config.model_checkpoint_dir = ''

  config.adam_args = dict(b1=0.9, b2=0.999, eps=1.0e-8, eps_root=0)

  # Number of training steps
  # num_steps = 300000
  num_steps = 200000  # SEAL

  config.num_steps = num_steps
  config.warmup_steps = 1000
  config.log_loss_every_steps = 100
  config.eval_every_steps = 10000
  config.checkpoint_every_steps = 2500

  # Learning rate
  config.schedule = 'cosine'  # constant, linear or cosine
  config.learning_rate = 5.0e-4
  config.end_learning_rate = 1.0e-5
  config.init_learning_rate = 0.00

  config.use_ema = False
  config.ema_decay = 0.995
  start_ema_after = 5000

  # DDM timescale params
  config.timestep = 1000
  config.batch_size = 16  # SEAL, 8 is default.
  config.num_recon_samples = config.batch_size

  config.start_ema_after = start_ema_after * max(config.batch_size // 8, 1)
  config.update_ema_every = max(config.batch_size // 8, 1)

  config.num_train_samples = 500
  config.num_test_samples = 16

  # Single integer or tuple.
  config.seed = None

  config.trial = 0  # Dummy for repeated runs.

  return config
