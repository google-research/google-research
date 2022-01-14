# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Base configuration.

config.render_width and config.crop_width must be added for use, as done in
config_lq.py, config_mq.py and config_hq.py
"""

import math

from dreamfields.config import coco_queries_val
import ml_collections



def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config(iters=10000):  # pylint: disable=invalid-name
  """Generate base config dict.

  Args:
    iters (int): Number of training iterations. Some hyperparameter schedules
      are based on this value, as well as the total training duration.

  Returns:
    config (ml_collections.ConfigDict): Configuration object.
  """


  # Hyperparams
  config = D(
      defragment_every=200,  # Manually defragment memory at this interval.

      ## Prompt engineering.
      query_template='{query}',

      ## Scene parameters and representation.
      jitter=True,  # Jitter strided samples along each ray.
      viewdirs=False,  # Condition NeRF's colors on viewing direction.
      posenc_dirs_deg=4,  # Number of frequencies in view positional encoding.
      num_samples=192,  # Number of strided samples along each ray during train.
      test=D(  # Parameters for validation view rendering, low quality videos.
          jitter=False,  # No jitter along ray during eval.
          white_bkgd=True,  # White background for eval.
          num_samples=256,  # More samples to reduce aliasing.
      ),
      test_hq=D(  # Parameters for high-quality evaluation videos.
          jitter=False,
          white_bkgd=True,
          num_samples=512,
          # Allow sampling outside scene bounds at test, due to a NaN error when
          # intersecting with a large number of samples.
          intersect_box=False,
      ),
      # Near and far bounds for sampling points along rays. Set based on scene
      # bounds so as to avoid clipping the cubical bounds.
      near=4. - math.sqrt(3) * 1,  # Camera radius - cube diagonal * mr1
      far=4. + math.sqrt(3) * 1,
      posenc_deg=8,  # Number of frequencies in x,y,z coordinate encoding.
      # Mask radiance to set scene bounds. The maximum norm of the (x,y,z)
      # coordinate where density is allowed to lie is linearly ramped from mr0
      # at the start of training to mr1 at mr_i_split iterations.
      mr_norm='inf',  # Use an infinity norm to define bounds. 'inf' or an int.
      mr_i_split=iters,
      mr0=1.,
      mr1=1.,
      white_bkgd=False,  # If True, composite renderings with a white bg.
      # Use Random Fourier Feature positional encodings instead of axis-aligned.
      fourfeat=True,
      # NeRF included noise before the density activation. Noise standard
      # deviation is linearly ramped from sn0 to sn1 over sn_i_split iterations.
      sn_i_split=iters,
      sn0=0.,
      sn1=0.,

      ## Camera sampling during training.
      focal_mult_range=[1.2, 1.2],  # During training, enlarge objects.
      th_range=[0, 360],  # Camera azimuth in degrees. Sample in 360 degrees.
      phi_range=[-30, -30],  # Camera elevation in degrees.
      rad_range=[4., 4.],  # Camera distance from center of the scene. Fixed.
      ema_scene_origin=True,  # Keep track of the center of mass of the scene.
      # Weight on experimental loss encouraging center of mass to be at 0.
      # This is disabled by setting zero_origin_lam = 0.
      zero_origin_lam=0.,

      ## MLP architecture.
      mlp_activation='swish',  # swish|relu.
      # Geometry network.
      features_early=[128],  # Dense layers before residual blocks.
      features_residual=[(256, 128)] * 4,  # Residual block feature dimensions.
      features_late=[128, 4],  # Features dimensions after concat viewdirs.
      parameterization='mipnerf',  # Only mipnerf is implemented.
      mipnerf=D(
          use_cov=True,  # If True, use integrated positional encoding (IPE).
          sigma_activation='softplus',  # softplus|relu. NeRF used relu.
          decay_start=1.,  # Disable coarse to fine annealing of IPE.
          decay_iters=1000,  # Duration of coarse to fine annealing.
      ),

      ## CLIP loss.
      # Training model. clip_<vit_b16|vit_b32|resnet_50|resnet_101|resnet_50x4>
      # pylint: disable=line-too-long
      # See https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/clip/model.py#L45
      # pylint: enable=line-too-long
      loss_model='clip_vit_b16',
      clip_width=224,  # Resize images to this size for input to CLIP.

      ## Augmentations.
      augment_backgrounds=True,
      checker_bg_nsq=8,
      noise_bg_prob=0.333,
      checker_bg_prob=0.333,
      fft_bg_prob=0.334,
      min_aug_acc=0.,
      bg_blur_std_range=[0., 10.],
      n_local_aug=8,  # Number of random crops per local device.

      ## Sparsify with transmittance loss.
      # 'neg_lam_transmittance_clipped' enables the transmittance loss, and
      # None disables it.
      transparency_loss='neg_lam_transmittance_clipped',
      acc_lam=0.5,  # Weight on transmittance loss.
      acc_lam_after=0,  # Immediately apply transmittance loss.
      # Target 50% opacity initially and anneal exponentially to target 10%
      # opacity over the first 500 iterations. Annealing in the target avoids
      # overshooting. Without annealing, some prompts produce 0 density scenes.
      acc_target0=0.5,
      acc_target1=0.1,
      acc_target_i_split=500,

      ## Optimization.
      # Ramp learning rate from lr0 to lr1 linearly for the first lr_i_split
      # iterations, then decay to lr2 until iters.
      lr_i_split=1500,
      lr0=1e-5,
      lr1=1e-4,
      lr2=1e-4,
      lr_cosine_decay=False,  # If True, use cosine lr decay. Otherwise, linear.
      adam_eps=1e-5,
      iters=iters,
      # Number of substeps per training iteration. Larger values slightly
      # increase throughput by reusing the same camera pose (and rays) across
      # substeps, and by JIT compiling larger blocks of code.
      substeps=1,
      # Apply the beta loss after beta_after iterations. This value dsables it.
      beta_after=2 * iters,

      ## Retrieval metric.
      retrieve_models=['clip_vit_b32'],  # Compute R-Precision with CLIP B/32.
      retrieve_widths=[224],  # Expected input resolution of retrieval model.
      # Negatives for R-Precision. The real query used for synthesis is added
      # to queries_r automatically if not provided.
      queries_r=coco_queries_val.queries,

      ## Logging.
      log_scalars_every=50,  # Log to tensorboard at this interval.
      flush_every=500,  # Flush tensorboard log writer at this interval.
      # Render a clip_width resolution image at this interval and compute
      # R-Precision using retrieve_models.
      render_every=1000,
      # If True, render a lq 300x300 video every video_every iterations.
      render_lq_video=True,
      video_every=5000,
      # If True, render a hq 400x400 video every hq_video_every iterations.
      render_hq_video=True,
      hq_video_every=10000,
      # Checkpointing options.
      checkpoint_every=1000,
      keep_every_n_steps=10000,
  )

  return config
