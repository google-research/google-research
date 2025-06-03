#!/bin/bash
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



GPUS="device=${1:-'0'}"
EXPNAME=${2:-'docker_run'}

docker run \
    --gpus $GPUS \
    --mount type=bind,source="$HOME"/diffusion_3d_logs,target=/gcs/xcloud-shared/$USER/diffusion_3d \
    -it diffusion_3d:latest \
    python diffusion_3d/main.py \
        --config.view_specific_templates='("{query}. on a white background.",)' \
        --config.seed=0 \
        --config.output_dir="/gcs/xcloud-shared/$USER/diffusion_3d/$(date +"%m-%d-%s")_$EXPNAME" \
        --config.query="a mushroom with a red cap" \
        --config.t_respace=256 \
        --config.t_segments="((256, 0, 1, 1),)" \
        --config.render_deformation=False \
        --config.optimize_deformation_codes=False \
        --config.optimize_azimuths=True \
        --config.optimize_elevations=False \
        --config.optimize_focal_mults=True \
        --config.optimize_radius=True \
        --config.optimize_loss_weights=False \
        --config.optimize_matching=False \
        --config.guide_after_timestep=-1 \
        --config.classifier_free_guidance_scale=7 \
        --config.fix_target_n_iter=30000 \
        --config.dont_optimize_n_iter=256 \
        --config.n_fixed_views=8 \
        --config.validate_every=5000 \
        --config.visualize_every=500 \
        --config.plot_losses_every=500 \
        --config.mlp_config.emb_color_config.n_levels=1 \
        --config.mlp_config.emb_color_config.n_features_per_level=4 \
        --config.mlp_config.emb_color_config.base_resolution=16 \
        --config.decayscales.color="(0., 0.)" \
        --config.per_point_deformation_lam=0 \
        --config.volsdf_eikonal_lam=0 \
        --config.depth_tv_lam="(0., 0.)"
