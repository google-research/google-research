# Copyright 2024 The Google Research Authors.
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

export DATA_ROOT=./data/NeRF/nerf_synthetic
export CKPT_ROOT=./data/Plenoctree/checkpoints/syn_sh16
export SCENE=lego
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz

python -m octree.optimization \
    --input $CKPT_ROOT/$SCENE/octrees/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz

python -m octree.evaluation \
    --input $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

# [Optional] Only used for in-browser viewing.
echo python -m octree.compression \
    $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --out_dir $CKPT_ROOT/$SCENE/ \
    --overwrite