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

#!/bin/bash
#
# A collection of useful flag groups to make experiment scripts more readable
# and maintainable.

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi

LAUNCH=python
LAUNCHER=( $DIR/../launcher.py )


TPU_FLAGS=( --allow_mixed_precision=True
            --defer_blurring=True )

RESNET_FLAGS=( --resnet_architecture=RESNET_V1 )
RESNET50x1_FLAGS=( "${RESNET_FLAGS[@]}"
                   --resnet_depth=50
                   --resnet_width=1 )
RESNET101x1_FLAGS=( "${RESNET_FLAGS[@]}"
                    --resnet_depth=101
                    --resnet_width=1 )
RESNET200x1_FLAGS=( "${RESNET_FLAGS[@]}"
                    --resnet_depth=200
                    --resnet_width=1 )

CIFAR10_FLAGS=( --input_fn=cifar10
                --shard_per_host=False
                --first_conv_kernel_size=3
                --first_conv_stride=1
                --use_initial_max_pool=False
                --image_size=32
                --blur_probability=0
                --apply_whitening=True
                --crop_area_range=0.2,1.0
                --use_pytorch_color_jitter=True
                --eval_crop_method=IDENTITY
                --crop_padding=0
                --label_smoothing=0 )

IMAGENET_FLAGS=( --input_fn=imagenet
                 --shard_per_host=True
                 --first_conv_kernel_size=7
                 --first_conv_stride=2
                 --use_initial_max_pool=True
                 --image_size=224
                 --blur_probability=0.5
                 --label_smoothing=0.05 )

CONTRASTIVE_FLAGS=( --stop_gradient_before_classification_head=True
                    --contrast_mode=ALL_VIEWS
                    --summation_location=OUTSIDE
                    --denominator_mode=ALL
                    --stage_1_contrastive_loss_weight=1.0
                    --stage_1_cross_entropy_loss_weight=0.0
                    --stage_2_contrastive_loss_weight=0.0
                    --stage_2_cross_entropy_loss_weight=1.0
                    --stage_1_use_encoder_weight_decay=True
                    --stage_1_use_projection_head_weight_decay=True
                    --stage_1_use_classification_head_weight_decay=False
                    --stage_2_use_encoder_weight_decay=False
                    --stage_2_use_projection_head_weight_decay=False
                    --stage_2_use_classification_head_weight_decay=True
                    --stage_1_update_encoder_batch_norm=True
                    --stage_2_update_encoder_batch_norm=False )
SUPCON_FLAGS=( "${CONTRASTIVE_FLAGS[@]}"
               --use_labels=True
               --normalize_embedding=True
               --use_bias_weight_decay=True
               --use_projection_batch_norm=False
               --projection_head_layers=128
               --zero_initialize_classifier=False
               --scale_by_temperature=True )
SIMCLR_FLAGS=( "${CONTRASTIVE_FLAGS[@]}"
                 --use_labels=False
                 --normalize_embedding=False
                 --use_bias_weight_decay=True
                 --use_projection_batch_norm=True
                 --projection_head_layers="2048,2048,128"
                 --zero_initialize_classifier=True
                 --scale_by_temperature=False )

XENT_FLAGS=( --stage_2_epochs=0
             --stage_2_warmup_epochs=0
             --stop_gradient_before_classification_head=False
             --stage_1_contrastive_loss_weight=0.0
             --stage_1_cross_entropy_loss_weight=1.0
             --stage_2_contrastive_loss_weight=0.0
             --stage_2_cross_entropy_loss_weight=0.0
             --stage_1_use_encoder_weight_decay=True
             --stage_1_use_projection_head_weight_decay=False
             --stage_1_use_classification_head_weight_decay=True
             --stage_2_use_encoder_weight_decay=False
             --stage_2_use_projection_head_weight_decay=False
             --stage_2_use_classification_head_weight_decay=False
             --stage_1_update_encoder_batch_norm=True
             --stage_2_update_encoder_batch_norm=False
             --normalize_embedding=False )
