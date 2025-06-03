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



inertia_list=(0.1 0.2 0.3)
cognitive_coeff_list=(0.1 0.2 0.3 0.4 0.5)
social_coeff_list=(0.2 0.3 0.4 0.5 0.6)
repel_coeff_list=(0.01 0.05 0.1)
step_length_list=(0.5 0.6 0.7 0.8 0.9 1.0)

while true
do
    inertia=${inertia_list[$RANDOM % ${#inertia_list[@]} ]}
    cognitive_coeff=${cognitive_coeff_list[$RANDOM % ${#cognitive_coeff_list[@]} ]}
    social_coeff=${social_coeff_list[$RANDOM % ${#social_coeff_list[@]} ]}
    repel_coeff=${repel_coeff_list[$RANDOM % ${#repel_coeff_list[@]} ]}
    step_length=${step_length_list[$RANDOM % ${#step_length_list[@]} ]}

    python -m search.py \
        -n legal_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length} \
        -e multitask \
        -d legal \
        -g 0,1,2,3,4 \
        --inertia $inertia \
        --cognitive_coeff $cognitive_coeff \
        --social_coeff $social_coeff \
        --repel_coeff $repel_coeff \
        --step_length $step_length \
        --starting_test_set_eval 1 \
        --fast_merge 1 \
        --project_name_wb legal_sweep \
        --weight_randomess 1 \
        --populate_initial_experts 1 \
        --initial_experts_num 20 \
        --starting_velocity_mode random \
        --repel_term 1 \
        --step_length_factor 0.95 \
        --restart_stray_particles 1 \
        --restart_patience 0.67
done