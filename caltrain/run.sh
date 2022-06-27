# Copyright 2022 The Google Research Authors.
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
set -e
set -x

virtualenv -p python3 env3
source env3/bin/activate

pip install -r caltrain/requirements.txt
export MPLBACKEND=Agg
DATA_DIR=caltrain/data
PLOT_DIR=caltrain/plots

python -m caltrain.download_data --data_dir=${DATA_DIR?}

python -m caltrain.plot_bias_heat_map --plot_dir=${PLOT_DIR?} --data_dir=${DATA_DIR?}
python -m caltrain.plot_calibration_errors --plot_dir=${PLOT_DIR?} --data_dir=${DATA_DIR?}
python -m caltrain.plot_ece_vs_tce --plot_dir=${PLOT_DIR?} --data_dir=${DATA_DIR?}
python -m caltrain.plot_glm_beta_eece_sece --plot_dir=${PLOT_DIR?} --data_dir=${DATA_DIR?}
python -m caltrain.plot_intro_ece_distribution --plot_dir=${PLOT_DIR?}
python -m caltrain.plot_intro_reliability_diagram --plot_dir=${PLOT_DIR?}
python -m caltrain.plot_tce_assumptions --plot_dir=${PLOT_DIR?}
