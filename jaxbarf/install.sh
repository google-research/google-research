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

# This installation was confirmed on a GCP VM with one P100 GPU created with
# a Deep Learning image, specifically a Debian 10 based Deep Learning
# image with CUDA 11.3 preinstalled.

# Create the virtual environment.
conda create --name jaxbarf pip python=3.9
conda activate jaxbarf

# Install the requirements.
pip install -r requirements.txt

# Enable GPU training with JAX.
pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
