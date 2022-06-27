# coding=utf-8
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

"""Setup dreamfields package."""

import setuptools

setuptools.setup(
    name="dreamfields",
    version="0.1.0",
    packages=setuptools.find_packages(include=["dreamfields", "dreamfields.*"]),
    description="Text to 3D",
    author="Google Research",
    install_requires=[
        "absl-py",
        "clu",
        "clip @ git+https://github.com/openai/CLIP.git",
        "dm_pix",
        "jax",
        "jaxlib",
        "flax",
        "matplotlib>=3.3.0",  # this version adds the turbo cmap
        "mediapy",
        "ml_collections",
        "numpy",
        "regex",
        "scenic @ git+git://github.com/google-research/scenic.git",
        "scipy",
        "tensorflow>=2.7.0",
        "tqdm",
        "torch"
    ],
)
