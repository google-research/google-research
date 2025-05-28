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

"""Installation for package."""

from setuptools import find_packages
from setuptools import setup


setup(
    name="hct",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "absl-py",
        "chex",
        "jax",
        "flax",
        "diffrax",
        "optax",
        "dm-haiku",
        "numpy",
        "jupyter",
        "jupyter_http_over_ws",
        "matplotlib",
    ]
)
