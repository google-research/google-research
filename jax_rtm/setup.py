# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Setup script for the jax_rtm package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="jax_rtm",
    version="0.1.0",
    description=(
        "JAX-Differentiable Radiative Transfer Model (RTM) for Satellite"
        " Simulator"
    ),
    author="Google Research Contrails Team",
    author_email="mccloskey@google.com",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "flax>=0.6.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "jax_rtm": [
            "data/params_992.json",
            "data/weather_85x85.npz",
            "data/weather_339x339.npz",
        ],
    },
)
