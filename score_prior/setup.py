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

"""Pip setup for https://pypi.org/project/score_prior/."""
import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="score_prior",
    version="0.1.0",
    author="Google LLC",
    author_email="bfeng@caltech.edu",
    description="Score-based priors for inverse problems in imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/google-research/score_prior",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "absl-py",
        "diffrax",
        "equinox",
        "flax",
        "jax",
        "jaxtyping",
        "ml-collections",
        "numpy",
        "optax",
        "scipy",
        "tensorflow",
        "tensorflow-datasets",
        "tensorflow-probability",
        "Pillow",
    ],
    python_requires=">=3.8",
)
