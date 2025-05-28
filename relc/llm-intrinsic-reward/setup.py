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

from setuptools import find_packages, setup


__version__ = "1.0.0.dev0"  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)

REQUIRED_PKGS = [
    "torch>=1.4.0",
    "transformers>=4.18.0",
    "numpy>=1.18.2",
    "accelerate",
    "datasets",
    "tyro>=0.5.7",
]
EXTRAS = {
    "test": ["parameterized", "pytest", "pytest-xdist", "accelerate", "peft>=0.4.0", "diffusers>=0.18.0"],
    "peft": ["peft>=0.4.0"],
    "diffusers": ["diffusers>=0.18.0"],
    "deepspeed": ["deepspeed>=0.9.5"],
    "dev": ["parameterized", "pytest", "pytest-xdist", "pre-commit", "peft>=0.4.0", "diffusers>=0.18.0"],
    "benchmark": ["wandb", "ghapi", "openrlbenchmark==0.2.1a5", "requests", "deepspeed"],
}

setup(
    name="trl",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    python_requires=">=3.7",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    version=__version__,
    description="A Pytorch implementation of PPO with intrinsic rewards (based on the TRL library).",
    author="Meng Cao",
    author_email="cadenc9020@gmail.com",
)
