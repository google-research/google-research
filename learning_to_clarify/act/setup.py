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

import os

from setuptools import find_packages, setup


__version__ = "0.1.0.dev0"

REQUIRED_PKGS = [line.strip() for line in open('requirements.txt')]


setup(
    name="act",
    include_package_data=True,
    #package_data={"act": ["commands/scripts/config/*", "commands/scripts/*"]},
    packages=find_packages("src", exclude={"tests"}),
    package_dir={"": "src"},
    install_requires=REQUIRED_PKGS,
    python_requires=">=3.10",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    version=__version__,
    description="ACT Algorithm Repository.",
    keywords="dpo, transformers, huggingface, gemma2, language modeling, rlhf",
)
