# coding=utf-8
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

# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""setup.py for activation_clustering."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="activation_clustering",
    version="0.1",
    packages=find_packages(),
    author="Yu-Han Liu",
    author_email="yuhanliu@google.com",
    install_requires=[
        "scikit-learn==0.19.2",
        "tensorflow-datasets==2.1.0",
        "tensorflow-gpu==2.1.0",
        "matplotlib==2.2.4",
        "scipy==1.2.2",
        "PyYaml==5.3",
        "jupyter==1.0.0",
        "dec_da @ https://github.com/dizcology/DEC-DA/archive/056079d05008da27961ab90cb68c66591ba2187f.zip",
    ],
)
