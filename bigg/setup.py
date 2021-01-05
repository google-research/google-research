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

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
# pylint: skip-file

from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop

import os
import subprocess
import platform
BASEPATH = os.path.dirname(os.path.abspath(__file__))


class custom_develop(develop):
    def run(self):
        original_cwd = os.getcwd()

        folders = [
            os.path.join(BASEPATH, 'bigg/model/tree_clib'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

        os.chdir(original_cwd)

        super().run()


setup(name='bigg',
    py_modules=['bigg'],
    install_requires=[
        'torch',
    ],
    cmdclass={
        'develop': custom_develop,
    }
)
