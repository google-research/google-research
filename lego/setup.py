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

# pylint: skip-file
from setuptools import setup
from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

import os
import subprocess
BASEPATH = os.path.dirname(os.path.abspath(__file__))

extlib_path = 'lego/ext_ops'
compile_args = ['-Wno-deprecated-declarations']
link_args = []
ext_modules = []

class custom_develop(develop):
    def run(self):
        original_cwd = os.getcwd()

        # build custom ops
        folders = [
           os.path.join(BASEPATH, 'lego/cpp_sampler'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

        os.chdir(original_cwd)

        super().run()


setup(name='lego',
      py_modules=['lego'],
      ext_modules=ext_modules,
      install_requires=[
          'torch>=1.7.0',
          'scikit-learn',
          'tensorboardX',
          'gym'
      ],
      cmdclass={
          'build_ext': BuildExtension,
          'develop': custom_develop,
        }
)
