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

"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension  # pylint: disable=unused-import
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution


__version__ = '0.0.1'
REQUIRED_PACKAGES = [
    'tensorflow >= 2.3.0',
]
project_name = 'tensorflow-sparse-conv-ops'

from setuptools.command.install import install  # pylint: disable=g-import-not-at-top, g-bad-import-order


class InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False

setup(
    name=project_name,
    version=__version__,
    description=(
        'tensorflow-sparse-conv-ops contains 2d/3d sparse convolution ops for TensorFlow'
    ),
    author='Google Inc.',
    author_email='alirezafathi@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    python_requires='>=3.6, <4',
    platforms=['manylinux2010_x86_64'],
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={'install': InstallPlatlib},
    # PyPI package information.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow sparse convolution machine learning',
)
