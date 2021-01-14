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

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to generate .whl."""

import os
import pathlib
import re
import setuptools
from setuptools.command.install import install


class BinaryDistribution(setuptools.Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


class InstallPlatlib(install):
  """Put .so's in platlib, not purelib; see github.com/google/or-tools/issues/616."""

  def finalize_options(self):
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


def get_long_description():
  here = os.path.abspath(os.path.dirname(__file__))
  with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    desc = "\n" + f.read()
  # Convert relative links to absolute.
  return re.sub(
      r"\(docs/(.+)\)",
      r"(https://github.com/google-research/google-research/blob/master/scann/docs/\g<1>)",
      desc)


setuptools.setup(
    name="scann",
    version="1.2.1",
    author="Google Inc.",
    url="https://github.com/google-research/google-research/tree/master/scann",
    author_email="opensource@google.com",
    description="Scalable Approximate Nearest Neighbor search library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=pathlib.Path("requirements.txt").read_text().splitlines(),
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache 2.0",
    keywords="machine learning",
)
