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

"""Lazy imports for heavy dependencies.

Based on:
https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/lazy_imports_lib.py
"""

import importlib
import tensorflow_datasets as tfds

utils = tfds.core.utils.py_utils


def _try_import(module_name):
  """Try importing a module, with an informative error message on failure."""
  try:
    mod = importlib.import_module(module_name)
    return mod
  except ImportError as e:
    err_msg = ("Failed importing {name}. This likely means that the dataset "
               "requires additional dependencies that have to be "
               "manually installed").format(name=module_name)
    utils.reraise(e, suffix=err_msg)


class LazyImporter(object):
  """Lazy importer for heavy dependencies.

  Some datasets require heavy dependencies for data generation. To allow for
  the default installation to remain lean, those heavy dependencies are
  lazily imported here.
  """

  @utils.classproperty
  @classmethod
  def cv2(cls):
    return _try_import("cv2")

  @utils.classproperty
  @classmethod
  def PIL_Image(cls):  # pylint: disable=invalid-name
    # TiffImagePlugin need to be activated explicitly on some systems
    # https://github.com/python-pillow/Pillow/blob/5.4.x/src/PIL/Image.py#L407
    _try_import("PIL.TiffImagePlugin")
    return _try_import("PIL.Image")

  @utils.classproperty
  @classmethod
  def pycocotools(cls):
    _try_import("pycocotools.coco")
    return _try_import("pycocotools")

  @utils.classproperty
  @classmethod
  def scipy(cls):
    _try_import("scipy.io")
    _try_import("scipy.ndimage")
    return _try_import("scipy")

lazy_imports = LazyImporter  # pylint: disable=invalid-name
