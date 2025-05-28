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

"""Collects all builders for the factors of influence dataset collection."""
import functools
import tensorflow_datasets as tfds

from factors_of_influence.fids import ade20k
from factors_of_influence.fids import berkeley_deep_drive
from factors_of_influence.fids import camvid
from factors_of_influence.fids import cityscapes
from factors_of_influence.fids import coco
from factors_of_influence.fids import indian_driving_dataset
from factors_of_influence.fids import isaid
from factors_of_influence.fids import isprs
from factors_of_influence.fids import kitti
from factors_of_influence.fids import mapillary
from factors_of_influence.fids import pascal
from factors_of_influence.fids import scannet
from factors_of_influence.fids import stanford_dogs
from factors_of_influence.fids import suim
from factors_of_influence.fids import sunrgbd
from factors_of_influence.fids import underwater_trash
from factors_of_influence.fids import vgallery
from factors_of_influence.fids import vkitti2
from factors_of_influence.fids import wilddash

from factors_of_influence.fids import fids_tfds  # pylint: disable=g-bad-import-order


VERSION = tfds.core.Version('5.0.0')
RELEASE_NOTES = {
    '5.0.0': 'Release July 2021 after integrity checking.',
    '5.0.1': 'Release October 2021, adding instance segmentation masks.',
    '5.0.2':
        'Release April 2022, correcting background classes for PContext, PVoc '
        'and IDD.',
}

fids_config = functools.partial(
    fids_tfds.FIDSConfig,
    version=VERSION,
    release_notes=RELEASE_NOTES,
)


class FidsAde20k(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(dataset=ade20k.ADE20k(), version=tfds.core.Version('5.0.1'))
  ]


class FidsBDD(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(
          dataset=berkeley_deep_drive.BDD(berkeley_deep_drive.ALL_SPARSE_107K),
          version=tfds.core.Version('5.0.1')),
      fids_config(
          dataset=berkeley_deep_drive.BDD(berkeley_deep_drive.ALL_DENSE_3K),
          version=tfds.core.Version('5.0.1')),
      fids_config(
          dataset=berkeley_deep_drive.BDD(berkeley_deep_drive.DETECTION_100K),
          version=tfds.core.Version('5.0.1')),
      fids_config(
          dataset=berkeley_deep_drive.BDD(berkeley_deep_drive.MSEG),
          version=tfds.core.Version('5.0.1')),
  ]


class FidsCamvid(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=camvid.CamVid())]


class FidsCityScapes(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=cityscapes.CityScapes)]


class FidsCOCO(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(
          dataset=coco.COCO(coco.ALL), version=tfds.core.Version('5.0.1')),
      fids_config(
          dataset=coco.COCO(coco.MSEG), version=tfds.core.Version('5.0.1')),
      fids_config(
          dataset=coco.COCO(coco.KEYPOINTS),
          version=tfds.core.Version('5.0.1')),
      fids_config(
          dataset=coco.COCO(coco.BOXES), version=tfds.core.Version('5.0.1')),
  ]


class FidsIDD(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(
          dataset=indian_driving_dataset.IDD,
          version=tfds.core.Version('5.0.2'))
  ]


class FidsISAID(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=isaid.ISAID())]


class FidsISPRS(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=isprs.ISPRS())]


class FidsKITTISegmentation(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=kitti.KITTISeg)]


class FidsMapillaryPublic(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=mapillary.MapillaryVistasPublic)]


class FidsPascalContext(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(
          dataset=pascal.PascalContext, version=tfds.core.Version('5.0.2'))
  ]


class FidsPascalVoc2012(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(dataset=pascal.PascalVOC, version=tfds.core.Version('5.0.2'))
  ]


class FidsScanNet(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=scannet.ScanNet20)]


class FidsStanfordDogs(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=stanford_dogs.StanfordDogs())]


class FidsSUIM(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=suim.SUIM())]


class FidsSUNRGBD(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(dataset=sunrgbd.SUNRGBD(sunrgbd.ALL)),
      fids_config(dataset=sunrgbd.SUNRGBD(sunrgbd.MSEG)),
      fids_config(dataset=sunrgbd.SUNRGBD(sunrgbd.DEPTH)),
  ]


class FidsUnderwaterTrash(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=underwater_trash.UnderwaterTrash())]


class FidsVGallery(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=vgallery.VGALLERY(vgallery.ALL))]


class FidsVKitti2(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [
      fids_config(dataset=vkitti2.VKITTI2(vkitti2.ALL)),
      fids_config(dataset=vkitti2.VKITTI2(vkitti2.CLONE)),
  ]


class FidsWildDash(fids_tfds.FIDSTFDS):
  BUILDER_CONFIGS = [fids_config(dataset=wilddash.WildDash19)]
