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

"""Image datasets."""

from tensorflow_datasets.image.abstract_reasoning import AbstractReasoning
from tensorflow_datasets.image.aflw2k3d import Aflw2k3d
from tensorflow_datasets.image.bigearthnet import Bigearthnet
from tensorflow_datasets.image.binarized_mnist import BinarizedMNIST
from tensorflow_datasets.image.binary_alpha_digits import BinaryAlphaDigits
from tensorflow_datasets.image.caltech import Caltech101
from tensorflow_datasets.image.caltech_birds import CaltechBirds2010
from tensorflow_datasets.image.cars196 import Cars196
from tensorflow_datasets.image.cassava import Cassava
from tensorflow_datasets.image.cats_vs_dogs import CatsVsDogs
from tensorflow_datasets.image.cbis_ddsm import CuratedBreastImagingDDSM
from tensorflow_datasets.image.celeba import CelebA
from tensorflow_datasets.image.celebahq import CelebAHq
from tensorflow_datasets.image.chexpert import Chexpert
from tensorflow_datasets.image.cifar import Cifar10
from tensorflow_datasets.image.cifar import Cifar100
from tensorflow_datasets.image.cifar10_1 import Cifar10_1
from tensorflow_datasets.image.cifar10_corrupted import Cifar10Corrupted
from tensorflow_datasets.image.clevr import CLEVR
from tensorflow_datasets.image.cmaterdb import Cmaterdb
from tensorflow_datasets.image.coco import Coco
from tensorflow_datasets.image.coco2014_legacy import Coco2014  # Deprecated
from tensorflow_datasets.image.coil100 import Coil100
from tensorflow_datasets.image.colorectal_histology import ColorectalHistology
from tensorflow_datasets.image.colorectal_histology import ColorectalHistologyLarge
from tensorflow_datasets.image.cycle_gan import CycleGAN
from tensorflow_datasets.image.deep_weeds import DeepWeeds
from tensorflow_datasets.image.diabetic_retinopathy_detection import DiabeticRetinopathyDetection
from tensorflow_datasets.image.downsampled_imagenet import DownsampledImagenet
from tensorflow_datasets.image.dsprites import Dsprites
from tensorflow_datasets.image.dtd import Dtd
from tensorflow_datasets.image.eurosat import Eurosat
from tensorflow_datasets.image.flowers import TFFlowers
from tensorflow_datasets.image.food101 import Food101
from tensorflow_datasets.image.horses_or_humans import HorsesOrHumans
from tensorflow_datasets.image.image_folder import ImageLabelFolder
from tensorflow_datasets.image.imagenet import Imagenet2012
from tensorflow_datasets.image.imagenet2012_corrupted import Imagenet2012Corrupted
from tensorflow_datasets.image.imagenet_resized import ImagenetResized
from tensorflow_datasets.image.kitti import Kitti
from tensorflow_datasets.image.lfw import LFW
from tensorflow_datasets.image.lsun import Lsun
from tensorflow_datasets.image.malaria import Malaria
from tensorflow_datasets.image.mnist import EMNIST
from tensorflow_datasets.image.mnist import FashionMNIST
from tensorflow_datasets.image.mnist import KMNIST
from tensorflow_datasets.image.mnist import MNIST
from tensorflow_datasets.image.mnist_corrupted import MNISTCorrupted
from tensorflow_datasets.image.omniglot import Omniglot
from tensorflow_datasets.image.open_images import OpenImagesV4
from tensorflow_datasets.image.oxford_flowers102 import OxfordFlowers102
from tensorflow_datasets.image.oxford_iiit_pet import OxfordIIITPet
from tensorflow_datasets.image.patch_camelyon import PatchCamelyon
from tensorflow_datasets.image.pet_finder import PetFinder
from tensorflow_datasets.image.places365_small import Places365Small
from tensorflow_datasets.image.quickdraw import QuickdrawBitmap
from tensorflow_datasets.image.resisc45 import Resisc45
from tensorflow_datasets.image.rock_paper_scissors import RockPaperScissors
from tensorflow_datasets.image.scene_parse_150 import SceneParse150
from tensorflow_datasets.image.shapes3d import Shapes3d
from tensorflow_datasets.image.smallnorb import Smallnorb
from tensorflow_datasets.image.so2sat import So2sat
from tensorflow_datasets.image.stanford_dogs import StanfordDogs
from tensorflow_datasets.image.stanford_online_products import StanfordOnlineProducts
from tensorflow_datasets.image.sun import Sun397
from tensorflow_datasets.image.svhn import SvhnCropped
from tensorflow_datasets.image.svhn_small import SvhnCroppedSmall
from tensorflow_datasets.image.the300w_lp import The300wLp
from tensorflow_datasets.image.uc_merced import UcMerced
from tensorflow_datasets.image.visual_domain_decathlon import VisualDomainDecathlon
from tensorflow_datasets.image.voc import Voc
from tensorflow_datasets.image.wider_face import WiderFace
