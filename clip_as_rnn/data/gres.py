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

"""grefer v0.1.

This interface provides access to gRefCOCO.

The following API functions are defined:
G_REFER      - REFER api class
getRefIds    - get ref ids that satisfy given filter conditions.
getAnnIds    - get ann ids that satisfy given filter conditions.
getImgIds    - get image ids that satisfy given filter conditions.
getCatIds    - get category ids that satisfy given filter conditions.
loadRefs     - load refs with the specified ref ids.
loadAnns     - load anns with the specified ann ids.
loadImgs     - load images with the specified image ids.
loadCats     - load category names with the specified category ids.
getRefBox    - get ref's bounding box [x, y, w, h] given the ref_id
showRef      - show image, segmentation or box of the referred object with the
               ref
getMaskByRef - get mask and area of the referred object given ref or ref ids
getMask      - get mask and area of the referred object given ref
showMask     - show mask of the referred object given ref
"""
# Adapted from
# https://github.com/yz93/LAVT-RIS/blob/main/data/dataset_refer_bert.py

# pylint: disable=all
import itertools
import json
import os
import os.path as osp
import pickle
import time
# pylint: disable=g-importing-member
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask
from skimage import io
import torch
from torch.utils import data


class G_REFER:
  """GRES dataset."""

  def __init__(self, data_root, dataset='grefcoco', splitBy='unc'):
    # provide data_root folder which contains grefcoco
    print('loading dataset %s into memory...' % dataset)
    self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
    self.DATA_DIR = osp.join(data_root, dataset)
    if dataset in ['grefcoco']:
      self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
    else:
      raise KeyError('No refer dataset is called [%s]' % dataset)

    tic = time.time()

    # load refs from data/dataset/refs(dataset).json
    self.data = {}
    self.data['dataset'] = dataset

    ref_file = osp.join(self.DATA_DIR, f'grefs({splitBy}).p')
    if osp.exists(ref_file):
      self.data['refs'] = pickle.load(open(ref_file, 'rb'), fix_imports=True)
    else:
      ref_file = osp.join(self.DATA_DIR, f'grefs({splitBy}).json')
      if osp.exists(ref_file):
        self.data['refs'] = json.load(open(ref_file, 'rb'))
      else:
        raise FileNotFoundError('JSON file not found')

    # load annotations from data/dataset/instances.json
    instances_file = osp.join(self.DATA_DIR, 'instances.json')
    instances = json.load(open(instances_file, 'r'))
    self.data['images'] = instances['images']
    self.data['annotations'] = instances['annotations']
    self.data['categories'] = instances['categories']

    # create index
    self.createIndex()
    print('DONE (t=%.2fs)' % (time.time() - tic))

  @staticmethod
  def _toList(x):
    return x if isinstance(x, list) else [x]

  @staticmethod
  def match_any(a, b):
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    return set(a) & set(b)

  def createIndex(self):
    # create sets of mapping
    # 1)  Refs: 	 	{ref_id: ref}
    # 2)  Anns: 	 	{ann_id: ann}
    # 3)  Imgs:		 	{image_id: image}
    # 4)  Cats: 	 	{category_id: category_name}
    # 5)  Sents:     	{sent_id: sent}
    # 6)  imgToRefs: 	{image_id: refs}
    # 7)  imgToAnns: 	{image_id: anns}
    # 8)  refToAnn:  	{ref_id: ann}
    # 9)  annToRef:  	{ann_id: ref}
    # 10) catToRefs: 	{category_id: refs}
    # 11) sentToRef: 	{sent_id: ref}
    # 12) sentToTokens: {sent_id: tokens}
    print('creating index...')
    # fetch info from instances
    Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
    Anns[-1] = None
    for ann in self.data['annotations']:
      Anns[ann['id']] = ann
      imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
    for img in self.data['images']:
      Imgs[img['id']] = img
    for cat in self.data['categories']:
      Cats[cat['id']] = cat['name']

    # fetch info from refs
    Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
    Sents, sentToRef, sentToTokens = {}, {}, {}
    availableSplits = []
    for ref in self.data['refs']:
      # ids
      ref_id = ref['ref_id']
      ann_id = ref['ann_id']
      category_id = ref['category_id']
      image_id = ref['image_id']

      if ref['split'] not in availableSplits:
        availableSplits.append(ref['split'])

      # add mapping related to ref
      if ref_id in Refs:
        print('Duplicate ref id')
      Refs[ref_id] = ref
      imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]

      category_id = self._toList(category_id)
      added_cats = []
      for cat in category_id:
        if cat not in added_cats:
          added_cats.append(cat)
          catToRefs[cat] = catToRefs.get(cat, []) + [ref]

      ann_id = self._toList(ann_id)
      refToAnn[ref_id] = [Anns[ann] for ann in ann_id]
      for ann_id_n in ann_id:
        annToRef[ann_id_n] = annToRef.get(ann_id_n, []) + [ref]

      # add mapping of sent
      for sent in ref['sentences']:
        Sents[sent['sent_id']] = sent
        sentToRef[sent['sent_id']] = ref
        sentToTokens[sent['sent_id']] = sent['tokens']

    # create class members
    self.Refs = Refs
    self.Anns = Anns
    self.Imgs = Imgs
    self.Cats = Cats
    self.Sents = Sents
    self.imgToRefs = imgToRefs
    self.imgToAnns = imgToAnns
    self.refToAnn = refToAnn
    self.annToRef = annToRef
    self.catToRefs = catToRefs
    self.sentToRef = sentToRef
    self.sentToTokens = sentToTokens
    self.availableSplits = availableSplits
    print('index created.')

  def getRefIds(self, image_ids=[], cat_ids=[], split=[]):
    image_ids = self._toList(image_ids)
    cat_ids = self._toList(cat_ids)
    split = self._toList(split)

    for s in split:
      if s not in self.availableSplits:
        raise ValueError(f'Invalid split name: {s}')

    refs = self.data['refs']

    if len(image_ids) > 0:
      lists = [self.imgToRefs[image_id] for image_id in image_ids]
      refs = list(itertools.chain.from_iterable(lists))
    if len(cat_ids) > 0:
      refs = [
          ref for ref in refs if self.match_any(ref['category_id'], cat_ids)
      ]
    if len(split) > 0:
      refs = [ref for ref in refs if ref['split'] in split]

    ref_ids = [ref['ref_id'] for ref in refs]
    return ref_ids

  def getAnnIds(self, image_ids=[], ref_ids=[]):
    image_ids = self._toList(image_ids)
    ref_ids = self._toList(ref_ids)

    if any([len(image_ids), len(ref_ids)]):
      if len(image_ids) > 0:
        lists = [
            self.imgToAnns[image_id]
            for image_id in image_ids
            if image_id in self.imgToAnns
        ]
        anns = list(itertools.chain.from_iterable(lists))
      else:
        anns = self.data['annotations']
      ann_ids = [ann['id'] for ann in anns]
      if len(ref_ids) > 0:
        lists = [self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]
        anns_by_ref_id = list(itertools.chain.from_iterable(lists))
        ann_ids = list(set(ann_ids).intersection(set(anns_by_ref_id)))
    else:
      ann_ids = [ann['id'] for ann in self.data['annotations']]

    return ann_ids

  def getImgIds(self, ref_ids=[]):
    ref_ids = self._toList(ref_ids)

    if len(ref_ids) > 0:
      image_ids = list(
          set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids])
      )
    else:
      image_ids = self.Imgs.keys()
    return image_ids

  def getCatIds(self):
    return self.Cats.keys()

  def loadRefs(self, ref_ids=[]):
    return [self.Refs[ref_id] for ref_id in self._toList(ref_ids)]

  def loadAnns(self, ann_ids=[]):
    if isinstance(ann_ids, str):
      ann_ids = int(ann_ids)
    return [self.Anns[ann_id] for ann_id in self._toList(ann_ids)]

  def loadImgs(self, image_ids=[]):
    return [self.Imgs[image_id] for image_id in self._toList(image_ids)]

  def loadCats(self, cat_ids=[]):
    return [self.Cats[cat_id] for cat_id in self._toList(cat_ids)]

  def getRefBox(self, ref_id):
    anns = self.refToAnn[ref_id]
    return [ann['bbox'] for ann in anns]  # [x, y, w, h]

  def showRef(self, ref, seg_box='seg'):
    ax = plt.gca()
    # show image
    image = self.Imgs[ref['image_id']]
    I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
    ax.imshow(I)
    # show refer expression
    for sid, sent in enumerate(ref['sentences']):
      print('%s. %s' % (sid + 1, sent['sent']))
    # show segmentations
    if seg_box == 'seg':
      ann_id = ref['ann_id']
      ann = self.Anns[ann_id]
      polygons = []
      color = []
      c = 'none'
      if type(ann['segmentation'][0]) == list:
        # polygon used for refcoco*
        for seg in ann['segmentation']:
          poly = np.array(seg).reshape((len(seg) / 2, 2))
          polygons.append(Polygon(poly, True, alpha=0.4))
          color.append(c)
        p = PatchCollection(
            polygons,
            facecolors=color,
            edgecolors=(1, 1, 0, 0),
            linewidths=3,
            alpha=1,
        )
        ax.add_collection(p)  # thick yellow polygon
        p = PatchCollection(
            polygons,
            facecolors=color,
            edgecolors=(1, 0, 0, 0),
            linewidths=1,
            alpha=1,
        )
        ax.add_collection(p)  # thin red polygon
      else:
        # mask used for refclef
        rle = ann['segmentation']
        m = mask.decode(rle)
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.array([2.0, 166.0, 101.0]) / 255
        for i in range(3):
          img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.5)))
    # show bounding-box
    elif seg_box == 'box':
      # ann_id = ref['ann_id']
      # ann = self.Anns[ann_id]
      bbox = self.getRefBox(ref['ref_id'])
      box_plot = Rectangle(
          (bbox[0], bbox[1]),
          bbox[2],
          bbox[3],
          fill=False,
          edgecolor='green',
          linewidth=3,
      )
      ax.add_patch(box_plot)

  def getMask(self, ann):
    if not ann:
      return None
    if ann['iscrowd']:
      raise ValueError('Crowd object')
    image = self.Imgs[ann['image_id']]
    if type(ann['segmentation'][0]) == list:  # polygon
      rle = mask.frPyObjects(
          ann['segmentation'], image['height'], image['width']
      )
    else:
      rle = ann['segmentation']

    m = mask.decode(rle)
    # sometimes there are multiple binary map (corresponding to multiple segs)
    m = np.sum(m, axis=2)
    m = m.astype(np.uint8)  # convert to np.uint8
    # compute area
    area = sum(mask.area(rle))  # should be close to ann['area']
    return {'mask': m, 'area': area}

  def getMaskByRef(self, ref=None, ref_id=None, merge=False):
    if not ref and not ref_id:
      raise ValueError
    if ref:
      ann_ids = ref['ann_id']
      ref_id = ref['ref_id']
    else:
      ann_ids = self.getAnnIds(ref_ids=ref_id)

    if ann_ids == [-1]:
      img = self.Imgs[self.Refs[ref_id]['image_id']]
      return {
          'mask': np.zeros([img['height'], img['width']], dtype=np.uint8),
          'empty': True,
      }

    anns = self.loadAnns(ann_ids)
    mask_list = [self.getMask(ann) for ann in anns if not ann['iscrowd']]

    if merge:
      merged_masks = sum([mask['mask'] for mask in mask_list])
      merged_masks[np.where(merged_masks > 1)] = 1
      return {'mask': merged_masks, 'empty': False}
    else:
      return mask_list

  def showMask(self, ref):
    M = self.getMask(ref)
    msk = M['mask']
    ax = plt.gca()
    ax.imshow(msk)


class GReferDataset(data.Dataset):

  def __init__(self, root, transform=None, split='val'):

    self.classes = []
    self.image_transforms = transform
    self.split = split
    self.refer = G_REFER(root)

    ref_ids = self.refer.getRefIds(split=self.split)
    img_ids = self.refer.getImgIds(ref_ids)

    all_imgs = self.refer.Imgs
    self.imgs = list(all_imgs[i] for i in img_ids)
    self.ref_ids = []
    # print(len(ref_ids))
    # print(len(self.imgs))
    self.sentence_raw = []
    # if we are testing on a dataset, test all sentences of an object;
    # o/w, we are validating during training, randomly sample one sentence
    # for efficiency
    for r in ref_ids:
      ref = self.refer.Refs[r]
      # ref_sentences = []
      # for i, (el, sent_id) in enumerate(zip(ref['sentences'],
      #                                       ref['sent_ids'])):
      for el in ref['sentences']:
        sentence_raw = el['raw']
        if len(sentence_raw) == 0:
          continue
        self.sentence_raw.append(sentence_raw)
        self.ref_ids.append(r)

    # print(len(self.sentence_raw))

  def get_classes(self):
    return self.classes

  def __len__(self):
    return len(self.ref_ids)

  def __getitem__(self, index):
    this_ref_id = self.ref_ids[index]
    this_img_id = self.refer.getImgIds(this_ref_id)
    this_img = self.refer.Imgs[this_img_id[0]]
    # print(this_ref_id, this_img_id)
    # print(len(self.ref_ids))
    img_path = os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])
    img = Image.open(img_path).convert('RGB')
    ref = self.refer.loadRefs(this_ref_id)
    # print("ref",ref)

    ref_mask_ann = self.refer.getMaskByRef(ref[0])
    if type(ref_mask_ann) == list:
      ref_mask_ann = ref_mask_ann[0]
    ref_mask = ref_mask_ann['mask']
    annot = np.zeros(ref_mask.shape)
    annot[ref_mask == 1] = 1

    target = Image.fromarray(annot.astype(np.uint8), mode='P')
    # print(np.array(target), np.unique(np.array(target).flatten()))
    if self.image_transforms is not None:
      # resize, from PIL to tensor, and mean and std normalization
      img = self.image_transforms(img)
      # target = self.target_transforms(target)
      target = torch.as_tensor(np.array(target, copy=True))
      # target = target.permute((2, 0, 1))
    sentence = self.sentence_raw[index]

    return img, img_path, target, sentence
