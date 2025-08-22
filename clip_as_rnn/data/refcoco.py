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

"""RefCOCO dataset."""

# Adapted from
# https://github.com/yz93/LAVT-RIS/blob/main/data/dataset_refer_bert.py
# pylint: disable=all
import itertools
import json
import os
import os.path as osp
import pickle as pickle
import sys
import time
# pylint: disable=g-importing-member
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask
import skimage.io as io
import torch
import torch.utils.data as data
from torchvision import transforms


class REFER:
  """RefCOCO dataset."""

  def __init__(self, data_root, dataset='refcoco', splitBy='unc', split='val'):
    # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
    # also provide dataset name and splitBy information
    # e.g., dataset = 'refcoco', splitBy = 'unc'
    print('loading dataset %s into memory...' % dataset)
    if dataset == 'refcocog':
      print('Split by {}!'.format(splitBy))
    self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
    self.DATA_DIR = osp.join(data_root, dataset)
    if dataset in ['refcoco', 'refcoco+', 'refcocog']:
      self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
    elif dataset == 'refclef':
      self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
    else:
      print('No refer dataset is called [%s]' % dataset)
      sys.exit()

    # load refs from data/dataset/refs(dataset).json
    tic = time.time()
    ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
    self.data = {}
    self.data['dataset'] = dataset
    # f = open(ref_file, 'r')
    self.data['refs'] = pickle.load(open(ref_file, 'rb'))

    # load annotations from data/dataset/instances.json
    instances_file = osp.join(self.DATA_DIR, 'instances.json')
    instances = json.load(open(instances_file, 'r'))
    self.data['images'] = instances['images']
    self.data['annotations'] = instances['annotations']
    self.data['categories'] = instances['categories']

    # create index
    self.createIndex()
    self.split = split
    print('DONE (t=%.2fs)' % (time.time() - tic))

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
    for ref in self.data['refs']:
      # ids
      ref_id = ref['ref_id']
      ann_id = ref['ann_id']
      category_id = ref['category_id']
      image_id = ref['image_id']

      # add mapping related to ref
      Refs[ref_id] = ref
      imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
      catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
      refToAnn[ref_id] = Anns[ann_id]
      annToRef[ann_id] = ref

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
    print('index created.')

  def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
    image_ids = image_ids if type(image_ids) == list else [image_ids]
    cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
    ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

    if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
      refs = self.data['refs']
    else:
      if not len(image_ids) == 0:
        refs = [self.imgToRefs[image_id] for image_id in image_ids]
        ref_ids = []
        for img_ref in refs:
          ref_ids.extend([ref['ref_id'] for ref in img_ref])
        return ref_ids
      else:
        refs = self.data['refs']
      if not len(cat_ids) == 0:
        refs = [ref for ref in refs if ref['category_id'] in cat_ids]
      if not len(ref_ids) == 0:
        refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
      if not len(split) == 0:
        if split in ['testA', 'testB', 'testC']:
          # we also consider testAB, testBC, ...
          refs = [ref for ref in refs if split[-1] in ref['split']]
        elif split in ['testAB', 'testBC', 'testAC']:
          # rarely used I guess...
          refs = [ref for ref in refs if ref['split'] == split]
        elif split == 'test':
          refs = [ref for ref in refs if 'test' in ref['split']]
        elif split == 'train' or split == 'val':
          refs = [ref for ref in refs if ref['split'] == split]
        else:
          print('No such split [%s]' % split)
          sys.exit()
    ref_ids = [ref['ref_id'] for ref in refs]
    return ref_ids

  def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
    image_ids = image_ids if type(image_ids) == list else [image_ids]
    cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
    ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

    if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
      ann_ids = [ann['id'] for ann in self.data['annotations']]
    else:
      if not len(image_ids) == 0:
        lists = [
            self.imgToAnns[image_id]
            for image_id in image_ids
            if image_id in self.imgToAnns
        ]  # list of [anns]
        anns = list(itertools.chain.from_iterable(lists))
      else:
        anns = self.data['annotations']
      if not len(cat_ids) == 0:
        anns = [ann for ann in anns if ann['category_id'] in cat_ids]
      ann_ids = [ann['id'] for ann in anns]
      # if not len(ref_ids) == 0:
      #   ids = set(ann_ids).intersection(
      #       set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids])
      #   )
    return ann_ids

  def getImgIds(self, ref_ids=[]):
    ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

    if not len(ref_ids) == 0:
      image_ids = list(
          set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids])
      )
    else:
      image_ids = self.Imgs.keys()
    return image_ids

  def getCatIds(self):
    return self.Cats.keys()

  def loadRefs(self, ref_ids=[]):
    if type(ref_ids) == list:
      return [self.Refs[ref_id] for ref_id in ref_ids]
    elif type(ref_ids) == int:
      return [self.Refs[ref_ids]]

  def loadAnns(self, ann_ids=[]):
    if type(ann_ids) == list:
      return [self.Anns[ann_id] for ann_id in ann_ids]
    elif type(ann_ids) == int or type(ann_ids) == unicode:
      return [self.Anns[ann_ids]]

  def loadImgs(self, image_ids=[]):
    if type(image_ids) == list:
      return [self.Imgs[image_id] for image_id in image_ids]
    elif type(image_ids) == int:
      return [self.Imgs[image_ids]]

  def loadCats(self, cat_ids=[]):
    if type(cat_ids) == list:
      return [self.Cats[cat_id] for cat_id in cat_ids]
    elif type(cat_ids) == int:
      return [self.Cats[cat_ids]]

  def getRefBox(self, ref_id):
    # ref = self.Refs[ref_id]
    ann = self.refToAnn[ref_id]
    return ann['bbox']  # [x, y, w, h]

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

  def getMask(self, ref):
    # return mask, area and mask-center
    ann = self.refToAnn[ref['ref_id']]
    image = self.Imgs[ref['image_id']]

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

  def showMask(self, ref):
    M = self.getMask(ref)
    msk = M['mask']
    ax = plt.gca()
    ax.imshow(msk)


class ReferDataset(data.Dataset):

  def __init__(
      self,
      root,
      dataset='refcoco',
      splitBy='google',
      image_transforms=None,
      target_transforms=None,
      split='train',
      eval_mode=False,
  ):

    self.classes = []
    self.image_transforms = image_transforms
    self.target_transforms = target_transforms
    self.split = split
    self.refer = REFER(root, dataset=dataset, splitBy=splitBy)

    ref_ids = self.refer.getRefIds(split=self.split)
    img_ids = self.refer.getImgIds(ref_ids)

    all_imgs = self.refer.Imgs
    self.imgs = list(all_imgs[i] for i in img_ids)
    self.ref_ids = ref_ids
    # print(len(ref_ids))
    # print(len(self.imgs))
    self.sentence_raw = []

    self.eval_mode = eval_mode
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
        ref_sentences.append(sentence_raw)
      self.sentence_raw.append(ref_sentences)
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

    ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
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


if __name__ == '__main__':

  def get_transform():
    transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])
    ]

    return transforms.Compose(transform)

  transform = get_transform()
  dataset_test = ReferDataset(
      root='/datasets/refseg',
      dataset='refcoco+',
      splitBy='google',
      image_transforms=transform,
      target_transforms=transform,
      split='train',
      eval_mode=False,
  )
  print('loaded')
  test_sampler = torch.utils.data.SequentialSampler(dataset_test)
  data_loader_test = torch.utils.data.DataLoader(
      dataset_test, batch_size=1, sampler=test_sampler, num_workers=1
  )

  for img, target, sentence in data_loader_test:
    # print(type(img),type(target))
    print(sentence)
    break
