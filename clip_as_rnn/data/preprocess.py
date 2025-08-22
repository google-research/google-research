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

"""Preprocess for referring datasets.

Adapted from
https://github.com/yz93/LAVT-RIS/blob/main/data/dataset_refer_bert.py
"""
# pylint: disable=all
from refer.refer import REFER
from torch.utils import data


class ReferDataset(data.Dataset):
  """Refer dataset."""

  def __init__(
      self,
      root,
      dataset='refcoco',
      splitBy='unc',
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
    print(len(ref_ids))
    print(len(self.imgs))
    # print(self.imgs)
    self.sentence_raw = []

    self.eval_mode = eval_mode
    # if we are testing on a dataset, test all sentences of an object;
    # o/w, we are validating during training, randomly sample one sentence for
    # efficiency
    for r in ref_ids:
      ref = self.refer.Refs[r]
      ref_sentences = []
      for el, _ in zip(ref['sentences'], ref['sent_ids']):
        sentence_raw = el['raw']
        ref_sentences.append(sentence_raw)

      self.sentence_raw.append(ref_sentences)
    # print(len(self.sentence_raw))

  def get_classes(self):
    return self.classes

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, index):
    this_img_id = self.imgs[index]['id']
    this_ref_ids = self.refer.getRefIds(this_img_id)
    this_img = self.refer.Imgs[this_img_id]
    refs = [self.refer.loadRefs(this_ref_id) for this_ref_id in this_ref_ids]

    batch_sentences = {}
    # batch_targets = {}
    for ref in refs:
      # Get sentence
      sentence_lis = []
      for el, _ in zip(ref[0]['sentences'], ref[0]['sent_ids']):
        sentence_raw = el['raw']
        sentence_lis.append(sentence_raw)
      batch_sentences.update({ref[0]['ref_id']: sentence_lis})

    return [this_img['file_name']], batch_sentences

  def get_ref(self):
    name_lis = []
    for i in range(len(self.ref_ids)):
      rid = self.ref_ids[i]
      # print(rid)
      ref = self.refer.loadRefs(rid)
      if ref[0]['file_name'] == '':
        print(1)
      # print(ref[0]['file_name'])
      # if ref[0]['file_name'] in name_lis:
      #     print("md")
      name_lis.append(ref[0]['file_name'])
      print(ref[0]['file_name'])
    # print(name_lis)
    print(len(name_lis))
    print(len(list(set(name_lis))))
