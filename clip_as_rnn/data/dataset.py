# Adapted from
# https://github.com/yz93/LAVT-RIS/blob/main/data/dataset_refer_bert.py
import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from refer.refer import REFER
import json


class ReferDataset(data.Dataset):
    def __init__(self,
                 root="/homes/55/runjia/storage/refcoco_data",
                 #  root="/datasets/jianhaoy/ReferSegmentation/data",
                 dataset="refcoco",
                 splitBy="google",
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False,
                 prompts_augment=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split
        self.refer = REFER(root, dataset=dataset,  splitBy=splitBy)
        self.prompt_augment = prompts_augment

        if self.prompt_augment:
            self.prompt_path = f'./data/{dataset}_{split}_{splitBy}.json'
            if os.path.exists(self.prompt_path):
                with open(self.prompt_path, 'r') as f:
                    self.prompt_dict_list = json.load(f)
            else:
                raise ValueError(
                    f'Prompt file {self.prompt_path} does not exist')

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
            ref_sentences = []
            for i, (el, sent_id) in enumerate(zip(ref['sentences'],
                                                  ref['sent_ids'])):
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
        img = Image.open(img_path).convert("RGB")
        ref = self.refer.loadRefs(this_ref_id)
        # print("ref",ref)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        target = Image.fromarray(annot.astype(np.uint8), mode="P")
        # print(np.array(target), np.unique(np.array(target).flatten()))
        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img = self.image_transforms(img)
            # target = self.target_transforms(target)
            target = torch.as_tensor(np.array(target, copy=True))
            # target = target.permute((2, 0, 1))
        sentence = self.sentence_raw[index]

        if self.prompt_augment:
            prompt = self.prompt_dict_list[index]['prompt']
            return img, img_path, target, prompt, sentence

        return img, img_path, target, sentence


if __name__ == "__main__":
    def get_transform():
        transform = [transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     # T.Normalize(mean=[0.485, 0.456, 0.406],
                     # std=[0.229, 0.224, 0.225])
                     ]

        return transforms.Compose(transform)

    transform = get_transform()
    dataset_test = ReferDataset(root="/datasets/refseg",
                                #  root="/datasets/jianhaoy/ReferSegmentation/data",
                                dataset="refcoco+",
                                splitBy="google",
                                image_transforms=transform,
                                target_transforms=transform,
                                split='train',
                                eval_mode=False)
    print("loaded")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   sampler=test_sampler,
                                                   num_workers=1)

    for img, target, sentence in data_loader_test:
        # print(type(img),type(target))
        print(sentence)
        break
