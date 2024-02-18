# Adapted from https://github.com/yz93/LAVT-RIS/blob/main/data/dataset_refer_bert.py
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from refer.refer import REFER

class ReferDataset(data.Dataset):

    def __init__(self,
                 root="/datasets/jianhaoy/ReferSegmentation/data",
                 dataset="refcoco",
                 splitBy="unc",
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.split = split
        self.refer = REFER(root, dataset=dataset,  splitBy=splitBy)

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
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]
            ref_sentences = []
            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
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
        
        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        refs = [self.refer.loadRefs(this_ref_id) for this_ref_id in this_ref_ids]
        if self.image_transforms is not None:
            img = self.image_transforms(img)
    
        batch_sentences = {}
        batch_targets = {}
        for ref in refs:
            # Get mask
            ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1
            target = Image.fromarray(annot.astype(np.uint8), mode="P")
            if self.target_transforms is not None:
                target = self.target_transforms(target)
            batch_targets.update({ref[0]['ref_id']:target})
            # Get sentence
            sentence_lis = []
            for i, (el, sent_id) in enumerate(zip(ref[0]['sentences'], ref[0]['sent_ids'])):
                sentence_raw = el['raw']
                sentence_lis.append(sentence_raw)
            batch_sentences.update({ref[0]['ref_id']:sentence_lis})
            
        return img, [this_img['file_name']], batch_targets, batch_sentences
    
            
            


if __name__ == "__main__":
    def get_transform():
        transform = [transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]

        return transforms.Compose(transform)
    
    transform = get_transform()
    dataset_test = ReferDataset(root="/datasets/jianhaoy/ReferSegmentation/data",
                 dataset="refcoco",
                 splitBy="google",
                 image_transforms=transform,
                 target_transforms=transform,
                 split='train',
                 eval_mode=False)
    # dataset_test.get_ref()
    print("loaded")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=1)
    
    for img,target,sentence in data_loader_test:
        print(type(img),type(target),type(sentence))
        print(sentence)
        break
        