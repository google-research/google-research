import os
import torch
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from PIL import Image

class COCOInstanceSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root='/datasets/jianhaoy/COCO/', annFile='/COCO/annotations/instances_val2017.json', transform=None):
        """
        Args:
            root (string): Root directory where images are downloaded.
            annFile (string): Path to the annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        masks = []
        for i in range(len(target)):
            mask = coco.annToMask(target[i])
            masks.append(mask)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        return img, masks

    def __len__(self):
        return len(self.ids)