import os
import torch
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np


# COCO_STUFF_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#                'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
#                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_STUFF_CLASSES  = ['person with clothes,people,human','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird avian',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack,bag',
                    'umbrella,parasol','handbag,purse','necktie','suitcase','frisbee',
                    'skis','sknowboard','sports ball','kite','baseball bat',
                    'glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','dessertspoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair seat','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor screen','laptop','mouse',
                    'remote control','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hairdrier,blowdrier','toothbrush',
                    ]




class COCOStuffDataset(torch.utils.data.Dataset):
    def __init__(self, root='/datasets/jianhaoy/COCO/', split='val', transform=None):
        """
        Args:
            root (string): Root directory where images are downloaded.
            annFile (string): Path to the annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.image_dir = os.path.join(root, 'images', f'{split}2017')
        self.ann_dir = os.path.join(root, 'annotations', f'{split}2017')
        self.images = os.listdir(self.image_dir)
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img)
        idx = self.images[index].split('.')[0]
        ann_path = os.path.join(self.ann_dir, f'{idx}_instanceTrainIds.png')
        ann = np.asarray(Image.open(ann_path), dtype=np.int32)
        
        return img, img_path, ann, idx
        

    def __len__(self):
        return len(self.images)