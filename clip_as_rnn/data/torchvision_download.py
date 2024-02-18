import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import Food101
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import StanfordCars
from torchvision.datasets import VOCSegmentation
from  PIL import Image

# dataset = VOCSegmentation(root="/datasets/jianhaoy/PASCAL", download=True)
dataset = Food101(root="/datasets/jianhaoy/Food101", download=True)