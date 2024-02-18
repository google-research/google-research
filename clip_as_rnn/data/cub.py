from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class CUB200Dataset(Dataset):
    def __init__(self, root='/scratch/network/ssd4/jianhaoy/CUB2002011/CUB_200_2011', split="test", img_transform=None, target_transform=None):
        '''
        从文件中读取图像，数据
        '''
        self.root = root  # 数据集路径
        # self.image_size = image_size  # 图像大小(正方形)
        self.transform = img_transform  # 图像的 transform 
        self.target_transform = target_transform  # 标签的 transform 

        # 构造数据集参数的各文件路径
        self.classes_file = os.path.join(root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(root, 'train_test_split.txt')  # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>
        
        self.id2class = self.create_bird_dictionary(self.classes_file)
        
        self.image_dir = os.path.join(root, 'images')  # 图像文件夹路径
        self.mask_dir = os.path.join(root, 'segmentations')  # mask 文件夹路径

        imgs_name_train, imgs_name_test, imgs_label_train, imgs_label_test, imgs_bbox_train, imgs_bbox_test = self._get_img_attributes()
        
        if split == 'train':
            # self.data = self._get_imgs(imgs_name_train, imgs_bbox_train)
            self.data = imgs_name_train
            self.label = imgs_label_train
        elif split == 'test':
            # self.data = self._get_imgs(imgs_name_test, imgs_bbox_test)
            self.data = imgs_name_test
            self.label = imgs_label_test

    def _get_img_id(self):
        ''' 读取张图片的 id，并根据 id 划分为测试集和训练集 '''
        imgs_id_train, imgs_id_test = [], []
        file = open(self.train_test_split_file, "r")
        for line in file:
            img_id, is_train = line.split()
            if is_train == "1":
                imgs_id_train.append(img_id)
            elif is_train == "0":
                imgs_id_test.append(img_id)
        file.close()
        return imgs_id_train, imgs_id_test

    def _get_img_class(self):
        ''' 读取每张图片的 class 类别 '''
        imgs_class = []
        file = open(self.image_class_labels_file, 'r')
        for line in file:
            _, img_class = line.split()
            imgs_class.append(img_class)
        file.close()
        return imgs_class

    def _get_bondingbox(self):
        ''' 获取图像边框 '''
        bondingbox = []
        file = open(self.bounding_boxes_file)
        for line in file:
            _, x, y, w, h = line.split()
            x, y, w, h = float(x), float(y), float(w), float(h)
            bondingbox.append((x, y, x+w, y+h))
        file.close()
        return bondingbox

    def _get_img_attributes(self):
        ''' 根据图片 id 读取每张图片的属性，包括名字(路径)、类别和边框，并分别按照训练集和测试集划分 '''
        imgs_name_train, imgs_name_test, imgs_label_train, imgs_label_test, imgs_bbox_train, imgs_bbox_test = [], [], [], [], [], []
        imgs_id_train, imgs_id_test = self._get_img_id()  # 获取训练集和测试集的 img_id
        imgs_bbox = self._get_bondingbox()  # 获取所有图像的 bondingbox
        imgs_class = self._get_img_class()  # 获取所有图像类别标签，按照 img_id 存储
        file = open(self.images_file)
        for line in file:
            img_id, img_name = line.split()
            if img_id in imgs_id_train:
                img_id = int(img_id)
                imgs_name_train.append(img_name)
                imgs_label_train.append(imgs_class[img_id-1]) # 下标从 0 开始
                imgs_bbox_train.append(imgs_bbox[img_id-1])
            elif img_id in imgs_id_test:
                img_id = int(img_id)
                imgs_name_test.append(img_name)
                imgs_label_test.append(imgs_class[img_id-1])
                imgs_bbox_test.append(imgs_bbox[img_id-1])
        file.close()
        return imgs_name_train, imgs_name_test, imgs_label_train, imgs_label_test, imgs_bbox_train, imgs_bbox_test

    def _get_imgs(self, imgs_name, imgs_bbox):
        ''' 遍历每一张图片的路径，读取图片信息 '''
        data = []
        for i in range(len(imgs_name)):
            img_path = os.path.join(self.root, 'images', imgs_name[i])
            img = self._convert_and_resize(img_path, imgs_bbox[i])
            data.append(img)
        return data

    def _convert_and_resize(self, img_path, img_bbox):
        ''' 将不是 'RGB' 模式的图像变为 'RGB' 格式，更改图像大小 '''
        img = Image.open(img_path).resize((self.image_size, self.image_size), box=img_bbox)
        img.show()
        if img.mode == 'L':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # print(type(img))
        return img

    def __getitem__(self, index):
        image_name = self.data[index]
        label = self.label[index]
        class_name = self.id2class[int(label)]
        image = Image.open(os.path.join(self.image_dir,image_name)).convert("RGB")
        mask_name = self.change_extension(image_name)
        target = self.process_target(Image.open(os.path.join(self.mask_dir,mask_name)))
        # print(image_name, class_name)
        # print(np.array(target).shape, np.unique(np.array(target).flatten()))
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(target)
        
        return image, os.path.join(self.image_dir,image_name), label, class_name

    def __len__(self):
        return len(self.data)

    def create_bird_dictionary(self,filename):
        # Initialize an empty dictionary
        bird_dict = {}

        # Open the file and read each line
        with open(filename, 'r') as file:
            for line in file:
                # Split the line into key and value components
                split_line = line.strip().split(' ')

                # The key is the first part, convert to int
                key = int(split_line[0])

                # The value is the second part, split by '.' and take the second half
                raw_value = split_line[1].split('.')[1]

                # Replace underscores in the value with spaces
                value = raw_value.replace('_', ' ')

                # Add the key-value pair to the dictionary
                bird_dict[key] = value

        return bird_dict
    
    def change_extension(self, filename, new_extension='.png'):
        base_name = os.path.splitext(filename)[0]
        return base_name + new_extension
    
    def process_target(self,image):
        # TODO: figure out ambuguous labels
        # Convert PIL image to NumPy array
        arr = np.array(image)
        # Create a new array where only 1s are preserved
        new_array = np.where(arr != 0, 1, arr)
        # Convert the NumPy array back to PIL image
        new_image = Image.fromarray(new_array.astype(np.uint8)*255)

        return new_image
            
if __name__ == "__main__":
    train_set = CUB200("./CUB_200_2011", train=True)  # 共 5994 张图片
    test_set = CUB200("./CUB_200_2011", train=False)  # 共 5794 张图片

