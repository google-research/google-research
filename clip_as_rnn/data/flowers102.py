import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from  PIL import Image
from scipy.io import loadmat

ID2CLASS={'21': 'fire lily',
 '3': 'canterbury bells',
 '45': 'bolero deep blue',
 '1': 'pink primrose',
 '34': 'mexican aster',
 '27': 'prince of wales feathers',
 '7': 'moon orchid',
 '16': 'globe-flower',
 '25': 'grape hyacinth',
 '26': 'corn poppy',
 '79': 'toad lily',
 '39': 'siam tulip',
 '24': 'red ginger',
 '67': 'spring crocus',
 '35': 'alpine sea holly',
 '32': 'garden phlox',
 '10': 'globe thistle',
 '6': 'tiger lily',
 '93': 'ball moss',
 '33': 'love in the mist',
 '9': 'monkshood',
 '102': 'blackberry lily',
 '14': 'spear thistle',
 '19': 'balloon flower',
 '100': 'blanket flower',
 '13': 'king protea',
 '49': 'oxeye daisy',
 '15': 'yellow iris',
 '61': 'cautleya spicata',
 '31': 'carnation',
 '64': 'silverbush',
 '68': 'bearded iris',
 '63': 'black-eyed susan',
 '69': 'windflower',
 '62': 'japanese anemone',
 '20': 'giant white arum lily',
 '38': 'great masterwort',
 '4': 'sweet pea',
 '86': 'tree mallow',
 '101': 'trumpet creeper',
 '42': 'daffodil',
 '22': 'pincushion flower',
 '2': 'hard-leaved pocket orchid',
 '54': 'sunflower',
 '66': 'osteospermum',
 '70': 'tree poppy',
 '85': 'desert-rose',
 '99': 'bromelia',
 '87': 'magnolia',
 '5': 'english marigold',
 '92': 'bee balm',
 '28': 'stemless gentian',
 '97': 'mallow',
 '57': 'gaura',
 '40': 'lenten rose',
 '47': 'marigold',
 '59': 'orange dahlia',
 '48': 'buttercup',
 '55': 'pelargonium',
 '36': 'ruby-lipped cattleya',
 '91': 'hippeastrum',
 '29': 'artichoke',
 '71': 'gazania',
 '90': 'canna lily',
 '18': 'peruvian lily',
 '98': 'mexican petunia',
 '8': 'bird of paradise',
 '30': 'sweet william',
 '17': 'purple coneflower',
 '52': 'wild pansy',
 '84': 'columbine',
 '12': "colt's foot",
 '11': 'snapdragon',
 '96': 'camellia',
 '23': 'fritillary',
 '50': 'common dandelion',
 '44': 'poinsettia',
 '53': 'primula',
 '72': 'azalea',
 '65': 'californian poppy',
 '80': 'anthurium',
 '76': 'morning glory',
 '37': 'cape flower',
 '56': 'bishop of llandaff',
 '60': 'pink-yellow dahlia',
 '82': 'clematis',
 '58': 'geranium',
 '75': 'thorn apple',
 '41': 'barbeton daisy',
 '95': 'bougainvillea',
 '43': 'sword lily',
 '83': 'hibiscus',
 '78': 'lotus lotus',
 '88': 'cyclamen',
 '94': 'foxglove',
 '81': 'frangipani',
 '74': 'rose',
 '89': 'watercress',
 '73': 'water lily',
 '46': 'wallflower',
 '77': 'passion flower',
 '51': 'petunia'}

class Flowers102Dataset(Flowers102):
    def __init__(self, root='/scratch/network/ssd4/jianhaoy/Flowers102/', split="test", download=False, img_transform=None, target_transform=None):
        super(Flowers102Dataset, self).__init__(root=root, split=split, transform=img_transform, download=download, target_transform=target_transform)
        self.mask_dir = self._base_folder / "segmim"
        # self.target_transform = transform
        self.idx_to_class = ID2CLASS
        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()
        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._mask_files = []
        for image_id in image_ids:
            self._mask_files.append(self.mask_dir / f"segmim_{image_id:05d}.jpg")
        
    def __getitem__(self, idx):
        image_file, class_id = self._image_files[idx], self.idx_to_class[str(self._labels[idx]+1)]
        image = Image.open(image_file).convert("RGB")
        label = Image.open(self._mask_files[idx])
        label = self.process_target(label)
        # print(image_file, class_id)
        # print(np.array(label), np.unique(np.array(label).flatten()))
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, str(image_file), label, class_id
        
    def process_target(self,image):
        # Convert PIL image to NumPy array
        arr = np.array(image)
        # Create a new array where only 1s are preserved
        new_array = np.where(arr != 0, 1, arr)
        # Convert the NumPy array back to PIL image
        new_image = Image.fromarray(new_array.astype(np.uint8))

        return new_image
            