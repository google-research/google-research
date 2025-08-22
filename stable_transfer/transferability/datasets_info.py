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

"""Contains (static) information about datasets.

1. Contains which datasets use a validation set instead of using a separate
    test set
2. Contains number of classes of classification datasets for training.
3. Contains preprocessing info for semantic segmentation datasets.
"""


DATASETS_WITH_VALIDATION_AS_TEST = [
    'food101', 'imagenette', 'ade20k', 'bdd', 'camvid', 'city', 'coco', 'idd',
    'isaid', 'isprs', 'kitti', 'mapillary', 'msegpcontext', 'pvoc', 'scannet',
    'sunrgbd',]


DATASET_NUM_CLASSES = {
    'oxford_flowers102': 102,
    'sun397': 397,
    'stanford_dogs': 120,
    'oxford_iiit_pet': 37,
    'caltech_birds2011': 200,
    'dtd': 47,
    'cifar10': 10,
    'imagenette': 10,
    'cifar100': 100,
}


SEGMENTATION_DATASET_INFO = {
    'ade20k': {
        'mean': [0.48456758055327015, 0.4656688196674575, 0.4243525517918916],
        'std': [0.2588946676768578, 0.25576863261712585, 0.2749972024401641],
        'dataset_name': 'fids_ade20k',
        'crop_size': (512, 512),
    },
    'bdd': {
        'mean': [0.36722284847810627, 0.41404042921213696, 0.42108752583194764],
        'std': [0.2526130760079844, 0.26953756308838606, 0.28668986155082293],
        'dataset_name': 'fids_bdd',
        'crop_size': (720, 1280),
    },
    'camvid': {
        'mean': [0.4135525214346383, 0.4267675639990583, 0.43429330870789346],
        'std': [0.31017532483387367, 0.31461703205620273, 0.31021052522748577],
        'dataset_name': 'fids_camvid',
        'crop_size': (624, 832),
    },
    'city': {
        'mean': [0.2868955263165162, 0.32513300997108135, 0.2838917598507516],
        'std': [0.18696374643849065, 0.19017338968162564, 0.18720214245271205],
        'dataset_name': 'fids_city_scapes',
        'crop_size': (512, 1024),
    },
    'coco': {
        'mean': [0.46584509286952447, 0.4467916652837197, 0.4027970027856512],
        'std': [0.2792570743798079, 0.27468881705627607, 0.2893893683946896],
        'dataset_name': 'fids_coco',
        'crop_size': (713, 713),
    },
    'idd': {
        'mean': [0.35481278866516913, 0.3666632612519479, 0.3591710136176604],
        'std': [0.2753931077147862, 0.2845505709972414, 0.3003046626952654],
        'dataset_name': 'fids_idd',
        'crop_size': (624, 832),
    },
    'isaid': {
        'mean': [0.3224990772393283, 0.32609559932461807, 0.30954792104727114],
        'std': [0.18431306043559967, 0.17849046116243392, 0.1734640796900857],
        'dataset_name': 'fids_isaid',
        'crop_size': (713, 713),
    },
    'isprs': {
        'mean': [0.33909638, 0.36251904, 0.34982737],
        'std': [0.14050266, 0.1387462, 0.13703808],
        'dataset_name': 'fids_isprs',
        'crop_size': (713, 713),
    },
    'kitti': {
        'mean': [0.3769289204886307, 0.3971275104183516, 0.3838184890366767],
        'std': [0.3092813268453932, 0.3189802872389981, 0.3298192904270604],
        'dataset_name': 'fids_kitti_segmentation',
        'crop_size': (512, 512),
    },
    'mapillary': {
        'mean': [0.414459302017456, 0.45885738262828446, 0.4649059742110866],
        'std': [0.2643241327324665, 0.275275065880243, 0.30314203519600535],
        'dataset_name': 'fids_mapillary_public',
        'crop_size': (768, 1024),
    },
    'msegpcontext': {
        'mean': [0.4534070723894893, 0.43937956556585933, 0.4024208506613706],
        'std': [0.27541526304174974, 0.27232753256708553, 0.28574396423927684],
        'dataset_name': 'fids_pascal_context',
        'crop_size': (420, 420),
    },
    'pvoc': {
        'mean': [0.4531086468922673, 0.43725320446227084, 0.39963537705323104],
        'std': [0.27576711572629464, 0.27251122900363556, 0.28560879332492456],
        'dataset_name': 'fids_pascal_voc2012',
        'crop_size': (420, 420),
    },
    'scannet': {
        'mean': [0.49658158589066304, 0.4494136849506602, 0.38261173016332106],
        'std': [0.29161244477060966, 0.28443821447289075, 0.2780922167303526],
        'dataset_name': 'fids_scan_net',
        'crop_size': (713, 713),
    },
    'suim': {
        'mean': [0.24383865491589002, 0.4224442819068281, 0.48276236871223316],
        'std': [0.22206708031370292, 0.22300285850150453, 0.2470457129108865],
        'dataset_name': 'fids_suim',
        'crop_size': (713, 713),
    },
    'sunrgbd': {
        'mean': [0.4894144839949708, 0.457240428822249, 0.428250524255643],
        'std': [0.27942549259025135, 0.2871466128411005, 0.29192820184660795],
        'dataset_name': 'fids_sunrgbd',
        'crop_size': (713, 713),
    },
    'vgallery': {
        'mean': [0.3976906719722632, 0.35323369684948275, 0.2562846881524604],
        'std': [0.2876926474514701, 0.29279828470237496, 0.2955533702235718],
        'dataset_name': 'fids_v_gallery',
        'crop_size': (576, 1024),
    },
    'vkitti2': {
        'mean': [0.3792748802264687, 0.38754103006986634, 0.30581123261946513],
        'std': [0.27025994787857666, 0.26975797265296797, 0.2712362348965233],
        'dataset_name': 'fids_v_kitti2',
        'crop_size': (375, 1242),
    },
}

