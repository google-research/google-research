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

"""Metadata of segmentation datasets in FIDS.

The goal of this file is to provide metadata of the segmentation tasks in the
FIDS datasets in a lightway manner without relying on other imports/code.

For each dataset some of its metadata is presented, including:
1. number of classes
2. image mean (per channel)
3. image std (per channel)
4. list with class names.
"""

METADATA = {
    'fids_ade20k': {
        'num_classes':
            151,
        'image_mean': [
            0.48456758055327015, 0.4656688196674575, 0.4243525517918916
        ],
        'image_std': [
            0.2588946676768578, 0.25576863261712585, 0.2749972024401641
        ],
        'classes': [
            'background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling',
            'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk',
            'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain',
            'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house',
            'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
            'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
            'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter',
            'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
            'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table',
            'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
            'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
            'bench', 'countertop', 'stove', 'palm', 'kitchen island',
            'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
            'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier',
            'awning', 'streetlight', 'booth', 'television receiver', 'airplane',
            'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator',
            'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship',
            'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
            'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
            'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
            'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
            'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
            'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier',
            'crt screen', 'plate', 'monitor', 'bulletin board', 'shower',
            'radiator', 'glass', 'clock', 'flag'
        ],
    },
    'fids_bdd': {
        'num_classes':
            20,
        'image_mean': [
            0.36722284847810627, 0.41404042921213696, 0.42108752583194764
        ],
        'image_std': [
            0.2526130760079844, 0.26953756308838606, 0.28668986155082293
        ],
        'classes': [
            'background', 'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ],
    },
    'fids_camvid': {
        'num_classes':
            32,
        'image_mean': [
            0.4135525214346383, 0.4267675639990583, 0.43429330870789346
        ],
        'image_std': [
            0.31017532483387367, 0.31461703205620273, 0.31021052522748577
        ],
        'classes': [
            'background', 'Animal', 'Archway', 'Bicyclist', 'Bridge',
            'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole',
            'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text',
            'MotorcycleScooter', 'OtherMoving', 'ParkingBlock', 'Pedestrian',
            'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky',
            'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree',
            'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Wall'
        ],
    },
    'fids_city_scapes': {
        'num_classes':
            33,
        'image_mean': [
            0.2868955263165162, 0.32513300997108135, 0.2838917598507516
        ],
        'image_std': [
            0.18696374643849065, 0.19017338968162564, 0.18720214245271205
        ],
        'classes': [
            'background', 'ego vehicle', 'rectification border', 'static',
            'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track',
            'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'caravan', 'trailer', 'train', 'motorcycle', 'bicycle'
        ],
    },
    'fids_coco': {
        'num_classes':
            134,
        'image_mean': [
            0.46584509286952447, 0.4467916652837197, 0.4027970027856512
        ],
        'image_std': [
            0.2792570743798079, 0.27468881705627607, 0.2893893683946896
        ],
        'classes': [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
            'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
            'light', 'mirror-stuff', 'net', 'pillow', 'platform',
            'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
            'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
            'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
            'window-blind', 'window-other', 'tree-merged', 'fence-merged',
            'ceiling-merged', 'sky-other-merged', 'cabinet-merged',
            'table-merged', 'floor-other-merged', 'pavement-merged',
            'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
            'food-other-merged', 'building-other-merged', 'rock-merged',
            'wall-other-merged', 'rug-merged'
        ],
    },
    'fids_idd': {
        'num_classes':
            35,
        'image_mean': [
            0.35481278866516913, 0.3666632612519479, 0.3591710136176604
        ],
        'image_std': [
            0.2753931077147862, 0.2845505709972414, 0.3003046626952654
        ],
        'classes': [
            'background', 'road', 'parking', 'drivable fallback', 'sidewalk',
            'rail track', 'non-drivable fallback', 'person', 'animal', 'rider',
            'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus',
            'caravan', 'trailer', 'vehicle fallback', 'curb', 'wall', 'fence',
            'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole',
            'polegroup', 'obs-str-bar-fallback', 'building', 'bridge', 'tunnel',
            'vegetation', 'sky', 'fallback background'
        ],
    },
    'fids_isaid': {
        'num_classes':
            16,
        'image_mean': [
            0.32249907723932625, 0.3260955993246167, 0.3095479210472695
        ],
        'image_std': [
            0.18431306043560025, 0.1784904611624337, 0.17346407969008562
        ],
        'classes': [
            'unlabeled', 'ship', 'storage_tank', 'baseball_diamond',
            'tennis_court', 'basketball_court', 'Ground_Track_Field', 'Bridge',
            'Large_Vehicle', 'Small_Vehicle', 'Helicopter', 'Swimming_pool',
            'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'
        ],
    },
    'fids_isprs': {
        'num_classes':
            7,
        'image_mean': [
            0.33909638362680655, 0.36251903574722705, 0.34982737497258753
        ],
        'image_std': [
            0.14050265616326077, 0.1387462049088704, 0.13703807506662904
        ],
        'classes': [
            'background', 'impervious_surfaces', 'building', 'low_vegetation',
            'tree', 'car', 'clutter'
        ],
    },
    'fids_kitti_segmentation': {
        'num_classes':
            33,
        'image_mean': [
            0.3769289204886307, 0.3971275104183516, 0.3838184890366767
        ],
        'image_std': [
            0.3092813268453932, 0.3189802872389981, 0.3298192904270604
        ],
        'classes': [
            'background', 'ego vehicle', 'rectification border', 'static',
            'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track',
            'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'caravan', 'trailer', 'train', 'motorcycle', 'bicycle'
        ],
    },
    'fids_mapillary_public': {
        'num_classes':
            66,
        'image_mean': [
            0.414459302017456, 0.45885738262828446, 0.4649059742110866
        ],
        'image_std': [
            0.2643241327324665, 0.275275065880243, 0.30314203519600535
        ],
        'classes': [
            'background', 'Bird', 'Ground Animal', 'Curb', 'Fence',
            'Guard Rail', 'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain',
            'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road',
            'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
            'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider',
            'Lane Marking - Crosswalk', 'Lane Marking - General', 'Mountain',
            'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner',
            'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
            'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth',
            'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame',
            'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)',
            'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus',
            'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle',
            'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle'
        ],
    },
    'fids_pascal_context': {
        'num_classes':
            60,
        'image_mean': [
            0.4534070723894893, 0.43937956556585933, 0.4024208506613706
        ],
        'image_std': [
            0.27541526304174974, 0.27232753256708553, 0.28574396423927684
        ],
        'classes': [
            'background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
            'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
            'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer',
            'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower',
            'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
            'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
            'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk',
            'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree',
            'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood'
        ],
    },
    'fids_pascal_voc2012': {
        'num_classes':
            21,
        'image_mean': [
            0.4531086468922673, 0.43725320446227084, 0.39963537705323104
        ],
        'image_std': [
            0.27576711572629464, 0.27251122900363556, 0.28560879332492456
        ],
        'classes': [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
        ],
    },
    'fids_scan_net': {
        'num_classes':
            41,
        'image_mean': [
            0.49658158589066304, 0.4494136849506602, 0.38261173016332106
        ],
        'image_std': [
            0.29161244477060966, 0.28443821447289075, 0.2780922167303526
        ],
        'classes': [
            'background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
            'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
            'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
            'mirror', 'floor mat', 'clothes', 'ceiling', 'books',
            'refridgerator', 'television', 'paper', 'towel', 'shower curtain',
            'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink',
            'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
            'otherprop'
        ],
    },
    'fids_suim': {
        'num_classes':
            8,
        'image_mean': [
            0.24383865491589002, 0.4224442819068281, 0.48276236871223316
        ],
        'image_std': [
            0.22206708031370292, 0.22300285850150453, 0.2470457129108865
        ],
        'classes': [
            'background', 'HD: Human divers', 'PF: Plants/sea-grass',
            'WR: Wrecks/ruins', 'RO: Robots/instruments',
            'RI: Reefs and invertebrates', 'FV: Fish and vertebrates',
            'SR: Sand/sea-floor (& rocks)'
        ],
    },
    'fids_sunrgbd': {
        'num_classes':
            38,
        'image_mean': [
            0.4894144839949705, 0.4572404288222465, 0.4282505242556435
        ],
        'image_std': [
            0.2794254925902518, 0.28714661284110093, 0.29192820184660856
        ],
        'classes': [
            'background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
            'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
            'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
            'mirror', 'floor mat', 'clothes', 'ceiling', 'books',
            'refridgerator', 'television', 'paper', 'towel', 'shower curtain',
            'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink',
            'lamp', 'bathtub', 'bag'
        ],
    },
    'fids_v_gallery': {
        'num_classes':
            10,
        'image_mean': [
            0.3976906719722626, 0.3532336968494831, 0.25628468815246086
        ],
        'image_std': [
            0.2876926474514705, 0.292798284702375, 0.2955533702235714
        ],
        'classes': [
            'Background', 'Wall', 'Ceiling', 'Sky', 'Door', 'Light', 'Floor',
            'Misc', 'Painting', 'Human'
        ],
    },
    'fids_v_kitti2': {
        'num_classes':
            15,
        'image_mean': [
            0.3792748802264694, 0.38754103006986584, 0.3058112326194656
        ],
        'image_std': [
            0.27025994787857655, 0.2697579726529681, 0.2712362348965231
        ],
        'classes': [
            'Background', 'Terrain', 'Sky', 'Tree', 'Vegetation', 'Building',
            'Road', 'GuardRail', 'TrafficSign', 'TrafficLight', 'Pole', 'Misc',
            'Truck', 'Car', 'Van'
        ],
    },
}
