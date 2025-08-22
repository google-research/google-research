// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Defines default camera pose for each scene.
 */

/**
 * Set initial camera pose depending on the scene.
 * @param {string} dirUrl Either points to a directory that contains scene files
 *  or to a json file that maps virtual filenames to download links.
 */
function setupInitialCameraPose(dirUrl) {
  initialPoses = {
    'default': {
      'position': [0.0, 0.0, -1.0],
      'lookat': [0.0, -0.2, 0.0],
    },
    'gardenvase': {
      'position':
          [-1.153626079913854, 0.09141117646357577, -0.04498930276675259],
      'lookat': [0.094277745998087, -0.40202745464891987, 0.06418813517430078],
    },
    'stump': {
      'position':
          [-0.09963341507482755, 0.1712403019316845, -1.0075092718648655],
      'lookat':
          [-0.012497478227366462, -0.3312853244063434, -0.011540956249269528],
    },
    'flowerbed': {
      'position': [0.4644830513864044, 0.38378557897492716, 0.7862872948871611],
      'lookat':
          [-0.001218062889751977, -0.07822484860437395, 0.011230485489593942],
    },
    'treehill': {
      'position': [-0.0817352399127311, 0.247149016630893, -0.7664196590552215],
      'lookat':
          [0.0701192403354998, -0.15835351676659565, 0.014803917156875946],
    },
    'bicycle': {
      'position': [-0.313440846842122, 0.1843653880795849, 1.0117971114204822],
      'lookat':
          [-0.017919925083087797, -0.23121641129545928, -0.09110198328931468],
    },
    'kitchenlego': {
      'position':
          [0.6881877686362137, -0.00024687513942323047, -0.5181017629499767],
      'lookat':
          [0.06593535336069208, -0.46322674677429077, 0.01335200641604152],
    },
    'fulllivingroom': {
      'position':
          [1.1539572663654272, -0.006785278327404387, -0.0972986385811351],
      'lookat':
          [-0.05581392405861873, -0.40202760746449473, 0.02985343723310108],
    },
    'kitchencounter': {
      'position':
          [-0.5252524747108798, 0.20722942027645458, -0.23619240897230126],
      'lookat':
          [0.07874489726541031, -0.37530269701679014, -0.004339540603794985],
    },
    'officebonsai': {
      'position': [0.5792121846621209, 0.2699942848622923, 0.6192179163390396],
      'lookat':
          [0.12415143497710077, -0.4270931529079502, -0.007119472237646017]
    },
  };

  /**
   * Quick helper function to set the lookat point regardless of camera
   * controls.
   * @param {number} x
   * @param {number} y
   * @param {number} z
   */
  function cameraLookAt(x, y, z) {
    if (gControls) {
      gControls.target.x = x;
      gControls.target.y = y;
      gControls.target.z = z;
    }
    /*if (gPointerLockControls) {
      gCamera.lookAt(x, y, z);
    }*/
  }

  function setCameraPose(d) {
    gCamera.position.x = d['position'][0];
    gCamera.position.y = d['position'][1];
    gCamera.position.z = d['position'][2];
    cameraLookAt(d['lookat'][0], d['lookat'][1], d['lookat'][2]);
  }

  setCameraPose(initialPoses['default']);
  for (let sceneName in initialPoses) {
    if (dirUrl.includes(sceneName)) {
      setCameraPose(initialPoses[sceneName]);
      break;
    }
  }
  gCamera.updateProjectionMatrix();
}