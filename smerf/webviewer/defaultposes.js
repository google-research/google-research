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
 *  Set initial camera pose depending on the scene.
 * @param {string} dirUrl The url where scene files are stored.
 * @param {!THREE.Vector3} submodelCenter The world-space center of the
 *   current submodel.
 */
function setupInitialCameraPose(dirUrl, submodelCenter) {
  initialPoses = {
    'default': {
      'position': [0.0, 0.0, 0.0],
      'lookat': [0.0, 0.0, 1.0],
    },
    'gardenvase': {
      'position':
          [-1.1868985500525444, 0.1898527233835131, -0.04923970470097733],
      'lookat':
          [-0.05581392405861873, -0.40202760746449473, 0.02985343723310108],
    },
    'stump': {
      'position': [0.0, 0.4, -0.8],
      'lookat': [0.0, -0.3, 0.0],
    },
    'flowerbed': {
      'position':
          [-0.02402388218043944, 0.11825367482140309, 0.907525093384825],
      'lookat':
          [0.016306507293821822, -0.15676691106539536, -0.016192691610482132],
    },
    'treehill': {
      'position': [-0.70994804046872, 0.19435986647308223, 0.30833533637897453],
      'lookat':
          [0.06327294888291587, -0.13299740290200024, 0.0037554887097183934],
    },
    'bicycle': {
      'position':
          [-0.4636408064933045, 0.49624791762954734, 0.8457540259646037],
      'lookat':
          [0.017170160491904368, -0.24649043500978007, -0.07787524806850904],
    },
    'kitchenlego': {
      'position':
          [-0.5872864419408019, 0.05633623000443683, -0.9472239198227385],
      'lookat': [0.07177184299031553, -0.4020277194862108, 0.04850453170234236],
    },
    'fulllivingroom': {
      'position':
          [1.1539572663654272, -0.006785278327404387, -0.0972986385811351],
      'lookat':
          [-0.05581392405861873, -0.40202760746449473, 0.02985343723310108],
    },
    'kitchencounter': {
      'position':
          [-0.7006764413546107, 0.2255633917824672, -0.46941182833135847],
      'lookat': [0.13197415755218864, -0.4020278046227117, 0.09221809216932579],
    },
    'officebonsai': {
      'position': [-0.4773314920559294, 0.05409730603092788, 1.014304107335418],
      'lookat':
          [0.11970974858222336, -0.40426664345968033, -0.019801655674420764],
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
    if (gOrbitControls) {
      gOrbitControls.target.x = x;
      gOrbitControls.target.y = y;
      gOrbitControls.target.z = z;
    }
    else if (gMapControls) {
      gMapControls.target.x =
        gCamera.position.x + (x - gCamera.position.x) * gCamera.near;
      gMapControls.target.y =
        gCamera.position.y + (y - gCamera.position.y) * gCamera.near;
      gMapControls.target.z =
        gCamera.position.z + (z - gCamera.position.z) * gCamera.near;
    }
    else {
      gCamera.lookAt(x, y, z);
    }
  }

  function setCameraPose(d) {
    gCamera.position.x = d['position'][0] + submodelCenter.x;
    gCamera.position.y = d['position'][1] + submodelCenter.y;
    gCamera.position.z = d['position'][2] + submodelCenter.z;
    cameraLookAt(
      d['lookat'][0] + submodelCenter.x,
      d['lookat'][1] + submodelCenter.y,
      d['lookat'][2] + submodelCenter.z);
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