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
 * @fileoverview Input event handling. 
 */

// With PointerLockControls we have to track key states ourselves.
/** @type {boolean} */
let gKeyW = false;
/** @type {boolean} */
let gKeyA = false;
/** @type {boolean} */
let gKeyS = false;
/** @type {boolean} */
let gKeyD = false;
/** @type {boolean} */
let gKeyQ = false;
/** @type {boolean} */
let gKeyE = false;
/** @type {boolean} */
let gKeyShift = false;

/**
 * Keeps track of frame times for smooth camera motion.
 * @type {!THREE.Clock}
 */
const gClock = new THREE.Clock();

/**
 * Adds event listeners to UI.
 */
function addHandlers() {
  document.addEventListener('keypress', function(e) {
    if (e.keyCode === 32 || e.key === ' ' || e.key === 'Spacebar') {
      if (gDisplayMode == DisplayModeType.DISPLAY_NORMAL) {
        gDisplayMode = DisplayModeType.DISPLAY_DIFFUSE;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_DIFFUSE) {
        gDisplayMode = DisplayModeType.DISPLAY_FEATURES;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_FEATURES) {
        gDisplayMode = DisplayModeType.DISPLAY_VIEW_DEPENDENT;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_VIEW_DEPENDENT) {
        gDisplayMode = DisplayModeType.DISPLAY_COARSE_GRID;
      } else /* gDisplayModeType == DisplayModeType.DISPLAY_COARSE_GRID */ {
        gDisplayMode = DisplayModeType.DISPLAY_NORMAL;
      }
      e.preventDefault();
    }
    if (e.key === 'i') {
      gStepMult += 1;
      console.log('num samples per voxel:', gStepMult);
      e.preventDefault();
    }
    if (e.key === 'o') {
      gStepMult -= 1;
      console.log('num samples per voxel:', gStepMult);
      e.preventDefault();
    }
  });
  document.addEventListener('keydown', function(e) {
    let key = e.key.toLowerCase();
    if (key === 'w') {
      gKeyW = true;
      e.preventDefault();
    }
    if (key === 'a') {
      gKeyA = true;
    }
    if (key === 's') {
      gKeyS = true;
      e.preventDefault();
    }
    if (key === 'd') {
      gKeyD = true;
      e.preventDefault();
    }
    if (key === 'q') {
      gKeyQ = true;
      e.preventDefault();
    }
    if (key === 'e') {
      gKeyE = true;
      e.preventDefault();
    }
    if (e.key === 'Shift') {
      gKeyShift = true;
      e.preventDefault();
    }
  });
  document.addEventListener('keyup', function(e) {
    let key = e.key.toLowerCase();
    if (key === 'w') {
      gKeyW = false;
      e.preventDefault();
    }
    if (key === 'a') {
      gKeyA = false;
    }
    if (key === 's') {
      gKeyS = false;
      e.preventDefault();
    }
    if (key === 'd') {
      gKeyD = false;
      e.preventDefault();
    }
    if (key === 'q') {
      gKeyQ = false;
      e.preventDefault();
    }
    if (key === 'e') {
      gKeyE = false;
      e.preventDefault();
    }
    if (e.key === 'Shift') {
      gKeyShift = false;
      e.preventDefault();
    }
  });
}

/**
 * Sets up the camera controls.
 * @param {string} mouseMode Either "orbit" or "fps".
 * @param {!HTMLElement} view The view.
 */
function setupCameraControls(mouseMode, view) {
  if (mouseMode && mouseMode == 'fps') {
    gPointerLockControls = new THREE.PointerLockControls(gCamera, view);
    let startButton = document.createElement('button');
    startButton.innerHTML = 'Click to enable mouse navigation';
    startButton.style = 'position: absolute;' +
        'top: 0;' +
        'width: 250px;' +
        'margin: 0 0 0 -125px;';
    viewSpaceContainer.appendChild(startButton);
    startButton.addEventListener('click', function() {
      gPointerLockControls.lock();
      gPointerLockControls.connect();
    }, false);
  } else {
    gOrbitControls = new THREE.OrbitControls(gCamera, view);
    // Disable damping until we have temporal reprojection for upscaling.
    // gOrbitControls.enableDamping = true;
    gOrbitControls.screenSpacePanning = true;
    gOrbitControls.zoomSpeed = 0.5;
  }
}

/**
 * Updates the camera based on user input.
 */
function updateCameraControls() {
  if (gOrbitControls) {
    gOrbitControls.update();
  } else {
    const elapsed = gClock.getDelta();
    let movementSpeed = 0.25;
    if (gKeyShift) {
      movementSpeed = 1;
    }
    let camForward = gCamera.getWorldDirection(new THREE.Vector3(0., 0., 0.));
    let upVec = new THREE.Vector3(0., 1., 0.);
    if (gKeyW) {
      // gPointerLockControls.moveForward undesirably restricts movement to the
      // X-Z-plane.
      gCamera.position =
          gCamera.position.addScaledVector(camForward, elapsed * movementSpeed);
    }
    if (gKeyA) {
      gPointerLockControls.moveRight(-elapsed * movementSpeed);
    }
    if (gKeyS) {
      gCamera.position = gCamera.position.addScaledVector(
          camForward, -elapsed * movementSpeed);
    }
    if (gKeyD) {
      gPointerLockControls.moveRight(elapsed * movementSpeed);
    }
    if (gKeyQ) {
      gCamera.position =
          gCamera.position.addScaledVector(upVec, -elapsed * movementSpeed);
    }
    if (gKeyE) {
      gCamera.position =
          gCamera.position.addScaledVector(upVec, elapsed * movementSpeed);
    }
  }
  gCamera.updateProjectionMatrix();
  gCamera.updateMatrixWorld();
}