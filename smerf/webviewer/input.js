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


/**
 * We control the camera using either orbit controls...
 * @type {?THREE.OrbitControls}
 */
let gOrbitControls = null;

/**
 * Map-controls, which are orbit controls with custom arguments, ...
 * @type {?THREE.OrbitControls}
 */
 let gMapControls = null;

 /**
  * ...or for large scenes we use FPS-style controls.
  * @type {?THREE.PointerLockControls}
  */
 let gPointerLockControls = null;

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
  let shaderEditor = document.getElementById('shader-editor');

  document.addEventListener('keypress', function(e) {
    if (document.activeElement === shaderEditor) {
      return;
    }
    if (e.keyCode === 32 || e.key === ' ' || e.key === 'Spacebar') {
      if (gDisplayMode == DisplayModeType.DISPLAY_NORMAL) {
        gDisplayMode = DisplayModeType.DISPLAY_DIFFUSE;
        console.log('Displaying DIFFUSE');
      } else if (gDisplayMode == DisplayModeType.DISPLAY_DIFFUSE) {
        gDisplayMode = DisplayModeType.DISPLAY_FEATURES;
        console.log('Displaying DISPLAY_FEATURES');
      } else if (gDisplayMode == DisplayModeType.DISPLAY_FEATURES) {
        gDisplayMode = DisplayModeType.DISPLAY_VIEW_DEPENDENT;
        console.log('Displaying DISPLAY_VIEW_DEPENDENT');
      } else if (gDisplayMode == DisplayModeType.DISPLAY_VIEW_DEPENDENT) {
        gDisplayMode = DisplayModeType.DISPLAY_COARSE_GRID;
        console.log('Displaying DISPLAY_COARSE_GRID');
      } else /* gDisplayModeType == DisplayModeType.DISPLAY_COARSE_GRID */ {
        gDisplayMode = DisplayModeType.DISPLAY_NORMAL;
        console.log('Displaying DISPLAY_NORMAL');
      }
      e.preventDefault();
    }
    if (e.key === 'r') {
      console.log('Recompile shader.');
      let material = getRayMarchScene().children[0].material;
      material.fragmentShader = shaderEditor.value;
      material.needsUpdate = true;
      e.preventDefault();
    }
    if (e.key === '?') {
      let position = gCamera.getWorldPosition(new THREE.Vector3(0., 0., 0.));
      let direction = gCamera.getWorldQuaternion(new THREE.Quaternion());
      console.log(`
// Camera Info:
gCamera.position.set(${position.x}, ${position.y}, ${position.z});
gCamera.quaternion.set(${direction.x}, ${direction.y}, ${direction.z}, ${
          direction.w});
`);
      e.preventDefault();
    }
  });
  document.addEventListener('keydown', function(e) {
    if (document.activeElement === shaderEditor) {
      return;
    }
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
    if (document.activeElement === shaderEditor) {
      return;
    }
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
 * @param {string} mouseMode Either "orbit", "fps" or "map".
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
    startButton.addEventListener('click', function() {
      gPointerLockControls.lock();
      gPointerLockControls.connect();
    }, false);

    const viewSpaceContainer = document.getElementById('viewspacecontainer');
    viewSpaceContainer.appendChild(startButton);
  } else if (mouseMode && mouseMode == 'map') {
    gMapControls = new THREE.OrbitControls(gCamera, view);
    gMapControls.panSpeed = 0.5 / gCamera.near;
    gMapControls.enableZoom = false;
    gMapControls.screenSpacePanning = false;
		gMapControls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      RIGHT: THREE.MOUSE.PAN
    };
		gMapControls.touches = {
      ONE: THREE.TOUCH.PAN,
      TWO: THREE.TOUCH.DOLLY_ROTATE,
    };
  } else { // mouseMode == 'orbit'
    gOrbitControls = new THREE.OrbitControls(gCamera, view);
    gOrbitControls.screenSpacePanning = true;
    gOrbitControls.zoomSpeed = 0.5;
    // Disable damping until we have temporal reprojection for upscaling.
    // gOrbitControls.enableDamping = true;
  }
}

/**
 * Updates the camera based on user input.
 */
function updateCameraControls() {
  if (gOrbitControls) {
    gOrbitControls.update();
  } else if (gMapControls) {
    gMapControls.update();
  } else if (gPointerLockControls) {
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
}