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
 * @fileoverview Render time benchmarking logic.
 */

/**
 * Whether the benchmark mode is currently in a cool-down state.
 */
let gIsCoolingDown = false;

/**
 * A list of frame timestamps, used for benchmarking.
 */
let gBenchmarkTimestamps = null;

let gFrameTimes = [];

/**
 * A dictionary of camera poses for benchmarking
 * @type {!object}
 */
let gBenchmarkCameras = {};

/**
 * Index of the current test camera that's being rendered for benchmarking.
 * @type {number}
 */
let gBenchmarkCameraIndex = 0;

/**
 * We use this constant as a prefix when saving benchmark output files.
 * @type {string}
 */
const gBenchmarkMethodName = 'blockmerf';

/**
 * We use this constant as a prefix when saving benchmark output files.
 * @type {?string}
 */
let gBenchmarkSceneName = null;

/**
 * Whether output images should be saved or not.
 * @type {boolean}
 */
let gSaveBenchmarkFrames = false;

/**
 * Shows the benchmark stats window and sets up the event listener for it.
 * @param {string} sceneName The name of the current scene.
 * @param {boolean} saveImages Should the benchmark images be saved to disk?
 */
function setupBenchmarkStats(sceneName, saveImages) {
  gBenchmarkSceneName = sceneName;
  gSaveBenchmarkFrames = saveImages;
  let benchmarkStats = document.getElementById('benchmark-stats');
  benchmarkStats.style.display = 'block';
  benchmarkStats.addEventListener('click', e => {
    gBenchmark = true;
  });
}

/**
 * Clears the benchmark stats content.
 * @param {!object} str ...
 */
function clearBenchmarkStats(str) {
  let benchmarkStats = document.getElementById('benchmark-stats');
  benchmarkStats.innerHTML = '';
}

/**
 * Adds a row of text to the benchmark stats window.
 * @param {!object} str ...
 */
function addBenchmarkRow(str) {
  let benchmarkStats = document.getElementById('benchmark-stats');
  benchmarkStats.innerHTML += str + '\n';
}

/**
 * Returns the benchmark stats output string.
 * @param {!object} str ...
 * @return {!object} ...
 */
function getBenchmarkStats(str) {
  const benchmarkStats = document.getElementById('benchmark-stats');
  return benchmarkStats.innerHTML;
}

/**
 * Loads the pose and projection matrices for the images used for benchmarking.
 * @param {!FilenameToLinkTranslator} filenameToLinkTranslator
 */
function loadBenchmarkCameras(filenameToLinkTranslator) {
  const benchmarkCamerasUrl =
      filenameToLinkTranslator.translate('test_frames.json');
  const benchmarkCamerasPromise = loadJSONFile(benchmarkCamerasUrl);
  benchmarkCamerasPromise.catch(error => {
    console.error(
        'Could not load test frames from: ' + benchmarkCamerasUrl +
        ', error: ' + error);
    return;
  });
  benchmarkCamerasPromise.then(parsed => {
    gBenchmarkCameras = parsed['test_frames'];
  });
}

/**
 * Sets the pose & projection matrix of the camera re-render a benchmark image.
 * @param {!THREE.PerspectiveCamera} camera The camera whose pose and projection
 *     matrix we're changing.
 * @param {number} index The index of the benchmark image want to re-render.
 */
function setBenchmarkCameraPose(camera, index) {
  camera.position.fromArray(gBenchmarkCameras[index]['position']);
  camera.setRotationFromMatrix(
      new THREE.Matrix4().fromArray(gBenchmarkCameras[index]['rotation']));
  camera.projectionMatrix.fromArray(gBenchmarkCameras[index]['projection']);
}

/**
 * Cools the GPU down between benchmarking runs.
 *
 * This function does the minimal work possible (i.e. clearing the screen to
 * a new color), to keep both the GPU driver and Javascript animation scheduler
 * active while also letting the GPU cores cool down.
 * @param {!object} t ...
 */
function cooldownFrame(t) {
  const alpha = 0.5 * (1.0 + Math.sin(t * Math.PI / 1000.0));
  let clearColor = new THREE.Color('#FFFFFF');
  clearColor.lerp(new THREE.Color('#A5C0E2'), alpha);
  gRenderer.setClearColor(clearColor, 1.0);
  gRenderer.clear();
  if (gStats) {
    gStats.update();
  }
  if (gIsCoolingDown) {
    requestAnimationFrame(cooldownFrame);
  }
}

/**
 * Returns the current timestamp formatted as a string.
 *
 * Example: "2023_11_22_1042"
 *
 * @returns {string}
 */
function formatTimestampAsString() {
  const date = new Date();
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  return `${date.getFullYear()}_${date.getMonth() + 1}_${date.getDate()}` +
      `_${hours}${minutes}`;
}

/**
 * Benchmarks performance by rendering test images while measuring frame times.
 *
 * You can use this function by calling it after all webgl calls have been
 * completed for a frame, just before the next call to requestAnimationFrame().
 *
 * Note however that this function has been designed to keep the GPU cool and
 * might want to delay the call to requestAnimationFrame() by a certain delay.
 * This is why defaultScheduleFrame is a parameter and why the return values
 * is a similar function (that may have an additional delay inserted).
 *
 * @param {!object} defaultScheduleFrame The function the renderer normally
 *   uses to schedule the next frame for rendering.
 * @returns {!object}
 */
function benchmarkPerformance(defaultScheduleFrame) {
  // These constants were tuned to get repeatable results in the bicycle scene
  // on an iPhone 15 Pro and a 2019 16" MacBook Pro with an AMD Radeon 5500M.
  const kCoolDownSeconds = 0.0;
  const kMaxFramesPerCamera = Math.max(4, Math.ceil(100 / gFrameMult));
  const kNumFramesToDiscard = Math.max(2, Math.ceil(0.1 * kMaxFramesPerCamera));

  // We start benchmarking only after gLastFrame has first been set.
  if (isLoading()) {
    return defaultScheduleFrame;
  }

  // We use the first frame after loading the scene to set up the
  // benchmarking state and cool the GPU down.
  if (!gBenchmarkTimestamps && !gIsCoolingDown) {
    setBenchmarkCameraPose(gCamera, 0);
    gBenchmarkTimestamps = [];

    if (kCoolDownSeconds > 0.0) {
      clearBenchmarkStats();
      addBenchmarkRow(`Cooling the GPU down for ${
          kCoolDownSeconds} seconds before benchmarking...`);
      gIsCoolingDown = true;
      requestAnimationFrame(cooldownFrame);
      return () => {
        setTimeout(() => {
          let s = new THREE.Vector2();
          gRenderer.getSize(s);
          clearBenchmarkStats();
          addBenchmarkRow(`frame timestamps (ms) at ${s.x}x${s.y}`);
          addBenchmarkRow('cam_idx ; start ; end ; mean frame time');
          gIsCoolingDown = false;
          defaultScheduleFrame();
        }, 1000 * kCoolDownSeconds);
      };
    }

    let s = new THREE.Vector2();
    gRenderer.getSize(s);
    clearBenchmarkStats();
    addBenchmarkRow(`frame timestamps (ms) at ${s.x}x${s.y}`);
    addBenchmarkRow('cam_idx ; start ; end ; mean frame time');
    return defaultScheduleFrame;
  }

  gBenchmarkTimestamps.push(window.performance.now());

  // Let the default frame scheduler proceed if we're still gathering frames.
  if (gBenchmarkTimestamps.length < kMaxFramesPerCamera) {
    return defaultScheduleFrame;
  }

  if (gSaveBenchmarkFrames) {
    frameAsPng = gRenderer.domElement.toDataURL('image/png');
    saveAs(frameAsPng, digits(gBenchmarkCameraIndex, 4) + '.png');
  }

  // Now that we have enough frames we can compute frame-time statistics.
  let benchmarkTimestamps = gBenchmarkTimestamps.slice(kNumFramesToDiscard);
  const numBenchmarkFrames = benchmarkTimestamps.length;
  const firstFrameTimestamp = benchmarkTimestamps[0];
  const lastFrameTimestamp = benchmarkTimestamps.pop();
  let meanTime = (lastFrameTimestamp - firstFrameTimestamp) /
      (gFrameMult * (numBenchmarkFrames - 1));
  gFrameTimes.push(meanTime);

  // Report them in the benchmark console.
  addBenchmarkRow(`${gBenchmarkCameraIndex} ; ${firstFrameTimestamp} ; ${
      lastFrameTimestamp} ; ${meanTime}`);

  // No more cameras: stop benchmarking, and store the results as a CSV file.
  if (++gBenchmarkCameraIndex >= gBenchmarkCameras.length) {
    console.log(gFrameTimes.reduce((a, b) => a + b, 0) / gFrameTimes.length);
    gBenchmark = false;
    const csvBlob =
        new Blob([getBenchmarkStats()], {type: 'text/plain;charset=utf-8'});
    const csvName = gBenchmarkMethodName + '_' + gBenchmarkSceneName + '_' +
        'frameMult_' + gFrameMult + '_' + formatTimestampAsString() + '.csv';
    saveAs(csvBlob, csvName);
    return defaultScheduleFrame;
  }

  // Otherwise, set things up for benchmarking the next camera pose and sleep
  // for the cooldown time to avoid biased results from thermal throttling.
  gBenchmarkTimestamps = [];
  setBenchmarkCameraPose(gCamera, gBenchmarkCameraIndex);
  if (kCoolDownSeconds > 0.0) {
    gIsCoolingDown = true;
    requestAnimationFrame(cooldownFrame);
    return () => {
      setTimeout(() => {
        gIsCoolingDown = false;
        defaultScheduleFrame();
      }, 1000 * kCoolDownSeconds);
    };
  }
  return defaultScheduleFrame;
}