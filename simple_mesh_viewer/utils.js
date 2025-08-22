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
 * @fileoverview Utility functions for file loading and more.
 */

/**
 * Formats the integer i as a string with "min" leading zeroes.
 * @param {number} i
 * @param {number} min
 * @return {string}
 */
function digits(i, min) {
  const s = '' + i;
  if (s.length >= min) {
    return s;
  } else {
    return ('00000' + s).substr(-min);
  }
}

/**
 * Updates the loading progress HTML elements.
 */
function updateLoadingProgress() {
  let imageProgress = document.getElementById('image-progress');
  if (gTotalBytesToDecode) {
    let progress = 100 * gBytesDecoded / gTotalBytesToDecode;
    progress = progress.toFixed(1);
    imageProgress.innerHTML = 'Progress: ' + progress + '%';
  } else {
    let gMebibytesDecoded = gBytesDecoded / (1024 * 1024);
    gMebibytesDecoded = gMebibytesDecoded.toFixed(1);
    imageProgress.innerHTML = ' Decoded: ' + gMebibytesDecoded + ' MiB';
  }
}


/**
 * Returns a promise that fires within a specified amount of time. Can be used
 * in an asynchronous function for sleeping.
 * @param {number} milliseconds Amount of time to sleep
 * @return {!Promise}
 */
function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/**
 * Safe fetching. Some servers restrict the number of requests and
 * respond with status code 429 ("Too Many Requests") when a threshold
 * is exceeded. When we encounter a 429 we retry after a short waiting period.
 * @param {!object} fetchFn Function that fetches the file.
 * @return {!Promise} Returns fetchFn's response.
 */
async function fetchAndRetryIfNecessary(fetchFn) {
  const response = await downloadFileWithProgress(fetchFn);
  if (response.status === 429) {
    await sleep(500);
    return fetchAndRetryIfNecessary(fetchFn);
  }
  return response;
}

/**
 * Downloads a file while maintaining progress.
 * @param {!object} fetchFn Function that fetches the file.
 * @return {!Promise} Returns fetchFn's response.
 */
async function downloadFileWithProgress(fetchFn) {
  try {
    const response = await fetchFn();
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const reader = response.body.getReader();

    let outputStream = new ReadableStream({
      start(controller) {
        async function read() {
          try {
            const {done, value} = await reader.read();
            if (done) {
              controller.close();
              return;
            }

            gBytesDecoded += value.byteLength;
            updateLoadingProgress();

            controller.enqueue(value);
            read();
          } catch (error) {
            console.error('Error reading the stream:', error);
            controller.error(error);
          }
        }
        read();
      }
    });
    return new Response(outputStream);
  } catch (error) {
    console.error('Error downloading the file:', error);
    throw error;
  }
}

/**
 * Loads PNG image from rgbaURL and decodes it to an Uint8Array.
 * @param {string} rgbaUrl The URL of the PNG image.
 * @return {!Promise<!Uint8Array>}
 */
function loadPNG(rgbaUrl) {
  let fetchFn = () => fetch(rgbaUrl, {method: 'GET', mode: 'cors'});
  const rgbaPromise = fetchAndRetryIfNecessary(fetchFn)
                          .then(response => {
                            return response.arrayBuffer();
                          })
                          .then(buffer => {
                            let data = new Uint8Array(buffer);
                            let pngDecoder = new PNG(data);
                            let pixels = pngDecoder.decodePixels();
                            return pixels;
                          });
  rgbaPromise.catch(error => {
    console.error(
        'Could not load PNG image from: ' + rgbaUrl + ', error: ' + error);
    return;
  });
  return rgbaPromise;
}

/**
 * Loads binary file from url and decodes it to an Uint8Array.
 * @param {string} url The URL of the binary file.
 * @return {!Promise<!Uint8Array>}
 */
function loadBinaryFile(url) {
  let fetchFn = () => fetch(url, {
    method: 'GET',
    mode: 'cors',
  });

  const promise = fetchAndRetryIfNecessary(fetchFn)
                      .then(response => {
                        return response.arrayBuffer();
                      })
                      .then(buffer => {
                        let data = new Uint8Array(buffer);
                        return data;
                      });
  promise.catch(error => {
    console.error(
        'Could not load binary file from: ' + url + ', error: ' + error);
    return;
  });
  return promise;
}

/**
 * Loads a text file.
 * @param {string} url URL of the file to be loaded.
 * @return {!Promise<string>}
 */
function loadTextFile(url) {
  let fetchFn = () => fetch(url, {
    method: 'GET',
    mode: 'cors',
  });
  return fetchAndRetryIfNecessary(fetchFn).then(response => response.text());
}

/**
 * Loads and parses a JSON file.
 * @param {string} url URL of the file to be loaded.
 * @return {!Promise<!object>} The parsed JSON file.
 */
function loadJSONFile(url) {
  let fetchFn = () => fetch(url, {method: 'GET', mode: 'cors'});
  return fetchAndRetryIfNecessary(fetchFn).then(response => response.json());
}

/**
 * Loads a GLB mesh file.
 * @param {string} url URL of the file to be loaded.
 * @return {!Promise<!object>} The parsed GLTF scene.
 */
function loadMesh(url) {
  return new Promise((resolve, reject) => {
    let loader = new THREE.GLTFLoader();
    let previouslyLoaded = 0;
    loader.load(
        url,
        (gltf) => {
          resolve(gltf);
        },
        (xhr) => {
          let deltaLoaded = xhr.loaded - previouslyLoaded;
          gBytesDecoded += deltaLoaded;
          previouslyLoaded = xhr.loaded;
          updateLoadingProgress();
        },
        (error) => {
          console.log(error);
          reject(error);
        });
  });
}

/**
 * Calculates PSNR for a given pair of images.
 * @param {!Uint8Array} original The reference image.
 * @param {!Uint8Array} compressed The degraded image.
 * @param {number=} maxPixelValue Maximum pixel value. Defaults to 255.
 * @return {number} PSNR value.
 */
function calculatePSNR(original, compressed, maxPixelValue = 255) {
  if (original.length !== compressed.length) {
    throw new Error('Input arrays must have the same length.');
  }
  let mse = 0;
  for (let i = 0; i < original.length; i++) {
    mse += Math.pow(original[i] - compressed[i], 2);
  }
  mse /= original.length;
  if (mse === 0) {
    return Infinity;  // PSNR is infinite for identical images
  }
  let psnr = 10 * Math.log10(Math.pow(maxPixelValue, 2) / mse);
  return psnr;
}

/**
 * Calculates the average for a given array of numbers.
 * @param {!object} arr Array of numbers.
 * @return {number} The average value.
 */
function calculateAverage(arr) {
  if (arr.length === 0) {
    return 0;
  }
  let sum = arr.reduce(function(total, currentValue) {
    return total + currentValue;
  }, 0);
  let average = sum / arr.length;
  return average;
}

/**
 * Obtain the parameters that were used to construct a projection matrix.
 * Assumes a WebGL-style matrix. WebGPU matrices are not supported.
 * @param {!THREE.Matrix4} projectionMatrix The projection matrix that is going
 *  to be modified in-place.
 * @return {!object} An object containing left, right, top, bottom, near, far.
 */
function getPerspectiveParameters(projectionMatrix) {
  // Obtain inputs.
  const te = projectionMatrix.elements;
  const x = te[0];
  const y = te[5];
  const a = te[8];
  const b = te[9];
  const c = te[10];
  const d = te[14];

  // Compute outputs.
  const near = (d / (c - 1));
  const far = (d / (c + 1));
  const left = (a - 1) / x * near;
  const right = (a + 1) / x * near;
  const top = (b + 1) / y * near;
  const bottom = (b - 1) / y * near;

  return {left, right, top, bottom, near, far};
}

/**
 * Adjusts only the near value of a projection matrix in-place.
 * @param {!THREE.Matrix4} projectionMatrix The projection matrix that is going
 *  to be modified in-place.
 * @param {number} newNear New near value.
 */
function adjustNearValueOfProjectionMatrix(projectionMatrix, newNear) {
  let params = getPerspectiveParameters(projectionMatrix);
  let r = newNear / params.near;
  projectionMatrix.makePerspective(
      params.left * r, params.right * r, params.top * r, params.bottom * r,
      newNear, params.far);
}

/**
 * Creates a DOM element that belongs to the given CSS class.
 * @param {string} what
 * @param {string} classname
 * @return {!HTMLElement}
 */
function create(what, classname) {
  const e = /** @type {!HTMLElement} */ (document.createElement(what));
  if (classname) {
    e.className = classname;
  }
  return e;
}

/**
 * Reports an error to the user by populating the error div with text.
 * @param {string} text
 */
function error(text) {
  const e = document.getElementById('Loading');
  e.textContent = 'Error: ' + text;
  e.style.display = 'block';
  hideLoading();
}


/**
 * Resizes a DOM element to the given dimensions.
 * @param {!Element} element
 * @param {number} width
 * @param {number} height
 */
function setDims(element, width, height) {
  element.style.width = width.toFixed(2) + 'px';
  element.style.height = height.toFixed(2) + 'px';
}

/**
 * Hides the loading prompt.
 */
function hideLoading() {
  let loading = document.getElementById('Loading');
  loading.style.display = 'none';

  let loadingContainer = document.getElementById('loading-container');
  loadingContainer.style.display = 'none';
}

/**
 * Checks whether the WebGL context is valid and the underlying hardware is
 * powerful enough. Otherwise displays a warning.
 * @return {boolean}
 */
function isRendererUnsupported() {
  let loading = document.getElementById('Loading');

  let gl = gRenderer.getContext();
  if (!gl) {
    loading.innerHTML = 'Error: WebGL2 context not found. Is your machine' +
        ' equipped with a discrete GPU?';
    return true;
  }

  let debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  if (!debugInfo) {
    loading.innerHTML = 'Error: Could not fetch renderer info. Is your' +
        ' machine equipped with a discrete GPU?';
    return true;
  }
  return false;
}