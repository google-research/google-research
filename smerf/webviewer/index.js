// Copyright 2023 The Google Research Authors.
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
 * @fileoverview Keeps track of whether loadOnFirstFrame has already been run.
 * @suppress {lintChecks}
 */
let gLoadOnFirstFrameRan = [];

/**
 * Number of already loaded textures.
 * @type {number}
 */
let gLoadedTextures = 0;

/**
 * Number of textures to load.
 * @type {number}
 */
let gNumTextures = 0;

/**
 * Our framerate display.
 * @type {?Object}
 */
let gStats = null;

/**
 * panel for current submodel
 */
let gSubmodelPanel = null;

/**
 * Allows forcing specific submodel for debugging
 */
let gSubmodelForceIndex = -1;

let gUseSubmodel = false;

let gSubmodelTransform = null;

let gDeferredMlp = null;

let gSubmodelScale = [];

let gSubmodelScaleFactor = [];

/**
 * Different display modes for debugging rendering.
 * @enum {number}
 */
const DisplayModeType = {
  /** Runs the full model with view dependence. */
  DISPLAY_NORMAL: 0,
  /** Disables the view-dependence network. */
  DISPLAY_DIFFUSE: 1,
  /** Only shows the latent features. */
  DISPLAY_FEATURES: 2,
  /** Only shows the view dependent component. */
  DISPLAY_VIEW_DEPENDENT: 3,
  /** Only shows the coarse block grid. */
  DISPLAY_COARSE_GRID: 4,
};

/**  @type {!DisplayModeType}  */
let gDisplayMode = DisplayModeType.DISPLAY_NORMAL;

/**
 * If true we evaluate run-time performance by re-rendering test viewpoints.
 * @type {boolean}
 */
let gBenchmark = false;

/**
 * Number of sample points per voxel.
 * @type {number}
 */
let gStepMult = 1;

/**
 * For benchmarking with vsync on: render this many redundant images per frame.
 * @type {number}
 */
let gFrameMult = 1;

/**
 * For large scenes with varying exposure we set this value to be the exposure
 * of the virtual camera (shutter_speed_in_seconds * iso / 1000).
 * @type {number}
 */
let gExposure = null;

/**
 * A web worker for parsing PNGs in a separate thread.
 * @type {*}
 */
let loadPNGWorker = (function() {
  /**
   * A singleton for loading PNGs asynchronously. This singleton owns a set of
   * web workers, which are used to offload CPU-heavy PNG decoding to separate
   * threads.
   */
  class GlobalLoadPNGWorker {
    /**
     * Initializes a GlobalLoadPNGWorker
     */
    constructor() {
      let that = this;

      // Instantiate one worker per core available. Leave one core for the UX
      // thread. Use a safe default value if navigator.hardwareConcurrency
      // isn't defined.
      let numWorkers = 8;
      if (navigator.hardwareConcurrency != null) {
        numWorkers = navigator.hardwareConcurrency - 2;
      }
      numWorkers = Math.max(1, numWorkers);

      // Create a pool of workers.
      this.workers = [];
      for (let i = 0; i < numWorkers; ++i) {
        let worker = new Worker('loadpng.worker.js');
        worker.onmessage = (e) => {
          that.onmessage(e);
        };
        this.workers.push(worker);
      }

      this.nextworker = 0;
      this.callbacks = {};
      this.i = 0;
    }

    /**
     * Fetches a PNG asynchronously.
     * @param {string} url URL of PNG to fetch.
     * @param {*} callback Callback to call with bytes from PNG.
     */
    submit(url, callback) {
      const i = this.i;
      this.callbacks[i] = callback;
      this.i += 1;

      const w = this.nextworker;
      const worker = this.workers[w];
      this.nextworker = (w + 1) % this.workers.length;

      worker.postMessage({'i': i, 'url': url});
    }

    /**
     * Callback for this.worker.
     * @param {*} e Event payload. `data` must contain `i` to know which
     *              callback to refer to and `result` to pass on.
     */
    onmessage(e) {
      const response = e.data;
      const i = response.i;
      const callback = this.callbacks[i];
      delete this.callbacks[i];
      callback(response.result);
    }
  }

  return new GlobalLoadPNGWorker();
})();

/**
 * The vertex shader for rendering a baked MERF scene with ray marching.
 * @const {string}
 */
const rayMarchVertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
uniform mat4 world_T_clip;

void main() {
  vec4 posClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = posClip;

  posClip /= posClip.w;
  vec4 nearPoint = world_T_clip * vec4(posClip.x, posClip.y, -1.0, 1.0);
  vec4 farPoint = world_T_clip * vec4(posClip.x, posClip.y, 1.0, 1.0);

  vOrigin = nearPoint.xyz / nearPoint.w;
  vDirection = normalize(farPoint.xyz / farPoint.w - vOrigin);
}
`;

/**
 * We build the ray marching shader programmatically, this string contains the
 * header for the shader.
 * @const {string}
 */
const rayMarchFragmentShaderHeader = `
precision highp float;

varying vec3 vOrigin;
varying vec3 vDirection;

uniform int displayMode;

uniform mat3 worldspaceROpengl;
uniform float nearPlane;

#ifdef USE_DISTANCE_GRID
uniform highp sampler3D distanceGrid;
uniform highp sampler3D occupancyGrid_L0;
#else
uniform highp sampler3D occupancyGrid_L0;
uniform highp sampler3D occupancyGrid_L1;
uniform highp sampler3D occupancyGrid_L2;
#ifndef USE_BITS
uniform highp sampler3D occupancyGrid_L3;
uniform highp sampler3D occupancyGrid_L4;
#endif
#endif

uniform vec4 bias_0[NUM_CHANNELS_ONE/4];
uniform vec4 bias_1[NUM_CHANNELS_TWO/4];
uniform vec4 bias_2[NUM_CHANNELS_THREE/4];

uniform highp sampler2D weightsZero;
uniform highp sampler2D weightsOne;
uniform highp sampler2D weightsTwo;

#ifdef USE_EXPOSURE
uniform float exposure;
#endif

uniform vec3 atlasSize;

uniform highp sampler3D sparseGridBlockIndices;
uniform highp sampler3D sparseGridDensity;
uniform highp sampler3D sparseGridRgb;
uniform highp sampler3D sparseGridFeatures;

// need to use texture arrays, otherwise we exceed max texture unit limit
uniform highp sampler2DArray planeDensity;
uniform highp sampler2DArray planeRgb;
uniform highp sampler2DArray planeFeatures;
`;

/**
 * Extends a dictionary.
 * @param {!object} obj Dictionary to extend
 * @param {!object} src Dictionary to be written into obj
 * @return {!object} Extended dictionary
 */
function extend(obj, src) {
  for (let key in src) {
    if (src.hasOwnProperty(key)) obj[key] = src[key];
  }
  return obj;
}

/**
 * Reports an error to the user by populating the error div with text.
 * @param {string} text
 */
function error(text) {
  const e = document.getElementById('error');
  e.textContent = text;
  e.style.display = 'block';
}

/**
 * Creates a DOM element that belongs to the given CSS class.
 * @param {string} what
 * @param {string} className
 * @return {!HTMLElement}
 */
function create(what, className) {
  const e = /** @type {!HTMLElement} */ (document.createElement(what));
  if (className) {
    e.className = className;
  }
  return e;
}

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
 * Returns true if the scene is still loading.
 */
function isLoading() {
  const loading = document.getElementById('Loading');
  return loading.style.display !== 'none';
}

/**
 * Executed whenever an image is loaded for updating the loading prompt.
 */
function onImageLoaded() {
  gLoadedTextures++;
  updateLoadingProgress();
}

/**
 * Updates the loading progress HTML elements.
 */
function updateLoadingProgress() {
  let imageProgress = document.getElementById('image-progress');
  const numTexturesString = gNumTextures > 0 ? gNumTextures : '?';
  imageProgress.innerHTML =
      'Loading images: ' + gLoadedTextures + '/' + numTexturesString;
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
  const response = await fetchFn();
  if (response.status === 429) {
    await sleep(500);
    return fetchAndRetryIfNecessary(fetchFn);
  }
  return response;
}

/**
 * Loads PNG image from rgbaURL and decodes it to an Uint8Array.
 * @param {string} rgbaUrl The URL of the PNG image.
 * @return {!Promise<!Uint8Array>}
 */
function loadPNG(rgbaUrl) {
  const result = new Promise((resolve) => {
    loadPNGWorker.submit(rgbaUrl, resolve);
  });
  return result;
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
 * @param {string} url URL of the file to be loaded
 * @return {!Promise<!object>} The parsed JSON file
 */
function loadJSONFile(url) {
  let fetchFn = () => fetch(url, {method: 'GET', mode: 'cors'});
  return fetchAndRetryIfNecessary(fetchFn).then(response => response.json());
}

/**
 * Creates an empty volume texture.
 * @param {number} width Width of the texture
 * @param {number} height Height of the texture
 * @param {number} depth Depth of the texture
 * @param {number} format Format of the texture
 * @param {number} filter Filter strategy of the texture
 * @return {!THREE.DataTexture3D} Volume texture
 */
function createEmptyVolumeTexture(width, height, depth, format, filter) {
  volumeTexture = new THREE.DataTexture3D(null, width, height, depth);
  volumeTexture.format = format;
  volumeTexture.generateMipmaps = false;
  volumeTexture.magFilter = volumeTexture.minFilter = filter;
  volumeTexture.wrapS = volumeTexture.wrapT = volumeTexture.wrapR =
      THREE.ClampToEdgeWrapping;
  volumeTexture.type = THREE.UnsignedByteType;
  return volumeTexture;
}

/**
 * Uploads a volume slice to a pre-existing volume texture.
 * @param {!Uint8Array} volumeSlice Data to be uploaded
 * @param {!THREE.DataTexture3D} texture Volume texture to upload to
 * @param {number} sliceIndex Index of the slice
 * @param {number} volumeWidth Width of the volume
 * @param {number} volumeHeight Height of the volume
 * @param {number} sliceDepth Depth of slice
 */
function uploadVolumeSlice(
    volumeSlice, texture, sliceIndex, volumeWidth, volumeHeight, sliceDepth) {
  const textureProperties = gRenderer['properties'].get(texture);
  let gl = gRenderer.getContext();

  let stride, glFormat;
  if (texture.format == THREE.RGBAFormat) {
    glFormat = gl.RGBA;
    stride = 4;
  } else if (texture.format == THREE.RGBFormat) {
    glFormat = gl.RGB;
    stride = 3;
  } else if (texture.format == THREE.LuminanceAlphaFormat) {
    glFormat = gl.LUMINANCE_ALPHA;
    stride = 2;
  } else if (texture.format == THREE.RedFormat) {
    glFormat = gl.RED;
    stride = 1;
  }

  let oldTexture = gl.getParameter(gl.TEXTURE_BINDING_3D);
  let textureHandle = textureProperties['__webglTexture'];
  gl.bindTexture(gl.TEXTURE_3D, textureHandle);
  // Upload row-by-row to work around bug with Intel + Mac OSX.
  // See https://crbug.com/654258.
  for (let z = 0; z < sliceDepth; ++z) {
    for (let y = 0; y < volumeHeight; ++y) {
      gl.texSubImage3D(
          gl.TEXTURE_3D, 0, 0, y, z + sliceIndex * sliceDepth, volumeWidth, 1,
          1, glFormat, gl.UNSIGNED_BYTE, volumeSlice,
          stride * volumeWidth * (y + volumeHeight * z));
    }
  }
  gl.bindTexture(gl.TEXTURE_3D, oldTexture);
}

/**
 * Loads a volume texture slice-by-slice.
 * @param {!object} uploadFn The texture upload function that is called when a
 *  new volume slice is received.
 * @param {number} volumeWidth Width of the volume
 * @param {number} volumeHeight Height of the volume
 * @param {number} numSlices Number of slices for the volume
 * @param {number} sliceDepth The depth of each slice
 * @param {string} filenamePrefix Prefix for all filenames. The slice index and
 *  the png file ending are appended to this string.
 * @param {?string} filename2Prefix (Optional) Prefix for secondary filenames.
 *  The slice index and the png file ending are appended to this string.
 * @param {string} fileExtension File extension for the slices we're loading.
 * @param {!object} filenameToLinkTranslator
 * @return {!Promise} Resolves when the texture is fully uploaded
 */
function loadVolumeTextureSliceBySlice(
    uploadFn,
    volumeWidth,
    volumeHeight,
    sliceDepth,
    numSlices,
    filenamePrefix,
    filename2Prefix,
    fileExtension,
    filenameToLinkTranslator,
) {
  let uploadPromises = [];

  for (let sliceIndex = 0; sliceIndex < numSlices; sliceIndex++) {
    let url = filenameToLinkTranslator.translate(
        `${filenamePrefix}_${digits(sliceIndex, 3)}.${fileExtension}`);
    let rgbaPromise = loadPNG(url);
    rgbaPromise = rgbaPromise.then(data => {
      onImageLoaded();
      return data;
    });
    let filePromises = [rgbaPromise];

    if (!!filename2Prefix) {
      let url2 = filenameToLinkTranslator.translate(
          `${filename2Prefix}_${digits(sliceIndex, 3)}.${fileExtension}`);
      let featurePromise = loadPNG(url2);
      featurePromise = featurePromise.then(data => {
        onImageLoaded();
        return data;
      });
      filePromises.push(featurePromise);
    }

    let uploadPromise = Promise.all(filePromises)
                            .then(images => {
                              if (!!filename2Prefix) {
                                uploadFn(
                                    images[0], images[1], sliceIndex,
                                    volumeWidth, volumeHeight, sliceDepth);
                              } else {
                                uploadFn(
                                    images[0], sliceIndex, volumeWidth,
                                    volumeHeight, sliceDepth);
                              }
                            })
                            .catch(console.error);
    uploadPromises.push(uploadPromise);
  }
  return Promise.all(uploadPromises);
}

/**
 * Loads a volume texture fully with just one upload call.
 * @param {!object} uploadFn The texture upload function that is called
 *  when the volume data is available as bytes.
 * @param {number} volumeWidth Width of the volume
 * @param {number} volumeHeight Height of the volume
 * @param {number} volumeDepth Depth of the volume
 * @param {string} filenamePrefix Prefix for all filenames. The slice index and
 *  he png file ending are appended to this string.
 * @param {?string} filename2Prefix (Optional) Prefix for secondary filenames.
 *  The slice index and the png file ending are appended to this string.
 * @param {string} fileExtension File extension for the slices we're loading.
 * @param {!object} filenameToLinkTranslator
 * @return {!Promise} Resolves when the texture is fully uploaded
 */
function loadVolumeTextureCompletely(
    uploadFn, volumeWidth, volumeHeight, volumeDepth, filenamePrefix,
    filename2Prefix, fileExtension, filenameToLinkTranslator) {
  let url =
      filenameToLinkTranslator.translate(`${filenamePrefix}.${fileExtension}`);
  let rgbaPromise = loadPNG(url);
  rgbaPromise = rgbaPromise.then(data => {
    onImageLoaded();
    return data;
  });
  let filePromises = [rgbaPromise];

  if (!!filename2Prefix) {
    let url2 = filenameToLinkTranslator.translate(
        `${filename2Prefix}.${fileExtension}`);
    let featurePromise = loadPNG(url2);
    featurePromise = featurePromise.then(data => {
      onImageLoaded();
      return data;
    });
    filePromises.push(featurePromise);
  }

  let uploadPromise =
      Promise.all(filePromises)
          .then(images => {
            if (!!filename2Prefix) {
              uploadFn(
                  images[0], images[1], volumeWidth, volumeHeight, volumeDepth);
            } else {
              uploadFn(images[0], volumeWidth, volumeHeight, volumeDepth);
            }
          })
          .catch(console.error);

  return uploadPromise;
}

/**
 * Creates three equally sized textures to hold triplanes.
 * @param {number} width Width of the texture
 * @param {number} height Height of the texture
 * @param {number} format Format of the texture
 * @return {!THREE.DataTexture2DArray} Texture array of size three
 */
function createEmptyTriplaneTextureArray(width, height, format) {
  let texture = new THREE.DataTexture2DArray(null, width, height, 3);
  texture.format = format;
  texture.generateMipmaps = false;
  texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.type = THREE.UnsignedByteType;
  return texture;
}

/**
 * Translates filenames to links.
 */
class FilenameToLinkTranslator {
  /**
   * Constructor.
   * @param {string} dirUrl The url where scene files are stored.
   * @param {?object} filenameToLink Dictionary that maps internal file names to
   *     download links.
   */
  constructor(dirUrl, filenameToLink) {
    this.dirUrl = dirUrl;
    this.filenameToLink = filenameToLink;
  }

  /**
   * Maps a virtual filename to an URL.
   * @param {string} filename Internal filename.
   * @return {string} Download URL.
   */
  translate(filename) {
    if (this.filenameToLink != null) {
      // Lookup download URL in map.
      return this.filenameToLink[filename];
    } else {
      // Simply preprend directory.
      return this.dirUrl + '/' + filename;
    }
  }
}


/**
 * Loads all assets for a single submodel.
 *
 * @param {!object} sceneParams scene_params.json file.
 * @param {string} viewDependenceNetworkShaderFunctions text of
 *     viewdependency.glsl.
 * @param {string} rayMarchFragmentShaderBody text of fragment.glsl.
 * @param {!FilenameToLinkTranslator} filenameToLinkTranslator object mapping
 *     fake filenames to URLs.
 * @param {number} nearPlane minimum distance from camera to start rendering
 * @returns {!THREE.ShaderMaterial}
 */
function loadSceneFiles(
    sceneParams, viewDependenceNetworkShaderFunctions,
    rayMarchFragmentShaderBody, filenameToLinkTranslator, nearPlane) {
  const fileExtension = sceneParams['export_array_format'] || 'png';

  // Determine which occupancy and/or distance grid files to load.
  // Load occupancy and distancegrids for empty space skipping.
  const resolutionToUse = sceneParams['triplane_resolution'];
  const voxelSizeToUse = sceneParams['triplane_voxel_size'];

  // We OOM on mobile devices if the PNG size is larger than 256^3, so
  // we need to split the larger volumes into these into charts.
  const kMaxNonSlicedVolumeSize = 256;
  const kNumEmptySpaceCharts = 8;

  let occupancyGridBlockSizes = [8, 16, 32, 64, 128];
  if (sceneParams['useBits']) {
    occupancyGridBlockSizes = [4, 16, 64];
  }
  if (sceneParams['useDistanceGrid']) {
    occupancyGridBlockSizes = [4];
  }

  function _loadVolumeTexture(texture, gridSize, filenamePrefix) {
    if (sceneParams['legacyGrids'] || gridSize <= kMaxNonSlicedVolumeSize) {
      gNumTextures++;
      loadVolumeTextureCompletely(
          (volumeData, width, height, depth) => {
            const volumeSize = Math.ceil(width * height * depth);
            for (let j = 0; j < volumeSize; j++) {
              texture.image.data[j] = volumeData[4 * j];
            }
            texture.needsUpdate = true;
          },
          gridSize, gridSize, gridSize, filenamePrefix, null, fileExtension,
          filenameToLinkTranslator);
    } else {
      gNumTextures += kNumEmptySpaceCharts;
      loadVolumeTextureSliceBySlice(
          (sliceData, sliceIndex, width, height, sliceDepth) => {
            const sliceSize = Math.ceil(width * height * sliceDepth);
            const offset = sliceIndex * sliceSize;
            for (let j = 0; j < sliceSize; j++) {
              texture.image.data[offset + j] = sliceData[4 * j];
            }
            texture.needsUpdate = true;
          },
          gridSize, gridSize, Math.ceil(gridSize / kNumEmptySpaceCharts),
          kNumEmptySpaceCharts, filenamePrefix, null, fileExtension,
          filenameToLinkTranslator);
    }
  }

  // Load the occupancy grid and distance grid textures used for empty space
  // skipping.
  let occupancyGridTextures = [];
  let occupancyGridSizes = [];
  let occupancyVoxelSizes = [];
  for (let occupancyGridIndex = 0;
       occupancyGridIndex < occupancyGridBlockSizes.length;
       occupancyGridIndex++) {
    // Assuming width = height = depth which typically holds when employing
    // scene contraction.
    const blockSize = occupancyGridBlockSizes[occupancyGridIndex];
    const gridSize = Math.ceil(resolutionToUse / blockSize);

    let occupancyGridTexture = createEmptyVolumeTexture(
        gridSize, gridSize, gridSize, THREE.RedFormat, THREE.NearestFilter);
    occupancyGridTexture.image.data =
        new Uint8Array(gridSize * gridSize * gridSize);
    _loadVolumeTexture(
        occupancyGridTexture, gridSize,
        'occupancy_grid_' + occupancyGridBlockSizes[occupancyGridIndex]);

    occupancyGridTextures.push(occupancyGridTexture);
    occupancyGridSizes.push(new THREE.Vector3(gridSize, gridSize, gridSize));
    occupancyVoxelSizes.push(voxelSizeToUse * blockSize);
  }

  let distanceGridTexture, distanceGridSize, distanceVoxelSize;
  if (sceneParams['useDistanceGrid']) {
    const blockSize = occupancyGridBlockSizes[0];
    const gridSize = Math.ceil(resolutionToUse / blockSize);

    distanceGridTexture = createEmptyVolumeTexture(
        gridSize, gridSize, gridSize, THREE.RedFormat, THREE.NearestFilter);
    distanceGridTexture.image.data =
        new Uint8Array(gridSize * gridSize * gridSize);
    _loadVolumeTexture(
        distanceGridTexture, gridSize, 'distance_grid_' + blockSize);

    distanceGridSize = new THREE.Vector3(gridSize, gridSize, gridSize);
    distanceVoxelSize = voxelSizeToUse * blockSize;
  }

  // Load the sparse grid's indirection grid.
  let sparseGridBlockIndicesUrl = filenameToLinkTranslator.translate(
      `sparse_grid_block_indices.${fileExtension}`);
  let sparseGridBlockIndicesPromise = loadPNG(sparseGridBlockIndicesUrl);

  // There will be two textures per sparse grid slice: one for
  // sparse_grid_rgb_and_density, one for sparse_grid_features.
  gNumTextures += 2 * sceneParams['num_slices'];

  // Load triplane assets.
  let planePromises = [];
  gNumTextures += 6;  // 3 planes, 2 textures per plane.
  for (let plane_idx = 0; plane_idx < 3; ++plane_idx) {
    // Load plane_rgb_and_density.
    let planeUrl = filenameToLinkTranslator.translate(
        `plane_rgb_and_density_${plane_idx}.${fileExtension}`);
    let planePromise = loadPNG(planeUrl);
    planePromise.then(onImageLoaded);
    planePromises.push(planePromise);

    // Load plane_features.
    planeUrl = filenameToLinkTranslator.translate(
        `plane_features_${plane_idx}.${fileExtension}`);
    planePromise = loadPNG(planeUrl);
    planePromises.push(planePromise);
    planePromise.then(onImageLoaded);
  }

  // Bind promises for all plane assets together.
  planePromises = Promise.all(planePromises);

  updateLoadingProgress();

  // Load sparse grid, note that textures are only allocated and
  // the actual loading is done progressively in loadOnFirstFrame.

  // Create empty volume textures that are later filled slice-by-slice.
  function _createEmptyAtlasVolumeTexture(format) {
    return createEmptyVolumeTexture(
        sceneParams['atlas_width'], sceneParams['atlas_height'],
        sceneParams['atlas_depth'], format, THREE.LinearFilter);
  }

  // All of these textures have the same dimensions and use linear
  // filtering.
  let sparseGridDensityTexture =
      _createEmptyAtlasVolumeTexture(THREE.LuminanceAlphaFormat);
  let sparseGridRgbTexture = _createEmptyAtlasVolumeTexture(THREE.RGBFormat);
  let sparseGridFeaturesTexture =
      _createEmptyAtlasVolumeTexture(THREE.RGBFormat);

  // The indirection grid uses nearest filtering and is loaded in one go.
  let v =
      sceneParams['sparse_grid_resolution'] / sceneParams['data_block_size'];
  let sparseGridBlockIndicesTexture =
      createEmptyVolumeTexture(v, v, v, THREE.RGBFormat, THREE.NearestFilter);

  // Update texture buffer for sparse_grid_block_indices.
  sparseGridBlockIndicesPromise.then(sparseGridBlockIndicesImage => {
    sparseGridBlockIndicesTexture.image.data = sparseGridBlockIndicesImage;
    sparseGridBlockIndicesTexture.needsUpdate = true;
  });

  // Allocate texture buffers for triplanes assets.
  let triplaneResolution = sceneParams['triplane_resolution'];
  let triplaneNumTexels = triplaneResolution * triplaneResolution;
  let triplaneGridSize =
      new THREE.Vector2(triplaneResolution, triplaneResolution);
  let planeDensityTexture = createEmptyTriplaneTextureArray(
      triplaneResolution, triplaneResolution, THREE.RedFormat);
  let planeRgbTexture = createEmptyTriplaneTextureArray(
      triplaneResolution, triplaneResolution, THREE.RGBFormat);
  let planeFeaturesTexture = createEmptyTriplaneTextureArray(
      triplaneResolution, triplaneResolution, THREE.RGBAFormat);

  // Once triplane assets are ready, populate textures.
  planePromises.then(planeData => {
    // Stack all textures into a single array.
    let planeDensityStack = new Uint8Array(3 * triplaneNumTexels);
    let planeRgbStack = new Uint8Array(3 * triplaneNumTexels * 3);
    let planeFeaturesStack = new Uint8Array(3 * triplaneNumTexels * 4);
    for (let plane_idx = 0; plane_idx < 3; plane_idx++) {
      let baseOffset = plane_idx * triplaneNumTexels;
      let planeRgbAndDensity = planeData[2 * plane_idx];
      let planeFeatures = planeData[2 * plane_idx + 1];
      for (let j = 0; j < triplaneNumTexels; j++) {
        planeDensityStack[baseOffset + j] = planeRgbAndDensity[j * 4 + 3];

        planeRgbStack[(baseOffset + j) * 3 + 0] = planeRgbAndDensity[j * 4 + 0];
        planeRgbStack[(baseOffset + j) * 3 + 1] = planeRgbAndDensity[j * 4 + 1];
        planeRgbStack[(baseOffset + j) * 3 + 2] = planeRgbAndDensity[j * 4 + 2];

        planeFeaturesStack[(baseOffset + j) * 4 + 0] = planeFeatures[j * 4 + 0];
        planeFeaturesStack[(baseOffset + j) * 4 + 1] = planeFeatures[j * 4 + 1];
        planeFeaturesStack[(baseOffset + j) * 4 + 2] = planeFeatures[j * 4 + 2];
        planeFeaturesStack[(baseOffset + j) * 4 + 3] = planeFeatures[j * 4 + 3];
      }
    }

    // Update texture buffers.
    planeDensityTexture.image.data = planeDensityStack;
    planeDensityTexture.needsUpdate = true;

    planeRgbTexture.image.data = planeRgbStack;
    planeRgbTexture.needsUpdate = true;

    planeFeaturesTexture.image.data = planeFeaturesStack;
    planeFeaturesTexture.needsUpdate = true;
  });

  // Assemble shader code from header, on-the-fly generated view-dependency
  // functions and body.
  let fragmentShaderSource = rayMarchFragmentShaderHeader;
  fragmentShaderSource += viewDependenceNetworkShaderFunctions;
  fragmentShaderSource += rayMarchFragmentShaderBody;
  fragmentShaderSource =
      rewriteViewDependenceDefinitions(sceneParams, fragmentShaderSource);

  let worldspaceROpengl = new THREE.Matrix3();
  worldspaceROpengl.set(-1, 0, 0, 0, 0, 1, 0, 1, 0);
  let minPosition = new THREE.Vector3(-2.0, -2.0, -2.0);

  // Hard code these values as constants rather than uniforms --- they
  // are not changing between submodels, and hardcoded constants allow
  // optiming shader compilers to do better work.
  fragmentShaderSource = '#define kMinPosition vec3(' +
      Number(minPosition.x).toFixed(10) + ', ' +
      Number(minPosition.y).toFixed(10) + ', ' +
      Number(minPosition.z).toFixed(10) + ')\n' + fragmentShaderSource;

  fragmentShaderSource =
      '#define kStepMult ' + gStepMult + '\n' + fragmentShaderSource;
  fragmentShaderSource = '#define kRangeFeaturesMin ' +
      Number(sceneParams['range_features'][0]).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kRangeFeaturesMax ' +
      Number(sceneParams['range_features'][1]).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kRangeDensityMin ' +
      Number(sceneParams['range_density'][0]).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kRangeDensityMax ' +
      Number(sceneParams['range_density'][1]).toFixed(10) + '\n' +
      fragmentShaderSource;

  // Pass uniforms to the shader. These are the variables that will be
  // manipulated over the course of rendering.
  let rayMarchUniforms = {
    // Neural network weights.
    'bias_0': {'value': null},
    'bias_1': {'value': null},
    'bias_2': {'value': null},
    'weightsZero': {'value': null},
    'weightsOne': {'value': null},
    'weightsTwo': {'value': null},

    // General rendering parameters.
    'displayMode': {'value': gDisplayMode - 0},
    'nearPlane': {'value': nearPlane},
    'minPosition': {'value': minPosition},

    'world_T_clip': {'value': new THREE.Matrix4()},
    'worldspaceROpengl': {'value': worldspaceROpengl},
    'exposure': {'value': gExposure},
  };

  occupancyUniforms = {};
  for (let i = 0; i < occupancyGridTextures.length; ++i) {
    let ri = occupancyGridTextures.length - i - 1;
    fragmentShaderSource = '#define kVoxelSizeOccupancy_L' + ri + ' ' +
        Number(occupancyVoxelSizes[i]).toFixed(10) + '\n' +
        fragmentShaderSource;
    fragmentShaderSource = '#define kGridSizeOccupancy_L' + ri + ' vec3(' +
        Number(occupancyGridSizes[i].x).toFixed(10) + ', ' +
        Number(occupancyGridSizes[i].y).toFixed(10) + ', ' +
        Number(occupancyGridSizes[i].z).toFixed(10) + ')\n' +
        fragmentShaderSource;
    occupancyUniforms['occupancyGrid_L' + ri] = {
      'value': occupancyGridTextures[i]
    };
  }
  rayMarchUniforms = extend(rayMarchUniforms, occupancyUniforms);

  if (sceneParams['useDistanceGrid']) {
    fragmentShaderSource = '#define USE_DISTANCE_GRID\n' + fragmentShaderSource;
    fragmentShaderSource = '#define kVoxelSizeDistance ' +
        Number(distanceVoxelSize).toFixed(10) + '\n' + fragmentShaderSource;
    fragmentShaderSource = '#define kGridSizeDistance vec3(' +
        Number(distanceGridSize.x).toFixed(10) + ', ' +
        Number(distanceGridSize.y).toFixed(10) + ', ' +
        Number(distanceGridSize.z).toFixed(10) + ')\n' + fragmentShaderSource;
    distanceUniforms = {
      'distanceGrid': {'value': distanceGridTexture},
    };
    rayMarchUniforms = extend(rayMarchUniforms, distanceUniforms);
  }

  let backgroundColor = new THREE.Color(0.0, 0.0, 0.0);
  if (sceneParams['backgroundColor']) {
    backgroundColor = new THREE.Color(sceneParams['backgroundColor']);
  }
  fragmentShaderSource = '#define kBackgroundColor vec3(' +
      Number(backgroundColor.r).toFixed(10) + ', ' +
      Number(backgroundColor.g).toFixed(10) + ', ' +
      Number(backgroundColor.b).toFixed(10) + ')\n' + fragmentShaderSource;

  if (gExposure || sceneParams['default_exposure']) {
    if (sceneParams['default_exposure']) {
      gExposure = parseFloat(sceneParams['default_exposure']);
    }
    fragmentShaderSource = '#define USE_EXPOSURE\n' + fragmentShaderSource;
  }

  if (sceneParams['deferred_rendering_mode'] === 'vfr') {
    fragmentShaderSource = '#define USE_VFR\n' + fragmentShaderSource;
  }

  if (sceneParams['merge_features_combine_op'] === 'coarse_sum') {
    fragmentShaderSource =
        '#define USE_FEATURE_CONCAT\n' + fragmentShaderSource;
  }

  if (sceneParams['useBits']) {
    fragmentShaderSource = '#define USE_BITS\n' + fragmentShaderSource;
  }

  if (sceneParams['useLargerStepsWhenOccluded']) {
    fragmentShaderSource =
        '#define LARGER_STEPS_WHEN_OCCLUDED\n' + fragmentShaderSource;
  }

  fragmentShaderSource = '#define kTriplaneVoxelSize ' +
      Number(sceneParams['triplane_voxel_size']).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kTriplaneGridSize vec2(' +
      Number(triplaneGridSize.x).toFixed(10) + ', ' +
      Number(triplaneGridSize.y).toFixed(10) + ')\n' + fragmentShaderSource;
  let triplaneUniforms = {
    'planeDensity': {'value': planeDensityTexture},
    'planeRgb': {'value': planeRgbTexture},
    'planeFeatures': {'value': planeFeaturesTexture},
  };
  rayMarchUniforms = extend(rayMarchUniforms, triplaneUniforms);

  fragmentShaderSource = '#define kDataBlockSize ' +
      Number(sceneParams['data_block_size']).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kSparseGridVoxelSize ' +
      Number(sceneParams['sparse_grid_voxel_size']).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kSparseGridGridSize vec3(' +
      Number(sceneParams['sparse_grid_resolution']).toFixed(10) + ', ' +
      Number(sceneParams['sparse_grid_resolution']).toFixed(10) + ', ' +
      Number(sceneParams['sparse_grid_resolution']).toFixed(10) + ')\n' +
      fragmentShaderSource;
  let sparseGridUniforms = {
    'sparseGridBlockIndices': {'value': sparseGridBlockIndicesTexture},
    'sparseGridDensity': {'value': sparseGridDensityTexture},
    'sparseGridRgb': {'value': sparseGridRgbTexture},
    'sparseGridFeatures': {'value': sparseGridFeaturesTexture},
    'atlasSize': {
      'value': new THREE.Vector3(
          sceneParams['atlas_width'], sceneParams['atlas_height'],
          sceneParams['atlas_depth'])
    },
  };
  rayMarchUniforms = extend(rayMarchUniforms, sparseGridUniforms);

  // Shader editor
  let shaderEditor = document.getElementById('shader-editor');
  shaderEditor.value = fragmentShaderSource;

  // Bundle uniforms, vertex and fragment shader in a material
  let rayMarchMaterial = new THREE.ShaderMaterial({
    uniforms: rayMarchUniforms,
    vertexShader: rayMarchVertexShader,
    fragmentShader: fragmentShaderSource,
    vertexColors: true,
  });
  rayMarchMaterial.side = THREE.DoubleSide;
  rayMarchMaterial.depthTest = false;
  rayMarchMaterial.needsUpdate = true;

  return rayMarchMaterial;
}


/**
 * Loads full scene representation.
 *
 * This includes all submodel assets, including allocation and download.
 *
 * This function should be called exactly once.
 *
 * @param {string} dirUrl Either points to a directory that contains scene files
 *                        or to a json file that maps virtual filenames to
 *                        download links
 * @param {number} width Width of the viewer frame
 * @param {number} height Height of the viewer frame
 * @param {!object} overrideParams A dictionary that contains overrides for the
 *   params in scene_params.json (e.g. combineMode, deferredMode or useBits).
 * @param {number} nearPlane Distance to the near clipping plane
 */
function loadScene(dirUrl, width, height, overrideParams, nearPlane) {
  // Check if dirUrl points to a json file or to a directory.
  if (dirUrl.includes('.json')) {
    // If this is the case, we fetch a JSON file that maps filenames to links.
    filenameToLinkPromise = loadJSONFile(dirUrl);
  } else {
    // Otherwise, the scene files directly lie at dirUrl and we create a
    // dummy promise that resolves immediately.
    filenameToLinkPromise = Promise.resolve(null);
  }

  // This variable is defined in progressive.js.
  gRayMarchScene = [];
  filenameToLinkPromise
      .then(filenameToLink => {
        // Mapping from fake filenames to real filenames under root directory
        // dirUrl.
        const filenameToLinkTranslator =
            new FilenameToLinkTranslator(dirUrl, filenameToLink);

        // Loads scene parameters (voxel grid size, view-dependence MLP).
        const sceneParamsUrl =
            filenameToLinkTranslator.translate('scene_params.json');
        const sceneParamsPromise = loadJSONFile(sceneParamsUrl);
        sceneParamsPromise.catch(error => {
          console.error(`Could not load scene params from: ${
              sceneParamsUrl}, error: ${error}`);
          return;
        });

        if (overrideParams['loadBenchmarkCameras']) {
          loadBenchmarkCameras(filenameToLinkTranslator);
        }

        // Some of the shader code is stored in seperate files.
        const textPromises = Promise.all([
          sceneParamsPromise, loadTextFile('viewdependency.glsl'),
          loadTextFile('fragment.glsl'),
          {filenameToLinkTranslator, filenameToLink}  // carried variables.
        ]);
        return textPromises;
      })
      .then(parsed => {
        // scene_params.json for this scene.
        // Translates filenames to full URLs.
        const [sceneParams, viewdependencyShader, fragmentShader, carry] =
            parsed;

        // Determine if there are multiple submodels or not. If so, figure out
        // how many.
        gUseSubmodel =
            (sceneParams.hasOwnProperty('num_local_submodels') &&
             sceneParams['num_local_submodels'] > 1);
        if (gUseSubmodel) {
          gSubmodelCount = sceneParams['num_local_submodels'];
          gCurrentSubmodel =
              sceneParams['sm_to_params'][sceneParams['submodel_idx']];
        }

        // Get links to scene_params.json files for each submodel.
        let sceneParamsPromises = [];
        for (let si = 0; si < gSubmodelCount; ++si) {
          // Get the submodel ids participating in this scene.
          const submodelId = sceneParams['params_to_sm'][si];
          // Construct path to scene_params.json for this submodel.
          sceneParamsPromises.push(
              loadJSONFile(carry.filenameToLinkTranslator.translate(
                  (gUseSubmodel ?
                       '../sm_' + String(submodelId).padStart(3, '0') + '/' :
                       '') +
                  'scene_params.json')));
        }

        // Wait for all scene_params.json files to be loaded.
        let sceneParamsPromise = Promise.all([
          {
            viewdependencyShader,
            fragmentShader,
            ...carry
          },  // carried variables.
          ...sceneParamsPromises,
        ]);
        return sceneParamsPromise;
      })
      .then(loaded => {
        let [carry, ...submodelSceneParams] = loaded;
        let sceneMaterial = [];
        let submodelFilenameToLinkTranslator = [];
        for (let si = 0; si < submodelSceneParams.length; ++si) {
          const submodelId = submodelSceneParams[si]['params_to_sm'][si];
          // Override the scene params using the URL GET variables.
          submodelSceneParams[si] =
              extend(submodelSceneParams[si], overrideParams);

          // Store the deferred MLP parameters in a global variable.
          gDeferredMlp = submodelSceneParams[si]['deferred_mlp'];

          // Also store the scaling factor for this submodel.
          gSubmodelScale[si] = submodelSceneParams[si]['submodel_scale'];
          gSubmodelScaleFactor[si] = gSubmodelScale[si] /
              Math.cbrt(submodelSceneParams[si]['num_submodels']);

          const mlpName = !!gDeferredMlp['ResampleDense_0/kernel'] ?
              'ResampleDense_' :
              'Dense_';

          // Validate the shapes of the loaded MLP parameters.
          //
          // WARNING: There must be EXACTLY three ResampleDense layers in the
          // DeferredMLP!!
          for (let li = 0; li < 3; li++) {
            let kernelShape = gDeferredMlp[mlpName + li + '/kernel']['shape'];
            let biasShape = gDeferredMlp[mlpName + li + '/bias']['shape'];
            if (mlpName === 'ResampleDense_') {
              let gridSize = kernelShape[1];

              // We assume that all grid dimensions are identical
              console.assert(
                  gridSize === kernelShape[2] && gridSize === kernelShape[3]);

              // We also require the grid shape and the bias shape to match.
              console.assert(
                  kernelShape[0] === biasShape[0] &&
                  kernelShape[1] === biasShape[1] &&
                  kernelShape[2] === biasShape[2] &&
                  kernelShape[3] === biasShape[3]);
            }
          }

          // Build fake filename-to-real-filename translator for this
          // submodel.
          submodelFilenameToLinkTranslator.push(new FilenameToLinkTranslator(
              dirUrl +
                  (gUseSubmodel ?
                       '/../sm_' + String(submodelId).padStart(3, '0') + '/' :
                       ''),
              carry.filenameToLink));

          // Load all assets related to this submodel. This is a blocking
          // operation. This operation will also allocate texture buffers
          // for all assets and populate them ASAP.
          sceneMaterial.push(loadSceneFiles(
              submodelSceneParams[si],
              carry.viewdependencyShader,
              carry.fragmentShader,
              submodelFilenameToLinkTranslator[si],
              nearPlane,
              ));

          // Create a proxy plane to draw on.
          const fullScreenPlane = new THREE.PlaneBufferGeometry(width, height);
          let fullScreenPlaneMesh =
              new THREE.Mesh(fullScreenPlane, sceneMaterial[si]);
          fullScreenPlaneMesh.position.z = -100;
          fullScreenPlaneMesh.frustumCulled = false;

          gRayMarchScene.push(new THREE.Scene());
          gRayMarchScene[si].add(fullScreenPlaneMesh);
          gRayMarchScene[si].autoUpdate = false;
          gLoadOnFirstFrameRan[si] = false;
        }

        // Now that we know the submodel scale we can set the camera pose.
        setupInitialCameraPose(
            dirUrl,
            submodelCenter(
                gCurrentSubmodel, submodelSceneParams[gCurrentSubmodel]));

        // Start rendering ASAP, forcing THREE.js to upload the textures.
        requestAnimationFrame(
            t => update(
                t, dirUrl, submodelFilenameToLinkTranslator,
                submodelSceneParams));
      });
}

/**
 * Initializes the application based on the URL parameters.
 */
function initFromParameters() {
  // HTTP GET query parameters
  const params = new URL(window.location.href).searchParams;

  // Base directory for all assets.
  const dirUrl = params.get('dir');

  // Screen size: <width>,<height>
  const size = params.get('s');

  // Controls platform-specific defaults: phone, low, medium, high. Not
  // const as benchmark=true can override it.
  let quality = params.get('quality');

  // Number of samples per voxel. Increase for slower rendering and fewer
  // artifacts.

  const stepMult = params.get('stepMult');
  if (stepMult) {
    gStepMult = parseInt(stepMult, 10);
  }
  const frameMult = params.get('frameMult');
  if (frameMult) {
    gFrameMult = parseInt(frameMult, 10);
  }

  // Manually specify exposure for exposure-aware models.
  const exposure = params.get('exposure');
  if (exposure) {
    gExposure = parseFloat(exposure);
  }

  // For manually overriding parameters in scene_params.json.
  let overrideParams = {};

  const benchmark = params.get('benchmark');
  if (benchmark && Boolean(benchmark)) {
    overrideParams['loadBenchmarkCameras'] = Boolean(benchmark);
    quality = 'high';

    const sceneNameChunks = dirUrl.split('/').slice(-2);
    setupBenchmarkStats(sceneNameChunks[0] + '_' + sceneNameChunks[1]);
  }

  // snerg, vfr
  const deferredMode = params.get('deferredMode');
  if (deferredMode) {
    overrideParams['deferred_rendering_mode'] = deferredMode;
  }

  // sum, concat_and_sum
  const combineMode = params.get('combineMode');
  if (combineMode && combineMode === 'concat_and_sum') {
    overrideParams['merge_features_combine_op'] = 'coarse_sum';
  }

  // are occupancy grids bitpacked?
  const useBits = params.get('useBits');
  if (useBits && Boolean(useBits)) {
    overrideParams['useBits'] = Boolean(useBits);
  }

  // Use distance grid for calculating step sizes.
  const useDistanceGrid = params.get('useDistanceGrid');
  if (useDistanceGrid && Boolean(useDistanceGrid)) {
    // A scene with a distance grid also has bit packed occupancy grids.
    overrideParams['useDistanceGrid'] = Boolean(useDistanceGrid);
    overrideParams['useBits'] =
        overrideParams['useBits'] || overrideParams['useDistanceGrid'];
  }

  // Load legacy scenes, where the distance & occupancy grids are stored
  // as a single monolithic file.
  const legacyGrids = params.get('legacyGrids');
  if (legacyGrids && Boolean(legacyGrids)) {
    overrideParams['legacyGrids'] = Boolean(legacyGrids);
  }

  // The background color (in hex, e.g. #FF0000 for red) that the scene is
  // rendered on top of. Defaults to black.
  const backgroundColor = params.get('backgroundColor');
  if (backgroundColor) {
    overrideParams['backgroundColor'] = '#' + backgroundColor;
  }

  const usageString =
      'To view a MERF scene, specify the following parameters in the URL:\n' +
      'dir: (Required) The URL to a MERF scene directory.\n' +
      'quality: (Optional) A quality preset (phone, low, medium or high).\n' +
      'mouseMode:  (Optional) How mouse navigation works: "orbit" for object' +
      ' centric scenes, "fps" for large scenes on a device with a mouse, or' +
      ' "map" for large scenes on a touchscreen device.\n' +
      'stepMult:  (Optional) Multiplier on how many steps to take per ray.\n' +
      'frameMult:  (Optional) For benchmarking with vsync on: render ' +
      ' frameMult redudant images per frame.\n' +
      'backgroundColor: (Optional) The background color (in hex, e.g. red is' +
      ' FF0000) the scene is rendered on top of. Defaults to black.\n' +
      'exposure: (Optional, only for large scenes) The exposure value of the' +
      ' virtual camera (shutter_speed_seconds * iso / 1000).\n' +
      'deferredMode (Optional, internal) The deferred rendering mode for' +
      ' view-dependence. Either "snerg" or "vfr".\n' +
      'combineMode (Optional, internal) How to combine the features from' +
      ' the triplanes with the features from the sparse 3D grid. Either ' +
      ' "sum" or "concat_and_sum".\n' +
      'useBits (Optional, internal) If true, use bit packing for a higher' +
      ' resolution occupancy grid.\n' +
      'useDistanceGrid (Optional, internal) If true, use a distance grid for' +
      ' empty space skipping instead of a hierarchy of occupancy grids.\n' +
      'benchmark (Optional) If true, sets quality=high and benchmarks' +
      '  rendertimes for the viewpoints in test_frames.json.\n' +
      's: (Optional) The dimensions as width,height. E.g. 640,360.\n' +
      'vfovy:  (Optional) The vertical field of view of the viewer.\n';

  if (!dirUrl) {
    error('dir is a required parameter.\n\n' + usageString);
    return;
  }

  // Default to filling the browser window, and rendering at full res.
  let frameBufferWidth = window.innerWidth -
      13;  // Body has a padding of 6 + 5px, we have a border of 2px.
  let frameBufferHeight = window.innerHeight - 19;
  let lowResFactor = parseInt(params.get('downscale') || 1, 10);

  if (size) {
    const match = size.match(/([\d]+),([\d]+)/);
    frameBufferWidth = parseInt(match[1], 10);
    frameBufferHeight = parseInt(match[2], 10);
  } else if (quality) {
    // No size specified, clip the max viewport size based on quality.
    if (quality == 'phone') {  // For iPhones.
      frameBufferWidth = Math.min(350, frameBufferWidth);
      frameBufferHeight = Math.min(600, frameBufferHeight);
    } else if (quality == 'low') {  // For laptops with integrated GPUs.
      frameBufferWidth = Math.min(1280, frameBufferWidth);
      frameBufferHeight = Math.min(800, frameBufferHeight);
    } else if (quality == 'medium') {  // For laptops with dicrete GPUs.
      frameBufferWidth = Math.min(1920, frameBufferWidth);
      frameBufferHeight = Math.min(1080, frameBufferHeight);
    }  // else assume quality is 'high' and render at full res.
  }

  // No downscale factor specified, estimate it from the quality setting.
  if (!params.get('downscale') && quality) {
    let maxPixelsPerFrame = frameBufferWidth * frameBufferHeight;
    if (quality == 'phone') {  // For iPhones.
      maxPixelsPerFrame = 300 * 450;
    } else if (quality == 'low') {  // For laptops with integrated GPUs.
      maxPixelsPerFrame = 600 * 250;
    } else if (quality == 'medium') {  // For laptops with dicrete GPUs.
      maxPixelsPerFrame = 1200 * 640;
    }  // else assume quality is 'high' and render at full res.

    while (frameBufferWidth * frameBufferHeight / lowResFactor >
           maxPixelsPerFrame) {
      lowResFactor++;
    }

    console.log('Automatically chose a downscaling factor of ' + lowResFactor);
  }
  overrideParams['useLargerStepsWhenOccluded'] = quality != 'high';

  // Near plane distance in world coordinates.
  const nearPlane = parseFloat(params.get('near') || 0.2);

  // FOV along screen height. Specified in degrees.
  const vfovy = parseFloat(params.get('vfovy') || 40.0);

  // Create container for viewport.
  const view = create('div', 'view');
  setDims(view, frameBufferWidth, frameBufferHeight);
  view.textContent = '';

  const viewSpaceContainer = document.getElementById('viewspacecontainer');
  viewSpaceContainer.style.display = 'inline-block';

  const viewSpace = document.querySelector('.viewspace');
  viewSpace.textContent = '';
  viewSpace.appendChild(view);

  let canvas = document.createElement('canvas');
  view.appendChild(canvas);

  // Add tool for visualizing framerate.
  gStats = Stats();
  gSubmodelPanel = gStats.addPanel(new Stats.Panel('SM', '#0ff', '#002'));
  gSubmodelPanel.update(gCurrentSubmodel);
  viewSpace.appendChild(gStats.dom);
  gStats.dom.style.position = 'absolute';

  // Set up a high performance WebGL context, making sure that anti-aliasing is
  // turned off.
  let gl = canvas.getContext('webgl2');
  gRenderer = new THREE.WebGLRenderer({
    canvas: canvas,
    context: gl,
    powerPreference: 'high-performance',
    alpha: false,
    stencil: false,
    precision: 'mediump',
    depth: false,
    antialias: false,
    desynchronized: true
  });

  // Set up the normal scene used for rendering.
  gCamera = new THREE.PerspectiveCamera(
      vfovy,
      Math.trunc(view.offsetWidth / lowResFactor) /
          Math.trunc(view.offsetHeight / lowResFactor),
      nearPlane, 100.0);
  setupProgressiveRendering(view, lowResFactor);
  gRenderer.autoClear = false;
  gRenderer.setSize(view.offsetWidth, view.offsetHeight);

  // Disable camera controls if we're benchmarking, since both OrbitControls
  // and PointerLockControls take ownership of the camera poses, making it
  // impossible to change programatically.
  if (!benchmark) {
    const mouseMode = params.get('mouseMode');
    setupCameraControls(mouseMode, view);
  }

  loadScene(
      dirUrl, Math.trunc(view.offsetWidth / lowResFactor),
      Math.trunc(view.offsetHeight / lowResFactor), overrideParams, nearPlane);
}

/**
 * Checks whether the WebGL context is valid and the underlying hardware is
 * powerful enough. Otherwise displays a warning.
 * @return {boolean}
 */
function isRendererUnsupported() {
  let loading = document.getElementById('Loading');

  let gl = document.getElementsByTagName('canvas')[0].getContext('webgl2');
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

/**
 * This function needs to run after the first frame has been rendered since we
 * are uploading to textures which only become valid after the first frame has
 * been rendered.
 * @param {string} dirUrl Either points to a directory that contains scene files
 *                        or to a json file that maps virtual filenames to
 *                        download links
 * @param {!FilenameToLinkTranslator} filenameToLinkTranslator
 * @param {!object} sceneParams Holds basic information about the scene like
 * @param {number} index which submodel to use.
 *grid dimensions
 **/
function loadOnFirstFrame(
    dirUrl, filenameToLinkTranslator, sceneParams, index) {
  if (gLoadOnFirstFrameRan[index]) return;

  // Early out if the renderer is not supported.
  if (isRendererUnsupported()) {
    gLoadOnFirstFrameRan[index] = true;
    let loadingContainer = document.getElementById('loading-container');
    loadingContainer.style.display = 'none';
    return;
  }

  // The 3D textures already have been allocated in loadScene. Here we flll
  // them with data slice-by-slice (overlapping download and texture upload).
  function _getUniform(uniformName) {
    return gRayMarchScene[index]
        .children[0]
        .material.uniforms[uniformName]['value'];
  }
  const sparseGridDensityTexture = _getUniform('sparseGridDensity');
  const sparseGridRgbTexture = _getUniform('sparseGridRgb');
  const sparseGridFeaturesTexture = _getUniform('sparseGridFeatures');

  function _loadVolumeTextureSliceBySlice(
      uploadFn, filenamePrefix, filename2Prefix) {
    const fileExtension = sceneParams['export_array_format'] || 'png';
    return loadVolumeTextureSliceBySlice(
        uploadFn, sceneParams['atlas_width'], sceneParams['atlas_height'],
        sceneParams['slice_depth'], sceneParams['num_slices'], filenamePrefix,
        filename2Prefix, fileExtension, filenameToLinkTranslator);
  }

  let uploadFn =
      (rgbaImage, featureImage, sliceIndex, volumeWidth, volumeHeight,
       sliceDepth) => {
        // The rgba image RGB channels hold RGB and the png's alpha channel
        // holds density. The feature RGBA channels are directly interpreted
        // as four feature channels and the A channel is also used as a
        // weight. We split apart rgba A channel for density and the feature A
        // channel for weight so we can seperately query these quantities
        // directly.
        let densityPixels =
            new Uint8Array(volumeWidth * volumeHeight * sliceDepth * 2);
        let rgbPixels =
            new Uint8Array(volumeWidth * volumeHeight * sliceDepth * 3);
        let featurePixels =
            new Uint8Array(volumeWidth * volumeHeight * sliceDepth * 3);
        for (let j = 0; j < volumeWidth * volumeHeight * sliceDepth; j++) {
          densityPixels[j * 2 + 0] = rgbaImage[j * 4 + 3];
          densityPixels[j * 2 + 1] = featureImage[j * 4 + 3];

          rgbPixels[j * 3 + 0] = rgbaImage[j * 4 + 0];
          rgbPixels[j * 3 + 1] = rgbaImage[j * 4 + 1];
          rgbPixels[j * 3 + 2] = rgbaImage[j * 4 + 2];

          featurePixels[j * 3 + 0] = featureImage[j * 4 + 0];
          featurePixels[j * 3 + 1] = featureImage[j * 4 + 1];
          featurePixels[j * 3 + 2] = featureImage[j * 4 + 2];
        }
        uploadVolumeSlice(
            densityPixels, sparseGridDensityTexture, sliceIndex, volumeWidth,
            volumeHeight, sliceDepth);
        uploadVolumeSlice(
            rgbPixels, sparseGridRgbTexture, sliceIndex, volumeWidth,
            volumeHeight, sliceDepth);
        uploadVolumeSlice(
            featurePixels, sparseGridFeaturesTexture, sliceIndex, volumeWidth,
            volumeHeight, sliceDepth);
      };
  let sparseGridTexturePromise = _loadVolumeTextureSliceBySlice(
      uploadFn, 'sparse_grid_rgb_and_density', 'sparse_grid_features');

  let allTexturesPromise = Promise.all([sparseGridTexturePromise]);
  allTexturesPromise.catch(errors => {
    console.error(
        'Could not load scene from: ' + dirUrl + ', errors:\n\t' + errors[0] +
        '\n\t' + errors[1] + '\n\t' + errors[2] + '\n\t' + errors[3]);
  });

  allTexturesPromise.then(texture => {
    hideLoading();
    console.log('Successfully loaded scene from: ' + dirUrl);
  });

  gLoadOnFirstFrameRan[index] = true;
}

function positionToSubmodel(pos, sceneParams) {
  if (gUseSubmodel == false) {
    return 0;
  }
  if (gSubmodelForceIndex >= 0) {
    return gSubmodelForceIndex;
  }
  let fixed_pos = new THREE.Vector3(-pos.x, pos.z, pos.y);
  let voxel_resolution = 2 / sceneParams['submodel_voxel_size'];
  let x_grid = fixed_pos.addScalar(1.0).divideScalar(2.0);
  x_grid = x_grid.multiplyScalar(voxel_resolution);
  let x_floor = x_grid.floor().clampScalar(0, voxel_resolution - 1);

  const linear_index =
      (x_floor.x * voxel_resolution + x_floor.y) * voxel_resolution + x_floor.z;
  return sceneParams['sm_to_params'][linear_index];
}

function submodelCenter(submodelId, sceneParams) {
  if (gUseSubmodel == false) {
    return new THREE.Vector3(0.0, 0.0, 0.0);
  }

  /* The submodels are ordered through z, y then x from negative to positive */
  let submodelVoxelSize = sceneParams['submodel_voxel_size'];
  let voxel_resolution = 2 / submodelVoxelSize;

  let submodelIndex = sceneParams['params_to_sm'][submodelId];
  let z_index = submodelIndex % voxel_resolution;
  let y_index =
      ((submodelIndex - z_index) / voxel_resolution) % voxel_resolution;
  let x_index =
      ((submodelIndex - z_index - y_index * voxel_resolution) /
       voxel_resolution / voxel_resolution);

  /* reorder for coordinate systems */
  x_index = voxel_resolution - 1 - x_index;
  [y_index, z_index] = [z_index, y_index];

  return new THREE.Vector3(
      (x_index + 0.5) * submodelVoxelSize - 1.0,
      (y_index + 0.5) * submodelVoxelSize - 1.0,
      (z_index + 0.5) * submodelVoxelSize - 1.0);
}

function submodelTransform(submodelId, sceneParams) {
  const submodel_position = submodelCenter(submodelId, sceneParams);
  const submodel_scale = sceneParams['submodel_scale'];

  let submodel_scale_matrix = new THREE.Matrix4();
  submodel_scale_matrix.makeScale(
      submodel_scale, submodel_scale, submodel_scale);
  let submodel_translate_matrix = new THREE.Matrix4();
  submodel_translate_matrix.makeTranslation(
      -submodel_position.x, -submodel_position.y, -submodel_position.z);
  submodel_matrix = new THREE.Matrix4();
  submodel_matrix.multiplyMatrices(
      submodel_scale_matrix, submodel_translate_matrix);

  return submodel_matrix;
}

/**
 * The main update function that gets called every frame.
 * @param {number} t elapsed time between frames (ms).
 * @param {string} dirUrl Either points to a directory that contains scene files
 *                        or to a json file that maps virtual filenames to
 *                        download links.
 * @param {!FilenameToLinkTranslator} filenameToLinkTranslator
 * @param {!object} sceneParams Holds basic information about the scene like
 *     grid dimensions.
 */
function update(t, dirUrl, filenameToLinkTranslators, sceneParams) {
  gCurrentSubmodel = positionToSubmodel(gCamera.position, sceneParams[0]);
  loadOnFirstFrame(
      dirUrl, filenameToLinkTranslators[gCurrentSubmodel],
      sceneParams[gCurrentSubmodel], gCurrentSubmodel);

  for (let i = 0; i < gFrameMult; ++i) {
    gSubmodelTransform =
        submodelTransform(gCurrentSubmodel, sceneParams[gCurrentSubmodel]);
    gSubmodelPanel.update(gCurrentSubmodel);

    updateCameraControls();
    // For benchmarking, we want to direcly set the projection matrix.
    if (!gBenchmark) {
      gCamera.updateProjectionMatrix();
    }
    gCamera.updateMatrixWorld();

    const currentSubmodelCenter =
        submodelCenter(gCurrentSubmodel, sceneParams[gCurrentSubmodel]);
    const submodelScale = gSubmodelScale[gCurrentSubmodel];
    let submodelCameraPosition = new THREE.Vector3().copy(gCamera.position);
    submodelCameraPosition.sub(currentSubmodelCenter);
    submodelCameraPosition.multiplyScalar(submodelScale);

    let shaderUniforms =
        gRayMarchScene[gCurrentSubmodel].children[0].material.uniforms;


    // Make sure to free up GPU memory from the previous frames.
    if (!!shaderUniforms['weightsZero']['value']) {
      shaderUniforms['weightsZero']['value'].dispose();
    }
    if (!!shaderUniforms['weightsOne']['value']) {
      shaderUniforms['weightsOne']['value'].dispose();
    }
    if (!!shaderUniforms['weightsTwo']['value']) {
      shaderUniforms['weightsTwo']['value'].dispose();
    }

    shaderUniforms['bias_0']['value'] =
        trilerpDeferredMlpBiases(gCurrentSubmodel, 0, submodelCameraPosition);
    shaderUniforms['bias_1']['value'] =
        trilerpDeferredMlpBiases(gCurrentSubmodel, 1, submodelCameraPosition);
    shaderUniforms['bias_2']['value'] =
        trilerpDeferredMlpBiases(gCurrentSubmodel, 2, submodelCameraPosition);

    shaderUniforms['weightsZero']['value'] =
        trilerpDeferredMlpKernel(gCurrentSubmodel, 0, submodelCameraPosition);
    shaderUniforms['weightsOne']['value'] =
        trilerpDeferredMlpKernel(gCurrentSubmodel, 1, submodelCameraPosition);
    shaderUniforms['weightsTwo']['value'] =
        trilerpDeferredMlpKernel(gCurrentSubmodel, 2, submodelCameraPosition);


    renderProgressively();
  }
  gStats.update();

  // By default we schedule the next frame ASAP, but the benchmark mode can
  // override this by replacing this lambda.
  let scheduleNextFrame = () => {
    requestAnimationFrame(
        t => update(t, dirUrl, filenameToLinkTranslators, sceneParams));
  };
  if (gBenchmark) {
    scheduleNextFrame = benchmarkPerformance(scheduleNextFrame);
  }
  scheduleNextFrame();
}

/**
 * Starts the volumetric scene viewer application.
 */
function start() {
  initFromParameters();
  addHandlers();
}

window.onload = start;
