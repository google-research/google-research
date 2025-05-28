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
 * @fileoverview Miscellaneous utilities for the webviewer.
 */

/**
 * Number of textures being fetched.
 * @type {number}
 */
let gNumTextures = 0;

/**
 * Number of already loaded textures.
 * @type {number}
 */
let gLoadedTextures = 0;

/**
 * Allows forcing specific submodel for debugging
 */
let gSubmodelForceIndex = -1;


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


function setupViewport(width, height) {
  gViewportDims = [width, height];
}


/**
 * Equivalent to range(n) in Python.
 */
function range(n) {
  return [...Array(n).keys()];
}


/**
 * Product of a set of numbers.
 * @param {array} xs
 * @return {number}
 */
function product(xs) {
  result = 1;
  for (let x of xs) {
    result *= x;
  }
  return result;
}


/**
 * Sum of a set of numbers
 */
function sum(xs) {
  result = 1;
  for (let x of xs) {
    result += x;
  }
  return result;
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


/** Show the loading prompt */
function showLoading() {
  let loading = document.getElementById('Loading');
  loading.style.display = 'block';

  let loadingContainer = document.getElementById('loading-container');
  loadingContainer.style.display = 'block';
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
function onImageFetch(value) {
  gNumTextures++;
  updateLoadingProgress();
  return value;
}

/**
 * Executed whenever an image is loaded for updating the loading prompt.
 */
function onImageLoaded(value) {
  gLoadedTextures++;
  updateLoadingProgress();
  return value;
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
 * Returns a promise that fires within a specified amount of time. Can be used
 * in an asynchronous function for sleeping.
 * @param {number} milliseconds Amount of time to sleep
 * @return {!Promise}
 */
function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}


/**
 * Given a submodel index, returns path to its scene_params.json file.
 *
 * @param {number} submodelId Submodel index.
 * @param {string} assetName Optional filename.
 * @return {string} Path to submodel assets.
 */
function submodelAssetPath(submodelId, assetName) {
  let prefix = '';
  if (gUseSubmodel) {
    const smIdx = String(submodelId).padStart(3, '0');
    prefix = `../sm_${smIdx}`;
    if (assetName == undefined) {
      return prefix;
    }
    return `${prefix}/${assetName}`;
  }
  return assetName;
}


/**
 * Determines appropriate submodel index for a position in world coordinates.
 */
function positionToSubmodel(xyz, sceneParams) {
  if (gUseSubmodel == false) {
    return 0;
  }
  if (gSubmodelForceIndex >= 0) {
    return gSubmodelForceIndex;
  }
  let fixed_xyz = new THREE.Vector3(-xyz.x, xyz.z, xyz.y);
  let voxel_resolution = 2 / sceneParams['submodel_voxel_size'];
  let x_grid = fixed_xyz.addScalar(1.0).divideScalar(2.0);
  x_grid = x_grid.multiplyScalar(voxel_resolution);
  let x_floor = x_grid.floor().clampScalar(0, voxel_resolution - 1);

  const linear_index =
      (x_floor.x * voxel_resolution + x_floor.y) * voxel_resolution + x_floor.z;
  return sceneParams['sm_to_params'][linear_index];
}


/**
 * Computes center of submodel in world coordinates.
 */
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



/**
 * Creates transform matrix from world coordinates to submodel coordinates.
 */
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
 * Loads binary asset from rgbaURL and decodes it to an Uint8Array.
 * @param {string} rgbaUrl The URL of the asset image.
 * @return {!Promise<!Uint8Array>}
 */
function loadAsset(rgbaUrl) {
  const result = new Promise((resolve) => {
    gLoadAssetsWorker.submit({url: rgbaUrl}, resolve);
  });
  return result;
}


/**
 * Merge slices into a single array with a web worker.
 */
function mergeSlices(asset, src, dst) {
  // Wait for all assets to arrive.
  let promises = asset.sliceAssets.map((sliceAsset) => sliceAsset.asset);

  // Nearly all calls to this function merge a list of assets sliced along the
  // depth dimension. The only exception to this is sparse grid density, which
  // must merge from >1 sources.

  let result = Promise.all(promises).then((rawAssets) => {
    // Replace promises with their actual values
    let rawSliceAssets = range(rawAssets.length).map((i) => {
      return {
        ...asset.sliceAssets[i],
        asset: rawAssets[i],
      };
    });
    // Forward request to worker.
    let rawAsset = {...asset, sliceAssets: rawSliceAssets};
    let request = {asset: rawAsset, src: src, dst: dst, fn: 'mergeSlices'};
    return new Promise((resolve) => {
      gCopySliceWorker.submit(request, resolve);
    });
  });

  return result;
}


/**
 * Merge slices of sparse grid density into a single array.
 */
function mergeSparseGridDensity(asset) {
  // Wait for all assets to arrive.
  let getAssetPromises = (assetSlices) =>
      Promise.all(assetSlices.sliceAssets.map((sliceAsset) => sliceAsset.asset));
  let rgbAndDensityPromise = getAssetPromises(asset.rgbAndDensityAsset);
  let featuresPromise = getAssetPromises(asset.featuresAsset);
  let promises = [rgbAndDensityPromise, featuresPromise];

  // Nearly all calls to this function merge a list of assets sliced along the
  // depth dimension. The only exception to this is sparse grid density, which
  // must merge from >1 sources.

  let result = Promise.all(promises).then((result) => {
    let rawRgbAndDensitySliceAssets = result[0];
    let rawFeaturesSliceAssets = result[1];

    // Replace promises with their actual values
    let reassembleSliceAssets = (originalSliceAsset, rawSliceAssets) => {
      let numSliceAssets = rawSliceAssets.length;
      let sliceAssets = range(numSliceAssets).map((i) => {
        return {...originalSliceAsset.sliceAssets[i], asset: rawSliceAssets[i]};
      });
      return {...originalSliceAsset, sliceAssets: sliceAssets};
    };
    let rawRgbAndDensityAsset = reassembleSliceAssets(
        asset.rgbAndDensityAsset, rawRgbAndDensitySliceAssets);
    let rawFeaturesAsset =
        reassembleSliceAssets(asset.featuresAsset, rawFeaturesSliceAssets);

    // Forward request to worker.
    let rawAsset = {
      assetType: asset.assetType,
      rgbAndDensityAsset: rawRgbAndDensityAsset,
      featuresAsset: rawFeaturesAsset,
    };
    let request = {
      asset: rawAsset,
      fn: 'mergeSparseGridDensity',
    };
    return new Promise((resolve) => {
      gCopySliceWorker.submit(request, resolve);
    });
  });

  return result;
}


/**
 * Get a field's value or return a default value.
 */
function getFieldOrDefault(obj, field, default_) {
  let result = obj[field];
  if (result == undefined) {
    return default_;
  }
  return result;
}


/**
 * Loads a text file.
 * @param {string} url URL of the file to be loaded.
 * @return {!Promise<string>}
 */
function loadTextFile(url) {
  let fetchFn = () => fetch(url, {method: 'GET', mode: 'cors'});
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
 * Translates filenames to links.
 */
class Router {
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


/** Format of a texture */
const Format = {
  RED: { numChannels: 1 },
  LUMINANCE_ALPHA: { numChannels: 2 },
  RGB: { numChannels: 3 },
  RGBA: { numChannels: 4 },
};

/** Where to copy inputs from */
const GridTextureSource = {
  RGBA_FROM_RGBA: {format: Format.RGBA, channels: [0, 1, 2, 3]},
  RGB_FROM_RGBA: {format: Format.RGBA, channels: [0, 1, 2]},
  RGB_FROM_RGB: {format: Format.RGB, channels: [0, 1, 2]},
  ALPHA_FROM_RGBA: {format: Format.RGBA, channels: [3]},
  RED_FROM_RED: {format: Format.RED, channels: [0]},
  LA_FROM_LUMINANCE_ALPHA: {format: Format.LUMINANCE_ALPHA, channels: [0, 1]},
};


/** Where to copy outputs to. **/
const GridTextureDestination = {
  RED_IN_RED: { format: Format.RED, channels: [0]},
  RGB_IN_RGB: { format: Format.RGB, channels: [0, 1, 2] },
  RGBA_IN_RGBA: { format: Format.RGBA, channels: [0, 1, 2, 3] },
  LA_IN_LUMINANCE_ALPHA: { format: Format.LUMINANCE_ALPHA, channels: [0, 1] },
  LUMINANCE_IN_LUMINANCE_ALPHA: { format: Format.LUMINANCE_ALPHA, channels: [0] },
  ALPHA_IN_LUMINANCE_ALPHA: { format: Format.LUMINANCE_ALPHA, channels: [1] },
};
