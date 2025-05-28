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
 * @fileoverview Utilities for fetching assets.
 */


/**
 * Fetch a set of occupancy grids.
 * @param {*} spec
 * @param {*} router
 * @returns
 */
function fetchScene(spec, router) {
  return {
    ...spec,
    occupancyGridsAsset: fetchAsset(spec.occupancyGridsSpec, router),
    distanceGridsAsset: fetchAsset(spec.distanceGridsSpec, router),
    triplaneAsset: fetchAsset(spec.triplaneSpec, router),
    sparseGridAsset: fetchAsset(spec.sparseGridSpec, router),
  };
}


/**
 * Fetch a set of occupancy grids.
 * @param {*} spec
 * @param {*} router
 * @returns
 */
function fetchOccupancyGrids(spec, router) {
  let gridAssets =
      spec.gridSpecs.map((gridSpec) => fetchAsset(gridSpec, router));
  return {...spec, gridAssets: gridAssets};
}


/**
 * Fetch assets for a sliced grid.
 * @param {*} spec
 * @param {*} router
 * @returns
 */
function fetchSlices(spec, router) {
  let sliceAssets =
      spec.sliceSpecs.map((sliceSpec) => fetchAsset(sliceSpec, router));
  return {...spec, sliceAssets: sliceAssets};
}


/**
 * Fetch triplane representation.
 * @param {*} spec
 * @param {*} router
 * @returns
 */
function fetchTriplane(spec, router) {
  let result = {...spec, featuresAsset: fetchAsset(spec.featuresSpec, router)};
  if (spec.separateRgbAndDensity) {
    result.rgbAsset = fetchAsset(spec.rgbSpec, router);
    result.densityAsset = fetchAsset(spec.densitySpec, router);
  } else {
    result.rgbAndDensityAsset = fetchAsset(spec.rgbAndDensitySpec, router);
  }
  return result;
}


/**
 * Fetch a flat, monolithic array.
 * @param {*} spec
 * @param {*} router
 * @returns
 */
function fetchArray(spec, router) {
  onImageFetch();
  const validateSize = (x) => {
    console.assert(x.length == product(spec.shape) * spec.numChannels, spec, x);
    return x;
  };
  const url = router.translate(spec.filename);
  const asset = loadAsset(url).then(validateSize).then(onImageLoaded);
  return {...spec, asset: asset};
}

/**
 * Fetches sparse grid assets.
 *
 * @param {*} spec
 * @param {*} router
 * @returns
 */
function fetchSparseGrid(spec, router) {
  let result = {
    ...spec,
    blockIndicesAsset: fetchAsset(spec.blockIndicesSpec, router),
    featuresAsset: fetchAsset(spec.featuresSpec, router),
  };
  if (spec.separateRgbAndDensity) {
    result.rgbAsset = fetchAsset(spec.rgbSpec, router);
    result.densityAsset = fetchAsset(spec.densitySpec, router);
  } else {
    result.rgbAndDensityAsset = fetchAsset(spec.rgbAndDensitySpec, router);
  }
  return result;
}


/**
 * Report that no fetch function is available.
 */
function notImplementedError(spec, router) {
  console.error(`${spec.assetType} is not yet implemented`, spec);
}

const gFetchRegistry = {
  'scene': fetchScene,

  // triplane
  'triplane': fetchTriplane,

  'triplane_rgb_and_density_slices': fetchSlices,
  'triplane_rgb_and_density_slice': fetchArray,

  'triplane_rgb_slices': fetchSlices,
  'triplane_rgb_slice': fetchArray,

  'triplane_density_slices': fetchSlices,
  'triplane_density_slice': fetchArray,

  'triplane_features_slices': fetchSlices,
  'triplane_features_slice': fetchArray,

  // distance grids
  'distance_grids': fetchOccupancyGrids,
  'distance_grid_slices': fetchSlices,
  'distance_grid_slice': fetchArray,

  // occupancy grids
  'occupancy_grids': fetchOccupancyGrids,
  'occupancy_grid_slices': fetchSlices,
  'occupancy_grid_slice': fetchArray,

  // sparse grid
  'sparse_grid': fetchSparseGrid,
  'sparse_grid_block_indices': fetchArray,

  'sparse_grid_rgb_and_density_slices': fetchSlices,
  'sparse_grid_rgb_and_density_slice': fetchArray,

  'sparse_grid_rgb_slices': fetchSlices,
  'sparse_grid_rgb_slice': fetchArray,

  'sparse_grid_density_slices': fetchSlices,
  'sparse_grid_density_slice': fetchArray,

  'sparse_grid_features_slices': fetchSlices,
  'sparse_grid_features_slice': fetchArray,
};


function fetchAsset(spec, router) {
  let fetchFn = gFetchRegistry[spec.assetType];
  if (fetchFn == undefined) {
    console.error(
        `Failed to find fetchFn for assetType ${spec.assetType}`, spec);
  }
  return fetchFn(spec, router);
}
