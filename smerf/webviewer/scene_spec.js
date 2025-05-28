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
 * @fileoverview Description of this file.
 */


function createSceneSpec(sceneParams) {
  return {
    assetType: 'scene',
    occupancyGridsSpec: createOccupancyGridsSpec(sceneParams),
    distanceGridsSpec: createDistanceGridsSpec(sceneParams),
    triplaneSpec: createTriplaneSpec(sceneParams),
    sparseGridSpec: createSparseGridSpec(sceneParams),
  };
}

function createOccupancyGridsSpec(sceneParams) {
  // Decide on which resolutions to load.
  let blockSizes = [8, 16, 32, 64, 128];
  if (sceneParams['useBits']) {
    blockSizes = [8, 32, 128];
  }
  if (sceneParams['useDistanceGrid']) {
    blockSizes = [8];
  }

  // Create one spec per occupancy grid.
  let occupancyGridSpecs = blockSizes.map(blockSize => {
    return createOccupancyGridSpec(sceneParams, 'occupancy_grid', blockSize);
  })

  // Create one spec for all occupancy grids.
  return {
    assetType: 'occupancy_grids',
    gridSpecs: occupancyGridSpecs,
    blockSizes: blockSizes
  };
}

function createDistanceGridsSpec(sceneParams) {
  if (!sceneParams['useDistanceGrid']) {
    return {assetType: 'distance_grids', gridSpecs: [], blockSizes: []};
  }

  let blockSize = 8;

  // Create one spec for all occupancy grids.
  return {
    assetType: 'distance_grids',
    gridSpecs: [createOccupancyGridSpec(sceneParams, 'distance_grid', blockSize)],
    blockSizes: [blockSize],
  };

}

function createSparseGridSpec(sceneParams) {
  const fileExtension = sceneParams['export_array_format'] || 'png';

  // Are RGB and density stored as two separate images?
  let separateRgbAndDensity = getFieldOrDefault(
      sceneParams, 'export_store_rgb_and_density_separately', false);

  let result = {
    assetType: 'sparse_grid',
    blockIndicesSpec: createSparseGridBlockIndicesSpec(sceneParams),
    separateRgbAndDensity: separateRgbAndDensity,
  };

  if (separateRgbAndDensity) {
    result.rgbSpec =
        createSparseGridAssetSpec(sceneParams, 'sparse_grid_rgb', 3);
    result.densitySpec =
        createSparseGridAssetSpec(sceneParams, 'sparse_grid_density', 2);
    // If RGB and density are stored separately, the final channel from features
    // is omitted. It's stored as the second channel in density instead.
    result.featuresSpec =
        createSparseGridAssetSpec(sceneParams, 'sparse_grid_features', 3);
  } else {
    result.rgbAndDensitySpec = createSparseGridAssetSpec(
        sceneParams, 'sparse_grid_rgb_and_density', 4);
    result.featuresSpec =
        createSparseGridAssetSpec(sceneParams, 'sparse_grid_features', 4);
  }

  return result;
}


function createSparseGridBlockIndicesSpec(sceneParams) {
  const fileExtension = sceneParams['export_array_format'] || 'png';
  let gridSize =
      sceneParams['sparse_grid_resolution'] / sceneParams['data_block_size'];
  return {
    assetType: 'sparse_grid_block_indices',
    filename: `sparse_grid_block_indices.${fileExtension}`,
    shape: [gridSize, gridSize, gridSize],
    numChannels: 3,
  };
}


function createSparseGridAssetSpec(sceneParams, prefix, numChannels) {
  const fileExtension = sceneParams['export_array_format'] || 'png';
  let numSlices = sceneParams['num_slices'];
  let width = sceneParams['atlas_width'];
  let height = sceneParams['atlas_height'];
  let depth = sceneParams['atlas_depth'];
  let sliceDepth = Math.ceil(depth / numSlices);

  // Create a spec for each slice.
  let sliceSpecs = [];
  for (let i = 0; i < numSlices; ++i) {
    const sliceIndex = digits(i, 3);
    filename = `${prefix}_${sliceIndex}.${fileExtension}`;
    sliceSpecs.push({
      assetType: `${prefix}_slice`,
      shape: [width, height, sliceDepth],
      numChannels: numChannels,
      sliceIndex: i,
      numSlices: numSlices,
      filename: filename,
    });
  }

  // Create a spec for all slices.
  return {
    assetType: `${prefix}_slices`,
    shape: [width, height, depth],
    numChannels: numChannels,
    sliceSpecs: sliceSpecs,
    numSlices: numSlices,
    mergeSlices: getMergeSlices(sceneParams),
  };
}


/**
 * Creates a spec for a potentially-sliced grid texture.
 */
function createOccupancyGridSpec(sceneParams, prefix, blockSize) {
  // 3D grids with a resolution higher than 256^3 are split into 8 sliced along
  // the depth dimension.
  const kMaxNonSlicedVolumeSize = 256;
  const kNumEmptySpaceCharts = 8;

  const fileExtension = sceneParams['export_array_format'] || 'png';
  const resolutionToUse = sceneParams['triplane_resolution'];
  const voxelSizeToUse = sceneParams['triplane_voxel_size'];

  // Number of voxels in each dimension.
  const gridSize = Math.ceil(resolutionToUse / blockSize);

  // Side-length of a occupancy grid voxel in squash coordinates.
  const voxelSize = voxelSizeToUse * blockSize;

  // Determine if grid is sliced or not.
  let exportSlicedGrids = getFieldOrDefault(
      sceneParams, 'export_slice_occupancy_and_distance_grids', true);
  const isSliced = (exportSlicedGrids && gridSize > kMaxNonSlicedVolumeSize);

  // Determine the number of color channels in the grid.
  let isPadded = getFieldOrDefault(
      sceneParams, 'export_pad_occupancy_and_distance_grids', true);
  let numChannels = isPadded ? 4 : 1;

  let sliceSpecs = [];
  if (sceneParams['legacyGrids'] || !isSliced) {
    // Grid is contained entirely within one file.
    let filename = `${prefix}_${blockSize}.${fileExtension}`;
    if (!sceneParams['legacyGrids']) {
      filename = `${prefix}_${blockSize}_000.${fileExtension}`;
    }
    sliceSpecs.push({
      assetType: `${prefix}_slice`,
      shape: [gridSize, gridSize, gridSize],
      numChannels: numChannels,
      sliceIndex: 0,
      numSlices: 1,
      filename: filename,
    });
  } else {
    // Grid is split across several different files.
    const sliceDepth = Math.ceil(gridSize / kNumEmptySpaceCharts);

    // Create a spec for each slice.
    for (let i = 0; i < kNumEmptySpaceCharts; ++i) {
      const sliceIndex = digits(i, 3);
      let filename = `${prefix}_${blockSize}_${sliceIndex}.${fileExtension}`;
      sliceSpecs.push({
        assetType: `${prefix}_slice`,
        shape: [gridSize, gridSize, sliceDepth],
        numChannels: numChannels,
        sliceIndex: i,
        numSlices: kNumEmptySpaceCharts,
        filename: filename,
      });
    }
  }

  // Create a spec for all slices.
  return {
    assetType: `${prefix}_slices`,
    shape: [gridSize, gridSize, gridSize],
    numChannels: numChannels,
    voxelSize: voxelSize,
    blockSize: blockSize,
    sliceSpecs: sliceSpecs,
    numSlices: kNumEmptySpaceCharts,
    mergeSlices: getMergeSlices(sceneParams),
  };
}

function createTriplaneSpec(sceneParams) {
  const gridSize = sceneParams['triplane_resolution'];
  const voxelSize = sceneParams['triplane_voxel_size'];
  let separateRgbAndDensity = getFieldOrDefault(
      sceneParams, 'export_store_rgb_and_density_separately', false);
  let result = {
    assetType: 'triplane',
    shape: [gridSize, gridSize, 3],
    numSlices: 3,
    voxelSize: voxelSize,
    separateRgbAndDensity: separateRgbAndDensity,
    featuresSpec: createTriplaneSlicesSpec(sceneParams, 'triplane_features', 4),
  };
  if (result.separateRgbAndDensity) {
    result.rgbSpec = createTriplaneSlicesSpec(sceneParams, 'triplane_rgb', 3);
    result.densitySpec =
        createTriplaneSlicesSpec(sceneParams, 'triplane_density', 1);
  } else {
    result.rgbAndDensitySpec =
        createTriplaneSlicesSpec(sceneParams, 'triplane_rgb_and_density', 4);
  }
  return result;
}

function createTriplaneSlicesSpec(sceneParams, prefix, numChannels) {
  const gridSize = sceneParams['triplane_resolution'];
  return {
    assetType: `${prefix}_slices`,
    shape: [gridSize, gridSize, 3],
    numChannels: numChannels,
    numSlices: 3,
    mergeSlices: getMergeSlices(sceneParams),
    sliceSpecs: range(3).map(
        (i) => createPlaneSliceSpec(sceneParams, prefix, numChannels, i)),
  };
}


function createPlaneSliceSpec(sceneParams, prefix, numChannels, sliceIndex) {
  const fileExtension = sceneParams['export_array_format'] || 'png';
  const gridSize = sceneParams['triplane_resolution'];
  // Filenames start with "plane", not "triplane".
  let filenamePrefix = prefix.replace(/^triplane_/, 'plane_');
  return {
    assetType: `${prefix}_slice`,
    shape: [gridSize, gridSize, 1],
    numChannels: numChannels,
    sliceIndex: sliceIndex,
    numSlices: 3,
    filename: `${filenamePrefix}_${sliceIndex}.${fileExtension}`,
  };
}

function getMergeSlices(sceneParams) {
  // Slices can only be merged if rgb and density are stored separately.
  const separateRgbAndDensity = getFieldOrDefault(
      sceneParams, 'export_store_rgb_and_density_separately', false);
  const mergeSlices =
      getFieldOrDefault(sceneParams, 'merge_slices', !separateRgbAndDensity);
  if (!separateRgbAndDensity && !mergeSlices) {
    throw new Error(
        'Slices must be merged when using "rgb_and_density" images. Please ' +
        're-export with export_store_rgb_and_density_separately=true and try ' +
        'again.');
  }
  return mergeSlices && separateRgbAndDensity;
}
