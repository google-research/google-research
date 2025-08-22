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
 * @fileoverview Populate WebGL textures with assets.
 */

/**
 * Populates all textures in a scene.
 */
function prepareScenePayload(asset) {
  return {
    textureType: 'scene',
    occupancyGridsPayload: prepareTexturePayload(asset.occupancyGridsAsset),
    distanceGridsPayload: prepareTexturePayload(asset.distanceGridsAsset),
    triplanePayload: prepareTexturePayload(asset.triplaneAsset),
    sparseGridPayload: prepareTexturePayload(asset.sparseGridAsset),
  }
}


function prepareOccupancyGridsPayload(asset) {
  let textureType = asset.assetType;
  let gridPayloads = asset.gridAssets.map(prepareTexturePayload);
  return {textureType, gridPayloads};
}


function prepareOccupancyGridPayload(asset) {
  if (asset.mergeSlices) {
    return prepareOccupancyGridMergedPayload(asset);
  } else {
    return prepareOccupancyGridSlicesPayload(asset);
  }
}

function prepareOccupancyGridMergedPayload(asset) {
  console.assert(asset.assetType.endsWith('_slices'), asset);
  let textureType = asset.assetType.replace(/_slices$/, '');
  let shape = asset.shape;
  let numChannels = asset.numChannels;
  let src = {
    1: GridTextureSource.RED_FROM_RED,
    4: GridTextureSource.ALPHA_FROM_RGBA,
  }[asset.numChannels];
  let payload = mergeSlices(asset, src, GridTextureDestination.RED_IN_RED);
  return {textureType, shape, numChannels, payload};
}

function prepareOccupancyGridSlicesPayload(asset) {
  console.assert(asset.assetType.endsWith('_slices'), asset);
  let textureType = asset.assetType.replace(/_slices$/, '');
  return {
    textureType: textureType,
    shape: asset.shape,
    numChannels: asset.numChannels,
    slicePayloads: asset.sliceAssets.map(prepareTexturePayload),
  };
}

function prepareOccupancyGridSlicePayload(asset) {
  console.assert(asset.assetType.endsWith('_slice'), asset);
  let payload = null;
  if (asset.numChannels == 1) {
    payload = asset.asset;
  } else if (asset.numChannels == 4) {
    payload = mergeSlices(
        {
          shape: asset.shape,
          numSlices: 1,
          sliceAssets: [{...asset, sliceIndex: 0, numSlices: 1}],
        },
        GridTextureSource.ALPHA_FROM_RGBA,
        GridTextureDestination.RED_IN_RED,
    );
  } else {
    throw new Error('Unrecognized number of input channels', asset);
  }
  return {
    textureType: asset.assetType,
    shape: asset.shape,
    numChannels: asset.numChannels,
    sliceIndex: asset.sliceIndex,
    numSlices: asset.numSlices,
    payload: payload,
  };
}


function prepareTriplanePayload(asset) {
  let result = {
    textureType: 'triplane',
    featuresPayload: preparePlanePayload(
        asset.featuresAsset,
        'triplane_features',
        GridTextureSource.RGBA_FROM_RGBA,
        GridTextureDestination.RGBA_IN_RGBA,
        ),
  };
  if (asset.separateRgbAndDensity) {
    result.rgbPayload = preparePlanePayload(
        asset.rgbAsset,
        'triplane_rgb',
        GridTextureSource.RGB_FROM_RGB,
        GridTextureDestination.RGB_IN_RGB,
    );
    result.densityPayload = preparePlanePayload(
        asset.densityAsset,
        'triplane_density',
        GridTextureSource.RED_FROM_RED,
        GridTextureDestination.RED_IN_RED,
    );
  } else {
    result.rgbPayload = preparePlanePayload(
        asset.rgbAndDensityAsset,
        'triplane_rgb',
        GridTextureSource.RGB_FROM_RGBA,
        GridTextureDestination.RGB_IN_RGB,
    );
    result.densityPayload = preparePlanePayload(
        asset.rgbAndDensityAsset,
        'triplane_density',
        GridTextureSource.ALPHA_FROM_RGBA,
        GridTextureDestination.RED_IN_RED,
    );
  }
  return result;
}


function preparePlanePayload(asset, dstKey, src, dst) {
  let result = {
    textureType: dstKey,
    shape: asset.shape,
    numChannels: asset.numChannels,
  };
  if (asset.mergeSlices) {
    result.payload = mergeSlices(asset, src, dst);
  } else {
    result.slicePayloads = asset.sliceAssets.map(
        (sliceAsset) => preparePlaneSlicePayload(sliceAsset, src, dst));
  }
  return result;
}

function preparePlaneSlicePayload(asset, src, dst) {
  let payload = null;
  if (src.format == dst.format && src.channels == dst.channels) {
    payload = asset.asset;
  } else {
    payload = mergeSlices(
        {
          shape: asset.shape,
          numSlices: 1,
          sliceAssets: [{...asset, sliceIndex: 0, numSlices: 1}],
        },
        src, dst);
  }
  return {
    textureType: asset.assetType,
    shape: asset.shape,
    sliceIndex: asset.sliceIndex,
    numSlices: asset.numSlices,
    numChannels: asset.numChannels,
    payload: payload
  };
}


function prepareArrayPayload(asset) {
  return {
    textureType: asset.assetType,
    payload: asset.asset,
    shape: asset.shape,
    numChannels: asset.numChannels
  };
}


function prepareSparseGridPayload(asset) {
  let result = {
    textureType: 'sparse_grid',
    blockIndicesPayload: prepareTexturePayload(asset.blockIndicesAsset),
    featuresPayload: prepareTexturePayload(asset.featuresAsset),
  };
  if (asset.separateRgbAndDensity) {
    result.rgbPayload = prepareTexturePayload(asset.rgbAsset);
    result.densityPayload = prepareTexturePayload(asset.densityAsset);
  } else {
    result.rgbPayload = prepareTexturePayload(asset.rgbAndDensityAsset);
    result.densityPayload = prepareSparseGridDensityPayload(asset);
  }
  return result;
}

function prepareSparseGridGenericPayload(asset) {
  if (asset.mergeSlices) {
    return prepareSparseGridGenericMergedPayload(asset);
  } else {
    return prepareSparseGridGenericSlicesPayload(asset);
  }
}

function prepareSparseGridGenericMergedPayload(asset) {
  let textureType = asset.assetType.replace(/_slices$/, '');
  if (textureType.includes('rgb_and_density')) {
    textureType = textureType.replace(/rgb_and_density$/, 'rgb');
  }
  let srcOpt = {
    2: GridTextureSource.LA_FROM_LUMINANCE_ALPHA,
    3: GridTextureSource.RGB_FROM_RGB,
    4: GridTextureSource.RGB_FROM_RGBA
  };
  let dstOpt = {
    2: GridTextureDestination.LA_IN_LUMINANCE_ALPHA,
    3: GridTextureDestination.RGB_IN_RGB,
    4: GridTextureDestination.RGB_IN_RGB
  };
  let payload = mergeSlices(
      asset,
      srcOpt[asset.numChannels],
      dstOpt[asset.numChannels],
  );

  return {
    textureType: textureType,
    shape: asset.shape,
    numChannels: asset.numChannels,
    payload: payload,
  };
}

function prepareSparseGridGenericSlicesPayload(asset) {
  let textureType = asset.assetType.replace(/_slices$/, '');
  if (textureType.includes('rgb_and_density')) {
    textureType = textureType.replace(/rgb_and_density$/, 'rgb');
  }
  return {
    textureType: textureType,
    shape: asset.shape,
    numChannels: asset.numChannels,
    numSlices: asset.numSlices,
    slicePayloads: asset.sliceAssets.map(prepareTexturePayload),
  };
}

function prepareSparseGridGenericSlicePayload(asset) {
  // This payload corresponds to a *slice* of a texture.
  let assetType = asset.assetType;

  let payload = null;
  if ((assetType == 'sparse_grid_rgb_slice' && asset.numChannels == 3) ||
      (assetType == 'sparse_grid_density_slice' && asset.numChannels == 2) ||
      (assetType == 'sparse_grid_features_slice' && asset.numChannels == 3)) {
    payload = asset.asset;
  } else {
    let srcOpts = {
      2: GridTextureSource.LA_FROM_LUMINANCE_ALPHA,
      3: GridTextureSource.RGB_FROM_RGB,
      4: GridTextureSource.RGB_FROM_RGBA
    };
    let dstOpts = {
      2: GridTextureDestination.LA_IN_LUMINANCE_ALPHA,
      3: GridTextureDestination.RGB_IN_RGB,
      4: GridTextureDestination.RGB_IN_RGB
    };
    payload = mergeSlices(
        {
          shape: asset.shape,
          numSlices: 1,
          sliceAssets: [{...asset, sliceIndex: 0, numSlices: 1}],
        },
        srcOpts[asset.numChannels], dstOpts[asset.numChannels]);
  }

  // Only the rgb part of rgb_and_density is extracted.
  let textureType = assetType;
  if (textureType.includes('rgb_and_density')) {
    textureType = textureType.replace(/_rgb_and_density$/, 'rgb');
  }

  // A slice requires no further processing.
  return {
    textureType: textureType,
    shape: asset.shape,
    numChannels: asset.numChannels,
    sliceIndex: asset.sliceIndex,
    numSlices: asset.numSlices,
    payload: payload,
  };
}


/**
 * Populates the sparse grid's density texture using the alpha channel from two
 * data sources. This is only necessary if separateRgbAndDensity is false.
 */
function prepareSparseGridDensityPayload(asset) {
  return {
    textureType: 'sparse_grid_density',
    shape: asset.rgbAndDensityAsset.shape,
    numChannels: 2,
    payload: mergeSparseGridDensity(asset),
  };
}


const gPrepareTexturePayloadRegistry = {
  'scene': prepareScenePayload,

  // triplane
  'triplane': prepareTriplanePayload,

  // distance grids
  'distance_grids': prepareOccupancyGridsPayload,
  'distance_grid_slices': prepareOccupancyGridPayload,
  'distance_grid_slice': prepareOccupancyGridSlicePayload,

  // occupancy grids
  'occupancy_grids': prepareOccupancyGridsPayload,
  'occupancy_grid_slices': prepareOccupancyGridPayload,
  'occupancy_grid_slice': prepareOccupancyGridSlicePayload,

  // sparse grid
  'sparse_grid': prepareSparseGridPayload,
  'sparse_grid_block_indices': prepareArrayPayload,

  'sparse_grid_rgb_and_density_slices': prepareSparseGridGenericPayload,
  'sparse_grid_rgb_and_density_slice': prepareSparseGridGenericSlicePayload,

  'sparse_grid_rgb_slices': prepareSparseGridGenericPayload,
  'sparse_grid_rgb_slice': prepareSparseGridGenericSlicePayload,

  'sparse_grid_density_slices': prepareSparseGridGenericPayload,
  'sparse_grid_density_slice': prepareSparseGridGenericSlicePayload,

  'sparse_grid_features_slices': prepareSparseGridGenericPayload,
  'sparse_grid_features_slice': prepareSparseGridGenericSlicePayload,
};


function prepareTexturePayload(asset) {
  let loadFn = gPrepareTexturePayloadRegistry[asset.assetType];
  if (loadFn == undefined) {
    console.error(
        `Failed to find loadFn for assetType ${asset.assetType}`, asset);
  }
  return loadFn(asset);
}
