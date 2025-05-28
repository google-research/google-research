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
 * @fileoverview Utilities for creating and initializing texture buffers.
 */

/**
 * Creates an empty volume texture.
 *
 * @param {number} width Width of the texture
 * @param {number} height Height of the texture
 * @param {number} depth Depth of the texture
 * @param {number} format Format of the texture
 * @param {number} filter Filter strategy of the texture
 * @return {!THREE.DataTexture3D} Volume texture
 */
function createEmptyVolumeTexture(width, height, depth, format, filter) {
  let volumeTexture = new THREE.DataTexture3D(null, width, height, depth);
  volumeTexture.internalFormat = getInternalFormat(format);
  volumeTexture.format = format;
  volumeTexture.generateMipmaps = false;
  volumeTexture.magFilter = volumeTexture.minFilter = filter;
  volumeTexture.wrapS = volumeTexture.wrapT = volumeTexture.wrapR =
      THREE.ClampToEdgeWrapping;
  volumeTexture.type = THREE.UnsignedByteType;
  volumeTexture.unpackAlignment = 1;
  gRenderer.initTexture(volumeTexture);
  return volumeTexture;
}


/**
 * Creates three empty, equally-sized textures to hold triplanes.
 *
 * @param {number} width Width of the texture
 * @param {number} height Height of the texture
 * @param {number} format Format of the texture
 * @return {!THREE.DataTexture2DArray} Texture array of size three
 */
function createEmptyTriplaneTextureArray(width, height, depth, format) {
  let texture = new THREE.DataTexture2DArray(null, width, height, depth);
  texture.internalFormat = getInternalFormat(format);
  texture.format = format;
  texture.generateMipmaps = false;
  texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.type = THREE.UnsignedByteType;
  texture.unpackAlignment = 1;
  gRenderer.initTexture(texture);
  return texture;
}


function getInternalFormat(format) {
  if (format == THREE.RedFormat) {
    return 'R8';
  } else if (format == THREE.LuminanceAlphaFormat) {
    return 'LUMINANCE_ALPHA';
  } else if (format == THREE.RGBFormat) {
    return 'RGB';
  } else if (format == THREE.RGBAFormat) {
    return 'RGBA';
  }
  throw new Error(`Unrecognized THREE.js format: ${format}`);
}


function createEmptySceneTexture(spec) {
  return {
    textureType: 'scene',
    occupancyGridsTexture: createEmptyTexture(spec.occupancyGridsSpec),
    distanceGridsTexture: createEmptyTexture(spec.distanceGridsSpec),
    triplaneTexture: createEmptyTexture(spec.triplaneSpec),
    sparseGridTexture: createEmptyTexture(spec.sparseGridSpec),
  };
}

function createEmptyOccupancyGridsTexture(spec) {
  let textureType = spec.assetType.replace(/_slices$/, '');
  let gridTextures = spec.gridSpecs.map(createEmptyTexture);
  return {textureType, gridTextures};
}

function createEmptyOccupancyGridTexture(spec) {
  let textureType = spec.assetType.replace(/_slices$/, '');
  let texture = createEmptyVolumeTexture(
      ...spec.shape, THREE.RedFormat, THREE.NearestFilter);
  return {textureType, texture};
}


function createEmptyTriplaneTexture(spec) {
  let result = {
    textureType: 'triplane',
    featuresTexture: createEmptyTexture(spec.featuresSpec),
  };
  if (spec.separateRgbAndDensity) {
    result.rgbTexture = createEmptyTexture(spec.rgbSpec);
    result.densityTexture = createEmptyTexture(spec.densitySpec);
  } else {
    let shape = spec.rgbAndDensitySpec.shape;
    result.rgbTexture = createEmptyTexture({
      assetType: 'triplane_rgb_slices',
      shape: shape,
    });
    result.densityTexture = createEmptyTexture({
      assetType: 'triplane_density_slices',
      shape: shape,
    });
  }
  return result;
}


function createEmptyTriplaneSlicesTexture(spec) {
  let textureType = spec.assetType.replace(/_slices$/, '');
  let format = {
    'triplane_density': THREE.RedFormat,
    'triplane_rgb': THREE.RGBFormat,
    'triplane_features': THREE.RGBAFormat,
  }[textureType];
  console.assert(format != undefined, spec);
  let texture = createEmptyTriplaneTextureArray(...spec.shape, format);
  return { textureType, texture };
}


function createEmptySparseGridTexture(spec) {
  let _createEmptyAtlasVolumeTexture = (spec, format) => {
    return createEmptyVolumeTexture(...spec.shape, format, THREE.LinearFilter);
  };

  // Determine which spec to use for rgb and density. This will change
  // depending on which assets were generated.
  let rgbSpec =
      spec.separateRgbAndDensity ? spec.rgbSpec : spec.rgbAndDensitySpec;
  let densitySpec =
      spec.separateRgbAndDensity ? spec.densitySpec : spec.rgbAndDensitySpec;

  let sparseGridRgbTexture =
      _createEmptyAtlasVolumeTexture(rgbSpec, THREE.RGBFormat);
  let sparseGridDensityTexture =
      _createEmptyAtlasVolumeTexture(densitySpec, THREE.LuminanceAlphaFormat);
  let sparseGridFeaturesTexture =
      _createEmptyAtlasVolumeTexture(spec.featuresSpec, THREE.RGBFormat);

  // The indirection grid uses nearest filtering and is loaded in one go.
  // uint8[64,64,64], 3 bytes per entry
  let sparseGridBlockIndicesTexture = createEmptyVolumeTexture(
      ...spec.blockIndicesSpec.shape, THREE.RGBFormat, THREE.NearestFilter);

  // Update texture buffer for sparse_grid_block_indices.
  return {
    textureType: 'sparse_grid',
    blockIndicesTexture: {
      textureType: 'sparse_grid_block_indices',
      texture: sparseGridBlockIndicesTexture,
    },
    rgbTexture: {
      textureType: 'sparse_grid_rgb',
      texture: sparseGridRgbTexture,
    },
    densityTexture: {
      textureType: 'sparse_grid_density',
      texture: sparseGridDensityTexture,
    },
    featuresTexture: {
      textureType: 'sparse_grid_features',
      texture: sparseGridFeaturesTexture,
    },
  };
}


const gCreateEmptyTextureRegistry = {
  'scene': createEmptySceneTexture,

  // triplane
  'triplane': createEmptyTriplaneTexture,
  'triplane_rgb_slices': createEmptyTriplaneSlicesTexture,
  'triplane_density_slices': createEmptyTriplaneSlicesTexture,
  'triplane_features_slices': createEmptyTriplaneSlicesTexture,

  // distance grids
  'distance_grids': createEmptyOccupancyGridsTexture,
  'distance_grid_slices': createEmptyOccupancyGridTexture,

  // occupancy grids
  'occupancy_grids': createEmptyOccupancyGridsTexture,
  'occupancy_grid_slices': createEmptyOccupancyGridTexture,

  // sparse grid
  'sparse_grid': createEmptySparseGridTexture,
};


function createEmptyTexture(spec) {
  let loadFn = gCreateEmptyTextureRegistry[spec.assetType];
  if (loadFn == undefined) {
    console.error(
        `Failed to find loadFn for assetType ${spec.assetType}`, spec);
  }
  return loadFn(spec);
}
