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
function populateScene(texture, payload) {
  let promises = [
    populateTexture(texture.occupancyGridsTexture, payload.occupancyGridsPayload),
    populateTexture(texture.distanceGridsTexture, payload.distanceGridsPayload),
    populateTexture(texture.triplaneTexture, payload.triplanePayload),
    populateTexture(texture.sparseGridTexture, payload.sparseGridPayload),
  ];
  return Promise.all(promises);
}


/**
 * Populates all occupancy grid textures.
 */
function populateOccupancyGridsTexture(texture, payload) {
  console.assert(
      texture.gridTextures.length == payload.gridPayloads.length,
      texture,
      payload,
  );
  let numGridTextures = texture.gridTextures.length;
  let promises = range(numGridTextures).map((i) => {
    return populateTexture(texture.gridTextures[i], payload.gridPayloads[i]);
  });
  return Promise.all(promises);
}


/**
 * Populates all textures related to the sparse grid.
 */
function populateSparseGridTexture(texture, payload) {
  let promises = [
    populateTexture(texture.blockIndicesTexture, payload.blockIndicesPayload),
    populateTexture(texture.rgbTexture, payload.rgbPayload),
    populateTexture(texture.densityTexture, payload.densityPayload),
    populateTexture(texture.featuresTexture, payload.featuresPayload),
  ];
  return Promise.all(promises);
}


/**
 * Populates all triplane textures.
 */
function populateTriplaneTexture(texture, payload) {
  return Promise.all([
    populateTexture(texture.rgbTexture, payload.rgbPayload),
    populateTexture(texture.densityTexture, payload.densityPayload),
    populateTexture(texture.featuresTexture, payload.featuresPayload),
  ]);
}


/**
 * Populates a single monolithic texture.
 */
async function populateArrayTexture(texture, payload) {
  if (payload.payload != null) {
    texture.texture.image.data = await payload.payload;
    texture.texture.needsUpdate = true;
  } else {
    throw new Error('Unclear how to ingest payload', texture, payload);
  }
}

/**
 * Populates a texture with or without slices.
 */
function populateArrayTextureWithWebGL(texture, payload) {
  if (payload.payload != null) {
    return populateArrayTextureMergedWithWebGL(texture, payload);
  } else if (payload.slicePayloads != null) {
    let promises = payload.slicePayloads.map(
        (slicePayload) =>
            populateArrayTextureSliceWithWebGL(texture, slicePayload));
    return Promise.all(promises);
  } else {
    throw new Error('Unclear how to ingest payload', texture, payload);
  }
}


/**
 * Populate's a slice of a target texture using the WebGL API.
 */
async function populateArrayTextureSliceWithWebGL(texture, payload) {
  let gl = gRenderer.getContext();
  const volumeWidth = payload.shape[0];
  const volumeHeight = payload.shape[1];
  const sliceDepth = payload.shape[2];
  const sliceIndex = payload.sliceIndex;

  let threeFormat = texture.texture.format;
  let {glFormat, glInternalFormat, numChannels} =
      threeFormatToOpenGLFormat(gl, threeFormat);

  // Wait for data to be ready.
  let srcData = await payload.payload;

  // Set target texture as OpenGL's current texture.
  const textureProperties = gRenderer['properties'].get(texture.texture);
  let newTexture = textureProperties['__webglTexture'];
  console.assert(newTexture != null, texture);

  // Both 3D textures and 2D texture arrays use gl.texSubImage3D, but different
  // context arguments are required.
  let glTextureBinding, glTextureType;
  if (texture.texture instanceof THREE.DataTexture3D) {
    glTextureBinding = gl.TEXTURE_BINDING_3D;
    glTextureType = gl.TEXTURE_3D;
  } else if (texture.texture instanceof THREE.DataTexture2DArray) {
    glTextureBinding = gl.TEXTURE_BINDING_2D_ARRAY;
    glTextureType = gl.TEXTURE_2D_ARRAY;
  }

  let oldTexture = gl.getParameter(glTextureBinding);
  gl.bindTexture(glTextureType, newTexture);
  let start = performance.mark(`${texture.textureType}-start`);
  gl.texSubImage3D(
      glTextureType,            // target
      0,                        // level
      0,                        // xoffset
      0,                        // yoffset
      sliceIndex * sliceDepth,  // zoffset
      volumeWidth,              // width
      volumeHeight,             // height
      sliceDepth,               // depth
      glFormat,                 // format
      gl.UNSIGNED_BYTE,         // type
      srcData,                  // srcData
      0,                        // srcOffset
  );
  let end = performance.mark(`${texture.textureType}-end`);
  performance.measure(
      `${texture.textureType}-duration`,
      `${texture.textureType}-start`,
      `${texture.textureType}-end`,
  )
  gl.bindTexture(glTextureType, oldTexture);
}


async function populateArrayTextureMergedWithWebGL(texture, payload) {
  return populateArrayTextureSliceWithWebGL(
      texture, {...payload, sliceIndex: 0, numSlices: 1});
}


/**
 * Converts THREE.js's texture format to WebGL's.
 */
function threeFormatToOpenGLFormat(gl, threeFormat) {
  if (threeFormat == THREE.RGBAFormat) {
    return {
      numChannels: 4,
      glFormat: gl.RGBA,
      glInternalFormat: gl.RGBA,
    };
  } else if (threeFormat == THREE.RGBFormat) {
    return {
      numChannels: 3,
      glFormat: gl.RGB,
      glInternalFormat: gl.RGB,
    };
  } else if (threeFormat == THREE.LuminanceAlphaFormat) {
    return {
      numChannels: 2,
      glFormat: gl.LUMINANCE_ALPHA,
      glInternalFormat: gl.LUMINANCE_ALPHA
    };
  } else if (threeFormat == THREE.RedFormat) {
    return {
      numChannels: 1,
      glFormat: gl.RED,
      glInternalFormat: gl.R8,
    };
  } else {
    throw new Error(`Unrecognized three format: ${threeFormat}`);
  }
}

/**
 * Registry for functions that can be reached via populateTexture().
 */
const gPopulateTextureRegistry = {
  'scene': populateScene,

  // triplane
  'triplane': populateTriplaneTexture,
  'triplane_rgb': populateArrayTextureWithWebGL,
  'triplane_density': populateArrayTextureWithWebGL,
  'triplane_features': populateArrayTextureWithWebGL,

  // distance grids
  'distance_grids': populateOccupancyGridsTexture,
  'distance_grid': populateArrayTextureWithWebGL,

  // occupancy grids
  'occupancy_grids': populateOccupancyGridsTexture,
  'occupancy_grid': populateArrayTextureWithWebGL,

  // sparse grid
  'sparse_grid': populateSparseGridTexture,
  'sparse_grid_block_indices': populateArrayTextureWithWebGL,
  'sparse_grid_rgb': populateArrayTextureWithWebGL,
  'sparse_grid_density': populateArrayTextureWithWebGL,
  'sparse_grid_features': populateArrayTextureWithWebGL,
};


/**
 * Entry point for populating textures.
 */
function populateTexture(texture, payload) {
  let loadFn = gPopulateTextureRegistry[texture.textureType];
  if (loadFn == undefined) {
    console.error(
        `Failed to find loadFn for assetType ${texture.textureType}`, texture);
  }
  return loadFn(texture, payload);
}