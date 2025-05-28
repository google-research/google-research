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
 * Compute the linearized index and corresponding weight for trilerp.
 *
 * @param {!THREE.Vector3} position
 * @param {!THREE.Vector3} cornerIndices
 * @param {number} gridSize
 * @return {!Array<number>}
 */
function computeTrilerpLocationsAndWeights(position, cornerIndices, gridSize) {
  // Convert the submodel-space position to 3D grid coordinates with half
  // voxel centers.
  let gridPosition = new THREE.Vector3().copy(position);
  gridPosition.addScalar(1.0);
  gridPosition.divideScalar(2.0);
  gridPosition.multiplyScalar(gridSize);
  gridPosition.subScalar(0.5);

  // Compute the xyz indices for the vertex specified by cornerIndices.
  const floorPosition = new THREE.Vector3().copy(gridPosition).floor();
  const ceilPosition = new THREE.Vector3().copy(gridPosition).ceil();
  let x = cornerIndices.x > 0 ? ceilPosition.x : floorPosition.x;
  let y = cornerIndices.y > 0 ? ceilPosition.y : floorPosition.y;
  let z = cornerIndices.z > 0 ? ceilPosition.z : floorPosition.z;

  // Clamp to the grid size.
  x = Math.min(Math.max(x, 0), gridSize - 1);
  y = Math.min(Math.max(y, 0), gridSize - 1);
  z = Math.min(Math.max(z, 0), gridSize - 1);

  // Transform the coordinates for to match the JAX coordinate system.
  x = gridSize - 1 - x;  // Reverse x.
  [y, z] = [z, y];       // Swap y and z.

  // And linearize the coordinates to a single 1D index.
  const index = (z * gridSize + y) * gridSize + x;

  // Finally, compute the trilinear interpolation weight for this sample.
  let wx = gridPosition.x - floorPosition.x;
  let wy = gridPosition.y - floorPosition.y;
  let wz = gridPosition.z - floorPosition.z;
  if (cornerIndices.x == 0) {
    wx = 1.0 - wx;
  }
  if (cornerIndices.y == 0) {
    wy = 1.0 - wy;
  }
  if (cornerIndices.z == 0) {
    wz = 1.0 - wz;
  }
  const w = wx * wy * wz;

  return [index, w];
}

/**
 * Trilinearly interpolate MLP kernel weights and return them as a texture.
 *
 * @param {number} submodel
 * @param {number} level
 * @param {number} position
 * @return {!THREE.DataTexture}
 */
function trilerpDeferredMlpKernel(submodel, level, position) {
  let newHeight, newWidth, weightsData;
  // Trilinearly interpolate the MLP weights if we have a grid of weights.
  if (!!gDeferredMlp['ResampleDense_' + level + '/kernel']) {
    const kernelDict = gDeferredMlp['ResampleDense_' + level + '/kernel'];
    const weights = kernelDict['data'];
    const gridSize = kernelDict['shape'][1];
    const width = kernelDict['shape'][4];
    const height = kernelDict['shape'][5];

    newHeight = makeMultipleOf(height, 4);
    newWidth = makeMultipleOf(width, 4);
    weightsData = new Float32Array(newWidth * newHeight);

    // Right now we define our grid in world-coordinates, but position is in
    // submodel-space, so we have to transform it back.
    let worldPosition = new THREE.Vector3().copy(position);
    worldPosition.divideScalar(getSubmodelScaleFactor(submodel));
    const submodelOffset =
        submodel * gridSize * gridSize * gridSize * width * height;

    for (let dx = 0; dx < 2; dx++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dz = 0; dz < 2; dz++) {
          const [mlpIndex, trilerpWeight] = computeTrilerpLocationsAndWeights(
              worldPosition, new THREE.Vector3(dx, dy, dz), gridSize);
          const weightOffset = submodelOffset + width * height * mlpIndex;
          for (let co = 0; co < newHeight; co++) {
            for (let ci = 0; ci < newWidth; ci++) {
              let index = co * newWidth + ci;
              let weight = 0.0;
              if (ci < width && co < height) {
                weight = weights[weightOffset + ci * height + co];
              }
              if (dx + dy + dz === 0) {
                weightsData[index] = trilerpWeight * weight;
              } else {
                weightsData[index] += trilerpWeight * weight;
              }
            }
          }
        }
      }
    }
  } else {  // Otherwise just set them directly.
    const kernelDict = gDeferredMlp['Dense_' + level + '/kernel'];
    const weights = kernelDict['data'];
    const width = kernelDict['shape'][0];
    const height = kernelDict['shape'][1];

    newHeight = makeMultipleOf(height, 4);
    newWidth = makeMultipleOf(width, 4);
    weightsData = new Float32Array(newWidth * newHeight);

    for (let co = 0; co < newHeight; co++) {
      for (let ci = 0; ci < newWidth; ci++) {
        let index = co * newWidth + ci;
        if (ci < width && co < height) {
          weightsData[index] = weights[ci * height + co];
        }
      }
    }
  }

  let weightsDataNew = new Float32Array(newWidth * newHeight);
  for (let j = 0; j < newWidth; j += 4) {
    for (let i = 0; i < newHeight; i++) {
      for (let c = 0; c < 4; c++) {
        weightsDataNew[(j / 4) * newHeight * 4 + i * 4 + c] =
            weightsData[(j / 4) * 4 + i * ((newWidth / 4) * 4) + c];
      }
    }
  }

  let texture = new THREE.DataTexture(
      weightsDataNew, 1, newWidth * newHeight / 4, THREE.RGBAFormat);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.type = THREE.FloatType;
  return texture;
}

/**
 * Trilinearly interpolate MLP biases and return them as a list of Vector4s.
 *
 * @param {number} submodel
 * @param {number} level
 * @param {number} position
 * @return {!Array<!THREE.Vector4>}
 */
function trilerpDeferredMlpBiases(submodel, level, position) {
  let biasList;

  // Trilinearly interpolate the MLP weights if we have a grid of weights.
  if (!!gDeferredMlp['ResampleDense_' + level + '/bias']) {
    const biasDict = gDeferredMlp['ResampleDense_' + level + '/bias'];
    const biases = biasDict['data'];
    const gridSize = biasDict['shape'][1];
    const height = biasDict['shape'][4];
    const newHeight = makeMultipleOf(height, 4);
    biasList = new Array(newHeight / 4);

    // Right now we define our grid in world-coordinates, but position is in
    // submodel-space, so we have to transform it back.
    let worldPosition = new THREE.Vector3().copy(position);
    worldPosition.divideScalar(getSubmodelScaleFactor(submodel));
    const submodelOffset = submodel * gridSize * gridSize * gridSize * height;

    for (let dx = 0; dx < 2; dx++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dz = 0; dz < 2; dz++) {
          const [mlpIndex, trilerpWeight] = computeTrilerpLocationsAndWeights(
              worldPosition, new THREE.Vector3(dx, dy, dz), gridSize);
          const biasOffset = submodelOffset + height * mlpIndex;
          for (let biasIndex = 0; biasIndex < newHeight / 4; ++biasIndex) {
            let vector = new THREE.Vector4(0.0, 0.0, 0.0, 0.0);
            for (let ci = 0; ci < 4; ci++) {
              if (biasIndex * 4 + 0 < newHeight) {
                vector.setComponent(
                    ci, biases[biasOffset + biasIndex * 4 + ci]);
              }
            }
            vector.multiplyScalar(trilerpWeight);
            if (dx + dy + dz === 0) {
              biasList[biasIndex] = vector;
            } else {
              biasList[biasIndex].add(vector);
            }
          }
        }
      }
    }
  } else {  // Otherwise just set them directly.
    const biasDict = gDeferredMlp['Dense_' + level + '/bias'];
    const biases = biasDict['data'];
    const height = biasDict['shape'][0];
    const newHeight = makeMultipleOf(height, 4);
    biasList = new Array(newHeight / 4);

    for (let biasIndex = 0; biasIndex < newHeight / 4; ++biasIndex) {
      let vector = new THREE.Vector4(0.0, 0.0, 0.0, 0.0);
      for (let ci = 0; ci < 4; ci++) {
        if (biasIndex * 4 + 0 < newHeight) {
          vector.setComponent(ci, biases[biasIndex * 4 + ci]);
        }
      }
      biasList[biasIndex] = vector;
    }
  }

  return biasList;
}

/**
 * Rewrite shader with view dependence definitions
 *
 * @param {!Object} scene_params
 * @param {string} shader
 * @return {string}
 */
function rewriteViewDependenceDefinitions(scene_params, shader) {
  let network_weights = getDeferredMlp();

  const mlpName =
      !!network_weights['ResampleDense_0/kernel'] ? 'ResampleDense_' : 'Dense_';
  const si = !!network_weights['ResampleDense_0/kernel'] ? 4 : 0;

  // Write bias values as uniform references.
  let fragmentShaderSource = shader;

  // Initialize output activations for each layer. The following code generates
  // lines like,
  //    intermediate_one[0] = bias_0[0]; intermediate_one[1] = bias_0[1]; ...
  let layer_output_variable_names =
      ['intermediate_one', 'intermediate_two', 'result'];
  for (let layerIndex = 0; layerIndex < 3; layerIndex++) {
    let width = network_weights[mlpName + layerIndex + '/bias'].shape[si];
    let inputVar = `bias_${layerIndex}`;
    let outputVar = layer_output_variable_names[layerIndex];
    let lines = [];
    for (let i = 0; i < width / 4; i++) {
      lines.push(`${outputVar}[${i}] = ${inputVar}[${i}];`);
    }
    let biasLines = lines.join(' ') + '\n';
    fragmentShaderSource = fragmentShaderSource.replace(
        new RegExp(`INITIALIZE_OUTPUT_ACTIVATIONS_${layerIndex}`, 'g'),
        biasLines);
  }

  let channelsZero =
      makeMultipleOf(network_weights[mlpName + '0/kernel'].shape[si], 4);
  let channelsOne =
      makeMultipleOf(network_weights[mlpName + '0/bias'].shape[si], 4);
  let channelsTwo =
      makeMultipleOf(network_weights[mlpName + '1/bias'].shape[si], 4);
  let channelsThree =
      makeMultipleOf(network_weights[mlpName + '2/bias'].shape[si], 4);
  let posEncScales = 4;

  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_ZERO', 'g'), channelsZero);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_POSENC_SCALES', 'g'), posEncScales.toString());
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_ONE', 'g'), channelsOne);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_TWO', 'g'), channelsTwo);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_THREE', 'g'), channelsThree);
  return fragmentShaderSource;
}

/**
 * @param {number} x
 * @param {number} y
 * @return {number}
 */
function makeMultipleOf(x, y) {
  if (x % y == 0) {
    return x;
  } else {
    return x + y - x % y;
  }
}
