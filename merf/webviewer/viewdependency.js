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
 * Creates a data texture containing MLP weights.
 *
 * @param {!Object} network_weights
 * @return {!THREE.DataTexture}
 */
function createNetworkWeightTexture(network_weights) {
  let width = network_weights.length;
  let height = network_weights[0].length;

  let weightsData = new Float32Array(width * height);
  for (let co = 0; co < height; co++) {
    for (let ci = 0; ci < width; ci++) {
      let index = co * width + ci;
      let weight = network_weights[ci][co];
      weightsData[index] = weight;
    }
  }

  let weightsDataNew = new Float32Array(width * height);
  for (let j = 0; j < width; j += 4) {
    for (let i = 0; i < height; i++) {
      for (let c = 0; c < 4; c++) {
        weightsDataNew[(j / 4) * height * 4 + i * 4 + c] =
            weightsData[(j / 4) * 4 + i * ((width / 4) * 4) + c];
      }
    }
  }
  weightsData = weightsDataNew;

  let texture = new THREE.DataTexture(
      weightsDataNew, 1, width * height / 4, THREE.RGBAFormat);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.type = THREE.FloatType;
  return texture;
}

/**
 * Creates shader code for the view-dependence MLP.
 *
 * This populates the shader code in viewDependenceNetworkShaderFunctions with
 * network weights and sizes as compile-time constants. The result is returned
 * as a string.
 *
 * @param {!Object} scene_params
 * @param {string} viewDependenceNetworkShaderFunctions
 * @return {string}
 */
function createViewDependenceFunctions(
    scene_params, viewDependenceNetworkShaderFunctions) {
  // For mat4mul, we need to make sure that widths/heights of matrices
  // are multiples of 4
  for (let layerIndex = 0; layerIndex < 3; layerIndex++) {
    weights = scene_params[layerIndex + '_weights'];
    bias = scene_params[layerIndex + '_bias'];
    width = weights.length;
    height = weights[0].length;
    new_width = makeMultipleOf(width, 4);
    new_height = makeMultipleOf(height, 4);
    new_weights = Array.from(Array(new_width), () => new Array(new_height));
    new_bias = Array(new_height);
    for (let j = 0; j < new_width; j++) {
      for (let i = 0; i < new_height; i++) {
        if (j < width && i < height) {
          new_weights[j][i] = weights[j][i];
          new_bias[i] = bias[i];
        } else {
          new_weights[j][i] = 0.0;
          new_bias[i] = 0.0;
        }
      }
    }
    scene_params[layerIndex + '_weights'] = new_weights;
    scene_params[layerIndex + '_bias'] = new_bias;
  }

  let network_weights = scene_params;

  // Write bias values as compile-time constants.
  let fragmentShaderSource = viewDependenceNetworkShaderFunctions;
  for (let layerIndex = 0; layerIndex < 3; layerIndex++) {
    let width = network_weights[layerIndex + '_bias'].length;
    let biasList = 'vec4(';
    for (let i = 0; i < width; i++) {
      let bias = network_weights[layerIndex + '_bias'][i];
      if (i % 4 == 0 && i != 0 && i != width - 1) {
        biasList += '), vec4(';
      }
      biasList += Number(bias).toFixed(7);
      if (i + 1 < width && (i + 1) % 4 != 0) {
        biasList += ', ';
      }
    }
    biasList += ')';
    fragmentShaderSource = fragmentShaderSource.replace(
        new RegExp('BIAS_LIST_' + layerIndex, 'g'), biasList);
  }

  let channelsZero = network_weights['0_weights'].length;
  let channelsOne = network_weights['0_bias'].length;
  let channelsTwo = network_weights['1_bias'].length;
  let channelsThree = network_weights['2_bias'].length;
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