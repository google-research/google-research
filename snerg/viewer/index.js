// Copyright 2021 The Google Research Authors.
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
 * The a global dictionary containing scene parameters.
 * @type {?Object}
 */
let gSceneParams = null;

/**
 * The timestamp of the last frame to be rendered, used to track performance.
 * @type {number}
 */
let gLastFrame = window.performance.now();

/**
 * The near plane used for rendering. Increasing this value somewhat speeds up
 * rendering, but this is most useful to show cross sections of the scene.
 * @type {number}
 */
let gNearPlane = 0.33;

/**
 * This scene renders the baked NeRF reconstruction using ray marching.
 * @type {?THREE.Scene}
 */
let gRayMarchScene = null;

/**
 * Progress counters for loading RGBA textures.
 * @type {number}
 */
let gLoadedRGBATextures = 0;

/**
 * Progress counters for loading feature textures.
 * @type {number}
 */
let gLoadedFeatureTextures = 0;

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
  DISPLAY_FEATURES: 2 ,
  /** Only shows the view dependent component. */
  DISPLAY_VIEW_DEPENDENT: 3 ,
  /** Only shows the coarse block grid. */
  DISPLAY_COARSE_GRID: 4,
  /** Only shows the block atlas structure. */
  DISPLAY_3D_ATLAS: 5,
};

/**  @type {!DisplayModeType}  */
let gDisplayMode = DisplayModeType.DISPLAY_NORMAL;

/**
 * Number of textures to load.
 * @type {number}
 */
let gNumTextures = 0;

/**
 * The THREE.js renderer object we use.
 * @type {?THREE.WebGLRenderer}
 */
let gRenderer = null;

/**
 * The perspective camera we use to view the scene.
 * @type {?THREE.PerspectiveCamera}
 */
let gCamera = null;

/**
 * We control the perspective camera above using OrbitControls.
 * @type {?THREE.OrbitControls}
 */
let gOrbitControls = null;

/**
 * An orthographic camera used to kick off ray marching with a
 * full-screen render pass.
 * @type {?THREE.OrthographicCamera}
 */
let gBlitCamera = null;

/**
 * Creates a float32 texture from a Float32Array of data.
 * @param {number} width
 * @param {number} height
 * @param {!Float32Array} data
 * @return {!THREE.DataTexture}
 */
function createFloatTextureFromData(width, height, data) {
  let texture = new THREE.DataTexture(data, width, height, THREE.RedFormat);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.type = THREE.FloatType;
  return texture;
}

/**
 * The vertex shader for rendering a baked NeRF scene with ray marching.
 * @const {string}
 */
const rayMarchVertexShader = `
  varying vec3 vOrigin;
  varying vec3 vDirection;
  uniform mat4 world_T_clip;

  void main() {
    vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = positionClip;

    positionClip /= positionClip.w;
    vec4 nearPoint = world_T_clip * vec4(positionClip.x, positionClip.y, -1.0, 1.0);
    vec4 farPoint = world_T_clip * vec4(positionClip.x, positionClip.y, 1.0, 1.0);

    vOrigin = nearPoint.xyz / nearPoint.w;
    vDirection = normalize(farPoint.xyz / farPoint.w - vOrigin);
  }
`;

/**
 * The fragment shader for rendering a baked NeRF scene with ray marching.
 * @const {string}
 */
const rayMarchFragmentShader = `
  varying vec3 vOrigin;
  varying vec3 vDirection;

  uniform int displayMode;
  uniform int ndc;

  uniform vec3 minPosition;
  uniform vec3 gridSize;
  uniform vec3 atlasSize;
  uniform float voxelSize;
  uniform float blockSize;
  uniform mat3 worldspace_R_opengl;
  uniform float nearPlane;

  uniform lowp sampler3D mapAlpha;
  uniform lowp sampler3D mapColor;
  uniform lowp sampler3D mapFeatures;
  uniform mediump sampler3D mapIndex;

  uniform mediump sampler2D weightsZero;
  uniform mediump sampler2D weightsOne;
  uniform mediump sampler2D weightsTwo;

  mediump float indexToPosEnc(vec3 dir, int index) {
    mediump float coordinate =
      (index % 3 == 0) ? dir.x : (
      (index % 3 == 1) ? dir.y : dir.z);
    if (index < 3) {
      return coordinate;
    }
    int scaleExponent = ((index - 3) % (3 * 4)) / 3;
    coordinate *= pow(2.0, float(scaleExponent));
    if ((index - 3) >= 3 * 4) {
      const float kHalfPi = 1.57079632679489661923;
      coordinate += kHalfPi;
    }
    return sin(coordinate);
  }

  mediump vec3 evaluateNetwork(
      lowp vec3 color, lowp vec4 features, mediump vec3 viewdir) {
    mediump float intermediate_one[NUM_CHANNELS_ONE] = float[](
      BIAS_LIST_ZERO
    );
    for (int j = 0; j < NUM_CHANNELS_ZERO; ++j) {
      mediump float input_value = 0.0;
      if (j < 27) {
        input_value = indexToPosEnc(viewdir, j);
      } else if (j < 30) {
        input_value =
          (j % 3 == 0) ? color.r : (
          (j % 3 == 1) ? color.g : color.b);
      } else {
        input_value =
          (j == 30) ? features.r : (
          (j == 31) ? features.g : (
          (j == 32) ? features.b : features.a));
      }
      if (abs(input_value) < 0.1 / 255.0) {
        continue;
      }
      for (int i = 0; i < NUM_CHANNELS_ONE; ++i) {
        intermediate_one[i] += input_value *
          texelFetch(weightsZero, ivec2(j, i), 0).x;
      }
    }

    mediump float intermediate_two[NUM_CHANNELS_TWO] = float[](
      BIAS_LIST_ONE
    );
    for (int j = 0; j < NUM_CHANNELS_ONE; ++j) {
      if (intermediate_one[j] <= 0.0) {
        continue;
      }
      for (int i = 0; i < NUM_CHANNELS_TWO; ++i) {
        intermediate_two[i] += intermediate_one[j] *
          texelFetch(weightsOne, ivec2(j, i), 0).x;
      }
    }

    mediump float result[NUM_CHANNELS_THREE] = float[](
      BIAS_LIST_TWO
    );
    for (int j = 0; j < NUM_CHANNELS_TWO; ++j) {
      if (intermediate_two[j] <= 0.0) {
        continue;
      }
      for (int i = 0; i < NUM_CHANNELS_THREE; ++i) {
        result[i] += intermediate_two[j] *
          texelFetch(weightsTwo, ivec2(j, i), 0).x;
      }
    }
    for (int i = 0; i < NUM_CHANNELS_THREE; ++i) {
      result[i] = 1.0 / (1.0 + exp(-result[i]));
    }

    return vec3(result[0], result[1], result[2]);
  }

  mediump vec3 convertOriginToNDC(vec3 origin, vec3 direction) {
    // We store the NDC scenes flipped, so flip back.
    origin.z *= -1.0;
    direction.z *= -1.0;

    const float near = 1.0;
    float t = -(near + origin.z) / direction.z;
    origin = origin * t + direction;

    // Hardcoded, worked out using approximate iPhone FOV of 67.3 degrees
    // and an image width of 1006 px.
    const float focal = 755.644;
    const float W = 1006.0;
    const float H = 756.0;
    float o0 = 1.0 / (W / (2.0 * focal)) * origin.x / origin.z;
    float o1 = -1.0 / (H / (2.0 * focal)) * origin.y / origin.z;
    float o2 = 1.0 + 2.0 * near / origin.z;

    origin = vec3(o0, o1, o2);
    origin.z *= -1.0;
    return origin;
  }

  mediump vec3 convertDirectionToNDC(vec3 origin, vec3 direction) {
    // We store the NDC scenes flipped, so flip back.
    origin.z *= -1.0;
    direction.z *= -1.0;

    const float near = 1.0;
    float t = -(near + origin.z) / direction.z;
    origin = origin * t + direction;

    // Hardcoded, worked out using approximate iPhone FOV of 67.3 degrees
    // and an image width of 1006 px.
    const float focal = 755.6440;
    const float W = 1006.0;
    const float H = 756.0;

    float d0 = 1.0 / (W / (2.0 * focal)) *
      (direction.x / direction.z - origin.x / origin.z);
    float d1 = -1.0 / (H / (2.0 * focal)) *
      (direction.y / direction.z - origin.y / origin.z);
    float d2 = -2.0 * near / origin.z;

    direction = normalize(vec3(d0, d1, d2));
    direction.z *= -1.0;
    return direction;
  }

  // Compute the atlas block index for a point in the scene using pancake
  // 3D atlas packing.
  mediump vec3 pancakeBlockIndex(
      mediump vec3 posGrid, float blockSize, ivec3 iBlockGridBlocks) {
    ivec3 iBlockIndex = ivec3(floor(posGrid / blockSize));
    ivec3 iAtlasBlocks = ivec3(atlasSize) / ivec3(blockSize + 2.0);
    int linearIndex = iBlockIndex.x + iBlockGridBlocks.x *
      (iBlockIndex.z + iBlockGridBlocks.z * iBlockIndex.y);

    mediump vec3 atlasBlockIndex = vec3(
      float(linearIndex % iAtlasBlocks.x),
      float((linearIndex / iAtlasBlocks.x) % iAtlasBlocks.y),
      float(linearIndex / (iAtlasBlocks.x * iAtlasBlocks.y)));

    // If we exceed the size of the atlas, indicate an empty voxel block.
    if (atlasBlockIndex.z >= float(iAtlasBlocks.z)) {
      atlasBlockIndex = vec3(-1.0, -1.0, -1.0);
    }

    return atlasBlockIndex;
  }

  mediump vec2 rayAabbIntersection(mediump vec3 aabbMin,
                                   mediump vec3 aabbMax,
                                   mediump vec3 origin,
                                   mediump vec3 invDirection) {
    mediump vec3 t1 = (aabbMin - origin) * invDirection;
    mediump vec3 t2 = (aabbMax - origin) * invDirection;
    mediump vec3 tMin = min(t1, t2);
    mediump vec3 tMax = max(t1, t2);
    return vec2(max(tMin.x, max(tMin.y, tMin.z)),
                min(tMax.x, min(tMax.y, tMax.z)));
  }

  void main() {
    // See the DisplayMode enum at the top of this file.
    // Runs the full model with view dependence.
    const int DISPLAY_NORMAL = 0;
    // Disables the view-dependence network.
    const int DISPLAY_DIFFUSE = 1;
    // Only shows the latent features.
    const int DISPLAY_FEATURES = 2;
    // Only shows the view dependent component.
    const int DISPLAY_VIEW_DEPENDENT = 3;
    // Only shows the coarse block grid.
    const int DISPLAY_COARSE_GRID = 4;
    // Only shows the 3D texture atlas.
    const int DISPLAY_3D_ATLAS = 5;

    // Set up the ray parameters in world space..
    float nearWorld = nearPlane;
    mediump vec3 originWorld = vOrigin;
    mediump vec3 directionWorld = normalize(vDirection);
    if (ndc != 0) {
      nearWorld = 0.0;
      originWorld = convertOriginToNDC(vOrigin, normalize(vDirection));
      directionWorld = convertDirectionToNDC(vOrigin, normalize(vDirection));
    }

    // Now transform them to the voxel grid coordinate system.
    mediump vec3 originGrid = (originWorld - minPosition) / voxelSize;
    mediump vec3 directionGrid = directionWorld;
    mediump vec3 invDirectionGrid = 1.0 / directionGrid;

    ivec3 iGridSize = ivec3(round(gridSize));
    int iBlockSize = int(round(blockSize));
    ivec3 iBlockGridBlocks = (iGridSize + iBlockSize - 1) / iBlockSize;
    ivec3 iBlockGridSize = iBlockGridBlocks * iBlockSize;
    mediump vec3 blockGridSize = vec3(iBlockGridSize);
    mediump vec2 tMinMax = rayAabbIntersection(
      vec3(0.0, 0.0, 0.0), gridSize, originGrid, invDirectionGrid);

    // Skip any rays that miss the scene bounding box.
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    if (tMinMax.x > tMinMax.y) {
      return;
    }

    mediump float t = max(nearWorld / voxelSize, tMinMax.x) + 0.5;
    mediump vec3 posGrid = originGrid + directionGrid * t;

    mediump vec3 blockMin = floor(posGrid / blockSize) * blockSize;
    mediump vec3 blockMax = blockMin + blockSize;
    mediump vec2 tBlockMinMax = rayAabbIntersection(
          blockMin, blockMax, originGrid, invDirectionGrid);
    mediump vec3 atlasBlockIndex;

    if (displayMode == DISPLAY_3D_ATLAS) {
      atlasBlockIndex = pancakeBlockIndex(posGrid, blockSize, iBlockGridBlocks);
    } else {
      atlasBlockIndex = 255.0 * texture(
        mapIndex, (blockMin + blockMax) / (2.0 * blockGridSize)).xyz;
    }

    lowp float visibility = 1.0;
    lowp vec3 color = vec3(0.0, 0.0, 0.0);
    lowp vec4 features = vec4(0.0, 0.0, 0.0, 0.0);
    int step = 0;
    int maxStep = int(ceil(length(gridSize)));

    while (step < maxStep && t < tMinMax.y && visibility > 1.0 / 255.0) {
      // Skip empty macroblocks.
      if (atlasBlockIndex.x > 254.0) {
        t = 0.5 + tBlockMinMax.y;
      } else { // Otherwise step through them and fetch RGBA and Features.
        mediump vec3 posAtlas = clamp(posGrid - blockMin, 0.0, blockSize);
        posAtlas += atlasBlockIndex * (blockSize + 2.0);
        posAtlas += 1.0; // Account for the one voxel padding in the atlas.

        if (displayMode == DISPLAY_COARSE_GRID) {
          color = atlasBlockIndex * (blockSize + 2.0) / atlasSize;
          features.rgb = atlasBlockIndex * (blockSize + 2.0) / atlasSize;
          features.a = 1.0;
          visibility = 0.0;
          continue;
        }

        // Do a conservative fetch for alpha!=0 at a lower resolution,
        // and skip any voxels which are empty. First, this saves bandwidth
        // since we only fetch one byte instead of 8 (trilinear) and most
        // fetches hit cache due to low res. Second, this is conservative,
        // and accounts for any possible alpha mass that the high resolution
        // trilinear would find.
        const int skipMipLevel = 2;
        const float miniBlockSize = float(1 << skipMipLevel);

        // Only fetch one byte at first, to conserve memory bandwidth in
        // empty space.
        lowp float atlasAlpha = texelFetch(
          mapAlpha, ivec3(posAtlas / miniBlockSize), skipMipLevel).x;

        if (atlasAlpha > 0.0) {
          // OK, we hit something, do a proper trilinear fetch at high res.
          mediump vec3 atlasUvw = posAtlas / atlasSize;
          atlasAlpha = textureLod(mapAlpha, atlasUvw, 0.0).x;

          // Only worth fetching the content if high res alpha is non-zero.
          if (atlasAlpha > 0.5 / 255.0) {
            lowp vec4 atlasRgba = vec4(0.0, 0.0, 0.0, atlasAlpha);
            atlasRgba.rgb = texture(mapColor, atlasUvw).rgb;
            if (displayMode != DISPLAY_DIFFUSE) {
              lowp vec4 atlasFeatures = texture(mapFeatures, atlasUvw);
              features += visibility * atlasFeatures;
            }
            color += visibility * atlasRgba.rgb;
            visibility *= 1.0 - atlasRgba.a;
          }
        }
        t += 1.0;
      }

      posGrid = originGrid + directionGrid * t;
      if (t > tBlockMinMax.y) {
        blockMin = floor(posGrid / blockSize) * blockSize;
        blockMax = blockMin + blockSize;
        tBlockMinMax = rayAabbIntersection(
              blockMin, blockMax, originGrid, invDirectionGrid);

        if (displayMode == DISPLAY_3D_ATLAS) {
          atlasBlockIndex = pancakeBlockIndex(
            posGrid, blockSize, iBlockGridBlocks);
        } else {
          atlasBlockIndex = 255.0 * texture(
            mapIndex, (blockMin + blockMax) / (2.0 * blockGridSize)).xyz;
        }
      }
      step++;
    }

    if (displayMode == DISPLAY_VIEW_DEPENDENT) {
      color = vec3(0.0, 0.0, 0.0) * visibility;
    } else if (displayMode == DISPLAY_FEATURES) {
      color = features.rgb;
    }

    // For forward-facing scenes, we partially unpremultiply alpha to fill
    // tiny holes in the rendering.
    lowp float alpha = 1.0 - visibility;
    if (ndc != 0 && alpha > 0.0) {
      lowp float filledAlpha = min(1.0, alpha * 1.5);
      color *= filledAlpha / alpha;
      alpha = filledAlpha;
      visibility = 1.0 - filledAlpha;
    }

    // Compute the final color, to save compute only compute view-depdence
    // for rays that intersected something in the scene.
    color = vec3(1.0, 1.0, 1.0) * visibility + color;
    const float kVisibilityThreshold = 254.0 / 255.0;
    if (visibility <= kVisibilityThreshold &&
        (displayMode == DISPLAY_NORMAL ||
         displayMode == DISPLAY_VIEW_DEPENDENT)) {
      color += evaluateNetwork(
        color, features, worldspace_R_opengl * normalize(vDirection));
    }

    gl_FragColor = vec4(color, 1.0);
}
`;

/**
 * Creates a material (i.e. shaders and texture bindings) for a SNeRG scene.
 *
 * First, this shader ray marches through an RGBA + FEATURE grid, stored as a
 * block-sparse matrix, skipping large macro-blocks of empty space wherever
 * possible. Then, once the ray opacity has saturated, the shader introduces
 * view dependence by running a small-MLP per ray. The MLP takes as input the
 * accumulated color, feature vectors as well as a positionally encoded
 * view-direction vector.
 *
 * @param {!Object} scene_params
 * @param {!THREE.Texture} alphaVolumeTexture
 * @param {!THREE.Texture} rgbVolumeTexture
 * @param {!THREE.Texture} featureVolumeTexture
 * @param {!THREE.Texture} atlasIndexTexture
 * @param {!THREE.Vector3} minPosition
 * @param {number} gridWidth
 * @param {number} gridHeight
 * @param {number} gridDepth
 * @param {number} blockSize
 * @param {number} voxelSize
 * @param {number} atlasWidth
 * @param {number} atlasHeight
 * @param {number} atlasDepth
 * @return {!THREE.Material}
 */
function createRayMarchMaterial(
    scene_params, alphaVolumeTexture, rgbVolumeTexture, featureVolumeTexture,
    atlasIndexTexture, minPosition, gridWidth, gridHeight, gridDepth, blockSize,
    voxelSize, atlasWidth, atlasHeight, atlasDepth) {
  // First set up the network weights.
  let network_weights = scene_params;
  let width = network_weights['0_weights'].length;
  let height = network_weights['0_weights'][0].length;

  let weightsDataZero = new Float32Array(width * height);
  for (let co = 0; co < height; co++) {
    for (let ci = 0; ci < width; ci++) {
      let index = co * width + ci;
      let weight = network_weights['0_weights'][ci][co];
      weightsDataZero[index] = weight;
    }
  }
  let weightsTexZero = createFloatTextureFromData(
      width, height, weightsDataZero);

  width = network_weights['0_bias'].length;
  height = 1;
  let biasListZero = '';
  for (let i = 0; i < width; i++) {
    let bias = network_weights['0_bias'][i];
    biasListZero += Number(bias).toFixed(7);
    if (i + 1 < width) {
      biasListZero += ', ';
    }
  }

  width = network_weights['1_weights'].length;
  height = network_weights['1_weights'][0].length;
  let weightsDataOne = new Float32Array(width * height);
  for (let co = 0; co < height; co++) {
    for (let ci = 0; ci < width; ci++) {
      let index = co * width + ci;
      let weight = network_weights['1_weights'][ci][co];
      weightsDataOne[index] = weight;
    }
  }
  let weightsTexOne =
      createFloatTextureFromData(width, height, weightsDataOne);

  width = network_weights['1_bias'].length;
  height = 1;
  let biasListOne = '';
  for (let i = 0; i < width; i++) {
    let bias = network_weights['1_bias'][i];
    biasListOne += Number(bias).toFixed(7);
    if (i + 1 < width) {
      biasListOne += ', ';
    }
  }

  width = network_weights['2_weights'].length;
  height = network_weights['2_weights'][0].length;
  let weightsDataTwo = new Float32Array(width * height);
  for (let co = 0; co < height; co++) {
    for (let ci = 0; ci < width; ci++) {
      let index = co * width + ci;
      let weight = network_weights['2_weights'][ci][co];
      weightsDataTwo[index] = weight;
    }
  }
  let weightsTexTwo =
      createFloatTextureFromData(width, height, weightsDataTwo);

  width = network_weights['2_bias'].length;
  height = 1;
  let biasListTwo = '';
  for (let i = 0; i < width; i++) {
    let bias = network_weights['2_bias'][i];
    biasListTwo += Number(bias).toFixed(7);
    if (i + 1 < width) {
      biasListTwo += ', ';
    }
  }

  let channelsZero = network_weights['0_weights'].length;
  let channelsOne = network_weights['0_bias'].length;
  let channelsTwo = network_weights['1_bias'].length;
  let channelsThree = network_weights['2_bias'].length;
  let posEncScales = 4;

  let fragmentShaderSource = rayMarchFragmentShader.replace(
      new RegExp('NUM_CHANNELS_ZERO', 'g'), channelsZero);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_POSENC_SCALES', 'g'), posEncScales.toString());
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_ONE', 'g'), channelsOne);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_TWO', 'g'), channelsTwo);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('NUM_CHANNELS_THREE', 'g'), channelsThree);

  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('BIAS_LIST_ZERO', 'g'), biasListZero);
  fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('BIAS_LIST_ONE', 'g'), biasListOne);
    fragmentShaderSource = fragmentShaderSource.replace(
      new RegExp('BIAS_LIST_TWO', 'g'), biasListTwo);

  // Now pass all the 3D textures as uniforms to the shader.
  let worldspace_R_opengl = new THREE.Matrix3();
  let M_dict = network_weights['worldspace_T_opengl'];
  worldspace_R_opengl['set'](
      M_dict[0][0], M_dict[0][1], M_dict[0][2],
      M_dict[1][0], M_dict[1][1], M_dict[1][2],
      M_dict[2][0], M_dict[2][1], M_dict[2][2]);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      'mapAlpha': {'value': alphaVolumeTexture},
      'mapColor': {'value': rgbVolumeTexture},
      'mapFeatures': {'value': featureVolumeTexture},
      'mapIndex': {'value': atlasIndexTexture},
      'displayMode': {'value': gDisplayMode - 0},
      'ndc' : {'value' : 0},
      'nearPlane' : { 'value' : 0.33},
      'blockSize': {'value': blockSize},
      'voxelSize': {'value': voxelSize},
      'minPosition': {'value': minPosition},
      'weightsZero': {'value': weightsTexZero},
      'weightsOne': {'value': weightsTexOne},
      'weightsTwo': {'value': weightsTexTwo},
      'world_T_clip': {'value': new THREE.Matrix4()},
      'worldspace_R_opengl': {'value': worldspace_R_opengl},
      'gridSize':
          {'value': new THREE.Vector3(gridWidth, gridHeight, gridDepth)},
      'atlasSize':
          {'value': new THREE.Vector3(atlasWidth, atlasHeight, atlasDepth)}
    },
    vertexShader: rayMarchVertexShader,
    fragmentShader: fragmentShaderSource,
    vertexColors: true,
  });

  material.side = THREE.DoubleSide;
  material.depthTest = false;
  return material;
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
 * @param {string} classname
 * @return {!HTMLElement}
 */
function create(what, classname) {
  const e = /** @type {!HTMLElement} */(document.createElement(what));
  if (classname) {
    e.className = classname;
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
 * Adds event listeners to UI.
 */
function addHandlers() {
  const view = document.querySelector('.view');
  view.addEventListener('keypress', function(e) {
    if (e.keyCode === 32 || e.key === ' ' || e.key === 'Spacebar') {
      if (gDisplayMode == DisplayModeType.DISPLAY_NORMAL) {
        gDisplayMode = DisplayModeType.DISPLAY_DIFFUSE;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_DIFFUSE) {
        gDisplayMode = DisplayModeType.DISPLAY_FEATURES;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_FEATURES) {
        gDisplayMode = DisplayModeType.DISPLAY_VIEW_DEPENDENT;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_VIEW_DEPENDENT) {
        gDisplayMode = DisplayModeType.DISPLAY_COARSE_GRID;
      } else if (gDisplayMode == DisplayModeType.DISPLAY_COARSE_GRID) {
        gDisplayMode = DisplayModeType.DISPLAY_3D_ATLAS;
      } else /* gDisplayModeType == DisplayModeType.DISPLAY_3D_ATLAS */ {
        gDisplayMode = DisplayModeType.DISPLAY_NORMAL;
      }
      e.preventDefault();
    }
  });
}

/**
 * Hides the Loading prompt.
 */
function hideLoading() {
  let loading = document.getElementById('Loading');
  loading.style.display = 'none';
}

/**
 * Updates the loading progress HTML elements.
 */
function updateLoadingProgress() {
  let texturergbprogress = document.getElementById('texturergbprogress');
  let texturefeaturesprogress =
      document.getElementById('texturefeaturesprogress');

  const textureString = gNumTextures > 0 ? gNumTextures : '?';
  texturergbprogress.innerHTML =
      'RGBA images: ' + gLoadedRGBATextures + '/' + textureString;
  texturefeaturesprogress.innerHTML =
      'feature images: ' + gLoadedFeatureTextures + '/' + textureString;
}

/**
 * Loads PNG image from rgbaURL and decodes it to an Uint8Array.
 * @param {string} rgbaUrl The URL of the PNG image.
 * @return {!Promise<!Uint8Array>}
 */
function loadPNG(rgbaUrl) {
  const rgbaPromise = fetch(rgbaUrl, {
                        method: 'GET',
                        mode: 'same-origin',
                      }).then(response => {
    return response.arrayBuffer();
  }).then(buffer => {
    let data = new Uint8Array(buffer);
    let pngDecoder = new PNG(data);
    let pixels = pngDecoder.decodePixels();
    return pixels;
  });
  rgbaPromise.catch(error => {
    console.error(
        'Could not PNG image from: ' + rgbaUrl + ', error: ' + error);
    return;
  });
  return rgbaPromise;
}

/**
 * Fills an existing 3D RGB texture and an existing 3D alpha texture
 * from {url}_%03d.png.
 *
 * @param {!THREE.DataTexture3D} texture_alpha The alpha texture to gradually
 *     fill with data.
 * @param {!THREE.DataTexture3D} texture_rgb The rgb texture to gradually fill
 *     with data.
 * @param {string} url The URL prefix for the texture to be loaded from.
 * @param {number} num_slices The number of images the volume is stored in.
 * @param {number} volume_width The width of the final volume texture.
 * @param {number} volume_height The height of the final volume texture.
 * @param {number} volume_depth The depth of the final volume texture.
 * @param {?function()} on_update A function to be called whenever an image has
 *     been loaded.
 *
 * @return {!Promise} A promise that completes when all RGBA images have been
 *     uploaded.
 */
function loadSplitVolumeTexturePNG(
    texture_alpha, texture_rgb, url, num_slices, volume_width, volume_height,
    volume_depth, on_update) {
  const slice_depth = 4;
  let uploadPromises = [];
  for (let i = 0; i < num_slices; i++) {
    let rgbaUrl = url + '_' + digits(i, 3) + '.png';
    let rgbaPromise = loadPNG(rgbaUrl);
    rgbaPromise = rgbaPromise.then(data => {
      on_update();
      return data;
    });

    uploadPromises[i] = new Promise(function(resolve, reject) {
      Promise.all([rgbaPromise, i])
          .then(values => {
            let rgbaPixels = values[0];
            let i = values[1];

            let rgbPixels = new Uint8Array(
                volume_width * volume_height * slice_depth * 3);
            let alphaPixels = new Uint8Array(
                volume_width * volume_height * slice_depth * 1);

            for (let j = 0; j < volume_width * volume_height * slice_depth;
                 j++) {
              rgbPixels[j * 3 + 0] = rgbaPixels[j * 4 + 0];
              rgbPixels[j * 3 + 1] = rgbaPixels[j * 4 + 1];
              rgbPixels[j * 3 + 2] = rgbaPixels[j * 4 + 2];
              alphaPixels[j] = rgbaPixels[j * 4 + 3];
            }

            // We unfortunately have to touch THREE.js internals to get access
            // to the texture handle and gl.texSubImage3D. Using dictionary
            // notation to make this code robust to minifcation.
            const rgbTextureProperties =
                gRenderer['properties'].get(texture_rgb);
            const alphaTextureProperties =
                gRenderer['properties'].get(texture_alpha);
            let gl = gRenderer.getContext();

            let oldTexture = gl.getParameter(gl.TEXTURE_BINDING_3D);
            gl.bindTexture(
                gl.TEXTURE_3D, rgbTextureProperties['__webglTexture']);
            gl.texSubImage3D(
                gl.TEXTURE_3D, 0, 0, 0, i * slice_depth, volume_width,
                volume_height, slice_depth, gl.RGB, gl.UNSIGNED_BYTE,
                rgbPixels, 0);
            gl.bindTexture(
                gl.TEXTURE_3D, alphaTextureProperties['__webglTexture']);
            gl.texSubImage3D(
                gl.TEXTURE_3D, 0, 0, 0, i * slice_depth, volume_width,
                volume_height, slice_depth, gl.RED, gl.UNSIGNED_BYTE,
                alphaPixels, 0);
            gl.bindTexture(gl.TEXTURE_3D, oldTexture);

            resolve(texture_rgb);
          })
          .catch(error => {
            reject(error);
          });
    });
  }

  return new Promise(function(resolve, reject) {
    Promise.all(uploadPromises).then(values => {
      resolve(values[0]);
    });
  });
}

/**
 * Fills an existing 3D RGBA texture from {url}_%03d.png.
 *
 * @param {!THREE.DataTexture3D} texture The texture to gradually fill with
 *     data.
 * @param {string} url The URL prefix for the texture to be loaded from.
 * @param {number} num_slices The number of images the volume is stored in.
 * @param {number} volume_width The width of the final volume texture.
 * @param {number} volume_height The height of the final volume texture.
 * @param {number} volume_depth The depth of the final volume texture.
 * @param {?function()} on_update A function to be called whenever an image has
 *     been loaded.
 *
 * @return {!Promise} A promise that completes when all RGBA images have been
 *     uploaded.
 */
function loadVolumeTexturePNG(
    texture, url, num_slices, volume_width, volume_height, volume_depth,
    on_update) {
  const slice_depth = 4;
  let uploadPromises = [];
  for (let i = 0; i < num_slices; i++) {
    let rgbaUrl = url + '_' + digits(i, 3) + '.png';
    let rgbaPromise = loadPNG(rgbaUrl);
    rgbaPromise = rgbaPromise.then(data => {
      on_update();
      return data;
    });

    uploadPromises[i] = new Promise(function(resolve, reject) {
      Promise.all([rgbaPromise, i])
          .then(values => {
            let rgbaImage = values[0];
            let i = values[1];

            // We unfortunately have to touch THREE.js internals to get access
            // to the texture handle and gl.texSubImage3D. Using dictionary
            // notation to make this code robust to minifcation.
            const textureProperties = gRenderer['properties'].get(texture);
            let gl = gRenderer.getContext();

            let oldTexture = gl.getParameter(gl.TEXTURE_BINDING_3D);
            let textureHandle = textureProperties['__webglTexture'];
            gl.bindTexture(gl.TEXTURE_3D, textureHandle);
            gl.texSubImage3D(
                gl.TEXTURE_3D, 0, 0, 0, i * slice_depth,
                volume_width, volume_height, slice_depth,
                gl.RGBA, gl.UNSIGNED_BYTE, rgbaImage, 0);
            gl.bindTexture(gl.TEXTURE_3D, oldTexture);

            resolve(texture);
          })
          .catch(error => {
            reject(error);
          });
    });
  }

  return new Promise(function(resolve, reject) {
    Promise.all(uploadPromises).then(values => {
      resolve(values[0]);
    });
  });
}

/**
 * Loads the initial SNeRG scene parameters..
 * @param {string} dirUrl
 * @param {number} width
 * @param {number} height
 * @returns {!Promise} A promise for when the initial scene params have loaded.
 */
function loadScene(dirUrl, width, height) {
  // Reset the texture loading window.
  gLoadedRGBATextures = gLoadedFeatureTextures = gNumTextures = 0;
  updateLoadingProgress();

  // Loads scene parameters (voxel grid size, NDC/no-NDC, view-dependence MLP).
  let sceneParamsUrl = dirUrl + '/' +
      'scene_params.json';
  let sceneParamsPromise = fetch(sceneParamsUrl, {
                             method: 'GET',
                             mode: 'same-origin',
                           }).then(response => {
    return response.json();
  });
  sceneParamsPromise.catch(error => {
    console.error(
        'Could not load scene params from: ' + sceneParamsUrl +
        ', error: ' + error);
    return;
  });

  // Load the indirection grid.
  const imageLoader = new THREE.ImageLoader();
  let atlasIndexUrl = dirUrl + '/' + 'atlas_indices.png';
  const atlasIndexPromise = new Promise(function(resolve, reject) {
    imageLoader.load(atlasIndexUrl, atlasIndexImage => {
      resolve(atlasIndexImage);
    }, undefined, () => reject(atlasIndexUrl));
  });

  let initializedPromise = Promise.all([sceneParamsPromise, atlasIndexPromise]);
  initializedPromise.then(values => {
    let parsed = values[0];
    let atlasIndexImage = values[1];

    // Start rendering ASAP, forcing THREE.js to upload the textures.
    requestAnimationFrame(render);

    gSceneParams = parsed;
    gSceneParams['dirUrl'] = dirUrl;
    gSceneParams['loadingTextures'] = false;
    gNumTextures = gSceneParams['num_slices'];

    // Create empty 3D textures for the loaders to incrementally fill with data.
    let rgbVolumeTexture = new THREE.DataTexture3D(
        null, gSceneParams['atlas_width'], gSceneParams['atlas_height'],
        gSceneParams['atlas_depth']);
    rgbVolumeTexture.format = THREE.RGBFormat;
    rgbVolumeTexture.generateMipmaps = false;
    rgbVolumeTexture.magFilter = rgbVolumeTexture.minFilter =
        THREE.LinearFilter;
    rgbVolumeTexture.wrapS = rgbVolumeTexture.wrapT =
        rgbVolumeTexture.wrapR = THREE.ClampToEdgeWrapping;
    rgbVolumeTexture.type = THREE.UnsignedByteType;

    let alphaVolumeTexture = new THREE.DataTexture3D(
        null, gSceneParams['atlas_width'], gSceneParams['atlas_height'],
        gSceneParams['atlas_depth']);
    alphaVolumeTexture.format = THREE.RedFormat;
    alphaVolumeTexture.generateMipmaps = true;
    alphaVolumeTexture.magFilter = THREE.LinearFilter;
    alphaVolumeTexture.minFilter = THREE.LinearMipmapNearestFilter;
    alphaVolumeTexture.wrapS = alphaVolumeTexture.wrapT =
        alphaVolumeTexture.wrapR = THREE.ClampToEdgeWrapping;
    alphaVolumeTexture.type = THREE.UnsignedByteType;

    let featureVolumeTexture = new THREE.DataTexture3D(
        null, gSceneParams['atlas_width'], gSceneParams['atlas_height'],
        gSceneParams['atlas_depth']);
    featureVolumeTexture.format = THREE.RGBAFormat;
    featureVolumeTexture.generateMipmaps = false;
    featureVolumeTexture.magFilter = featureVolumeTexture.minFilter =
        THREE.LinearFilter;
    featureVolumeTexture.wrapS = featureVolumeTexture.wrapT =
        featureVolumeTexture.wrapR = THREE.ClampToEdgeWrapping;
    featureVolumeTexture.type = THREE.UnsignedByteType;

    let atlasIndexTexture = new THREE.DataTexture3D(
        atlasIndexImage,
        Math.ceil(gSceneParams['grid_width'] / gSceneParams['block_size']),
        Math.ceil(gSceneParams['grid_height'] / gSceneParams['block_size']),
        Math.ceil(gSceneParams['grid_depth'] / gSceneParams['block_size']));
    atlasIndexTexture.format = THREE.RGBAFormat;
    atlasIndexTexture.generateMipmaps = false;
    atlasIndexTexture.magFilter = atlasIndexTexture.minFilter =
        THREE.NearestFilter;
    atlasIndexTexture.wrapS = atlasIndexTexture.wrapT =
        atlasIndexTexture.wrapR = THREE.ClampToEdgeWrapping;
    atlasIndexTexture.type = THREE.UnsignedByteType;

    let fullScreenPlane = new THREE.PlaneBufferGeometry(width, height);
    let rayMarchMaterial = createRayMarchMaterial(
        gSceneParams, alphaVolumeTexture, rgbVolumeTexture,
        featureVolumeTexture, atlasIndexTexture,
        new THREE.Vector3(
            gSceneParams['min_x'], gSceneParams['min_y'],
            gSceneParams['min_z']),
        gSceneParams['grid_width'], gSceneParams['grid_height'],
        gSceneParams['grid_depth'], gSceneParams['block_size'],
        gSceneParams['voxel_size'], gSceneParams['atlas_width'],
        gSceneParams['atlas_height'], gSceneParams['atlas_depth']);

    let fullScreenPlaneMesh = new THREE.Mesh(fullScreenPlane, rayMarchMaterial);
    fullScreenPlaneMesh.position.z = -100;
    fullScreenPlaneMesh.frustumCulled = false;

    gRayMarchScene = new THREE.Scene();
    gRayMarchScene.add(fullScreenPlaneMesh);
    gRayMarchScene.autoUpdate = false;

    gBlitCamera = new THREE.OrthographicCamera(
        width / -2, width / 2, height / 2, height / -2, -10000, 10000);
    gBlitCamera.position.z = 100;
  });

  initializedPromise.catch(errors => {
    console.error(
        'Could not load scene from: ' + dirUrl + ', errors:\n\t' + errors[0] +
        '\n\t' + errors[1] + '\n\t' + errors[2] + '\n\t' + errors[3]);
  });

  return initializedPromise;
}

/**
 * Initializes the application based on the URL parameters.
 */
function initFromParameters() {
  const params = new URL(window.location.href).searchParams;
  const dirUrl = params.get('dir');
  const size = params.get('s');

  const usageString =
      'To view a SNeRG scene, specify the following parameters in the URL:\n' +
      '(Required) The URL to a SNeRG scene directory.\n' +
      's: (Optional) The dimensions as width,height. E.g. 640,360.\n' +
      'vfovy:  (Optional) The vertical field of view of the viewer.';

  if (!dirUrl) {
    error('dir is a required parameter.\n\n' + usageString);
    return;
  }

  let width = 1280;
  let height = 720;
  if (size) {
    const match = size.match(/([\d]+),([\d]+)/);
    width = parseInt(match[1], 10);
    height = parseInt(match[2], 10);
  }

  gNearPlane = parseFloat(params.get('near') || 0.33);
  const vfovy = parseFloat(params.get('vfovy') || 35);

  loadScene(dirUrl, width, height);

  const view = create('div', 'view');
  setDims(view, width, height);
  view.textContent = '';

  const viewSpaceContainer = document.getElementById('viewspacecontainer');
  viewSpaceContainer.style.display = 'inline-block';

  const viewSpace = document.querySelector('.viewspace');
  viewSpace.textContent = '';
  viewSpace.appendChild(view);

  let canvas = document.createElement('canvas');
  view.appendChild(canvas);

  // Set up a high performance WebGL context, making sure that anti-aliasing is
  // truned off.
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

  gCamera = new THREE.PerspectiveCamera(
      72, canvas.offsetWidth / canvas.offsetHeight, gNearPlane, 100.0);
  gCamera.aspect = view.offsetWidth / view.offsetHeight;
  gCamera.fov = vfovy;
  gRenderer.autoClear = false;
  gRenderer.setSize(view.offsetWidth, view.offsetHeight);

  gOrbitControls = new THREE.OrbitControls(gCamera, view);
  gOrbitControls.screenSpacePanning = true;
  gOrbitControls.zoomSpeed = 0.5;
}

/**
 * Set up code that needs to run once after the  scene parameters have loaded.
 */
function loadOnFirstFrame() {
  // Early out if we've already run this function.
  if (gSceneParams['loadingTextures']) {
    return;
  }

  // Set up the camera controls for the scene type.
  gOrbitControls.target.x = 0.0;
  gOrbitControls.target.y = 0.0;
  gOrbitControls.target.z = 0.0;

  if (gSceneParams['ndc']) {
    gCamera.position.x = 0.0;
    gCamera.position.y = 0.0;
    gCamera.position.z = -0.25;
    gOrbitControls.panSpeed = 2.0;
    gOrbitControls.minDistance = 0.05;
    gOrbitControls.maxDistance = 0.3;
    gOrbitControls.mouseButtons.LEFT = THREE.MOUSE.PAN;
  } else {
    gCamera.position.x = 0.0;
    gCamera.position.y = 1.0;
    gCamera.position.z = -4.0;
  }

  gOrbitControls.position = gCamera.position;
  gOrbitControls.position0 = gCamera.position;

  gCamera.updateProjectionMatrix();
  gOrbitControls.update();

  // Now that the 3D textures have been allocated, we can start slowly filling
  // them with data.
  const alphaVolumeTexture =
      gRayMarchScene.children[0].material.uniforms['mapAlpha']['value'];
  const rgbVolumeTexture =
      gRayMarchScene.children[0].material.uniforms['mapColor']['value'];
  const featureVolumeTexture =
      gRayMarchScene.children[0].material.uniforms['mapFeatures']['value'];

  let rgbVolumeTexturePromise = loadSplitVolumeTexturePNG(
      alphaVolumeTexture, rgbVolumeTexture, gSceneParams['dirUrl'] + '/rgba',
      gNumTextures, gSceneParams['atlas_width'], gSceneParams['atlas_height'],
      gSceneParams['atlas_depth'], function() {
        gLoadedRGBATextures++;
        updateLoadingProgress();
      });
  let featureVolumeTexturePromise = loadVolumeTexturePNG(
      featureVolumeTexture, gSceneParams['dirUrl'] + '/feature', gNumTextures,
      gSceneParams['atlas_width'], gSceneParams['atlas_height'],
      gSceneParams['atlas_depth'], function() {
        gLoadedFeatureTextures++;
        updateLoadingProgress();
      });

  let allTexturesPromise =
      Promise.all([rgbVolumeTexturePromise, featureVolumeTexturePromise]);
  allTexturesPromise.catch(errors => {
    console.error(
        'Could not load scene from: ' + gSceneParams['dirUrl'] +
        ', errors:\n\t' + errors[0] + '\n\t' + errors[1] + '\n\t' + errors[2] +
        '\n\t' + errors[3]);
  });

  // After all the textures have been loaded, we build mip maps for alpha
  // to enable accelerated ray marching inside each macroblock.
  allTexturesPromise.then(texture => {
    const alphaTextureProperties =
        gRenderer['properties'].get(alphaVolumeTexture);
    let gl = gRenderer.getContext();
    let oldTexture = gl.getParameter(gl.TEXTURE_BINDING_3D);
    gl.bindTexture(gl.TEXTURE_3D, alphaTextureProperties['__webglTexture']);
    gl.generateMipmap(gl.TEXTURE_3D);
    gl.bindTexture(gl.TEXTURE_3D, oldTexture);

    hideLoading();
    console.log('Successfully loaded scene from: ' + gSceneParams['dirUrl']);
  });

  // Now set the loading textures flag so this function runs only once.
  gSceneParams['loadingTextures'] = true;
}

/**
 * Updates the frame rate counter using exponential fall-off smoothing.
 */
function updateFPSCounter() {
  let currentFrame = window.performance.now();
  let milliseconds = currentFrame - gLastFrame;
  let oldMilliseconds = 1000 /
      (parseFloat(document.getElementById('fpsdisplay').innerHTML) || 1.0);

  // Prevent the FPS from getting stuck by ignoring frame times over 2 seconds.
  if (oldMilliseconds > 2000 || oldMilliseconds < 0) {
    oldMilliseconds = milliseconds;
  }
  let smoothMilliseconds = oldMilliseconds * (0.75) + milliseconds * 0.25;
  let smoothFps = 1000 / smoothMilliseconds;
  gLastFrame = currentFrame;
  document.getElementById('fpsdisplay').innerHTML = smoothFps.toFixed(1);
}

/**
 * The main render function that gets called every frame.
 * @param {number} t
 */
function render(t) {
  loadOnFirstFrame();

  gOrbitControls.update();
  gCamera.updateMatrix();
  gRenderer.setRenderTarget(null);
  gRenderer.clear();

  let world_T_camera = gCamera.matrixWorld;
  let camera_T_clip = new THREE.Matrix4();
  camera_T_clip.getInverse(gCamera.projectionMatrix);
  let world_T_clip = new THREE.Matrix4();
  world_T_clip.multiplyMatrices(world_T_camera, camera_T_clip);

  gRayMarchScene.children[0].material.uniforms['world_T_clip']['value'] =
      world_T_clip;
  gRayMarchScene.children[0].material.uniforms['displayMode']['value'] =
      gDisplayMode - 0;
  gRayMarchScene.children[0].material.uniforms['ndc']['value'] =
      gSceneParams['ndc'] - 0;
  gRayMarchScene.children[0].material.uniforms['nearPlane']['value'] =
      gNearPlane;
  gRenderer.render(gRayMarchScene, gBlitCamera);

  updateFPSCounter();
  requestAnimationFrame(render);
}

/**
 * Starts the volumetric object viewer application.
 */
function start() {
  initFromParameters();
  addHandlers();
}

start();
