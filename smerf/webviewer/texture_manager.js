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

/**
 * Possible states for a ray march texture buffer.
 */
const NEEDS_NEW_SUBMODEL = -1;
const LOADING = 0;
const READY = 1;

/**
 * Mapping from integers to registered submodel scene content.
 */
let gSubmodelSceneContents = {};

/**
 * Maximum number of payloads to keep in memory. Use this to limit Host
 * RAM consumption.
 */
let gSubmodelCacheSize = 10;

/**
 * Ping pong buffers. texture will be initialized to cotnain a structure
 * matching that of create_texture.js
 */
let gRayMarchTextureBuffers = [
  {si: 0, state: NEEDS_NEW_SUBMODEL, texture: null},
  {si: 0, state: NEEDS_NEW_SUBMODEL, texture: null},
];

/**
 * Index of the active entry in gRayMarchTextureBuffers;
 */
let gActiveRayMarchTextureBuffer = 0;

/**
 * Three.js scene instance. The same instance is used for all submodels.
 */
let gRayMarchScene = null;

/**
 * Get the global three.js scene.
 */
function getRayMarchScene() {
  if (gRayMarchScene == null) {
    throw new Error('gRayMarchScene has not been initialized yet!');
  }
  return gRayMarchScene;
}

/**
 * Get the index of the active submodel.
 */
function getActiveSubmodelIndex() {
  return gRayMarchTextureBuffers[gActiveRayMarchTextureBuffer].si;
}

/**
 * Get the content of the active submodel.
 */
function getActiveSubmodelContent() {
  return getSubmodelContent(getActiveSubmodelIndex());
}

/**
 * Get the scale of a submodel
 */
function getSubmodelScale(si) {
  let content = getSubmodelContent(si);
  let submodelScale = content.params['submodel_scale'];
  return submodelScale;
}

/**
 * Get the scale factor of a submodel.
 */
function getSubmodelScaleFactor(si) {
  let content = getSubmodelContent(si);
  let submodelScale = getSubmodelScale(si);
  let submodelResolution = Math.cbrt(content.params['num_submodels']);
  let submodelScaleFactor = submodelScale / submodelResolution;
  return submodelScaleFactor;
}

/**
 * Get the content for a submodel.
 */
function getSubmodelContent(si) {
  return gSubmodelSceneContents[si];
}

/**
 * Register content for a submodel.
 */
function registerSubmodelContent(i, content) {
  gSubmodelSceneContents[i] = content;
}


/**
 * Get global Deferred MLP parameters.
 */
function getDeferredMlp() {
  console.assert(gDeferredMlp != null);
  return gDeferredMlp;
}


/**
 * Set the global Deferred MLP parameters.
 */
function registerDeferredMlp(deferredMlp) {
  validateDeferredMlp(deferredMlp);
  gDeferredMlp = deferredMlp;
}


/**
 * Get size of allocated texture assets.
 */
function getCurrentTextureUsageInBytes() {
  let numBytes = 0;
  for (rmtb of gRayMarchTextureBuffers) {
    numBytes += getTextureSizeInBytes(rmtb.texture);
  }
  return numBytes;
}

/**
 * Attempt to set the active submodel.
 *
 * This operation is best-effort: if the requested submodel's textures are
 * ready, it will switch to them. If they are not, it will start the process
 * of preparing them. Call getActiveSubmodelIndex() ater this function to
 * determine if this call succeeded or not.
 */
function setCurrentRayMarchScene(si) {
  let activeBufferIdx = gActiveRayMarchTextureBuffer;
  let activeBuffer = gRayMarchTextureBuffers[activeBufferIdx];
  let otherBufferIdx = (activeBufferIdx + 1) % 2;
  let otherBuffer = gRayMarchTextureBuffers[otherBufferIdx];

  if (getSubmodelContent(si) == null) {
    // Requested submodel doesn't exist. Don't attempt to load it.
    return Promise.resolve();
  }

  // Update for LRU cache.
  getSubmodelContent(si).lastTouched = Date.now();

  if (si == activeBuffer.si && activeBuffer.state >= LOADING) {
    // Wait for this buffer to finish loading.
    return Promise.resolve();
  }

  if (si == otherBuffer.si && otherBuffer.state == READY) {
    // Switch to other buffer.
    console.log(`Switching to buffer ${otherBufferIdx} for submodel #${si}`);
    let sceneContent = getSubmodelContent(si);
    setTextureUniforms(sceneContent.params, otherBuffer.texture);
    gActiveRayMarchTextureBuffer = otherBufferIdx;
    return Promise.resolve();
  }

  if (otherBuffer.state >= LOADING && otherBuffer.state < READY) {
    // The other buffer is busy loading. Don't try to claim it.
    return Promise.resolve();
  }

  // Claim this buffer and start loading it.
  console.log(
      `Preparing texture buffer #${otherBufferIdx} for submodel #${si}`);
  otherBuffer.si = si;
  otherBuffer.state = LOADING;
  showLoading();

  return Promise.resolve()
        .then(() => {
          // Prepare texture buffers for use.
          reinitializeSparseGridTextures(otherBuffer);
          // Fetch assets now if they haven't been already.
          let content = getSubmodelContent(otherBuffer.si);
          if (content.payload == null) {
            console.log(`Fetching assets for submodel #${otherBuffer.si}`);
            let asset = fetchAsset(content.spec, content.router);
            let payload = prepareTexturePayload(asset);
            content.payload = payload;
          }
          // Populate texture with assets.
          console.log(`Populating textures for submodel #${
              otherBuffer.si} into buffer #${otherBufferIdx}`);
          return populateTexture(otherBuffer.texture, content.payload);
        }).then(() => {
          otherBuffer.state = READY;
          console.log(`Submodel #${otherBuffer.si} is ready for rendering`);
          hideLoading();
        });
}

/**
 * Limit the number of submodel payloads in Host RAM.
 */
function garbageCollectSubmodelPayloads() {
  // Draw up a list of candidate submodels to delete. Anything that has a
  // payload is a candidate.
  let candidates = [];
  for (let si of Object.keys(gSubmodelSceneContents)) {
    let content = getSubmodelContent(si);
    if (content.payload == null) {
      continue;
    }
    candidates.push({
      lastTouched: content.lastTouched || 0,
      si: si,
    });
  }

  // Sort submodel idxs by last touched, oldest first.
  let oldestFirst = (a, b) => {
    return a.lastTouched - b.lastTouched;
  };
  candidates.sort(oldestFirst);

  // Delete payload field from old submodels.
  for (let i = 0; i < candidates.length - gSubmodelCacheSize; ++i) {
    let si = candidates[i].si;
    console.log(`Deleting payload for submodel #${si}`);
    getSubmodelContent(si).payload = null;
  }

}

/**
 * Initialize scene content. This is a lightweight operation.
 */
function initializeSceneContent(sceneParams, router) {
  return {
    spec: createSceneSpec(sceneParams),
    params: sceneParams,
    router: router,
    payload: null,
  };
}


/**
 * Re-initializes texture buffers for sparse grid. These textures are not
 * reusable yet, as their shape can vary between submodels.
 */
function reinitializeSparseGridTextures(rmtb) {
  let texture = rmtb.texture.sparseGridTexture;

  // Dispose of existing textures.
  texture.blockIndicesTexture.texture.dispose();
  texture.rgbTexture.texture.dispose();
  texture.densityTexture.texture.dispose();
  texture.featuresTexture.texture.dispose();

  // Create new textures.
  let sparseGridSpec = getSubmodelContent(rmtb.si).spec.sparseGridSpec;
  rmtb.texture.sparseGridTexture = createEmptyTexture(sparseGridSpec);
}


/**
 * Initializes ping-pong texture buffers.
 */
async function initializePingPongBuffers(si) {
  // Instantiate the three.js scene without textures.
  let sceneContent = getSubmodelContent(si);
  gRayMarchScene = await initializeRayMarchScene(si, sceneContent);

  // Instantiate textures for the ping pong buffers.
  for (let rmtb of gRayMarchTextureBuffers) {
    rmtb.texture = createEmptyTexture(sceneContent.spec);
  }

  // Assign texture uniforms from the first buffer to the scene.
  setTextureUniforms(sceneContent.params, gRayMarchTextureBuffers[0].texture);
  gActiveRayMarchTextureBuffer = 0;
}


async function initializeDeferredMlp(si) {
  // Instantiate the three.js scene without textures.
  let sceneContent = getSubmodelContent(si);
  let sceneParams = sceneContent.params;

  if (sceneParams['export_store_deferred_mlp_separately']) {
    let url = sceneContent.router.translate('deferred_mlp.json');
    return loadJSONFile(url).then(registerDeferredMlp);
  }
  // DeferredMLP is stored in sceneParams.
  return registerDeferredMlp(sceneParams['deferred_mlp']);
}

/**
 * Assign sceneTexture's texture assets to the global ray march scene's
 * uniforms.
 */
function setTextureUniforms(sceneParams, sceneTexture) {
  let rayMarchUniforms = getRayMarchScene().children[0].material.uniforms;

  // Occupancy grids
  let occupancyGridTextures = sceneTexture.occupancyGridsTexture.gridTextures;
  let numOccupancyGrids = occupancyGridTextures.length;
  for (let i = 0; i < numOccupancyGrids; ++i) {
    let texture = occupancyGridTextures[i];
    let ri = numOccupancyGrids - i - 1;
    rayMarchUniforms['occupancyGrid_L' + ri]['value'] = texture.texture;
  }

  // Distance grid
  if (sceneParams['useDistanceGrid']) {
    let texture = sceneTexture.distanceGridsTexture.gridTextures[0].texture;
    rayMarchUniforms['distanceGrid']['value'] = texture;
  }

  // triplane
  let triplaneTexture = sceneTexture.triplaneTexture;
  rayMarchUniforms['planeDensity']['value'] =
      triplaneTexture.densityTexture.texture;
  rayMarchUniforms['planeRgb']['value'] = triplaneTexture.rgbTexture.texture;
  rayMarchUniforms['planeFeatures']['value'] =
      triplaneTexture.featuresTexture.texture;

  // sparse grid
  let sparseGridTexture = sceneTexture.sparseGridTexture;
  rayMarchUniforms['sparseGridBlockIndices']['value'] =
      sparseGridTexture.blockIndicesTexture.texture;
  rayMarchUniforms['sparseGridDensity']['value'] =
      sparseGridTexture.densityTexture.texture;
  rayMarchUniforms['sparseGridRgb']['value'] =
      sparseGridTexture.rgbTexture.texture;
  rayMarchUniforms['sparseGridFeatures']['value'] =
      sparseGridTexture.featuresTexture.texture;
  rayMarchUniforms['atlasSize']['value'] = new THREE.Vector3(
      sceneParams['atlas_width'],
      sceneParams['atlas_height'],
      sceneParams['atlas_depth'],
  );
}


function getTextureSizeInBytes(sceneTexture) {
  let numBytes = 0.0;
  let getTextureSize = (texture) => {
    if (texture == null) {
      return 0;
    }
    let image = texture.texture.image;
    return image.height * image.width * image.depth;
  };

  // Occupancy grids
  let occupancyGridTextures = sceneTexture.occupancyGridsTexture.gridTextures;
  let numOccupancyGrids = occupancyGridTextures.length;
  for (let i = 0; i < numOccupancyGrids; ++i) {
    let texture = occupancyGridTextures[i];
    numBytes += getTextureSize(texture) * 1;
  }

  // Distance grid
  if (sceneTexture.distanceGridsTexture.gridTextures.length > 0) {
    let texture = sceneTexture.distanceGridsTexture.gridTextures[0];
    numBytes += getTextureSize(texture) * 1;
  }

  // triplane
  let triplaneTexture = sceneTexture.triplaneTexture;
  numBytes += getTextureSize(triplaneTexture.rgbTexture) * 3;
  numBytes += getTextureSize(triplaneTexture.densityTexture) * 1;
  numBytes += getTextureSize(triplaneTexture.featuresTexture) * 4;

  // sparse grid
  let sparseGridTexture = sceneTexture.sparseGridTexture;
  numBytes += getTextureSize(sparseGridTexture.blockIndicesTexture) * 1;
  numBytes += getTextureSize(sparseGridTexture.rgbTexture) * 3;
  numBytes += getTextureSize(sparseGridTexture.densityTexture) * 1;
  numBytes += getTextureSize(sparseGridTexture.featuresTexture) * 4;

  return numBytes;
}


/**
 * Initializes global ray march scene using a reference submodel's scene
 * content.
 *
 * Uniforms for texture buffers are set to null. We assume that the reference
 * submodel content is more-or-less identical across all submodels. No reference
 * is made to the shape of texture assets.
 */
async function initializeRayMarchScene(si, sceneContent) {
  let sceneParams = sceneContent.params;
  let sceneSpec = sceneContent.spec;

  // Assemble shader code from header, on-the-fly generated view-dependency
  // functions and body.
  let fragmentShaderSource = kRayMarchFragmentShaderHeader;
  fragmentShaderSource += await loadTextFile('viewdependency.glsl');
  fragmentShaderSource += await loadTextFile('fragment.glsl');
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

  fragmentShaderSource = '#define kSubmodelScale ' +
      Number(getSubmodelScale(si)).toFixed(10) + '\n' + fragmentShaderSource;
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
    'minPosition': {'value': minPosition},

    'world_T_cam': {'value': new THREE.Matrix4()},
    'cam_T_clip': {'value': new THREE.Matrix4()},
    'worldspaceROpengl': {'value': worldspaceROpengl},
  };

  occupancyUniforms = {};
  let occupancyGridSpecs = sceneSpec.occupancyGridsSpec.gridSpecs;
  let numOccupancyGrids = occupancyGridSpecs.length;
  for (let i = 0; i < numOccupancyGrids; ++i) {
    // Initialize occupancy grid shader
    let spec = occupancyGridSpecs[i];
    let ri = numOccupancyGrids - i - 1;
    fragmentShaderSource = '#define kVoxelSizeOccupancy_L' + ri + ' ' +
        Number(spec.voxelSize).toFixed(10) + '\n' +
        fragmentShaderSource;
    fragmentShaderSource = '#define kGridSizeOccupancy_L' + ri + ' vec3(' +
        Number(spec.shape[0]).toFixed(10) + ', ' +
        Number(spec.shape[1]).toFixed(10) + ', ' +
        Number(spec.shape[2]).toFixed(10) + ')\n' +
        fragmentShaderSource;

    // Initialize occupancy grid uniforms
    occupancyUniforms['occupancyGrid_L' + ri] = {'value': null};
  }
  rayMarchUniforms = extend(rayMarchUniforms, occupancyUniforms);

  if (sceneParams['useDistanceGrid']) {
    // Initialize distance grid shader
    let spec = sceneSpec.distanceGridsSpec.gridSpecs[0];
    fragmentShaderSource = '#define USE_DISTANCE_GRID\n' + fragmentShaderSource;
    fragmentShaderSource = '#define kVoxelSizeDistance ' +
        Number(spec.voxelSize).toFixed(10) + '\n' + fragmentShaderSource;
    fragmentShaderSource = '#define kGridSizeDistance vec3(' +
        Number(spec.shape[0]).toFixed(10) + ', ' +
        Number(spec.shape[1]).toFixed(10) + ', ' +
        Number(spec.shape[2]).toFixed(10) + ')\n' + fragmentShaderSource;

    // Initialize distance grid uniforms
    distanceUniforms = {'distanceGrid': {'value': null}};
    rayMarchUniforms = extend(rayMarchUniforms, distanceUniforms);
  }

  let backgroundColor = new THREE.Color(0.5, 0.5, 0.5);
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
    exposureUniforms = {'exposure': {'value': gExposure}};
    rayMarchUniforms = extend(rayMarchUniforms, exposureUniforms);
  }

  const activation =
      sceneParams['activation'] ? sceneParams['activation'] : 'elu';
  fragmentShaderSource =
      '#define ACTIVATION_FN ' + activation + '\n' + fragmentShaderSource;

  if (sceneParams['feature_gating'] === null ||
      sceneParams['feature_gating'] === undefined ||
      sceneParams['feature_gating'] === true) {
    fragmentShaderSource =
        '#define USE_FEATURE_GATING\n' + fragmentShaderSource;
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
    fragmentShaderSource = '#define kVisibilityDelay ' +
        Number(sceneParams['step_size_visibility_delay']).toFixed(10) + '\n' +
        fragmentShaderSource;
  }

  // Initialize triplane shader
  let triplaneGridSize = new THREE.Vector2(...sceneSpec.triplaneSpec.shape);
  fragmentShaderSource = '#define kTriplaneVoxelSize ' +
      Number(sceneParams['triplane_voxel_size']).toFixed(10) + '\n' +
      fragmentShaderSource;
  fragmentShaderSource = '#define kTriplaneGridSize vec2(' +
      Number(triplaneGridSize.x).toFixed(10) + ', ' +
      Number(triplaneGridSize.y).toFixed(10) + ')\n' + fragmentShaderSource;

  // Initialize triplane uniforms
  let triplaneUniforms = {
    'planeDensity': {'value': null},
    'planeRgb': {'value': null},
    'planeFeatures': {'value': null},
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
    'sparseGridBlockIndices': {'value': null},
    'sparseGridDensity': {'value': null},
    'sparseGridRgb': {'value': null},
    'sparseGridFeatures': {'value': null},
    'atlasSize': {'value': null},
  };
  rayMarchUniforms = extend(rayMarchUniforms, sparseGridUniforms);

  // Shader editor
  let shaderEditor = document.getElementById('shader-editor');
  shaderEditor.value = fragmentShaderSource;

  // Bundle uniforms, vertex and fragment shader in a material
  let rayMarchMaterial = new THREE.ShaderMaterial({
    uniforms: rayMarchUniforms,
    vertexShader: kRayMarchVertexShader,
    fragmentShader: fragmentShaderSource,
    vertexColors: true,
  });
  rayMarchMaterial.side = THREE.DoubleSide;
  rayMarchMaterial.depthTest = false;
  rayMarchMaterial.needsUpdate = true;

  const plane = new THREE.PlaneBufferGeometry(...gViewportDims);
  let mesh = new THREE.Mesh(plane, rayMarchMaterial);
  mesh.position.z = -100;
  mesh.frustumCulled = false;

  let scene = new THREE.Scene();
  scene.add(mesh);
  scene.autoUpdate = false;

  return scene;
}


/**
 * Validates shape of DeferredMLP parameters.
 */
function validateDeferredMlp(deferredMlp) {
  const mlpName =
      !!deferredMlp['ResampleDense_0/kernel'] ? 'ResampleDense' : 'Dense';

  // WARNING: There must be EXACTLY three ResampleDense layers in the
  // DeferredMLP!!
  for (let li = 0; li < 3; li++) {
    const layerName = `${mlpName}_${li}`;
    let kernelShape = deferredMlp[`${layerName}/kernel`]['shape'];
    let biasShape = deferredMlp[`${layerName}/bias`]['shape'];
    if (mlpName === 'ResampleDense') {
      let gridSize = kernelShape[1];

      // We assume that all grid dimensions are identical
      console.assert(
          gridSize === kernelShape[2] && gridSize === kernelShape[3]);

      // We also require the grid shape and the bias shape to match.
      console.assert(
          kernelShape[0] === biasShape[0] && kernelShape[1] === biasShape[1] &&
          kernelShape[2] === biasShape[2] && kernelShape[3] === biasShape[3]);
    }
  }
}
