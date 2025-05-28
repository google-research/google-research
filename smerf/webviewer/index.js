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
 * @fileoverview Main driver for web viewer.
 */


/**
 * panel for current submodel
 */
let gSubmodelPanel = null;
let gVMemPanel = null;

/**
 * Number of sample points per voxel.
 * @type {number}
 */
let gStepMult = 1;


/**
 * For large scenes with varying exposure we set this value to be the exposure
 * of the virtual camera (shutter_speed_in_seconds * iso / 1000).
 * @type {number}
 */
let gExposure = null;


/**
 * Loads full scene representation.
 *
 * This includes all submodel assets, including allocation and download.
 *
 * This function should be called exactly once.
 *
 * @param {string} dirUrl Either points to a directory that contains scene files
 *                        or to a json file that maps virtual filenames to
 *                        download links
 * @param {!object} overrideParams A dictionary that contains overrides for the
 *   params in scene_params.json (e.g. combineMode, deferredMode or useBits).
 */
function loadScene(dirUrl, overrideParams) {
  // Check if dirUrl points to a json file or to a directory.
  let filenameToLinkPromise;
  if (dirUrl.includes('.json')) {
    // If this is the case, we fetch a JSON file that maps filenames to links.
    filenameToLinkPromise = loadJSONFile(dirUrl);
  } else {
    // Otherwise, the scene files directly lie at dirUrl and we create a
    // dummy promise that resolves immediately.
    filenameToLinkPromise = Promise.resolve(null);
  }

  // This variable is defined in progressive.js.
  filenameToLinkPromise
      .then(filenameToLink => {
        // Mapping from fake filenames to real filenames under root directory
        // dirUrl.
        const router = new Router(dirUrl, filenameToLink);

        // Loads scene parameters (voxel grid size, view-dependence MLP).
        const sceneParamsUrl = router.translate('scene_params.json');
        const sceneParamsPromise = loadJSONFile(sceneParamsUrl);

        if (overrideParams['loadBenchmarkCameras']) {
          loadBenchmarkCameras(router);
        }

        // Some of the shader code is stored in seperate files.
        return Promise.all([sceneParamsPromise, {router, filenameToLink}]);
      })
      .then(parsed => {
        // scene_params.json for this scene.
        // Translates filenames to full URLs.
        const [sceneParams, carry] = parsed;

        // Determine if there are multiple submodels or not. If so, figure out
        // how many.
        let initialSubmodelIndex = 0;
        gUseSubmodel =
            (sceneParams.hasOwnProperty('num_local_submodels') &&
             sceneParams['num_local_submodels'] > 1);
        if (gUseSubmodel) {
          // Override default submodel to the user chose by URL.
          gSubmodelCount = sceneParams['num_local_submodels'];
          initialSubmodelIndex =
              sceneParams['sm_to_params'][sceneParams['submodel_idx']];
        }

        // Get links to scene_params.json files for each submodel.
        let sceneParamsPromises = [];
        for (let si = 0; si < gSubmodelCount; ++si) {
          // Get the submodel ids participating in this scene.
          const submodelId = sceneParams['params_to_sm'][si];
          // Find path to its scene_params.json file.
          const filePath = carry.router.translate(
              submodelAssetPath(submodelId, 'scene_params.json'));
          // Construct path to scene_params.json for this submodel.
          sceneParamsPromises.push(loadJSONFile(filePath));
        }

        // Wait for all scene_params.json files to be loaded.
        return Promise.all(
            [{...carry, initialSubmodelIndex}, ...sceneParamsPromises]);
      })
      .then(loaded => {
        let [carry, ...submodelSceneParams] = loaded;
        for (let si = 0; si < submodelSceneParams.length; ++si) {
          // Override the scene params using the URL GET variables.
          submodelSceneParams[si] =
              extend(submodelSceneParams[si], overrideParams);

          // Build fake-filename-to-real-filename translator for this
          // submodel.
          const submodelId = submodelSceneParams[si]['params_to_sm'][si];
          let subDirUrl = dirUrl;
          if (gUseSubmodel) {
            subDirUrl = `${subDirUrl}/${submodelAssetPath(submodelId)}`;
          }
          let submodelRouter = new Router(subDirUrl, carry.filenameToLink);

          // Load all assets related to this submodel. This is not a blocking
          // operation.
          // TODO: Consider loading this content on-demand and using an LRU
          // cache to bound memory usage.
          let submodelContent =
              initializeSceneContent(submodelSceneParams[si], submodelRouter);
          console.log(`spec for submodel #${si}:`, submodelContent.spec);

          // Register submodel content with the texture manager.
          registerSubmodelContent(si, submodelContent);
        }

        // Now that we know the submodel scale we can set the camera pose.
        let si = carry.initialSubmodelIndex;
        setupInitialCameraPose(
            dirUrl,
            submodelCenter(si, getSubmodelContent(si).params),
        );

        // Instantiate scene & texture buffers.
        return Promise.all([si, initializeDeferredMlp(si)]);
      }).then(([si, _]) => {
        return initializePingPongBuffers(si);
      }).then(() => {
        return requestAnimationFrame(renderNextFrame);
      });
}

/**
 * Initializes the application based on the URL parameters.
 */
function initFromParameters() {
  // HTTP GET query parameters
  const params = new URL(window.location.href).searchParams;

  // Base directory for all assets.
  const dirUrl = params.get('dir');

  // Screen size: <width>,<height>
  const size = params.get('s');

  // Controls platform-specific defaults: phone, low, medium, high. Not
  // const as benchmark=true can override it.
  let quality = params.get('quality');

  // Number of samples per voxel. Increase for slower rendering and fewer
  // artifacts.

  const stepMult = params.get('stepMult');
  if (stepMult) {
    gStepMult = parseInt(stepMult, 10);
  }
  const frameMult = params.get('frameMult');
  if (frameMult) {
    gFrameMult = parseInt(frameMult, 10);
  }

  // Manually specify exposure for exposure-aware models.
  const exposure = params.get('exposure');
  if (exposure) {
    gExposure = parseFloat(exposure);
  }

  // For manually overriding parameters in scene_params.json.
  let overrideParams = {};

  const benchmarkParam = params.get('benchmark');
  const benchmark = benchmarkParam &&
      (benchmarkParam.toLowerCase() === 'time' ||
       benchmarkParam.toLowerCase() === 'quality');
  if (benchmark) {
    overrideParams['loadBenchmarkCameras'] = true;
    quality = 'high';
    const sceneNameChunks = dirUrl.split('/').slice(-2);
    setupBenchmarkStats(
        sceneNameChunks[0] + '_' + sceneNameChunks[1],
        benchmarkParam.toLowerCase() === 'quality');
  }

  // snerg, vfr
  const deferredMode = params.get('deferredMode');
  if (deferredMode) {
    overrideParams['deferred_rendering_mode'] = deferredMode;
  }

  // sum, concat_and_sum
  const combineMode = params.get('combineMode');
  if (combineMode && combineMode === 'concat_and_sum') {
    overrideParams['merge_features_combine_op'] = 'coarse_sum';
  }

  // are occupancy grids bitpacked?
  const useBits = params.get('useBits');
  if (useBits) {
    overrideParams['useBits'] = useBits.toLowerCase() === 'true';
  }

  // Use distance grid for calculating step sizes.
  const useDistanceGrid = params.get('useDistanceGrid');
  if (useDistanceGrid) {
    overrideParams['useDistanceGrid'] =
        useDistanceGrid.toLowerCase() === 'true';
  }

  // Load legacy scenes, where the distance & occupancy grids are stored
  // as a single monolithic file.
  const legacyGrids = params.get('legacyGrids');
  if (legacyGrids) {
    overrideParams['legacyGrids'] = legacyGrids.toLowerCase() === 'true';
  }

  // Sets the activation function of the DeferredMLP. Either "relu" or "elu".
  // Defaults to elu.
  const activation = params.get('activation');
  if (activation) {
    overrideParams['activation'] = activation;
  }

  // Whether to use feature gating for the triplanes. Either "true" or "false".
  // Defaults to true.
  const featureGating = params.get('featureGating');
  if (featureGating) {
    overrideParams['feature_gating'] = featureGating.toLowerCase() === 'true';
  }

  // Limit the number of cached submodel payloads.
  const submodelCacheSize = params.get('submodelCacheSize');
  if (submodelCacheSize) {
    gSubmodelCacheSize = Number(submodelCacheSize);
  }

  // Merge slices of assets together before binding to WebGL texture.
  const mergeSlices = params.get('mergeSlices');
  if (mergeSlices) {
    overrideParams['merge_slices'] = mergeSlices == 'true';
  }

  // The background color (in hex, e.g. #FF0000 for red) that the scene is
  // rendered on top of. Defaults to medium grey.
  const backgroundColor = params.get('backgroundColor');
  if (backgroundColor) {
    overrideParams['backgroundColor'] = '#' + backgroundColor;
  }

  const usageString =
      'To view a MERF scene, specify the following parameters in the URL:\n' +
      'dir: (Required) The URL to a MERF scene directory.\n' +
      'quality: (Optional) A quality preset (phone, low, medium or high).\n' +
      'mouseMode:  (Optional) How mouse navigation works: "orbit" for object' +
      ' centric scenes, "fps" for large scenes on a device with a mouse, or' +
      ' "map" for large scenes on a touchscreen device.\n' +
      'stepMult:  (Optional) Multiplier on how many steps to take per ray.\n' +
      'frameMult:  (Optional) For benchmarking with vsync on: render ' +
      ' frameMult redudant images per frame.\n' +
      'backgroundColor: (Optional) The background color (in hex, e.g. red is' +
      ' FF0000) the scene is rendered on top of. Defaults to grey.\n' +
      'exposure: (Optional, only for large scenes) The exposure value of the' +
      ' virtual camera (shutter_speed_seconds * iso / 1000).\n' +
      'deferredMode (Optional, internal) The deferred rendering mode for' +
      ' view-dependence. Either "snerg" or "vfr".\n' +
      'combineMode (Optional, internal) How to combine the features from' +
      ' the triplanes with the features from the sparse 3D grid. Either ' +
      ' "sum" or "concat_and_sum".\n' +
      'useBits (Optional, internal) If true, use bit packing for a higher' +
      ' resolution occupancy grid.\n' +
      'useDistanceGrid (Optional, internal) If true, use a distance grid for' +
      ' empty space skipping instead of a hierarchy of occupancy grids.\n' +
      'activation (Optional, internal) The activation function for the' +
      ' DeferredMLP: "elu" (default) or "relu" (for older models).\n' +
      'featureGating (Optional, internal) If true, use feature gating for the' +
      ' triplanes. Set to false for older MERF scenes.\n' +
      'benchmark (Optional) If "time" or "quality", sets quality=high and' +
      ' renders the viewpoints in [scene_dir]/test_frames.json. If set to' +
      ' "time", only frame times are reported. If set to "quality", then' +
      ' the rendered images are also downloaded as "%04d.png".\n' +
      's: (Optional) The dimensions as width,height. E.g. 640,360.\n' +
      'vfovy:  (Optional) The vertical field of view of the viewer.\n';

  if (!dirUrl) {
    error('dir is a required parameter.\n\n' + usageString);
    return;
  }

  // Default to filling the browser window, and rendering at full res.
  let frameBufferWidth = window.innerWidth -
      13;  // Body has a padding of 6 + 5px, we have a border of 2px.
  let frameBufferHeight = window.innerHeight - 19;
  let lowResFactor = parseInt(params.get('downscale') || 1, 10);

  if (size) {
    const match = size.match(/([\d]+),([\d]+)/);
    frameBufferWidth = parseInt(match[1], 10);
    frameBufferHeight = parseInt(match[2], 10);
  } else if (quality) {
    // No size specified, clip the max viewport size based on quality.
    if (quality == 'phone') {  // For iPhones.
      frameBufferWidth = Math.min(350, frameBufferWidth);
      frameBufferHeight = Math.min(600, frameBufferHeight);
    } else if (quality == 'low') {  // For laptops with integrated GPUs.
      frameBufferWidth = Math.min(1280, frameBufferWidth);
      frameBufferHeight = Math.min(800, frameBufferHeight);
    } else if (quality == 'medium') {  // For laptops with dicrete GPUs.
      frameBufferWidth = Math.min(1920, frameBufferWidth);
      frameBufferHeight = Math.min(1080, frameBufferHeight);
    }  // else assume quality is 'high' and render at full res.
  }

  // No downscale factor specified, estimate it from the quality setting.
  let stepSizeVisibilityDelay = 0.99;
  if (!params.get('downscale') && quality) {
    let maxPixelsPerFrame = frameBufferWidth * frameBufferHeight;
    if (quality == 'phone') {  // For iPhones.
      maxPixelsPerFrame = 300 * 450;
      stepSizeVisibilityDelay = 0.8;
    } else if (quality == 'low') {  // For laptops with integrated GPUs.
      maxPixelsPerFrame = 600 * 250;
      stepSizeVisibilityDelay = 0.8;
    } else if (quality == 'medium') {  // For laptops with dicrete GPUs.
      maxPixelsPerFrame = 1200 * 640;
      stepSizeVisibilityDelay = 0.95;
    }  // else assume quality is 'high' and render at full res.

    while (frameBufferWidth * frameBufferHeight / lowResFactor >
           maxPixelsPerFrame) {
      lowResFactor++;
    }

    console.log('Automatically chose a downscaling factor of ' + lowResFactor);
  }
  overrideParams['useLargerStepsWhenOccluded'] = true;
  overrideParams['step_size_visibility_delay'] = stepSizeVisibilityDelay;

  // Near plane distance in world coordinates.
  const nearPlane = parseFloat(params.get('near') || 0.2);

  // FOV along screen height. Specified in degrees.
  const vfovy = parseFloat(params.get('vfovy') || 40.0);

  // Create container for viewport.
  const view = create('div', 'view');
  setDims(view, frameBufferWidth, frameBufferHeight);
  view.textContent = '';

  const viewSpaceContainer = document.getElementById('viewspacecontainer');
  viewSpaceContainer.style.display = 'inline-block';

  const viewSpace = document.querySelector('.viewspace');
  viewSpace.textContent = '';
  viewSpace.appendChild(view);

  let canvas = document.createElement('canvas');
  view.appendChild(canvas);

  // Add tool for visualizing framerate.
  gStats = Stats();
  gStats.dom.style.position = 'absolute';
  viewSpace.appendChild(gStats.dom);

  gSubmodelPanel = gStats.addPanel(new Stats.Panel('SM', '#0ff', '#002'));
  gSubmodelPanel.update(getActiveSubmodelIndex());

  gVMemPanel = gStats.addPanel(new Stats.Panel('MB VRAM', '#0ff', '#002'));
  gVMemPanel.update(0);

  // Show FPS; hide other panels.
  gStats.showPanel(0);

  // Set up a high performance WebGL context, making sure that anti-aliasing is
  // turned off.
  let gl = canvas.getContext('webgl2', {
    powerPreference: 'high-performance',
    alpha: false,
    stencil: false,
    precision: 'highp',
    depth: false,
    antialias: false,
    desynchronized: false,
    preserveDrawingBuffer:
        benchmarkParam && benchmarkParam.toLowerCase() === 'quality',
  });
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gRenderer = new THREE.WebGLRenderer({
    canvas: canvas,
    context: gl,
  });

  // Set up the normal scene used for rendering.
  gCamera = new THREE.PerspectiveCamera(
      vfovy,
      Math.trunc(view.offsetWidth / lowResFactor) /
          Math.trunc(view.offsetHeight / lowResFactor),
      nearPlane, 100.0);
  setupProgressiveRendering(view, lowResFactor);
  gRenderer.autoClear = false;
  gRenderer.setSize(view.offsetWidth, view.offsetHeight);

  // Disable camera controls if we're benchmarking, since both OrbitControls
  // and PointerLockControls take ownership of the camera poses, making it
  // impossible to change programatically.
  if (!benchmark) {
    const mouseMode = params.get('mouseMode');
    setupCameraControls(mouseMode, view);
  }

  let width = Math.trunc(view.offsetWidth / lowResFactor);
  let height = Math.trunc(view.offsetHeight / lowResFactor);
  setupViewport(width, height);

  loadScene(dirUrl, overrideParams);
}

/**
 * The main update function that gets called every frame.
 *
 * @param {number} t elapsed time between frames (ms).
 */
function renderNextFrame(t) {
  // Delete old submodels to keep memory usage in check.
  garbageCollectSubmodelPayloads();

  // Attempt to set the current ray march scene. This will kick off the process
  // of instantiating a new scene if necessary.
  let submodelIndex =
      positionToSubmodel(gCamera.position, getActiveSubmodelContent().params);
  setCurrentRayMarchScene(submodelIndex);

  // setCurrentRayMarchScene() may not actually change the scene. Use the
  // index of the current active submodel instead.
  submodelIndex = getActiveSubmodelIndex();

  let sceneParams = getSubmodelContent(submodelIndex).params;

  for (let i = 0; i < gFrameMult; ++i) {
    gSubmodelTransform = submodelTransform(submodelIndex, sceneParams);

    gSubmodelPanel.update(submodelIndex);
    gVMemPanel.update(getCurrentTextureUsageInBytes() / 1e6);

    updateCameraControls();

    // For benchmarking, we want to direcly set the projection matrix.
    if (!gBenchmark) {
      gCamera.updateProjectionMatrix();
    }
    gCamera.updateMatrixWorld();

    const currentSubmodelCenter = submodelCenter(submodelIndex, sceneParams);
    const submodelScale = getSubmodelScale(submodelIndex);
    let submodelCameraPosition = new THREE.Vector3().copy(gCamera.position);
    submodelCameraPosition.sub(currentSubmodelCenter);
    submodelCameraPosition.multiplyScalar(submodelScale);

    let shaderUniforms = getRayMarchScene().children[0].material.uniforms;

    // Make sure to free up GPU memory from the previous frames.
    if (!!shaderUniforms['weightsZero']['value']) {
      shaderUniforms['weightsZero']['value'].dispose();
    }
    if (!!shaderUniforms['weightsOne']['value']) {
      shaderUniforms['weightsOne']['value'].dispose();
    }
    if (!!shaderUniforms['weightsTwo']['value']) {
      shaderUniforms['weightsTwo']['value'].dispose();
    }

    shaderUniforms['bias_0']['value'] =
        trilerpDeferredMlpBiases(submodelIndex, 0, submodelCameraPosition);
    shaderUniforms['bias_1']['value'] =
        trilerpDeferredMlpBiases(submodelIndex, 1, submodelCameraPosition);
    shaderUniforms['bias_2']['value'] =
        trilerpDeferredMlpBiases(submodelIndex, 2, submodelCameraPosition);

    shaderUniforms['weightsZero']['value'] =
        trilerpDeferredMlpKernel(submodelIndex, 0, submodelCameraPosition);
    shaderUniforms['weightsOne']['value'] =
        trilerpDeferredMlpKernel(submodelIndex, 1, submodelCameraPosition);
    shaderUniforms['weightsTwo']['value'] =
        trilerpDeferredMlpKernel(submodelIndex, 2, submodelCameraPosition);


    renderProgressively();
  }
  gStats.update();

  // By default we schedule the next frame ASAP, but the benchmark mode can
  // override this by replacing this lambda.
  let scheduleNextFrame = () => {
    requestAnimationFrame(renderNextFrame);
  };
  if (gBenchmark) {
    scheduleNextFrame = benchmarkPerformance(scheduleNextFrame);
  }
  scheduleNextFrame();
}

/**
 * Starts the volumetric scene viewer application.
 */
function start() {
  initFromParameters();
  addHandlers();
}

window.onload = start;
