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
 * @fileoverview Global state of the web viewer.
 */

/**
 * Our framerate display.
 * @type {?Object}
 */
let gStats = null;

/**
 * If enabled, expect multiple submodels.
 */
let gUseSubmodel = false;

/**
 * Transform from world coordinates to the current submodel.
 */
let gSubmodelTransform = null;

/**
 * Deferred MLP parameters for the current submodel.
 */
let gDeferredMlp = null;

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
  DISPLAY_FEATURES: 2,
  /** Only shows the view dependent component. */
  DISPLAY_VIEW_DEPENDENT: 3,
  /** Only shows the coarse block grid. */
  DISPLAY_COARSE_GRID: 4,
};

/**  @type {!DisplayModeType}  */
let gDisplayMode = DisplayModeType.DISPLAY_NORMAL;

/**
 * If true we evaluate run-time performance by re-rendering test viewpoints.
 * @type {boolean}
 */
let gBenchmark = false;

/**
 * For benchmarking with vsync on: render this many redundant images per frame.
 * @type {number}
 */
let gFrameMult = 1;

/**
 * A web worker for parsing binary assets in a separate thread.
 * @type {*}
 */
let gLoadAssetsWorker = new WorkerPool(4, "loadpng.worker.js");


/**
 * A web worker for merging slices together.
 * @type {*}
 */
let gCopySliceWorker = new WorkerPool(4, "copyslices.worker.js");

/**
 * The vertex shader for rendering a baked MERF scene with ray marching.
 * @const {string}
 */
const kRayMarchVertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
uniform mat4 world_T_cam;
uniform mat4 cam_T_clip;

void main() {
  vec4 posClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = posClip;
  posClip /= posClip.w;

  vec4 originCam = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 nearPointCam = cam_T_clip * vec4(posClip.x, posClip.y, -1.0, 1.0);
  nearPointCam /= -nearPointCam.z;

  vec4 originWorld = world_T_cam * originCam;
  vec4 nearPointWorld = world_T_cam * nearPointCam;
  vOrigin = originWorld.xyz / originWorld.w;
  vDirection = nearPointWorld.xyz / nearPointWorld.w - vOrigin;
}
`;

/**
 * We build the ray marching shader programmatically, this string contains the
 * header for the shader.
 * @const {string}
 */
const kRayMarchFragmentShaderHeader = `
precision highp float;

varying vec3 vOrigin;
varying vec3 vDirection;

uniform int displayMode;

uniform mat3 worldspaceROpengl;
uniform float nearPlane;

#ifdef USE_DISTANCE_GRID
uniform highp sampler3D distanceGrid;
uniform highp sampler3D occupancyGrid_L0;
#else
uniform highp sampler3D occupancyGrid_L0;
uniform highp sampler3D occupancyGrid_L1;
uniform highp sampler3D occupancyGrid_L2;
#ifndef USE_BITS
uniform highp sampler3D occupancyGrid_L3;
uniform highp sampler3D occupancyGrid_L4;
#endif
#endif

uniform vec4 bias_0[NUM_CHANNELS_ONE/4];
uniform vec4 bias_1[NUM_CHANNELS_TWO/4];
uniform vec4 bias_2[NUM_CHANNELS_THREE/4];

uniform highp sampler2D weightsZero;
uniform highp sampler2D weightsOne;
uniform highp sampler2D weightsTwo;

#ifdef USE_EXPOSURE
uniform float exposure;
#endif

uniform vec3 atlasSize;

uniform highp sampler3D sparseGridBlockIndices;
uniform highp sampler3D sparseGridDensity;
uniform highp sampler3D sparseGridRgb;
uniform highp sampler3D sparseGridFeatures;

// need to use texture arrays, otherwise we exceed max texture unit limit
uniform highp sampler2DArray planeDensity;
uniform highp sampler2DArray planeRgb;
uniform highp sampler2DArray planeFeatures;
`;


/**
 * The THREE.js renderer object we use.
 * @type {?THREE.WebGLRenderer}
 */
let gRenderer = null;


/**
 * The number of submodels
 */
let gSubmodelCount = 1;


/**
 * The perspective camera we use to view the scene.
 * @type {?THREE.PerspectiveCamera}
 */
let gCamera = null;


let gViewportDims = [640, 480];
