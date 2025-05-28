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
 * @fileoverview Global variables shared among files.
 */

/**
 * Keeps track of whether loadOnFirstFrame has already been run.
 * @type {boolean}
 */
let gFirstFrameRendered = false;

/**
 * Our framerate display.
 * @type {?Object}
 */
let gStats = null;

/**
 * Number of channels for encoding appearance.
 * @type {number}
 */
let gNumChannels = 24;

/**
 * For historic reasons we only store four channels per file.
 * @type {number}
 */
let gNumChannelsPerChunk = 4;

/**
 * Since we only store four channels per file we need multiple files to hold
 * our 24-channel texture representation.
 * @type {number}
 */
let gNumChannelChunks = gNumChannels / gNumChannelsPerChunk;

/**
 * Frame is rendered `gFrameMult` times. This allows us to benchmark
 * frame rates that exceeded the display refresh rate.
 * @type {number}
 */
let gFrameMult = 1;

/** @type {!THREE.PerspectiveCamera} */
let gCamera = null;

/** @type {?THREE.OrbitControls} */
let gControls = null;

/** @type {?THREE.WebGL2Renderer} */
let gRenderer = null;

/** @type {?THREE.Scene} */
let gScene = null;

/** @type {?THREE.ShaderMaterial} */
let gMaterial = null;

/** @type {?THREE.ShaderMaterial} */
let gDeferredMaterial = null;

/**
 * This g-buffer holds world space positions for deferred rendering.
 * @type {?THREE.WebGLRenderTarget}
 */
let gPositionWorldRenderTarget = null;

/**
 * This is a plane proxy geometry for the deferred rendering pass.
 * @type {?THREE.Mesh}
 */
let gDeferredMesh = null;

/**
 * This scene only contains the proxy plane used for deferred rendering.
 * @type {?THREE.Scene}
 * */
let gDeferredScene = null;

/**
 * The sparse grids are resampled into these feature buffers in seperate
 * rendering passes.
 * @type {?THREE.WebGLRenderTarget}
 */
let gFeatureMaps = [];

/**
 * This volume texture is progressively populated in the render loop.
 * @type {?THREE.DataTexture3D}
 */
let gSparseGridFeaturesTexture = [];

/**
 * Different display modes for debugging rendering.
 * @enum {number}
 */
const DisplayModeType = {
  /** Runs the full model with view dependence. */
  DISPLAY_FULL: 0,
  /** Disables the view-dependence network. */
  DISPLAY_DIFFUSE: 1,
  /** Only shows the view dependent component. */
  DISPLAY_VIEW_DEPENDENT: 2,
  /** Visualizes the surface normals of the mesh. */
  DISPLAY_NORMALS: 3,
  /** Visualizes the mesh using diffuse shading and a white albedo. */
  DISPLAY_SHADED: 4,
  /** Visualizes the depth map as 1/z. */
  DISPLAY_DEPTH: 5,
};

/**  @type {!DisplayModeType}  */
let gDisplayMode = DisplayModeType.DISPLAY_FULL;

/**
 * Number of bytes that already have been decoded for keeping track of
 * download progress.
 * @type {number}
 */
let gBytesDecoded = 0;

/**
 * Total number of bytes to decode for keeping track of download progress.
 * @type {number}
 */
let gTotalBytesToDecode = null;
