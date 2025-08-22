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
 * @fileoverview Progressive rendering.
 */

/**
 * The THREE.js renderer object we use.
 * @type {?THREE.WebGLRenderer}
 */
let gRenderer = null;

/**
 * This scene renders the baked NeRF reconstruction using ray marching.
 * @type {?THREE.Scene}
 */
let gRayMarchScene = null;

/**
 * The perspective camera we use to view the scene.
 * @type {?THREE.PerspectiveCamera}
 */
let gCamera = null;

/**
 * This is the main bookkeeping scene for progressive upsampling, as it
 * keeps track of multiple low-res frames, and their corresponding filter
 * weights.
 * @type {?THREE.Scene}
 */
let gSceneAccumulate = null;

/**
 * A lower res orthographic camera used to kick off ray marching
 * with a full-screen render pass.
 * @type {?THREE.OrthographicCamera}
 */
let gLowResBlitCamera = null;

/**
 * A higher res orthographic camera used to perform full-resolution
 * post-processing passes.
 * @type {?THREE.OrthographicCamera}
 */
let gHighResBlitCamera = null;

/**
 * Keeps track of the camera transformation matrix, so we can turn off
 * progressive rendering when the camera moves
 * @type {?THREE.Matrix4}
 */
let gOldMatrixWorld = null;

/**
 * Keeps track of the camera projection matrix, so we can turn off
 * progressive rendering when the camera zooms in or out.
 * @type {?THREE.Matrix4}
 */
let gOldProjectionMatrix = null;

/**
 * Counts the current frame number, used for random sampling.
 * @type {number}
 */
let gFrameIndex = 0;

/**
 * This is a half-res rendertarget used for progressive rendering.
 * @type {?THREE.WebGLRenderTarget}
 */
let gLowResTexture = null;


/**
 * @param {!THREE.Texture} textureLowRes
 * @param {!THREE.Texture} textureHistory
 * @param {!THREE.Vector2} lowResolution
 * @param {!THREE.Vector2} highResolution
 * @return {!THREE.Material}
 */
function createAccumulateMaterial(
    textureLowRes, textureHistory, lowResolution, highResolution) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      'mapLowRes': {'value': textureLowRes},
      'mapHistory': {'value': textureHistory},
      'lowResolution': {'value': lowResolution},
      'highResolution': {'value': highResolution},
      'jitterOffset': {'value': new THREE.Vector2(0.0, 0.0)},
      'emaAlpha': {'value': 0.15},
    },
    vertexShader: accumulateVertexShader,
    fragmentShader: accumulateFragmentShader,
  });

  return material;
}

/**
 * These are the ping-pong buffers used for progressive upsampling. Every frame
 * we read from one buffer, and write into the other. This allows us to maintain
 * a history of multiple low-res frames, and their corresponding filter
 * weights.
 * @type {!Array<?THREE.WebGLRenderTarget>}
 */
const gAccumulationTextures = [null, null];

/** @const {string}  */
const normalizeVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/** @const {string} */
const normalizeFragmentShader = `
varying vec2 vUv;
uniform sampler2D map;
void main() {
  gl_FragColor = texture2D(map, vUv);
  if (gl_FragColor.a > 0.0) {
    gl_FragColor.rgb /= gl_FragColor.a;
  }
  gl_FragColor.a = 1.0;
}
`;

/**
 * @param {!THREE.Texture} texture
 * @return {!THREE.Material}
 */
function createNormalizeMaterial(texture) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      'map': {'value': texture},
    },
    vertexShader: normalizeVertexShader,
    fragmentShader: normalizeFragmentShader,
  });

  return material;
}

/**
 * Blits a texture into the framebuffer, normalizing the result using
 * the alpha channel. I.e. pixel_out = pixel_in.rgba / pixel_in.a.
 * @type {?THREE.Scene}
 */
let gSceneNormalize = null;

/** @const {string}  */
const accumulateVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/** @const {string} */
const accumulateFragmentShader = `
varying vec2 vUv;
uniform vec2 lowResolution;
uniform vec2 highResolution;
uniform vec2 jitterOffset;
uniform float emaAlpha;

uniform sampler2D mapLowRes;
uniform sampler2D mapHistory;

float pixelFilter(vec2 pixelCenter, vec2 sampleCenter) {
  vec2 delta = pixelCenter - sampleCenter;
  float squaredNorm = dot(delta, delta);
  return exp(-2.29 * squaredNorm);
}

void main() {
  // First we need to compute the coordinates of the pixel centers
  // in the low resolution grid by compensating for the camera jitter.
  // Note that the offset is defined in clip space [-1,1]^2, so we need
  // to multiply it by 0.5 to make it valid in texture space [0,1]^2.
  vec2 compensatedUnitCoords = vUv - jitterOffset * 0.5;

  // Now compute the integer coordinates in the low resolution grid for each
  // adjacent texel.
  ivec2 lowResCoords00 = ivec2(compensatedUnitCoords * lowResolution - 0.5);
  ivec2 lowResCoords01 = ivec2(0, 1) + lowResCoords00;
  ivec2 lowResCoords10 = ivec2(1, 0) + lowResCoords00;
  ivec2 lowResCoords11 = ivec2(1, 1) + lowResCoords00;

  float mask00 =
    min(lowResCoords00.x, lowResCoords00.y) < 0 ||
    lowResCoords00.x >= int(lowResolution.x) ||
    lowResCoords00.y >= int(lowResolution.y) ? 0.0 : 1.0;
  float mask01 =
    min(lowResCoords01.x, lowResCoords01.y) < 0 ||
    lowResCoords01.x >= int(lowResolution.x) ||
    lowResCoords01.y >= int(lowResolution.y) ? 0.0 : 1.0;
  float mask10 =
    min(lowResCoords10.x, lowResCoords10.y) < 0 ||
    lowResCoords10.x >= int(lowResolution.x) ||
    lowResCoords10.y >= int(lowResolution.y) ? 0.0 : 1.0;
  float mask11 =
    min(lowResCoords11.x, lowResCoords11.y) < 0 ||
    lowResCoords11.x >= int(lowResolution.x) ||
    lowResCoords11.y >= int(lowResolution.y) ? 0.0 : 1.0;

  // We also need to keep track of the high resolution counterparts of these
  // coordinates, so we can compute the pixel reconstruction filter weights.
  vec2 compensatedHighResCoords = highResolution * compensatedUnitCoords;
  vec2 highResCoords00 =
      highResolution * (vec2(lowResCoords00) + 0.5) / lowResolution;
  vec2 highResCoords01 =
      highResolution * (vec2(lowResCoords01) + 0.5) / lowResolution;
  vec2 highResCoords10 =
      highResolution * (vec2(lowResCoords10) + 0.5) / lowResolution;
  vec2 highResCoords11 =
      highResolution * (vec2(lowResCoords11) + 0.5) / lowResolution;

  vec4 lowResColor = vec4(0.0, 0.0, 0.0, 0.0);
  lowResColor += mask00 * vec4(
    texelFetch(mapLowRes,lowResCoords00, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords00);
  lowResColor += mask01 * vec4(
    texelFetch(mapLowRes, lowResCoords01, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords01);
  lowResColor += mask10 * vec4(
    texelFetch(mapLowRes, lowResCoords10, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords10);
  lowResColor += mask11 * vec4(
    texelFetch(mapLowRes, lowResCoords11, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords11);

  vec4 historyColor = texture2D(mapHistory, vUv);
  gl_FragColor = emaAlpha * lowResColor + (1.0 - emaAlpha) * historyColor;
}
`;

/**
 * Sets up the state needed for progressive rendering.
 * @param {!HTMLElement} view The view.
 * @param {number} lowResFactor The downsampling factor that determines the
 * initial render resolution.
 */
function setupProgressiveRendering(view, lowResFactor) {
  gHighResBlitCamera = new THREE.OrthographicCamera(
      view.offsetWidth / -2, view.offsetWidth / 2, view.offsetHeight / 2,
      view.offsetHeight / -2, -10000, 10000);
  gHighResBlitCamera.position.z = 100;
  let fullScreenPlane =
      new THREE.PlaneBufferGeometry(view.offsetWidth, view.offsetHeight);

  gLowResTexture = new THREE.WebGLRenderTarget(
      Math.trunc(view.offsetWidth / lowResFactor),
      Math.trunc(view.offsetHeight / lowResFactor), {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        type: THREE.UnsignedByteType,
        format: THREE.RGBFormat
      });
  gAccumulationTextures[0] =
      new THREE.WebGLRenderTarget(view.offsetWidth, view.offsetHeight, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        type: THREE.FloatType,
        format: THREE.RGBAFormat
      });
  gAccumulationTextures[1] =
      new THREE.WebGLRenderTarget(view.offsetWidth, view.offsetHeight, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        type: THREE.FloatType,
        format: THREE.RGBAFormat
      });

  let fullScreenAccumulateQuad = new THREE.Mesh(
      fullScreenPlane,
      createAccumulateMaterial(
          gLowResTexture.texture, gAccumulationTextures[1],
          new THREE.Vector2(
              Math.trunc(view.offsetWidth / lowResFactor),
              Math.trunc(view.offsetHeight / lowResFactor)),
          new THREE.Vector2(view.offsetWidth, view.offsetHeight)));
  fullScreenAccumulateQuad.position.z = -100;
  gSceneAccumulate = new THREE.Scene();
  gSceneAccumulate.add(fullScreenAccumulateQuad);
  gSceneAccumulate.autoUpdate = false;

  let fullScreenNormalizeQuad = new THREE.Mesh(
      fullScreenPlane,
      createNormalizeMaterial(gAccumulationTextures[0].texture));
  fullScreenNormalizeQuad.position.z = -100;
  gSceneNormalize = new THREE.Scene();
  gSceneNormalize.add(fullScreenNormalizeQuad);
  gSceneNormalize.autoUpdate = false;

  gLowResBlitCamera = new THREE.OrthographicCamera(
      Math.trunc(view.offsetWidth / lowResFactor) / -2,
      Math.trunc(view.offsetWidth / lowResFactor) / 2,
      Math.trunc(view.offsetHeight / lowResFactor) / 2,
      Math.trunc(view.offsetHeight / lowResFactor) / -2, -10000, 10000);
  gLowResBlitCamera.position.z = 100;

  gOldProjectionMatrix = gCamera.projectionMatrix.clone();
  gOldMatrixWorld = gCamera.matrixWorld.clone();
}

/**
 * Implements progressive rendering.
 */
function renderProgressively() {
  let cameraMoved = !gCamera.projectionMatrix.equals(gOldProjectionMatrix) ||
      !gCamera.matrixWorld.equals(gOldMatrixWorld);

  gRenderer.setRenderTarget(gLowResTexture);
  gRenderer.clear();

  //
  // For progressive upsampling, jitter the camera matrix within the pixel
  // footprint.
  //

  // We start by forming a set of jitter offsets that touch every high
  // resolution pixel center.
  const downSamplingFactor =
      gAccumulationTextures[0].width / gLowResTexture.width;
  const isEven = (downSamplingFactor % 2) == 0;
  // These values assume an even downsampling factor.
  let jitterOffset = 0.5;
  let endIndex = Math.trunc(downSamplingFactor / 2);
  if (!isEven) {  // But it's not that hard to correct for this assumption.
    jitterOffset = 0.5;
    endIndex += 1;
  }
  let samples_x = [];
  let samples_y = [];
  for (let i = 0; i < endIndex; i++) {
    for (let j = 0; j < endIndex; j++) {
      samples_x.push((jitterOffset + i) / downSamplingFactor);
      samples_y.push((jitterOffset + j) / downSamplingFactor);

      samples_x.push(-(jitterOffset + i) / downSamplingFactor);
      samples_y.push((jitterOffset + j) / downSamplingFactor);

      samples_x.push((jitterOffset + i) / downSamplingFactor);
      samples_y.push(-(jitterOffset + j) / downSamplingFactor);

      samples_x.push(-(jitterOffset + i) / downSamplingFactor);
      samples_y.push(-(jitterOffset + j) / downSamplingFactor);
    }
  }

  // To set up the jitter properly we need to update the projection matrices of
  // both our cameras in tandem:
  // 1) the orthographic blit matrix that kicks off the ray march, and
  // 2) the perspective projection matrix which computes ray origins/directions.
  let sample_index = gFrameIndex % samples_x.length;
  let offset_x = samples_x[sample_index];
  let offset_y = samples_y[sample_index];

  // First update the orthographic camera, which uses coordinates in
  //   resolution * [-0.5,0,5]^2.
  gLowResBlitCamera.left = offset_x + gLowResTexture.width / -2;
  gLowResBlitCamera.right = offset_x + gLowResTexture.width / 2;
  gLowResBlitCamera.top = offset_y + gLowResTexture.height / 2;
  gLowResBlitCamera.bottom = offset_y + gLowResTexture.height / -2;
  gLowResBlitCamera.updateProjectionMatrix();

  // After this we will be working with clip space cameras, that have
  // coordinates in
  //   [-1,1]^2.
  // So we need to scale the offset accordingly.
  offset_x *= 2.0 / gLowResTexture.width;
  offset_y *= 2.0 / gLowResTexture.height;

  // Now adjust the projection matrix that computes the ray parameters.
  let clip_T_camera = gCamera.projectionMatrix.clone();
  clip_T_camera.elements[8] += offset_x;
  clip_T_camera.elements[9] += offset_y;

  //
  // Now we can do the volume rendering at a lower resolution.
  //

  let camera_T_clip = new THREE.Matrix4();
  camera_T_clip.getInverse(clip_T_camera);

  let world_T_camera = gCamera.matrixWorld;
  let world_T_clip = new THREE.Matrix4();
  world_T_clip.multiplyMatrices(world_T_camera, camera_T_clip);

  gRayMarchScene.children[0].material.uniforms['world_T_clip']['value'] =
      world_T_clip;
  gRayMarchScene.children[0].material.uniforms['displayMode']['value'] =
      gDisplayMode - 0;
  gRayMarchScene.children[0].material.uniforms['stepMult']['value'] = gStepMult;
  gRenderer.render(gRayMarchScene, gLowResBlitCamera);

  //
  // Finally collect these low resolution samples into our high resolution
  // accumulation bufer.
  //

  // With more subsampling we need to average more aggressively over time. This
  // is controled by emaAlpha (exponential moving average), which averages more
  // when the value gets smaller. This formula for setting emaAlpha was hand-
  // tuned to work well in gardenvase.
  let emaAlpha = Math.min(1.0, Math.sqrt(0.1 / samples_x.length));
  if (cameraMoved) {
    gFrameIndex = 0;
    emaAlpha = 1.0;
  }

  let accumulationTargetIndex = gFrameIndex % 2;
  let accumulationReadIndex = 1 - accumulationTargetIndex;
  gRenderer.setRenderTarget(gAccumulationTextures[accumulationTargetIndex]);
  gSceneAccumulate.children[0].material.uniforms['mapHistory']['value'] =
      gAccumulationTextures[accumulationReadIndex].texture;
  gSceneAccumulate.children[0].material.uniforms['jitterOffset']['value'] =
      new THREE.Vector2(offset_x, offset_y);
  gSceneAccumulate.children[0].material.uniforms['emaAlpha']['value'] =
      emaAlpha;
  gRenderer.clear();
  gRenderer.render(gSceneAccumulate, gHighResBlitCamera);

  gRenderer.setRenderTarget(null);
  gSceneNormalize.children[0].material.uniforms['map']['value'] =
      gAccumulationTextures[accumulationTargetIndex].texture;
  gRenderer.clear();
  gRenderer.render(gSceneNormalize, gHighResBlitCamera);

  gFrameIndex++;
  gOldProjectionMatrix = gCamera.projectionMatrix.clone();
  gOldMatrixWorld = gCamera.matrixWorld.clone();
}
