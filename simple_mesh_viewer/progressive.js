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
 * This is a half-res render target used for progressive rendering.
 * @type {?THREE.WebGLRenderTarget}
 */
let gLowResTexture = null;

/**
 * Sqrt of the number of samples for anti-aliasing.
 * @type {number}
 */
let gAaFactor = 1;

/**
 * TAA sample offsets.
 */
let gSamples = null;

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
      'nearPlane': {'value': 0.2},
      'farPlane': {'value': 100.0},
      'world_T_clip': {'value': new THREE.Matrix4()},
      'history_clip_T_world': {'value': new THREE.Matrix4()},
      'cameraPos': {'value': new THREE.Vector3(0.0, 0.0, 0.0)},
      'mapLowRes': {'value': textureLowRes},
      'mapHistory': {'value': textureHistory},
      'lowResolution': {'value': lowResolution},
      'highResolution': {'value': highResolution},
      'jitterOffset': {'value': new THREE.Vector2(0.0, 0.0)},
      'emaAlpha': {'value': 0.15},
      'maxBoxScale': {'value': 0.0},
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
uniform vec2 resolution;
uniform sampler2D map;
void main() {
  // These coefficients were found with local optimization (Adam, L2 loss) to
  // undo the effect of our pixel reconstruction filter exp(-2.29 * x^2) for an
  // image consisting of uniform random noise.
  const float cornerWeight = -0.01341202;
  const float crossWeight = 0.00847795;
  const float centerWeight = 1.0197363;

  vec2 invResolution = 1.0 / resolution;
  vec3 sharpened = vec3(0.0, 0.0, 0.0);
  #define ADD_SAMPLE(dx, dy, w) {\
    vec4 color = texture2D(map, vUv + vec2(dx, dy) * invResolution);\
    if (color.a > 0.0) { color.rgb /= color.a; }\
    sharpened += w * color.rgb;\
  }
  ADD_SAMPLE(-1, -1, cornerWeight)
  ADD_SAMPLE(-1,  1, cornerWeight)
  ADD_SAMPLE( 1, -1, cornerWeight)
  ADD_SAMPLE( 1,  1, cornerWeight)
  ADD_SAMPLE( 0,  0, centerWeight)
  ADD_SAMPLE( 0, -1, crossWeight)
  ADD_SAMPLE( 0,  1, crossWeight)
  ADD_SAMPLE(-1,  0, crossWeight)
  ADD_SAMPLE( 1,  0, crossWeight)

  gl_FragColor = vec4(sharpened, 1.0);
}
`;

/**
 * @param {!THREE.Texture} texture
 * @param {!THREE.Vector2} resolution
 * @return {!THREE.Material}
 */
function createNormalizeMaterial(texture, resolution) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      'map': {'value': texture},
      'resolution': {'value': resolution},
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
uniform mat4 world_T_clip;
uniform float nearPlane;
uniform float farPlane;

varying vec2 vUv;
varying vec3 vDirection;

void main() {
  vUv = uv;
  vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = positionClip;

  positionClip /= positionClip.w;
  vec4 nearPoint = world_T_clip * vec4(positionClip.x, positionClip.y, -1.0, 1.0);
  vec4 farPoint = world_T_clip * vec4(positionClip.x, positionClip.y, 1.0, 1.0);
  vDirection = farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w;
  vDirection /= farPlane - nearPlane;
}
`;

/** @const {string} */
const accumulateFragmentShader = `
precision highp float;
varying vec2 vUv;
varying vec3 vDirection;

uniform vec2 lowResolution;
uniform vec2 highResolution;
uniform vec2 jitterOffset;
uniform float emaAlpha;
uniform float maxBoxScale;

uniform sampler2D mapLowRes;

uniform vec3 cameraPos;
uniform mat4 history_clip_T_world;
uniform sampler2D mapHistory;

// From Filament: https://github.com/google/filament/blob/main/filament/src/materials/antiAliasing/taa.mat
// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead
// of 16.
//      https://therealmjp.github.io/
// Some optimizations from here:
//      http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
// Optimized to 5 taps by removing the corner samples
// And modified for mediump support
vec4 sampleTextureCatmullRom(
  const sampler2D tex, const highp vec2 uv, const highp vec2 texSize) {
  // We're going to sample a a 4x4 grid of texels surrounding the target UV
  // coordinate. We'll do this by rounding down the sample location to get the
  // exact center of our "starting" texel. The starting texel will be at
  // location [1, 1] in the grid, where [0, 0] is the top left corner.

  highp vec2 samplePos = uv * texSize;
  highp vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

  // Compute the fractional offset from our starting texel to our original
  // sample location, which we'll feed into the Catmull-Rom spline function to
  // get our filter weights.
  highp vec2 f = samplePos - texPos1;
  highp vec2 f2 = f * f;
  highp vec2 f3 = f2 * f;

  // Compute the Catmull-Rom weights using the fractional offset that we
  // calculated earlier. These equations are pre-expanded based on our knowledge
  // of where the texels will be located, which lets us avoid having to evaluate
  // a piece-wise function.
  vec2 w0 = f2 - 0.5 * (f3 + f);
  vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  vec2 w3 = 0.5 * (f3 - f2);
  vec2 w2 = 1.0 - w0 - w1 - w3;

  // Work out weighting factors and sampling offsets that will let us use
  // bilinear filtering to simultaneously evaluate the middle 2 samples from
  // the 4x4 grid.
  vec2 w12 = w1 + w2;

  // Compute the final UV coordinates we'll use for sampling the texture
  highp vec2 texPos0 = texPos1 - vec2(1.0);
  highp vec2 texPos3 = texPos1 + vec2(2.0);
  highp vec2 texPos12 = texPos1 + w2 / w12;

  highp vec2 invTexSize = 1.0 / texSize;
  texPos0  *= invTexSize;
  texPos3  *= invTexSize;
  texPos12 *= invTexSize;

  float k0 = w12.x * w0.y;
  float k1 = w0.x  * w12.y;
  float k2 = w12.x * w12.y;
  float k3 = w3.x  * w12.y;
  float k4 = w12.x * w3.y;

  vec4 result =   textureLod(tex, vec2(texPos12.x, texPos0.y),  0.0) * k0
                + textureLod(tex, vec2(texPos0.x,  texPos12.y), 0.0) * k1
                + textureLod(tex, vec2(texPos12.x, texPos12.y), 0.0) * k2
                + textureLod(tex, vec2(texPos3.x,  texPos12.y), 0.0) * k3
                + textureLod(tex, vec2(texPos12.x, texPos3.y),  0.0) * k4;

  result *= 1.0 / (k0 + k1 + k2 + k3 + k4);

  // we could end-up with negative values
  result = max(vec4(0), result);

  return result;
}

vec4 sampleTextureLanczos(
  const sampler2D tex, const vec2 uv, const vec2 texSize, const int r) {
  const float kPi = 3.1415926535897932384626433832795;
  const float kEps = 1e-10;
  float rInv = 1.0 / float(r);

  // Use macros here to maximize the potential for compiler inlining.
  #define BIAS(x) (kPi * ((x) + kEps))
  #define SINC(x) (sin(BIAS((x))) / BIAS((x)))
  #define lanczosKernel1D(x) (SINC((x)) * SINC((x) * (rInv)))

  vec2 centerPosf = uv * texSize;
  vec2 centerPosi = floor(centerPosf - 0.5) + 0.5;
  vec4 result = vec4(0.0);
  for (int dy = -r; dy <= r; dy++) {
    float yy = centerPosi.y + float(dy);
    float dyy = min(abs(yy - centerPosf.y), float(r));
    float wy = lanczosKernel1D(dyy);
    for (int dx = -r; dx <= r; dx++) {
      float xx = centerPosi.x + float(dx);
      float dxx = min(abs(xx - centerPosf.x), float(r));
      float wx = lanczosKernel1D(dxx);
      result += wx * wy * texelFetch(tex, ivec2(xx, yy), 0).rgba;
    }
  }
  return max(vec4(0.0), result);
}

// From Filament: https://github.com/google/filament/blob/main/filament/src/materials/antiAliasing/taa.mat
vec3 RGB_YCoCg(const vec3 c) {
  float Y  = dot(c.rgb, vec3( 1, 2,  1) * 0.25);
  float Co = dot(c.rgb, vec3( 2, 0, -2) * 0.25);
  float Cg = dot(c.rgb, vec3(-1, 2, -1) * 0.25);
  return vec3(Y, Co, Cg);
}

// From Filament: https://github.com/google/filament/blob/main/filament/src/materials/antiAliasing/taa.mat
vec3 YCoCg_RGB(const vec3 c) {
  float Y  = c.x;
  float Co = c.y;
  float Cg = c.z;
  float r = Y + Co - Cg;
  float g = Y + Cg;
  float b = Y - Co - Cg;
  return vec3(r, g, b);
}

vec3 clipToBox(
  const vec3 colorAabbMin,
  const vec3 colorAabbMax,
  const vec3 originColor,
  const vec3 historyColor) {
  vec3 colorDir = historyColor - originColor;
  vec3 inverseColorDir = 1.0 / colorDir;
  vec3 tMax = (colorAabbMax - originColor) * inverseColorDir;
  vec3 tMin = (colorAabbMin - originColor) * inverseColorDir;
  tMax = max(tMax, tMin);
  tMin = min(tMax, tMin);
  return originColor + colorDir * saturate(
    min(tMax.x, min(tMax.y, tMax.z)));
}

float pixelFilter(vec2 pixelCenter, vec2 sampleCenter) {
  vec2 delta = pixelCenter - sampleCenter;
  float squaredNorm = dot(delta, delta);
  return exp(-2.29 * squaredNorm);
}

float maskOutOfBounds(ivec2 pos, vec2 resolution) {
  return
   min(pos.x, pos.y) < 0 ||
   pos.x >= int(resolution.x) ||
   pos.y >= int(resolution.y) ?
   0.0 : 1.0;
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

  float mask00 = maskOutOfBounds(lowResCoords00, lowResolution);
  float mask01 = maskOutOfBounds(lowResCoords01, lowResolution);
  float mask10 = maskOutOfBounds(lowResCoords10, lowResolution);
  float mask11 = maskOutOfBounds(lowResCoords11, lowResolution);

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

  vec4 lowResColorAndDisparity00 = texelFetch(mapLowRes, lowResCoords00, 0);
  vec4 lowResColorAndDisparity01 = texelFetch(mapLowRes, lowResCoords01, 0);
  vec4 lowResColorAndDisparity10 = texelFetch(mapLowRes, lowResCoords10, 0);
  vec4 lowResColorAndDisparity11 = texelFetch(mapLowRes, lowResCoords11, 0);

  // We store the reconstruction kernel weight in the alpha channnel.
  vec4 lowResColor =
    mask00 * vec4(lowResColorAndDisparity00.rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords00) +
    mask01 * vec4(lowResColorAndDisparity01.rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords01) +
    mask10 * vec4(lowResColorAndDisparity10.rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords10) +
    mask11 * vec4(lowResColorAndDisparity11.rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords11);

  // Dilate FG objects by biasing towards the nearest depth.
  // This version has edge artifacts IMO.
  float lowResDisparity = max(
    max(
      mask00 * lowResColorAndDisparity00.a,
      mask01 * lowResColorAndDisparity01.a
    ),
    max(
      mask10 * lowResColorAndDisparity10.a,
      mask11 * lowResColorAndDisparity11.a
    )
  );
  /* Averaging disparity is a softer way of encouraging FG depths.
  float lowResDisparity =
  mask00 * lowResColorAndDisparity00.a *
   pixelFilter(compensatedHighResCoords, highResCoords00) +
  mask01 * lowResColorAndDisparity01.a *
   pixelFilter(compensatedHighResCoords, highResCoords01) +
  mask10 * lowResColorAndDisparity10.a *
   pixelFilter(compensatedHighResCoords, highResCoords10) +
  mask11 * lowResColorAndDisparity11.a *
   pixelFilter(compensatedHighResCoords, highResCoords11);
  lowResDisparity /= lowResColor.a;
  */

  // Use variance clamping to limit ghosting from reprojected colors.
  ivec2 lowResCoordsNearest = ivec2(compensatedUnitCoords * lowResolution);
  vec3 mean = vec3(0.0, 0.0, 0.0);
  vec3 squaredMean = vec3(0.0, 0.0, 0.0);
  vec3 minColor = vec3(1.0, 1.0, 1.0);
  vec3 maxColor = vec3(0.0, 0.0, 0.0);

  #define BOUND_SAMPLE(dx, dy) {\
    ivec2 lowResCoords = lowResCoordsNearest + ivec2(dx, dy);\
    if (maskOutOfBounds(lowResCoords, lowResolution) > 0.0) {\
      vec3 neighborColor = texelFetch(mapLowRes, lowResCoords, 0).rgb;\
      neighborColor = RGB_YCoCg(neighborColor);\
      mean += neighborColor;\
      squaredMean += neighborColor * neighborColor;\
      minColor = min(minColor, neighborColor);\
      maxColor = max(maxColor, neighborColor);\
    }\
  }

  // Following [Karis2014] we avoid box artifacts by computing the color
  // bbox as the average of the cross neighborhood and the 3x3 neighborhood.
  // First compute the cross neighborhood statistics.
  BOUND_SAMPLE( 0,  0)
  BOUND_SAMPLE( 0, -1)
  BOUND_SAMPLE(-1,  0)
  BOUND_SAMPLE( 0,  1)
  BOUND_SAMPLE( 1,  0)
  vec3 crossMean = mean * (1.0 / 5.0);
  vec3 crossSquaredMean = squaredMean * (1.0 / 5.0);
  vec3 crossSigma = sqrt(crossSquaredMean - crossMean * crossMean);
  vec3 crossMinColor = max(minColor, crossMean - crossSigma);
  vec3 crossMaxColor = min(maxColor, crossMean + crossSigma);

  // Then, reuse these samples to compute the 3x3 neighborhood statistics.
  BOUND_SAMPLE(-1, -1)
  BOUND_SAMPLE(-1,  1)
  BOUND_SAMPLE( 1, -1)
  BOUND_SAMPLE( 1,  1)
  mean *= 1.0 / 9.0;
  squaredMean *= 1.0 / 9.0;
  vec3 sigma = sqrt(squaredMean - mean * mean);
  minColor = max(minColor, mean - sigma);
  maxColor = min(maxColor, mean + sigma);

  // And average the two bboxes.
  minColor = 0.5 * (minColor + crossMinColor);
  maxColor = 0.5 * (maxColor + crossMaxColor);

  // Expand the color bbox if the low-res pixel samples are far away from our
  // target pixel. Tuned to the smallest stable value in bicycle & gardenvase.
  float clipBoxFactor = 1.0 + maxBoxScale * (1.0 - exp(-lowResColor.a));
  vec3 boxCenter = 0.5 * (minColor + maxColor);
  minColor = boxCenter + clipBoxFactor * (minColor - boxCenter);
  maxColor = boxCenter + clipBoxFactor * (maxColor - boxCenter);

  // Now reproject the previously rendered frame.
  vec3 positionWorld = cameraPos + vDirection / lowResDisparity;
  vec4 positionHistoryClip = history_clip_T_world * vec4(positionWorld, 1.0);
  vec2 historyUv = 0.5 * (positionHistoryClip.xy / positionHistoryClip.w + 1.0);
  //vec4 historyColor = sampleTextureCatmullRom(mapHistory, historyUv,
  //  highResolution);
  vec4 historyColor = sampleTextureLanczos(mapHistory, historyUv,
    highResolution, 3);

  // Clip the reprojected color to match the colors around the central pixel.
  vec3 lowResRgb = lowResColor.rgb / max(0.00000001, lowResColor.a);
  vec3 historyRgb = historyColor.rgb / max(0.00000001, historyColor.a);
  lowResRgb = RGB_YCoCg(lowResRgb);
  historyRgb = RGB_YCoCg(historyRgb);
  historyRgb = clipToBox(minColor, maxColor, lowResRgb, historyRgb);
  historyRgb = YCoCg_RGB(historyRgb);
  historyColor.rgb = historyRgb * historyColor.a;

  // Discard history samples that were outside the view frustum.
  float blendAlpha = emaAlpha;
  if (historyUv.x <= 0.0 || historyUv.y <= 0.0 ||
      historyUv.x >= 1.0 || historyUv.y >= 1.0) {
    blendAlpha = 1.0;
  }

  gl_FragColor = blendAlpha * lowResColor + (1.0 - blendAlpha) * historyColor;
}
`;

/**
 * Sets up the state needed for progressive rendering.
 * @param {!HTMLElement} view The view.
 * @param {number} renderFactor The downsampling factor that determines the
 * initial render resolution.
 * @param {number} aaFactor Sqrt of the number of aa samples to take.
 * @param {string} sampler sequence type. Must be one of {grid, halton, pmj2bn}
 */
function setupProgressiveRendering(view, renderFactor, aaFactor, sampler) {
  gAaFactor = aaFactor;
  gSamples = getSampleSequence(sampler, renderFactor, aaFactor);

  gHighResBlitCamera = new THREE.OrthographicCamera(
      view.offsetWidth / -2, view.offsetWidth / 2, view.offsetHeight / 2,
      view.offsetHeight / -2, -10000, 10000);
  gHighResBlitCamera.position.z = 100;
  let fullScreenPlane =
      new THREE.PlaneBufferGeometry(view.offsetWidth, view.offsetHeight);

  gLowResTexture = new THREE.WebGLRenderTarget(
      Math.trunc(view.offsetWidth / renderFactor),
      Math.trunc(view.offsetHeight / renderFactor), {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        type: THREE.FloatType,
        format: THREE.RGBAFormat
      });
  gAccumulationTextures[0] =
      new THREE.WebGLRenderTarget(view.offsetWidth, view.offsetHeight, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        type: THREE.FloatType,
        format: THREE.RGBAFormat
      });
  gAccumulationTextures[1] =
      new THREE.WebGLRenderTarget(view.offsetWidth, view.offsetHeight, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        type: THREE.FloatType,
        format: THREE.RGBAFormat
      });

  let fullScreenAccumulateQuad = new THREE.Mesh(
      fullScreenPlane,
      createAccumulateMaterial(
          gLowResTexture.texture, gAccumulationTextures[1],
          new THREE.Vector2(
              Math.trunc(view.offsetWidth / renderFactor),
              Math.trunc(view.offsetHeight / renderFactor)),
          new THREE.Vector2(view.offsetWidth, view.offsetHeight)));
  fullScreenAccumulateQuad.position.z = -100;
  gSceneAccumulate = new THREE.Scene();
  gSceneAccumulate.add(fullScreenAccumulateQuad);
  gSceneAccumulate.autoUpdate = false;

  let fullScreenNormalizeQuad = new THREE.Mesh(
      fullScreenPlane,
      createNormalizeMaterial(
          gAccumulationTextures[0].texture,
          new THREE.Vector2(view.offsetWidth, view.offsetHeight)));
  fullScreenNormalizeQuad.position.z = -100;
  gSceneNormalize = new THREE.Scene();
  gSceneNormalize.add(fullScreenNormalizeQuad);
  gSceneNormalize.autoUpdate = false;

  gLowResBlitCamera = new THREE.OrthographicCamera(
      Math.trunc(view.offsetWidth / renderFactor) / -2,
      Math.trunc(view.offsetWidth / renderFactor) / 2,
      Math.trunc(view.offsetHeight / renderFactor) / 2,
      Math.trunc(view.offsetHeight / renderFactor) / -2, -10000, 10000);
  gLowResBlitCamera.position.z = 100;

  gOldProjectionMatrix = gCamera.projectionMatrix.clone();
  gOldMatrixWorld = gCamera.matrixWorld.clone();
}


/**
 * Retrieves the sample sequence used for anti-aliasing.
 * @param {string} sampler The sampler to use.
 * @param {number} renderFactor Render factor to use.
 * @param {number} aaFactor The square root of the number of AA samples to use.
 * @return {!object} The sample sequence.
 */
function getSampleSequence(sampler, renderFactor, aaFactor) {
  let samples_x = [];
  let samples_y = [];

  if (sampler == 'grid') {
    // These values assume an even downsampling factor.
    const isEven = (renderFactor % 2) == 0;
    let jitterOffset = 0.5;
    let endIndex = Math.trunc(renderFactor / 2);
    if (!isEven) {  // But it's not that hard to correct for this assumption.
      jitterOffset = 0.5;
      endIndex += 1;
    }

    for (let i = 0; i < endIndex; i++) {
      for (let j = 0; j < endIndex; j++) {
        samples_x.push((jitterOffset + i) / renderFactor);
        samples_y.push((jitterOffset + j) / renderFactor);

        samples_x.push(-(jitterOffset + i) / renderFactor);
        samples_y.push((jitterOffset + j) / renderFactor);

        samples_x.push((jitterOffset + i) / renderFactor);
        samples_y.push(-(jitterOffset + j) / renderFactor);

        samples_x.push(-(jitterOffset + i) / renderFactor);
        samples_y.push(-(jitterOffset + j) / renderFactor);
      }
    }
  } else if (sampler == 'halton') {
    function halton(base, index) {
      let f = 1.0;
      let result = 0.0;
      while (index > 0) {
        f = f / base;
        result += f * (index % base);
        index = Math.trunc(index / base);
      }
      return result;
    }

    for (let i = 0; i < aaFactor; i++) {
      for (let j = 0; j < aaFactor; j++) {
        let idx = i * aaFactor + j;
        samples_x.push(halton(2, idx) - 0.5);
        samples_y.push(halton(3, idx) - 0.5);
      }
    }
  } else if (sampler == 'pmj2bn') {
    let pmj_bn = [
      [0.020536686622164797, 0.99358349494336828],
      [0.55996322041922764, 0.49485891637128582],
      [0.98614172826964674, 0.53012074488569283],
      [0.48423032933205351, 0.0061863672661581917],
      [0.26627072739697794, 0.71944896648320367],
      [0.75811124157392429, 0.21755206455609233],
      [0.72334460477747287, 0.76343005317928114],
      [0.22116834368615787, 0.2522156422010382],
      [0.4360376011781899, 0.81388045304864975],
      [0.93270208483403105, 0.32768005324420368],
      [0.56541114799432568, 0.68170263153148392],
      [0.06534873652662973, 0.17199599651752659],
      [0.16657279714041395, 0.56650317720183951],
      [0.6627117086593719, 0.063519082822104167],
      [0.82741495010832666, 0.93595943382544289],
      [0.34580065746363065, 0.43224288785505183],
      [0.18868554611662058, 0.80938302930852535],
      [0.68949282267939505, 0.31224086751522284],
      [0.81050821457431843, 0.68863233556355807],
      [0.31147923987963044, 0.22340847081153406],
      [0.43799628812406116, 0.5578762001431673],
      [0.93981161372446587, 0.060940353877743311],
      [0.53116499024849451, 0.94461718415758966],
      [0.061780436139939457, 0.4385332950416484],
      [0.31287568015261874, 0.90514814452373105],
      [0.84419543012640663, 0.40529347211096678],
      [0.65281902027373984, 0.59621596217649486],
      [0.15240072028489182, 0.095522791470850008],
      [0.097517184499875773, 0.65562905369769853],
      [0.59398644921073618, 0.15469771795523632],
      [0.90373818333777212, 0.84796099935682467],
      [0.40037176015678316, 0.34601784005925068],
      [0.12562133792041313, 0.92059625916854504],
      [0.63988105939198281, 0.41525435496342905],
      [0.86969963894227698, 0.57859765200134905],
      [0.34233858899578717, 0.078165643867913373],
      [0.38306085885247459, 0.67136595149859757],
      [0.8765244056900714, 0.17130330037864336],
      [0.61171275779560796, 0.83752916981278225],
      [0.12201120160034358, 0.32949487246332093],
      [0.29498668505953723, 0.78105805200408929],
      [0.79451502824581277, 0.28084146510215308],
      [0.70418208636176005, 0.73607421269058926],
      [0.20463986011229796, 0.18753557204711663],
      [0.046825890909320211, 0.5035845106085175],
      [0.51447760400633469, 0.030398468983112721],
      [0.9533618515682083, 0.9691410419601455],
      [0.46538319375371162, 0.46878473202219006],
      [0.078942978996503474, 0.86145914334545737],
      [0.57883319391768351, 0.35971044866817514],
      [0.92054077096553755, 0.6403424616224318],
      [0.42131032681469138, 0.13849823201786821],
      [0.36190853945888568, 0.60998899960434594],
      [0.83037793097846269, 0.10981617919704351],
      [0.6868057486793574, 0.88936558108294883],
      [0.18519172473559864, 0.39005656414425821],
      [0.48561583491324695, 0.9572838856903999],
      [0.97455284024614275, 0.45342986300244748],
      [0.53181717189498923, 0.54655934387227678],
      [0.00035727250841476689, 0.046179446294617503],
      [0.23484076587496966, 0.70447874500739494],
      [0.73468439914393613, 0.24884360740319522],
      [0.78050995040833393, 0.79647397382854901],
      [0.26478915192144253, 0.29678912093926002]
    ];

    for (let i = 0; i < aaFactor * aaFactor; ++i) {
      samples_x.push(pmj_bn[i][0] - 0.5);
      samples_y.push(pmj_bn[i][1] - 0.5);
    }
  }
  return [samples_x, samples_y];
}

/**
 * Implements progressive rendering.
 */
function renderProgressively() {
  const renderFactor = gAccumulationTextures[0].width / gLowResTexture.width;

  // Early out by rendering the frame at full res.
  if (renderFactor == 1 && gAaFactor == 1) {
    renderFrame();
    return;
  }

  const [samples_x, samples_y] = gSamples;

  let cameraMoved = !gCamera.projectionMatrix.equals(gOldProjectionMatrix) ||
      !gCamera.matrixWorld.equals(gOldMatrixWorld);

  // To set up the jitter properly we need to update the projection matrices
  // of both our cameras in tandem:
  // 1) the orthographic blit matrix that kicks off the ray march, and
  // 2) the perspective projection matrix which computes ray origins/directions.
  let sample_index = gFrameIndex % samples_x.length;
  let offset_x = samples_x[sample_index];
  let offset_y = samples_y[sample_index];


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
  gRenderer.setRenderTarget(gLowResTexture);
  renderFrame(clip_T_camera);

  //
  // Finally collect these low resolution samples into our high resolution
  // accumulation bufer.
  //

  // With more subsampling we need to average more aggressively over time.
  // This is controlled by emaAlpha (exponential moving average), which
  // averages more when the value gets smaller. This formula was hand tuned to
  // the highest flicker-free value in bicycle at 720p with 0.66 spp.
  let emaAlpha = Math.min(1.0, 0.33 / Math.pow(samples_x.length, 2.0 / 3.0));

  // Decreasing emaAlpha when the camera is not moving sometimes helps with
  // single frame quality. However, it doesn't seem to make a difference for
  // moderate upsampling factors between 1 and 0.66 spp.
  if (!cameraMoved) {
    emaAlpha *= 1.0;
  }

  // When upsampling, we sometimes get into a situation where the color
  // clamping boxes of adjacent samples are in conflict. This results in
  // flickering that can't be addressed with emaAlpha. To work around this
  // we scale up the size of the clamping box based on distance from the pixel
  // samples in the low res image to the high res pixel we're trying to
  // recover. This value tunes how quickly that bbox grows with distance, and
  //  was tuned to look good in bicycle and gardenvase for 1spp and 0.5spp.
  let maxBoxScale = Math.max(0.0, renderFactor * renderFactor - 1.0);

  // Set up matrices for temporal reprojection
  let world_T_clip = new THREE.Matrix4();
  world_T_clip.multiplyMatrices(
      gCamera.matrixWorld, gCamera.projectionMatrixInverse);
  let history_cam_T_world = new THREE.Matrix4();
  history_cam_T_world.getInverse(gOldMatrixWorld);
  let history_clip_T_world = new THREE.Matrix4();
  history_clip_T_world.multiplyMatrices(
      gOldProjectionMatrix, history_cam_T_world);

  let accumulationTargetIndex = gFrameIndex % 2;
  let accumulationReadIndex = 1 - accumulationTargetIndex;
  gRenderer.setRenderTarget(gAccumulationTextures[accumulationTargetIndex]);
  gSceneAccumulate.children[0].material.uniforms['nearPlane']['value'] =
      gCamera.near;
  gSceneAccumulate.children[0].material.uniforms['farPlane']['value'] =
      gCamera.far;
  gSceneAccumulate.children[0].material.uniforms['world_T_clip']['value'] =
      world_T_clip;
  gSceneAccumulate.children[0]
      .material.uniforms['history_clip_T_world']['value'] =
      history_clip_T_world;
  gSceneAccumulate.children[0].material.uniforms['cameraPos']['value'] =
      gCamera.position;
  gSceneAccumulate.children[0].material.uniforms['mapHistory']['value'] =
      gAccumulationTextures[accumulationReadIndex].texture;
  gSceneAccumulate.children[0].material.uniforms['jitterOffset']['value'] =
      new THREE.Vector2(offset_x, offset_y);
  gSceneAccumulate.children[0].material.uniforms['emaAlpha']['value'] =
      emaAlpha;
  gSceneAccumulate.children[0].material.uniforms['maxBoxScale']['value'] =
      maxBoxScale;
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
