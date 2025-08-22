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
 * @fileoverview Main file implementing the spherical gaussian based mesh
 * viewer.
 */

/**
 * This vertex shader is used for rendering a g-buffer containing world space
 * positions, which are lated used during deferred rendering.
 */
const positionVertexShaderSource = `
varying vec3 vPositionWorld;

void main() {
  vec3 positionWorld = position;
  vec4 positionClip = projectionMatrix *
    modelViewMatrix * vec4(positionWorld, 1.0);
  gl_Position = positionClip;
  vPositionWorld = positionWorld;
}
`;

/**
 * This fragment shader is used for rendering a g-buffer containing world space
 * positions, which are lated used during deferred rendering.
 */
const positionFragmentShaderSource = `
varying vec3 vPositionWorld;

void main() {
  gl_FragColor = vec4(vPositionWorld, 1.0);
}
`;

/**
 * This vertex shader is used for the deferred rendering pass.
 */
const deferredVertexShaderSource = `
varying vec2 vUv;
varying vec3 vDirection;

uniform mat3 worldspace_R_opengl;
uniform mat4 world_T_clip;

void main() {
  vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = positionClip;

  positionClip /= positionClip.w;
  vec4 nearPoint = world_T_clip *
    vec4(positionClip.x, positionClip.y, -1.0, 1.0);
  vec4 farPoint = world_T_clip * vec4(positionClip.x, positionClip.y, 1.0, 1.0);
  vec3 origin = nearPoint.xyz / nearPoint.w;
  vDirection = worldspace_R_opengl *
    normalize(farPoint.xyz / farPoint.w - origin);

  vUv = uv;
}
`;

/**
 * This fragment shader is used for the deferred rendering pass. Here the actual
 * queries to the texture representation takes place and spherical gaussians
 * are being decoded.
 */
const deferredFragmentShaderSource = `
uniform highp sampler2DArray triplane;

uniform mat4 camera_T_world;
uniform mat4 world_T_clip;

uniform sampler2D positionWorldMap;
uniform float near;
uniform float far;
varying vec2 vUv;
varying vec3 vDirection;

uniform int displayMode;
uniform vec3 sparseGridGridSize;
uniform float sparseGridVoxelSize;
uniform vec3 atlasSize;
uniform float dataBlockSize;
uniform highp sampler2D sparseGridFeatures_00;
uniform highp sampler2D sparseGridFeatures_01;
uniform highp sampler2D sparseGridFeatures_02;
uniform highp sampler2D sparseGridFeatures_03;
uniform highp sampler2D sparseGridFeatures_04;
uniform highp sampler2D sparseGridFeatures_05;

#define GRID_MIN vec3(-2.0, -2.0, -2.0)

uniform mat3 worldspace_R_opengl;
uniform float sceneScaleFactor;

uniform float rangeDiffuseRgbMin;
uniform float rangeDiffuseRgbMax;
uniform float rangeColorMin;
uniform float rangeColorMax;
uniform float rangeMeanMin;
uniform float rangeMeanMax;
uniform float rangeScaleMin;
uniform float rangeScaleMax;

#define SIGMOID(DTYPE) DTYPE sigmoid(DTYPE x) { return 1.0 / (1.0 + exp(-x)); }
SIGMOID(vec3)
SIGMOID(vec4)

#define DENORMALIZE(DTYPE)\
DTYPE denormalize(DTYPE x, float min, float max) {\
    return min + x * (max - min);\
}
DENORMALIZE(float)
DENORMALIZE(vec3)
DENORMALIZE(vec4)

// Component-wise maximum
float max3 (vec3 v) {
  return max (max (v.x, v.y), v.z);
}

// Projective contraction
vec3 contract(vec3 x) {
  vec3 xAbs = abs(x);
  float xMax = max3(xAbs);
  if (xMax <= 1.0) {
    return x;
  }
  float scale = 1.0 / xMax;
  vec3 z = scale * x;
  // note that z.a = sign(z.a) where a is the the argmax component
  if (xAbs.x >= xAbs.y && xAbs.x >= xAbs.z) {
    z.x *= (2.0 - scale); // argmax = 0
  } else if (xAbs.y >= xAbs.x && xAbs.y >= xAbs.z) {
    z.y *= (2.0 - scale); // argmax = 1
  } else {
    z.z *= (2.0 - scale); // argmax = 2
  }
  return z;
}

vec3 evalSphericalGaussian(vec3 direction, vec3 mean, float scale, vec3 color) {
  color = sigmoid(color);
  mean = normalize(mean);
  scale = abs(scale);
  return color * exp(scale * (dot(direction, mean) - 1.0));
}

vec3 compute_sh_shading(vec3 n) {
  // SH coefficients for the "Eucalyptus Grove" scene from
  // "An Efficient Representation for Irradiance Environment Maps"
  // [Ravamoorthi & Hanrahan, 2001]
  vec3 c[9] = vec3[](
      vec3(0.38, 0.43, 0.45),
      vec3(0.29, 0.36, 0.41),
      vec3(0.04, 0.03, 0.01),
      vec3(-0.10, -0.10, -0.09),
      vec3(-0.06, -0.06, -0.04),
      vec3(0.01, -0.01, -0.05),
      vec3(-0.09, -0.13, -0.15),
      vec3(-0.06, -0.05, -0.04),
      vec3(0.02, -0.00, -0.05)
  );

  // From the SH shading implementation in three js:
  // https://github.com/mrdoob/three.js/blob/master/src/math/SphericalHarmonics3.js
  vec3 color = c[0] * 0.282095;

  color += c[1] * 0.488603 * n.y;
  color += c[2] * 0.488603 * n.z;
  color += c[3] * 0.488603 * n.x;

  color += c[4] * 1.092548 * (n.x * n.y);
  color += c[5] * 1.092548 * (n.y * n.z);
  color += c[7] * 1.092548 * (n.x * n.z);
  color += c[6] * 0.315392 * (3.0 * n.z * n.z - 1.0);
  color += c[8] * 0.546274 * (n.x * n.x - n.y * n.y);

  // Brighten everything up a bit with and-tuned constants.
  return 1.66 * color + vec3(0.1, 0.1, 0.1);
}

void main() {
  // See the DisplayMode enum at the top of this file.
  // Runs the full model with view dependence.
  const int DISPLAY_FULL = 0;
  // Disables the view-dependence network.
  const int DISPLAY_DIFFUSE = 1;
  // Only shows the view dependent component.
  const int DISPLAY_VIEW_DEPENDENT = 2;
  // Visualizes the surface normals of the mesh.
  const int DISPLAY_NORMALS = 3;
  // Visualizes the mesh using diffuse shading and a white albedo.
  const int DISPLAY_SHADED = 4;
  // Visualizes the depth map as 1/z.
  const int DISPLAY_DEPTH = 5;

  vec3 normal;
  vec3 positionWorld = texture2D(positionWorldMap, vUv).rgb;
  if (displayMode == DISPLAY_NORMALS ||
    displayMode == DISPLAY_SHADED) {
    vec3 positionWorldRot = worldspace_R_opengl * positionWorld;
    normal = normalize(cross(dFdx(positionWorldRot), dFdy(positionWorldRot)));
  }

  float depth = -(camera_T_world * vec4(positionWorld, 1.0)).z;

  // Query triplanes
  vec3 z = contract(positionWorld * sceneScaleFactor);
  vec3 posTriplaneGrid = (z - GRID_MIN) / TRIPLANE_VOXEL_SIZE;
  vec3[3] planeUv;
  planeUv[0] = vec3(posTriplaneGrid.yz / TRIPLANE_SIZE, 0.0);
  planeUv[1] = vec3(posTriplaneGrid.xz / TRIPLANE_SIZE, 6.0);
  planeUv[2] = vec3(posTriplaneGrid.xy / TRIPLANE_SIZE, 12.0);

  vec3 diffuse = vec3(0.0, 0.0, 0.0);
  vec4[4] v_00;
  if (displayMode == DISPLAY_FULL || displayMode == DISPLAY_DIFFUSE ||
    displayMode == DISPLAY_VIEW_DEPENDENT) {
    v_00[0] = texture2D(sparseGridFeatures_00, vUv);
    v_00[1] = texture(triplane, planeUv[0]);
    v_00[2] = texture(triplane, planeUv[1]);
    v_00[3] = texture(triplane, planeUv[2]);

    for (int k = 0; k < 4; k++) {
      diffuse += denormalize(v_00[k].rgb, rangeDiffuseRgbMin,
        rangeDiffuseRgbMax);
    }

    diffuse = sigmoid(diffuse);
  }

  vec3 viewDependence;
  if (displayMode == DISPLAY_FULL || displayMode == DISPLAY_VIEW_DEPENDENT) {
    vec4[4] v_01;
    vec4[4] v_02;
    vec4[4] v_03;
    vec4[4] v_04;
    vec4[4] v_05;

    // Read from sparse grid that has been resampeled into a 2D map by the
    // prior render passes.
    v_01[0] = texture2D(sparseGridFeatures_01, vUv);
    v_02[0] = texture2D(sparseGridFeatures_02, vUv);
    v_03[0] = texture2D(sparseGridFeatures_03, vUv);
    v_04[0] = texture2D(sparseGridFeatures_04, vUv);
    v_05[0] = texture2D(sparseGridFeatures_05, vUv);

    // Read from triplanes.
    v_01[1] = texture(triplane, planeUv[0] + vec3(0.0, 0.0, 1.0));
    v_02[1] = texture(triplane, planeUv[0] + vec3(0.0, 0.0, 2.0));
    v_03[1] = texture(triplane, planeUv[0] + vec3(0.0, 0.0, 3.0));
    v_04[1] = texture(triplane, planeUv[0] + vec3(0.0, 0.0, 4.0));
    v_05[1] = texture(triplane, planeUv[0] + vec3(0.0, 0.0, 5.0));

    v_01[2] = texture(triplane, planeUv[1] + vec3(0.0, 0.0, 1.0));
    v_02[2] = texture(triplane, planeUv[1] + vec3(0.0, 0.0, 2.0));
    v_03[2] = texture(triplane, planeUv[1] + vec3(0.0, 0.0, 3.0));
    v_04[2] = texture(triplane, planeUv[1] + vec3(0.0, 0.0, 4.0));
    v_05[2] = texture(triplane, planeUv[1] + vec3(0.0, 0.0, 5.0));

    v_01[3] = texture(triplane, planeUv[2] + vec3(0.0, 0.0, 1.0));
    v_02[3] = texture(triplane, planeUv[2] + vec3(0.0, 0.0, 2.0));
    v_03[3] = texture(triplane, planeUv[2] + vec3(0.0, 0.0, 3.0));
    v_04[3] = texture(triplane, planeUv[2] + vec3(0.0, 0.0, 4.0));
    v_05[3] = texture(triplane, planeUv[2] + vec3(0.0, 0.0, 5.0));

    vec3 c0 = vec3(0.0, 0.0, 0.0);
    vec3 m0 = vec3(0.0, 0.0, 0.0);
    float s0 = 0.0;

    vec3 c1 = vec3(0.0, 0.0, 0.0);
    vec3 m1 = vec3(0.0, 0.0, 0.0);
    float s1 = 0.0;

    vec3 c2 = vec3(0.0, 0.0, 0.0);
    vec3 m2 = vec3(0.0, 0.0, 0.0);
    float s2 = 0.0;

    for (int k = 0; k < 4; k++) {
      c0 += denormalize(vec3(v_00[k].a, v_01[k].rg), rangeColorMin,
        rangeColorMax);
      m0 += denormalize(vec3(v_01[k].ba, v_02[k].r), rangeMeanMin,
        rangeMeanMax);
      s0 += denormalize(v_02[k].g, rangeScaleMin, rangeScaleMax);

      c1 += denormalize(vec3(v_02[k].ba, v_03[k].r), rangeColorMin,
        rangeColorMax);
      m1 += denormalize(v_03[k].gba, rangeMeanMin, rangeMeanMax);
      s1 += denormalize(v_04[k].r, rangeScaleMin, rangeScaleMax);

      c2 += denormalize(v_04[k].gba, rangeColorMin, rangeColorMax);
      m2 += denormalize(v_05[k].rgb, rangeMeanMin, rangeMeanMax);
      s2 += denormalize(v_05[k].a, rangeScaleMin, rangeScaleMax);
    }

    vec3 directionWorld = normalize(vDirection);
    viewDependence = evalSphericalGaussian(
        directionWorld, m0, s0, c0);
    viewDependence += evalSphericalGaussian(
      directionWorld, m1, s1, c1);
    viewDependence += evalSphericalGaussian(
      directionWorld, m2, s2, c2);
  }

  vec3 color;
  if (displayMode == DISPLAY_FULL) {
      color = diffuse + viewDependence;
  } else if (displayMode == DISPLAY_DIFFUSE) {
      color = diffuse;
  } else if (displayMode == DISPLAY_VIEW_DEPENDENT) {
      color = viewDependence;
  } else if (displayMode == DISPLAY_NORMALS) {
      color = 0.5 * (normal + 1.0);
  } else if (displayMode == DISPLAY_DEPTH) {
   color = vec3(1.0 / max(0.000001, depth));
  } else /* displayMode == DISPLAY_SHADED */ {
      color = compute_sh_shading(vec3(normal.x, normal.z, normal.y));
  }
  gl_FragColor = vec4(color, 1.0 / depth);
}
`;

/**
 * This vertex shader is used for a render pass that resamples
 * the sparse grid texture into 2D buffers.
 */
const featureMapVertexShaderSource = `
varying vec2 vUv;

uniform mat3 worldspace_R_opengl;
uniform mat4 world_T_clip;

void main() {
  vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = positionClip;

  positionClip /= positionClip.w;
  vec4 nearPoint = world_T_clip *
    vec4(positionClip.x, positionClip.y, -1.0, 1.0);
  vec4 farPoint = world_T_clip * vec4(positionClip.x, positionClip.y, 1.0, 1.0);
  vec3 origin = nearPoint.xyz / nearPoint.w;

  vUv = uv;
}
`;

/**
 * This fragment shader is used for a render pass that resamples
 * the sparse grid texture into 2D buffers.
 */
const featureMapFragmentShaderSource = `

varying vec2 vUv;

uniform sampler2D positionWorldMap;
uniform vec3 sparseGridGridSize;
uniform float sparseGridVoxelSize;
uniform vec3 atlasSize;
uniform float dataBlockSize;
uniform highp sampler3D sparseGridFeatures;
uniform highp sampler3D sparseGridBlockIndices;

#define GRID_MIN vec3(-2.0, -2.0, -2.0)

uniform mat3 worldspace_R_opengl;
uniform float sceneScaleFactor;

// Component-wise maximum
float max3 (vec3 v) {
  return max (max (v.x, v.y), v.z);
}

// Projective contraction
vec3 contract(vec3 x) {
  vec3 xAbs = abs(x);
  float xMax = max3(xAbs);
  if (xMax <= 1.0) {
    return x;
  }
  float scale = 1.0 / xMax;
  vec3 z = scale * x;
  // note that z.a = sign(z.a) where a is the the argmax component
  if (xAbs.x >= xAbs.y && xAbs.x >= xAbs.z) {
    z.x *= (2.0 - scale); // argmax = 0
  } else if (xAbs.y >= xAbs.x && xAbs.y >= xAbs.z) {
    z.y *= (2.0 - scale); // argmax = 1
  } else {
    z.z *= (2.0 - scale); // argmax = 2
  }
  return z;
}

void main() {
  ivec3 iGridSize = ivec3(round(sparseGridGridSize));
  int iBlockSize = int(round(dataBlockSize));
  ivec3 iBlockGridBlocks = (iGridSize + iBlockSize - 1) / iBlockSize;
  ivec3 iBlockGridSize = iBlockGridBlocks * iBlockSize;
  vec3 blockGridSize = vec3(iBlockGridSize);

  vec3 positionWorld = texture2D(positionWorldMap, vUv).rgb;

  // Query sparse grid.
  vec3 z = contract(positionWorld * sceneScaleFactor);

  // With half-voxel offset.
  vec3 posSparseGrid = (z - GRID_MIN) / sparseGridVoxelSize - 0.5;
  vec3 atlasBlockMin =
    floor(posSparseGrid / dataBlockSize) * dataBlockSize;
  vec3 atlasBlockMax = atlasBlockMin + dataBlockSize;
  vec3 atlasBlockIndex =
    255.0 * texture(sparseGridBlockIndices, (atlasBlockMin + atlasBlockMax) /
                                  (2.0 * blockGridSize)).xyz;

  vec3 posAtlas = clamp(posSparseGrid - atlasBlockMin, 0.0, dataBlockSize);

  posAtlas += atlasBlockIndex * (dataBlockSize + 1.0);
  posAtlas += 0.5;
  vec3 atlasUvw = posAtlas / atlasSize;
  gl_FragColor = texture(sparseGridFeatures, atlasUvw);
}
`;

/**
 * This function is called every frame to update controls and render a frame.
 * @param {number} now Current time stamp.
 * @param {string} dirUrl Either points to a directory that contains scene files
 *  or to a json file that maps virtual filenames to download links.
 * @param {!FilenameToLinkTranslator} filenameToLinkTranslator Maps
 *  internal filenames to URLs.
 * @param {!object} sceneParams Holds basic information about the scene like
 *  grid dimensions.
 */
function update(now, dirUrl, filenameToLinkTranslator, sceneParams) {
  if (gControls) {
    gControls.update();
  }

  // For benchmarking, we want to direcly set the projection matrix.
  if (!gBenchmark) {
    gCamera.updateProjectionMatrix();
    gCamera.updateMatrixWorld();
  }

  gScene.traverse(function(child) {
    if (child.isMesh) {
      if (gFirstFrameRendered) {
        child.frustumCulled = true;
      }
    }
  });
  renderProgressively();
  loadOnFirstFrame(dirUrl, filenameToLinkTranslator, sceneParams);
  gStats.update();

  gFirstFrameRendered = true;

  // By default we schedule the next frame ASAP, but the benchmark mode can
  // override this by replacing this lambda.
  let scheduleNextFrame = () => {
    requestAnimationFrame(
        t => update(t, dirUrl, filenameToLinkTranslator, sceneParams));
  };
  if (gBenchmark) {
    scheduleNextFrame = benchmarkPerformance(scheduleNextFrame);
  }
  scheduleNextFrame();
}

/**
 * Renders a frame.
 *
 * @param {?THREE.Matrix4} clip_T_camera (Optional) A (potentially jittered)
 *  projection matrix.
 */
function renderFrame(clip_T_camera) {
  if (!clip_T_camera) {
    clip_T_camera = gCamera.projectionMatrix.clone();
  }
  let camera_T_clip = new THREE.Matrix4();
  camera_T_clip.getInverse(clip_T_camera);

  // Manually override the projection matrix, and restore it at the end
  // of this function.
  let oldProjectionMatrix = gCamera.projectionMatrix.clone();
  let oldProjectionMatrixInverse = gCamera.projectionMatrixInverse.clone();
  gCamera.projectionMatrix = clip_T_camera;
  gCamera.projectionMatrixInverse = camera_T_clip;

  let world_T_camera = gCamera.matrixWorld;
  let world_T_clip = new THREE.Matrix4();
  world_T_clip.multiplyMatrices(world_T_camera, camera_T_clip);

  gDeferredMaterial.uniforms['world_T_clip']['value'] = world_T_clip;
  gDeferredMaterial.uniforms['displayMode']['value'] = gDisplayMode - 0;
  gDeferredMaterial.uniforms['camera_T_world']['value'] =
      gCamera.matrixWorldInverse;

  for (let rep = 0; rep < gFrameMult; rep++) {
    // Keep track of the original render target, so we can restore it
    // for the final render pass.
    const outputRenderTarget = gRenderer.getRenderTarget();

    // First render a 2D map containing world positions.
    gRenderer.setRenderTarget(gPositionWorldRenderTarget);
    gRenderer.clear(true, true, true);
    gRenderer.render(gScene, gCamera);

    // Read world position in and render 2D feature maps.
    if (gDisplayMode == DisplayModeType.DISPLAY_FULL ||
        gDisplayMode == DisplayModeType.DISPLAY_DIFFUSE ||
        gDisplayMode == DisplayModeType.DISPLAY_VIEW_DEPENDENT) {
      gDeferredMesh.material = gRenderFeatureMapMaterial;
      for (let i = 0; i < gNumChannelChunks; i++) {
        gRenderFeatureMapMaterial.uniforms['sparseGridFeatures']['value'] =
            gSparseGridFeaturesTexture[i];
        gRenderer.setRenderTarget(gFeatureMaps[i]);
        gRenderer.clear(true, true, true);
        gRenderer.render(gDeferredScene, gBlitCamera);
      }
    }

    // Read 2D feature maps in and compute final pixel value.
    gDeferredMesh.material = gDeferredMaterial;
    gRenderer.setRenderTarget(outputRenderTarget);  // Reset render target.
    gRenderer.clear(true, true, true);
    gRenderer.render(gDeferredScene, gBlitCamera);
  }

  // Restore the original projection matrix.
  gCamera.projectionMatrix = oldProjectionMatrix;
  gCamera.projectionMatrixInverse = oldProjectionMatrixInverse;
}

document.addEventListener('keypress', function(e) {
  if (e.keyCode === 32 || e.key === ' ' || e.key === 'Spacebar') {
    const renderModeDiv = document.getElementById('rendermode');
    if (gDisplayMode == DisplayModeType.DISPLAY_FULL) {
      gDisplayMode = DisplayModeType.DISPLAY_DIFFUSE;
      renderModeDiv.textContent = 'Diffuse only (press space to toggle)';
    } else if (gDisplayMode == DisplayModeType.DISPLAY_DIFFUSE) {
      gDisplayMode = DisplayModeType.DISPLAY_VIEW_DEPENDENT;
      renderModeDiv.textContent = 'View-dependent only (press space to toggle)';
    } else if (gDisplayMode == DisplayModeType.DISPLAY_VIEW_DEPENDENT) {
      gDisplayMode = DisplayModeType.DISPLAY_NORMALS;
      renderModeDiv.textContent = 'Displaying normals (press space to toggle)';
    } else if (gDisplayMode == DisplayModeType.DISPLAY_NORMALS) {
      gDisplayMode = DisplayModeType.DISPLAY_SHADED;
      renderModeDiv.textContent = 'Showing shaded mesh (press space to toggle)';
    } else if (gDisplayMode == DisplayModeType.DISPLAY_SHADED) {
      gDisplayMode = DisplayModeType.DISPLAY_DEPTH;
      renderModeDiv.textContent = 'Showing disparity (press space to toggle)';
    } else /*gDisplayMode == DisplayModeType.DISPLAY_DEPTH */ {
      gDisplayMode = DisplayModeType.DISPLAY_FULL;
      renderModeDiv.textContent = 'Full rendering (press space to toggle)';
    }
    e.preventDefault();
  }
  if (e.key === 'c') {
    let s = '';
    s += '\'position\': [' + gCamera.position.x + ', ' + gCamera.position.y +
        ', ' + gCamera.position.z + '],\n';
    s += '\'lookat\': [' + gControls.target.x + ', ' + gControls.target.y +
        ', ' + gControls.target.z + '],\n';
    console.log(s);
    e.preventDefault();
  }
  if (e.key === 'i') {
    gFrameMult += 1;
    console.log('gFrameMult:', gFrameMult);
    e.preventDefault();
  }
  if (e.key === 'o') {
    gFrameMult -= 1;
    console.log('gFrameMult:', gFrameMult);
    e.preventDefault();
  }
});

/**
 * Translates filenames to links.
 */
class FilenameToLinkTranslator {
  /**
   * Constructor.
   * @param {string} dirUrl The url where scene files are stored.
   * @param {?object} filenameToLink Dictionary that maps interal file names to
   *     download links.
   */
  constructor(dirUrl, filenameToLink) {
    this.dirUrl = dirUrl;
    this.filenameToLink = filenameToLink;
  }

  /**
   * Maps a virtual filename to an URL.
   * @param {string} filename Internal filename.
   * @return {string} Download URL.
   */
  translate(filename) {
    if (this.filenameToLink != null) {
      // Lookup download URL in map.
      return this.filenameToLink[filename];
    } else {
      // Simply preprend directory.
      return this.dirUrl + '/' + filename;
    }
  }
}

/**
 * Loads all scenes files consisting of the mesh and the appearance
 * representation.
 */
function loadScene() {
  const params = new URL(window.location.href).searchParams;
  const dirUrl = params.get('scene');

  const usageString =
      'To load a scene, specify the following parameters in the URL:\n' +
      'scene: (Required) URL, where the scene files are stored.\n' +
      'renderFactor: (Optional, float) Upsampling factor. Defaults to no' +
      ' upsampling, i.e. factor=1.\n' +
      'aaFactor: (Optional, int) TAA sample amount. Defaults to 4.' +
      'Ignored for sampler=grid.\n' +
      'sampler: (Optional, str) sample sequence for TAA. Defaults to' +
      ' halton, but can be grid or pmj2bn.\n' +
      'frameMult:  (Optional) For benchmarking with vsync on: render' +
      ' frameMult redudant images per frame. Defaults to 1.\n' +
      's: (Optional) The dimensions as width,height. E.g. 640,360. A value of' +
      ' auto leads to using the scene\'s training resolutions.\n' +
      'vfovy:  (Optional) The vertical field of view of the viewer. Defaults' +
      ' to 32.';

  if (!dirUrl) {
    error('scene is a required parameter.\n\n`' + usageString);
  }

  // Screen size: <width>,<height>
  const size = params.get('s') || '1280,720';

  // FOV along screen height. Specified in degrees.
  const vfovy = parseFloat(params.get('vfovy') || 32.0);

  if (params.get('frameMult') != null) {
    gFrameMult = parseInt(params.get('frameMult'));
    console.log('gFrameMult:', gFrameMult);
  }

  let sceneNameToResolution = {};
  sceneNameToResolution['bicycle'] = [1237, 822];
  sceneNameToResolution['flowerbed'] = [1256, 828];
  sceneNameToResolution['gardenvase'] = [1297, 840];
  sceneNameToResolution['stump'] = [1245, 825];
  sceneNameToResolution['treehill'] = [1267, 832];
  sceneNameToResolution['fulllivingroom'] = [1557, 1038];
  sceneNameToResolution['kitchencounter'] = [1558, 1038];
  sceneNameToResolution['kitchenlego'] = [1558, 1039];
  sceneNameToResolution['officebonsai'] = [1559, 1039];

  const benchmarkParam = params.get('benchmark');
  const benchmark = benchmarkParam &&
      (benchmarkParam.toLowerCase() === 'time' ||
       benchmarkParam.toLowerCase() === 'quality');
  const sceneNameChunks = dirUrl.split('/').slice(-2);
  if (benchmark) {
    console.log('Benchmark mode activated.');
    const defaultBenchmarkMotion =
        benchmarkParam.toLowerCase() === 'time' ? 0.0 : 0.05;
    benchmarkMotion =
        parseFloat(params.get('benchmarkMotion') || defaultBenchmarkMotion);
    const numFrameMultIncrements =
        benchmarkParam.toLowerCase() === 'time' ? 2 : 0;
    setupBenchmarkStats(
        sceneNameChunks[0] + '_' + sceneNameChunks[1],
        benchmarkParam.toLowerCase() === 'quality', benchmarkMotion,
        numFrameMultIncrements);
  }

  // Body has a padding of 5 + 5px, we have a border of 2px.
  let frameBufferWidth = window.innerWidth - 12;
  let frameBufferHeight = window.innerHeight - 20;
  renderFactor = parseFloat(params.get('renderFactor') || 1.0);
  aaFactor = parseInt(params.get('aaFactor') || 4, 10);
  sampler = params.get('sampler') || 'halton';

  // Obtain total decoded size to calculate download progress.
  totalBytesToDecode = {
    'gardenvase': 1010522189,
    'stump': 1212188261,
    'flowerbed': 1372500709,
    'treehill': 1639728254,
    'bicycle': 1458295469,
    'kitchenlego': 1133288129,
    'fulllivingroom': 1130651520,
    'kitchencounter': 1149619705,
    'officebonsai': 981507937,
  };
  for (let sceneName in totalBytesToDecode) {
    if (dirUrl.includes(sceneName)) {
      gTotalBytesToDecode = totalBytesToDecode[sceneName];
      break;
    }
  }

  if (size) {
    const match = size.match(/([\d]+),([\d]+)/);
    if (size === 'auto') {
      frameBufferWidth = sceneNameToResolution[sceneNameChunks[1]][0];
      frameBufferHeight = sceneNameToResolution[sceneNameChunks[1]][1];
      console.log('auto resolution:', frameBufferWidth, frameBufferHeight);
    } else {
      frameBufferWidth = parseInt(match[1], 10);
      frameBufferHeight = parseInt(match[2], 10);
    }
  }

  const view = create('div', 'view');
  setDims(view, frameBufferWidth, frameBufferHeight);
  view.textContent = '';

  const viewSpaceContainer = document.getElementById('viewspacecontainer');
  viewSpaceContainer.style.display = 'inline-block';

  const viewSpace = document.querySelector('.viewspace');
  viewSpace.textContent = '';
  viewSpace.appendChild(view);

  canvas = document.createElement('canvas');
  view.appendChild(canvas);

  gStats = Stats();
  viewSpace.appendChild(gStats.dom);
  gStats.dom.style.position = 'absolute';

  // Set up a high performance WebGL context, making sure that anti-aliasing is
  // truned off.
  let gl = canvas.getContext('webgl2', {
    alpha: false,
    premultipliedAlpha: false,
    powerPreference: 'high-performance',
    stencil: false,
    precision: 'highp',
    depth: true,
    antialias: false,
    preserveDrawingBuffer:
        benchmarkParam && benchmarkParam.toLowerCase() === 'quality',
  });
  gRenderer = new THREE.WebGLRenderer({
    canvas: canvas,
    context: gl,
  });
  gRenderer.autoClear = false;
  gRenderer.sortObjects = false;
  gRenderer.autoClearColor = false;
  gRenderer.autoClearStencil = false;
  gRenderer.autoClearDepth = false;
  gRenderer.setClearColor(new THREE.Color('rgb(255, 255, 255)'), 1.0);
  gRenderer.setSize(view.offsetWidth, view.offsetHeight);
  gRenderer.clear(true, true, true);

  gScene = new THREE.Scene();
  const near = 0.25;
  const far = 100.0;  // previously 1000.0
  gCamera = new THREE.PerspectiveCamera(
      vfovy,
      Math.trunc(view.offsetWidth / renderFactor) /
          Math.trunc(view.offsetHeight / renderFactor),
      near, far);

  if (!benchmark) {
    gControls = new THREE.OrbitControls(gCamera, view);
    gControls.screenSpacePanning = true;
    setupInitialCameraPose(dirUrl);
  }

  setupProgressiveRendering(view, renderFactor, aaFactor, sampler);

  // Setup deferred rendering.
  gPositionWorldRenderTarget =
      new THREE.WebGLRenderTarget(view.offsetWidth, view.offsetHeight, {
        minFilter: THREE.NearestFilter,
        maxFilter: THREE.NearestFilter,
        type: THREE.FloatType,
        format: THREE.RGBAFormat,
      });
  gBlitCamera = new THREE.OrthographicCamera(
      view.offsetWidth / -2, view.offsetWidth / 2, view.offsetHeight / 2,
      view.offsetHeight / -2, -10000, 10000);
  gBlitCamera.position.z = 100;

  let worldspace_R_opengl = new THREE.Matrix3();
  worldspace_R_opengl['set'](-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);

  // Setup 2D buffers that the sparse grid 3D textures are resampled into, which
  // improves cache coherency.
  for (let i = 0; i < gNumChannelChunks; i++) {
    featureMap =
        new THREE.WebGLRenderTarget(view.offsetWidth, view.offsetHeight, {
          minFilter: THREE.NearestFilter,
          maxFilter: THREE.NearestFilter,
          type: THREE.UnsignedByteType,
          format: THREE.RGBAFormat,
        });
    gFeatureMaps.push(featureMap);
  }

  // Check if dirUrl points to a json file or to a directory.
  if (dirUrl.includes('.json')) {
    // If this is the case, we fetch a JSON file that maps filenames to links.
    filenameToLinkPromise = loadJSONFile(dirUrl);
  } else {
    // Otherwise, the scene files directly lie at dirUrl and we create a
    // dummy promise that resolves immediately.
    filenameToLinkPromise = Promise.resolve(null);
  }

  filenameToLinkPromise.then(filenameToLink => {
    const filenameToLinkTranslator =
        new FilenameToLinkTranslator(dirUrl, filenameToLink);

    // Loads scene parameters.
    let sceneParamsUrl =
        filenameToLinkTranslator.translate('scene_params.json');
    let sceneParamsPromise = loadJSONFile(sceneParamsUrl);

    if (benchmark) {
      loadBenchmarkCameras(filenameToLinkTranslator);
    }

    sceneParamsPromise.then(sceneParams => {
      if (!('scene_scale_factor' in sceneParams)) {
        sceneParams['scene_scale_factor'] = 1.0;
      }

      // Load the indirection grid.
      let sparseGridBlockIndicesUrl =
          filenameToLinkTranslator.translate('sparse_grid_block_indices.raw');
      let sparseGridBlockIndicesPromise =
          loadBinaryFile(sparseGridBlockIndicesUrl);

      // Load sparse grid, note that textures are only allocated and
      // the actual loading is done progressively in loadOnFirstFrame.
      let sparseGridBlockIndicesTexture = null;

      // Create empty volume textures that are lated filled slice-by-slice.
      function _createEmptyAtlasVolumeTexture() {
        return createEmptyVolumeTexture(
            sceneParams['atlas_width'], sceneParams['atlas_height'],
            sceneParams['atlas_depth'], THREE.RGBAFormat, THREE.LinearFilter);
      }
      for (let channelChunkIndex = 0; channelChunkIndex < gNumChannelChunks;
           ++channelChunkIndex) {
        gSparseGridFeaturesTexture.push(_createEmptyAtlasVolumeTexture());
      }

      // The indirection grid uses nearest filtering and is loaded in one go.
      let v = sceneParams['sparse_grid_resolution'] /
          sceneParams['data_block_size'];
      sparseGridBlockIndicesTexture = createEmptyVolumeTexture(
          v, v, v, THREE.RGBFormat, THREE.NearestFilter);
      sparseGridBlockIndicesPromise.then(sparseGridBlockIndicesImage => {
        sparseGridBlockIndicesTexture.image.data = sparseGridBlockIndicesImage;
        sparseGridBlockIndicesTexture.needsUpdate = true;
      });


      // Load triplane features.
      let planePromises = [];
      for (let planeIndex = 0; planeIndex < 3; ++planeIndex) {
        for (let channelChunkIndex = 0; channelChunkIndex < gNumChannelChunks;
             ++channelChunkIndex) {
          let planeUrl = filenameToLinkTranslator.translate(
              'plane_features_' + planeIndex + '_' +
              digits(channelChunkIndex, 2) + '.raw');
          let planePromise = loadBinaryFile(planeUrl);
          planePromises.push(planePromise);
        }
      }
      planePromises = Promise.all(planePromises);

      // Create triplane textures.
      let triplaneResolution = sceneParams['triplane_resolution'];
      let triplaneTexture = createEmptyTriplaneTextureArray(
          triplaneResolution, triplaneResolution, THREE.RGBAFormat);

      // Load triplanes.
      let triplaneBytesPerImage = triplaneResolution * triplaneResolution * 4;
      planePromises.then(inputImages => {
        let outputData =
            new Uint8Array(3 * gNumChannelChunks * triplaneBytesPerImage);

        let imageIndex = 0;
        let targetIndex = 0;
        for (let planeIndex = 0; planeIndex < 3; planeIndex++) {
          for (let channelChunkIndex = 0; channelChunkIndex < gNumChannelChunks;
               ++channelChunkIndex) {
            inputData = inputImages[imageIndex];
            for (let sourceIndex = 0; sourceIndex < triplaneBytesPerImage;
                 ++sourceIndex) {
              outputData[targetIndex] = inputData[sourceIndex];
              targetIndex++;
            }
            imageIndex++;
          }
        }
        triplaneTexture.image.data = outputData;
        triplaneTexture.needsUpdate = true;
      });

      let uniforms = {
        'worldspace_R_opengl': {'value': worldspace_R_opengl},
        'positionWorldMap': {'value': gPositionWorldRenderTarget.texture},
        'near': {'value': near},
        'far': {'value': far},
        'world_T_clip': {'value': new THREE.Matrix4()},
        'camera_T_world': {'value': new THREE.Matrix4()},
        'displayMode': {'value': gDisplayMode - 0},
        'dataBlockSize': {'value': sceneParams['data_block_size']},
        'sparseGridVoxelSize': {'value': sceneParams['sparse_grid_voxel_size']},
        'sparseGridGridSize': {
          'value': new THREE.Vector3(
              sceneParams['sparse_grid_resolution'],
              sceneParams['sparse_grid_resolution'],
              sceneParams['sparse_grid_resolution'])
        },
        'atlasSize': {
          'value': new THREE.Vector3(
              sceneParams['atlas_width'], sceneParams['atlas_height'],
              sceneParams['atlas_depth'])
        },
        'sceneScaleFactor': {'value': sceneParams['scene_scale_factor']},
        'triplane': {'value': triplaneTexture},

        'rangeDiffuseRgbMin':
            {'value': sceneParams['ranges']['diffuse_rgb']['min']},
        'rangeDiffuseRgbMax':
            {'value': sceneParams['ranges']['diffuse_rgb']['max']},
        'rangeColorMin': {'value': sceneParams['ranges']['color']['min']},
        'rangeColorMax': {'value': sceneParams['ranges']['color']['max']},
        'rangeMeanMin': {'value': sceneParams['ranges']['mean']['min']},
        'rangeMeanMax': {'value': sceneParams['ranges']['mean']['max']},
        'rangeScaleMin': {'value': sceneParams['ranges']['scale']['min']},
        'rangeScaleMax': {'value': sceneParams['ranges']['scale']['max']},
      };

      for (let channelChunkIndex = 0; channelChunkIndex < gNumChannelChunks;
           ++channelChunkIndex) {
        uniforms['sparseGridFeatures_' + digits(channelChunkIndex, 2)] = {
          'value': gFeatureMaps[channelChunkIndex].texture
        };
      }

      let triplaneResolutionFormatted = triplaneResolution.toFixed(1);
      let shaderDefinitions = '#define TRIPLANE_SIZE vec2(' +
          triplaneResolutionFormatted + ', ' + triplaneResolutionFormatted +
          ')\n';
      shaderDefinitions += '#define TRIPLANE_VOXEL_SIZE ' +
          sceneParams['triplane_voxel_size'] + '\n';

      gDeferredMaterial = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: deferredVertexShaderSource,
        fragmentShader: shaderDefinitions + deferredFragmentShaderSource,
      });

      let featureMapUniforms = {
        'worldspace_R_opengl': {'value': worldspace_R_opengl},
        'positionWorldMap': {'value': gPositionWorldRenderTarget.texture},
        'world_T_clip': {'value': new THREE.Matrix4()},
        'dataBlockSize': {'value': sceneParams['data_block_size']},
        'sparseGridVoxelSize': {'value': sceneParams['sparse_grid_voxel_size']},
        'sparseGridGridSize': {
          'value': new THREE.Vector3(
              sceneParams['sparse_grid_resolution'],
              sceneParams['sparse_grid_resolution'],
              sceneParams['sparse_grid_resolution'])
        },
        'atlasSize': {
          'value': new THREE.Vector3(
              sceneParams['atlas_width'], sceneParams['atlas_height'],
              sceneParams['atlas_depth'])
        },
        'sparseGridBlockIndices': {'value': sparseGridBlockIndicesTexture},
        'sceneScaleFactor': {'value': sceneParams['scene_scale_factor']},
        'sparseGridFeatures': {'value': gSparseGridFeaturesTexture[0]},
      };

      gRenderFeatureMapMaterial = new THREE.ShaderMaterial({
        uniforms: featureMapUniforms,
        vertexShader: featureMapVertexShaderSource,
        fragmentShader: featureMapFragmentShaderSource,
      });

      let fullScreenPlane =
          new THREE.PlaneBufferGeometry(view.offsetWidth, view.offsetHeight);
      gDeferredMesh = new THREE.Mesh(fullScreenPlane, gDeferredMaterial);
      gDeferredScene = new THREE.Scene();
      gDeferredScene.add(gDeferredMesh);

      gMaterial = new THREE.ShaderMaterial({
        vertexShader: positionVertexShaderSource,
        fragmentShader: positionFragmentShaderSource,
      });

      // Load the mesh.
      meshUrl =
          filenameToLinkTranslator.translate('viewer_mesh_post_gltfpack.glb');
      meshPromise = loadMesh(meshUrl);
      meshPromise.then(gltf => {
        gltf.scene.traverse(child => {
          if (child.isMesh) {
            // Ensure that all meshes get uploaded on the first frame.
            child.frustumCulled = false;
            child.material = gMaterial;
          }
        });
        gScene.add(gltf.scene);
        requestAnimationFrame(
            t => update(t, dirUrl, filenameToLinkTranslator, sceneParams));
      });
    });
  });
}

/**
 * Creates three equally sized textures to hold triplanes.
 * @param {number} width Width of the texture.
 * @param {number} height Height of the texture.
 * @param {number} format Format of the texture.
 * @return {!THREE.DataTexture2DArray} Texture array of size three.
 */
function createEmptyTriplaneTextureArray(width, height, format) {
  let texture =
      new THREE.DataTexture2DArray(null, width, height, 3 * gNumChannelChunks);
  texture.format = format;
  texture.generateMipmaps = false;
  texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.type = THREE.UnsignedByteType;
  return texture;
}

/**
 * Creates an empty volume texture.
 * @param {number} width Width of the texture.
 * @param {number} height Height of the texture.
 * @param {number} depth Depth of the texture.
 * @param {number} format Format of the texture.
 * @param {number} filter Filter strategy of the texture.
 * @return {!THREE.DataTexture3D} Volume texture.
 */
function createEmptyVolumeTexture(width, height, depth, format, filter) {
  volumeTexture = new THREE.DataTexture3D(null, width, height, depth);
  volumeTexture.format = format;
  volumeTexture.generateMipmaps = false;
  volumeTexture.magFilter = volumeTexture.minFilter = filter;
  volumeTexture.wrapS = volumeTexture.wrapT = volumeTexture.wrapR =
      THREE.ClampToEdgeWrapping;
  volumeTexture.type = THREE.UnsignedByteType;
  return volumeTexture;
}

/**
 * Uploads a volume slice to a pre-existing volume texture.
 * @param {!Uint8Array} volumeSlice Data to be uploaded
 * @param {!THREE.DataTexture3D} texture Volume texture to upload to
 * @param {number} sliceIndex Index of the slice
 * @param {number} volumeWidth Width of the volume
 * @param {number} volumeHeight Height of the volume
 * @param {number} sliceDepth Depth of slice
 */
function uploadVolumeSlice(
    volumeSlice, texture, sliceIndex, volumeWidth, volumeHeight, sliceDepth) {
  const textureProperties = gRenderer['properties'].get(texture);
  let gl = gRenderer.getContext();

  let stride, glFormat;
  if (texture.format == THREE.RGBAFormat) {
    glFormat = gl.RGBA;
    stride = 4;
  } else if (texture.format == THREE.RGBFormat) {
    glFormat = gl.RGB;
    stride = 3;
  } else if (texture.format == THREE.RedFormat) {
    glFormat = gl.RED;
    stride = 1;
  }

  let oldTexture = gl.getParameter(gl.TEXTURE_BINDING_3D);
  let textureHandle = textureProperties['__webglTexture'];
  gl.bindTexture(gl.TEXTURE_3D, textureHandle);
  // Upload row-by-row to work around bug with Intel + Mac OSX.
  // See https://crbug.com/654258.
  for (let z = 0; z < sliceDepth; ++z) {
    for (let y = 0; y < volumeHeight; ++y) {
      gl.texSubImage3D(
          gl.TEXTURE_3D, 0, 0, y, z + sliceIndex * sliceDepth, volumeWidth, 1,
          1, glFormat, gl.UNSIGNED_BYTE, volumeSlice,
          stride * volumeWidth * (y + volumeHeight * z));
    }
  }
  gl.bindTexture(gl.TEXTURE_3D, oldTexture);
}

/**
 * Loads a volume texture slice-by-slice.
 * @param {!object} uploadFn Function that is called when a new volume slice is
 *  received.  Uploads function to texture.
 * @param {number} numSlices Total number of slices
 * @param {number} sliceDepth Depth of slice
 * @param {number} volumeWidth Width of the volume
 * @param {number} volumeHeight Height of the volume
 * @param {number} volumeDepth Depth of the volume
 * @param {string} filenamePrefix The string all filenames start with. The slice
 *  index and the png file ending are appended to this string.
 * @param {!object} filenameToLinkTranslator
 * @return {!Promise} Resolves when the texture is fully uploaded
 */
function loadVolumeTextureSliceBySlice(
    uploadFn, numSlices, sliceDepth, volumeWidth, volumeHeight, volumeDepth,
    filenamePrefix, filenameToLinkTranslator) {
  let uploadPromises = [];
  for (let sliceIndex = 0; sliceIndex < numSlices; sliceIndex++) {
    let url = filenameToLinkTranslator.translate(
        filenamePrefix + '_' + digits(sliceIndex, 3) + '.raw');
    let rgbaPromise = loadBinaryFile(url);
    rgbaPromise = rgbaPromise.then(data => {
      return data;
    });

    let uploadPromise = new Promise(function(resolve, reject) {
      rgbaPromise
          .then(rgbaImage => {
            uploadFn(
                rgbaImage, sliceIndex, volumeWidth, volumeHeight, sliceDepth);
            resolve();
          })
          .catch(error => {
            reject(error);
          });
    });
    uploadPromises.push(uploadPromise);
  }
  return Promise.all(uploadPromises);
}

/**
 * This function needs to run after the first frame has been rendered since we
 * are uploading to textures which only become valid after the first frame has
 * been rendered.
 * @param {string} dirUrl Either points to a directory that contains scene files
 *  or to a json file that maps virtual filenames to download links.
 * @param {!FilenameToLinkTranslator} filenameToLinkTranslator
 * @param {!object} sceneParams Holds basic information about the scene like
 *  grid dimensions.
 **/
function loadOnFirstFrame(dirUrl, filenameToLinkTranslator, sceneParams) {
  if (gFirstFrameRendered) return;

  // Early out if the renderer is not supported.
  if (isRendererUnsupported()) {
    gFirstFrameRendered = true;
    let loadingContainer = document.getElementById('loading-container');
    loadingContainer.style.display = 'none';
    return;
  }

  function _loadVolumeTextureSliceBySlice(uploadFn, filenamePrefix) {
    return loadVolumeTextureSliceBySlice(
        uploadFn, sceneParams['num_slices'], sceneParams['slice_depth'],
        sceneParams['atlas_width'], sceneParams['atlas_height'],
        sceneParams['atlas_depth'], filenamePrefix, filenameToLinkTranslator);
  }

  let allTexturesPromise = [];
  for (let channelChunkIndex = 0; channelChunkIndex < gNumChannelChunks;
       ++channelChunkIndex) {
    // The pngs RGBA channels are directly interpreted as four feature channels.
    let uploadFn =
        (rgbaImage, sliceIndex, volumeWidth, volumeHeight, sliceDepth) => {
          uploadVolumeSlice(
              rgbaImage, gSparseGridFeaturesTexture[channelChunkIndex],
              sliceIndex, volumeWidth, volumeHeight, sliceDepth);
        };

    let sparseGridFeaturesTexturePromise = _loadVolumeTextureSliceBySlice(
        uploadFn, 'sparse_grid_features_' + digits(channelChunkIndex, 2));
    allTexturesPromise.push(sparseGridFeaturesTexturePromise);
  }
  allTexturesPromise = Promise.all(allTexturesPromise);

  allTexturesPromise.catch(errors => {
    console.error(
        'Could not load scene from: ' + dirUrl + ', errors:\n\t' + errors[0] +
        '\n\t' + errors[1] + '\n\t' + errors[2] + '\n\t' + errors[3]);
  });

  allTexturesPromise.then(texture => {
    hideLoading();
    console.log('Successfully loaded scene from: ' + dirUrl);
  });
}

window.onload = loadScene;
