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
 * Web worker responsible for copying bytes from sliced RGBA arrays to other
 * targets.
 */

importScripts("utils.js");

/**
 * Copy asset.asset's contents into dstData using the format described by src &
 * dst.
 */
async function copySliceToBuffer(dstData, asset, src, dst) {
  console.assert(src.channels.length == dst.channels.length, src, dst);

  // Determine the offset for the output for this slice. Assume that each
  // slice has the same shape.
  let numPixels = product(asset.shape);
  let baseOutOffset = numPixels * dst.format.numChannels * asset.sliceIndex;

  // Wait for data to be ready.
  let srcData = await asset.asset;

  // Copy this asset's contents into dstData.
  for (let px = 0; px < numPixels; ++px) {
    // Buffer offsets for this pixel.
    let inOffset = px * src.format.numChannels;
    let outOffset = baseOutOffset + px * dst.format.numChannels;
    for (let i = 0; i < src.channels.length; ++i) {
      let inChannel = src.channels[i];
      let outChannel = dst.channels[i];
      dstData[outOffset + outChannel] = srcData[inOffset + inChannel];
    }
  }
}


/**
 * Process an asset of type *_slices.
 */
function processSlices(e) {
  const i = e.data.i;
  let { asset, src, dst } = e.data.request;

  let numPixels = product(asset.shape);
  let result = new Uint8Array(numPixels * dst.format.numChannels);

  let promises = asset.sliceAssets.map((sliceAsset) => {
    return copySliceToBuffer(result, sliceAsset, src, dst);
  });

  return Promise.all(promises).then(() => {
    self.postMessage({i: i, result: result}, [result.buffer]);
  });
}


/**
 * Process sparse grid's density.
 */
function processSparseGridDensity(e) {
  const i = e.data.i;
  let { asset } = e.data.request;

  let numSlices = asset.rgbAndDensityAsset.numSlices;
  console.assert(asset.rgbAndDensityAsset.numSlices == numSlices, asset);
  console.assert(asset.featuresAsset.numSlices == numSlices, asset);

  // Create a buffer for a luminance-alpha texture.
  let result = new Uint8Array(product(asset.rgbAndDensityAsset.shape) * 2);

  // Populate texture as assets arrive.
  let promises = range(numSlices).map((i) => {
    let rgbSliceAsset = asset.rgbAndDensityAsset.sliceAssets[i];
    let featuresSliceAsset = asset.featuresAsset.sliceAssets[i];

    // Copy density features to buffer. This requires merging the alpha values
    // from two different data sources.
    let copyLuminance = copySliceToBuffer(
        result,
        rgbSliceAsset,
        GridTextureSource.ALPHA_FROM_RGBA,
        GridTextureDestination.LUMINANCE_IN_LUMINANCE_ALPHA,
    );

    let copyAlpha = copySliceToBuffer(
        result,
        featuresSliceAsset,
        GridTextureSource.ALPHA_FROM_RGBA,
        GridTextureDestination.ALPHA_IN_LUMINANCE_ALPHA,
    );

    return Promise.all([copyLuminance, copyAlpha]);
  });

  return Promise.all(promises).then(() => {
    self.postMessage({i: i, result: result}, [result.buffer]);
  });
}

/**
 * Main entry point.
 */
function main(e) {
  if (e.data.request.fn == 'mergeSlices') {
    return processSlices(e);
  } else if (e.data.request.fn == 'mergeSparseGridDensity') {
    return processSparseGridDensity(e);
  } else {
    throw new Error(`Unrecognized request`, e.data.request);
  }
}

// main() is called every time a new message comes in.
self.onmessage = main;

