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
 * Web worker script for parsing PNG files in a separate thread.
 *
 * We use a pure JavaScript library for parsing PNGs as there is no
 * way to access decoded PNGs bytes directly. Solutions that rely on
 * <canvas> elements are lossy thanks to alpha premultiplication.
 *
 */

importScripts("deps/zlib.js");
importScripts("deps/png.js");

let gWorkQueue = [];
let gLiveRequests = {};
let kMaxLiveRequests = 100;
let kNumSleepMilliseconds = 250;


/**
 * Return a promise that sleeps for a given number of milliseconds.
 * @param {number} ms ...
 * @return {!Promise} ...
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Safe fetching. Some servers restrict the number of requests and
 * respond with status code 429 ("Too Many Requests") when a threshold
 * is exceeded. When we encounter a 429 we retry after a short waiting period.
 * @param {!object} fetchFn Function that fetches the file.
 * @return {!Promise} Returns fetchFn's response.
 */
async function fetchAndRetryIfNecessary(fetchFn) {
  let response;
  while (true) {
    response = await fetchFn();
    if (response.status === 429) {
      await sleep(kNumSleepMilliseconds);
      continue;
    }
    break;
  }
  return response;
}

/**
 * Decode a PNG file.
 *
 * @param {!Uint8Array} array Bytes to decode.
 * @return {!object} ...
 */
function parsePNG(array) {
  let pngDecoder = new PNG(array);
  let pixels = pngDecoder.decodePixels();
  return pixels;
}


/**
 * Process a request by fetching a URL, decoding its contents, and sending the
 * result back to the main thread.
 * @param {!object} e ...
 * @return {!object} ...
 */
function processRequest(e) {
  const i = e.data.i;
  let { url } = e.data.request;

  // Trim *.gz and trust the browser to take care of gzip decoding.
  if (url.endsWith('.gz')) {
    url = url.substring(0, url.length-3);
  }

  // The following promise runs freely till the computation chain completes.
  const promise =
      fetchAndRetryIfNecessary(() => {
        return fetch(url, {method: 'GET', mode: 'cors'});
      })
      .then(response => {
        return response.arrayBuffer();
      })
      .then(buffer => {
        let content = new Uint8Array(buffer);
        if (url.endsWith('.raw')) {
          return content;  // no further processing required.
        } else if (url.endsWith('.png')) {
          return parsePNG(content);
        } else {
          console.error(`Unrecognized filetype for ${url}`);
          return null;
        }
      })
      .then(array => {
        self.postMessage({i: i, result: array}, [array.buffer]);
        // Ensure that the array was handed off to the main thread.
        console.assert(array.byteLength == 0);

        // self.postMessage({i: i, result: array});
      })
      .catch(error => {
        console.error(`Could not load asset from: ${url}, error: ${error}`);
        return null;
      });

  return promise;
}

/**
 * Main loop.
 */
async function main() {
  while (true) {
    let numLiveRequests = Object.keys(gLiveRequests).length;
    if (numLiveRequests >= kMaxLiveRequests) {
      await sleep(kNumSleepMilliseconds);
      continue;
    }
    if (gWorkQueue.length == 0) {
      await sleep(kNumSleepMilliseconds);
      continue;
    }
    let request = gWorkQueue.shift();
    gLiveRequests[request.data.i] = request;
    processRequest(request).finally(() => {
      delete gLiveRequests[request.data.i];
    });
  }
}

/**
 * Push a work item onto the queue.
 * @param {!object} e ...
 */
self.onmessage = function(e) {
  gWorkQueue.push(e);
};


main();
