// Copyright 2024 The Google Research Authors.
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

/**
 * Safe fetching. Some servers restrict the number of requests and
 * respond with status code 429 ("Too Many Requests") when a threshold
 * is exceeded. When we encounter a 429 we retry after a short waiting period.
 * @param {!object} fetchFn Function that fetches the file.
 * @return {!Promise} Returns fetchFn's response.
 */
async function fetchAndRetryIfNecessary(fetchFn) {
  const response = await fetchFn();
  if (response.status === 429) {
    await sleep(500);
    return fetchAndRetryIfNecessary(fetchFn);
  }
  return response;
}

/**
 * Fetches and decodes a PNG image.
 * @param {*} e  Event payload. The data attribute must contain the `i` to
 *               identify which callback to use and `url` to know which PNG to
 *               load.
 */
self.onmessage = function(e) {
  const i = e.data.i;
  let url = e.data.url;

  // Trim *.gz and trust the browser to take care of gzip decoding.
  if (url.endsWith('.gz')) {
    url = url.substring(0, url.length-3);
  }

  // The following promise runs freely till the computation chain completes.
  fetchAndRetryIfNecessary(() => {
        return fetch(url, {method: 'GET', mode: 'cors'});
      })
      .then(response => { return response.arrayBuffer(); })
      .then(buffer => {
        let content = new Uint8Array(buffer);
        if (url.endsWith('.raw')) {
          return content;  // no further processing required.
        } else {
          console.error(`Unrecognized filetype for ${url}`);
          return null;
        }
      })
      .then(buffer => {
        self.postMessage({i: i, result: buffer});
      })
      .catch(error => {
        console.error(`Could not load assert from: ${url}, error: ${error}`);
        return null;
      });
};
