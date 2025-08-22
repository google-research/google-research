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
 * A web worker for parsing binary assets in a separate thread.
 * @type {*}
 */

/**
 * A singleton managing a collection of web workers.
 */
class WorkerPool {
  /**
   * Initializes a WorkerPool
   */
  constructor(numWorkers, filename) {
    let that = this;
    numWorkers = numWorkers || 2;

    // Create a pool of workers.
    this.workers = [];
    for (let i = 0; i < numWorkers; ++i) {
      let worker = new Worker(filename);
      worker.onmessage = (e) => {
        that.onmessage(e);
      };
      this.workers.push(worker);
    }

    this.nextworker = 0;
    this.callbacks = {};
    this.i = 0;
  }

  /**
   * Submit task to web worker.
   */
  submit(request, callback) {
    const i = this.i;
    this.callbacks[i] = callback;
    this.i += 1;

    const w = this.nextworker;
    const worker = this.workers[w];
    this.nextworker = (w + 1) % this.workers.length;

    worker.postMessage({i, request});
  }

  /**
   * Callback for this.worker.
   */
  onmessage(e) {
    const response = e.data;
    const i = response.i;
    const callback = this.callbacks[i];
    callback(response.result);
    delete this.callbacks[i];
  }
}