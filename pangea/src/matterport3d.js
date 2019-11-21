// Copyright 2019 The Google Research Authors.
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
 * @fileoverview A graph instance for the Matterport3D indoor simulator.
 * https://niessner.github.io/Matterport/
 * This implementation can be adapted for other environments.
 */

{
  // Constants used for three.js rendering.
  const DEFAULT_RADIUS = 1000000;

  /**
   * A graph instance for the Matterport3D indoor simulator.
   * https://niessner.github.io/Matterport/
   */
  var Matterport3D = class extends Graph {
    /**
     * Constructs the navigation graph.
     * @param {string} scanID A house in the simulator.
     * @param {string} directory Local directory or Google cloud storage
     * (GCS) bucket to store static data. The code below adopts a flat
     * directory structure. All files live in the root. See this doc for
     * file naming conventions:
     * https://github.com/niessner/Matterport/blob/master/data_organization.md
     */
    constructor(scanID, directory) {
      super();
      this._directory = directory;

      var filename = this._getConnectivityFilename(scanID);
      var promise = d3.json(filename);

      var data, scope = this;
      promise.then(_data => scope._data = _data);

      // Private attributes.

      this._cache = {};
      this._data = data;
      this._promise = promise;
      this._scanID = scanID;
    }

    // Private methods.

    /**
     * Returns the filename for the graph connectivity file. Redefine this
     * method to match your directory structure.
     * @param {string} scanID A house in the simulator.
     * @return {string} The filename.
     */
    _getConnectivityFilename(scanID) {
      return this._directory + scanID + '_connectivity.json';
    }

    /**
     * Returns a list of filenames. One for each side of a cube mapping
     * ordered px nx py ny pz nz.
     * @param {string} scanID A house in the simulator.
     * @param {string} panoID A node in the graph.
     * @return {!Array<string>} The filenames.
     */
    _getBoxTextureFilenames(scanID, panoID) {
      return [2, 4, 0, 5, 1, 3].map(i => {
        return this._directory + panoID + '_skybox' + i + '_sami.jpg';
      });
    }

    /**
     * Returns an object containing camera information.
     * @param {string} node A node in the graph.
     * @return {?Object} An object containing camera information.
     */
    _getDatum(node) {
      var datum;

      this._data.forEach(_datum => {
        if (_datum.image_id == node && _datum.included) {
          datum = _datum;
        }
      });

      return datum;
    }

    // Graph class API.

    /**
     * Returns a list of the nodes in the graph.
     * @return {!Array<string>} A list of nodes.
     */
    getNodes() {
      var nodes = [];

      this._data.forEach(datum => {
        if (datum.included) nodes.push(datum.image_id);
      });

      return nodes;
    }

    /**
     * Returns a list of nodes adjacent to the one given.
     * @param {string} node A node in the graph.
     * @return {!Array<string>} A list of nodes.
     */
    getNeighbors(node) {
      var datum = this._getDatum(node);
      var neighbors = [];

      datum.visible.forEach((bool, i) => {
        var _datum = this._data[i];
        if (bool && _datum.included) neighbors.push(_datum.image_id);
      });

      return neighbors;
    }

    /**
     * Returns a homography matrix transforming vectors from camera to
     * world coordinates.
     * @param {string} node A node in the graph.
     * @return {!THREE.Matrix4} A homography matrix.
     */
    getCameraMatrix(node) {
      var datum = this._getDatum(node);

      return new THREE.Matrix4().fromArray(datum.pose).transpose();
    }

    /**
     * Returns the distance of the camera from the floor.
     * @param {string} node A node in the graph.
     * @return {number} The camera height.
     */
    getCameraHeight(node) {
      return this._getDatum(node).height;
    }

    /**
     * Returns a list of textures. The panorama texture mapping.
     * @param {string} node A node in the graph.
     * @return {!Array<!THREE.Texture>} The panorama texture mapping.
     */
    getCameraTextures(node) {
      // Caches the textures for efficiency.
      if (!this._cache.hasOwnProperty(node)) {
        var loader = new THREE.TextureLoader();
        var filenames = this._getBoxTextureFilenames(this._scanID, node);
        this._cache[node] = filenames.map(f => loader.load(f));
      }

      return this._cache[node];
    }

    /**
     * Returns the geometry of the texture mapping. Default is equirectancular.
     * @param {number=} radius The viewing horizon of the geometry.
     * @return {!THREE.Geometry} The geometry of the texture mapping.
     */
    getCameraGeometry(radius = DEFAULT_RADIUS) {
      return new THREE.BoxGeometry(radius, radius, radius);
    }

    /**
     * Converts from Matterport's z-up coordinates to three.js's y-up
     * coordinates.
     */
    localToWorld(object) {
      object.rotation.x = -Math.PI / 2;
    }

    /**
     * Returns a boolean or promise indicating initialization.
     * @return {!Promise <boolean>} Whether or not the graph is
     * initialized.
     */
    isInitialized() {
      return this._promise.then(() => true, () => false);
    }
  };
}

(global => {
  if (typeof define === 'function' && define.amd) {
    define([], () => Matterport3D);

  } else if (typeof module !== 'undefined' && typeof exports === 'object') {
    module.exports = Matterport3D;

  } else if (global !== undefined) {
    global.Matterport3D = Matterport3D;
  }
})(this);
