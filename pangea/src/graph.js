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
 * @fileoverview Abstract class for a navigation graph.
 */

{
  // Constants used for three.js rendering.
  const DEFAULT_RADIUS = 1000000;

  /**
   * Abstract class for a navigation graph.
   */
  var Graph = class {
    /**
     * Constructor for a navigation graph.
     */
    constructor() {
      this.isGraph = true;
    }

    /**
     * Returns a list of the nodes in the graph.
     * @return { !Array< string > } - A list of nodes.
     */
    getNodes() {
      return [];
    }

    /**
     * Returns a list of nodes adjacent to the one given.
     * @param { string } node - A node in the graph.
     * @return { !Array< string > } - A list of nodes.
     */
    getNeighbors(node) {
      return [];
    }

    /**
     * Returns a homography matrix transforming vectors from camera to
     * world coordinates.
     * @param { string } node - A node in the graph.
     * @return { !THREE.Matrix4 } - A homography matrix.
     */
    getCameraMatrix(node) {
      return new THREE.Matrix4();
    }

    /**
     * Returns the distance of the camera from the floor.
     * @param { string } node - A node in the graph.
     * @return { number } - The camera height.
     */
    getCameraHeight(node) {
      return 0;
    }

    /**
     * Returns a list of textures. The panorama texture mapping.
     * @param { string } node - A node in the graph.
     * @return { !Array< !THREE.Texture > } - The panorama texture mapping.
     */
    getCameraTextures(node) {
      return [new THREE.Color(0xffffff)];
    }

    /**
     * Returns the geometry of the texture mapping. Default is
     * equirectancular.
     * @param { number= } radius - The viewing horizon of the geometry.
     * @return { !THREE.Geometry } - The geometry of the texture mapping.
     */
    getCameraGeometry(radius = DEFAULT_RADIUS) {
      return new THREE.SphereGeometry(radius, 100, 100);
    }

    /**
     * Returns an object whose interior texture mapping is the panorama.
     * @param { string } node - A node in the graph.
     * @param { number= } radius - The viewing horizon of the geometry.
     * @return { !THREE.Object3D } - An object representing the node.
     */
    getCameraObject(node, radius = DEFAULT_RADIUS) {
      var geometry = this.getCameraGeometry(radius);
      var material = this.getCameraTextures(node).map(texture => {
        return new THREE.MeshBasicMaterial({

          map: texture,
          side: THREE.BackSide

        });
      });

      var object = new THREE.Mesh(geometry, material);
      object.scale.multiplyScalar(-1);
      object.applyMatrix(this.getCameraMatrix(node));

      return object;
    }

    /**
     * Converts from local to three.js coordinates.
     */
    localToWorld(object) {}

    /**
     * Returns a boolean or promise indicating initialization.
     * @return { boolean } - Whether or not the graph is initialized.
     */
    isInitialized() {
      return true;
    }
  };
}

(global => {
  if (typeof define === 'function' && define.amd) {
    define([], () => Graph);

  } else if (typeof module !== 'undefined' && typeof exports === 'object') {
    module.exports = Graph;

  } else if (global !== undefined) {
    global.Graph = Graph;
  }
})(this);
