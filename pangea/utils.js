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
 * @fileoverview Matterport utils.
 */


class Matterport {

  /**
   * Constructor for the Matterport class.
   * @param {string} directory - The directory containing simulator data.
   */
  constructor(directory) {

    var scope = this;

    // Dependencies.

    if (typeof THREE == 'undefined') {
      console.error('ERROR: matterport requires THREE');
    }

    if (typeof d3 == 'undefined') {
      console.error('ERROR: matterport requires d3');
    }

    scope.directory = directory;
    scope.cd = `${scope.directory}connectivity/`;
    scope.sd = `${scope.directory}scans/`;

    scope.format = 'matterport_skybox_images';

    // Matterport uses a z-up coordinate system.

    scope.up = new THREE.Vector3(0, 0, 1);

  }

  /**
   * Promises a graph.
   * @param {string} scan - An identifier for a graph.
   * @return {!Promise} graph - Promises a navigation graph.
   */
  graph(scan) {

    var scope = this;

    var url = `${scope.cd}${scan}_connectivity.json`;
    var promise = d3.json(url, {credentials: 'include'});

    return promise.then(function(data) {

      var graph = {};

      for (var i = 0; i < data.length; i++) {

        var node = {};

        var key = data[i]['image_id'];
        node.height = data[i]['height'];
        node.included = data[i]['included'];

        var scale = new THREE.Matrix4();
        scale.makeScale(1, -1, -1);

        node.matrix = new THREE.Matrix4();
        node.matrix.fromArray(data[i]['pose']);
        node.matrix.transpose();
        node.matrix.multiply(scale);

        node.inverse = new THREE.Matrix4();
        node.inverse.getInverse(node.matrix);

        node.visibility = [];
        node.neighbors = [];

        for (var j = 0; j < data.length; j++) {

          var adjacent = data[j]['image_id'];

          if (data[i]['visible'][j]) {
            node.visibility.push(adjacent);
          }

          if (data[i]['unobstructed'][j]) {
            node.neighbors.push(adjacent);
          }

        }

        graph[key] = node;

      }

      return graph;

    });


  }

  /**
   * Promises a texture.
   * @param {string} scan - An identifier for a graph.
   * @param {string} pano - A node in the graph.
   * @return {!THREE.Texture} - The scene background texture.
   */
  texture(scan, pano) {

    var scope = this;

    var loader = new THREE.CubeTextureLoader();
    loader.setCrossOrigin('use-credentials');

    return loader.load([
      `${scope.sd}${scan}/${scope.format}/${pano}_skybox2_sami.jpg`,
      `${scope.sd}${scan}/${scope.format}/${pano}_skybox4_sami.jpg`,
      `${scope.sd}${scan}/${scope.format}/${pano}_skybox0_sami.jpg`,
      `${scope.sd}${scan}/${scope.format}/${pano}_skybox5_sami.jpg`,
      `${scope.sd}${scan}/${scope.format}/${pano}_skybox1_sami.jpg`,
      `${scope.sd}${scan}/${scope.format}/${pano}_skybox3_sami.jpg`,
    ]);

  }

}


(function(root) {

  if (typeof define === 'function' && define.amd) {

    define([], function() {

      return Matterport;

    });

  } else if (typeof module !== 'undefined'
             && typeof exports === 'object') {

    module.exports = Matterport;

  } else if (root !== undefined) {

    root.Matterport = Matterport;

  }

})(this);
