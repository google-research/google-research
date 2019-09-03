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
 * @fileoverview Pangea is a lightweight platform for annotating grounded
 * environmental actions in photo-realistic environment simulators
 * (e.g., Matterport3D). Pangea can be adapted to your data crowd-sourcing
 * service of choice (e.g., Amazon mechanical turk).
 */

var SCANDIR = './scans/';
var CONNDIR = './connectivity/';

/**
 * Texture loader for the Matterport3D simulator.
 * @param {string} scanId
 * @param {string} panoId
 * @return {!THREE.LoadingManager}
 */
function getTexture(scanId, panoId) {
  var loader = new THREE.CubeTextureLoader();
  loader.setCrossOrigin('use-credentials');
  return loader.load([
    `${SCANDIR}${scanId}/matterport_skybox_images/${panoId}_skybox2_sami.jpg`,
    `${SCANDIR}${scanId}/matterport_skybox_images/${panoId}_skybox4_sami.jpg`,
    `${SCANDIR}${scanId}/matterport_skybox_images/${panoId}_skybox0_sami.jpg`,
    `${SCANDIR}${scanId}/matterport_skybox_images/${panoId}_skybox5_sami.jpg`,
    `${SCANDIR}${scanId}/matterport_skybox_images/${panoId}_skybox1_sami.jpg`,
    `${SCANDIR}${scanId}/matterport_skybox_images/${panoId}_skybox3_sami.jpg`,
  ]);
}

/**
 * Constructs a node in the navigation graph. This node is a 3D object that can
 * be rendered onto the scene and contains information about the camera
 * orientation, visibility and navigability of other nodes in the graph. This
 * is a helper function that is called by getGraph().
 * @param {!Object} data
 * @param {number} ix
 * @return {!THREE.Object3D}
 */
function getNode(data, ix) {
  // Nodes are rendered onto the scene as cylinders that have been dropped to
  // ground level using the camera height which is provided by the simulator.
  var geom = new THREE.CylinderBufferGeometry(0.1, 0.1, 0.2, 128);
  var matt = new THREE.MeshPhongMaterial({color: 0xFFFFFF});
  var node = new THREE.Mesh(geom, matt);
  node.included = data[ix]['included'];
  node.height = data[ix]['height'];
  node.name = data[ix]['image_id'];
  node.visible = false;
  // The camera pose matrices provided by the Matterport3D simulator use a
  // different coordinate system than THREE.js. The following lines correct
  // for this discrepancy.
  var u = data[ix]['pose'].slice(0); u[11] -= node.height;
  var X = new THREE.Matrix4().fromArray(u).transpose();
  var F = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
  var G = new THREE.Matrix4().makeRotationY(Math.PI / 2);
  var Y = G.multiply(F.multiply(X));
  node.applyMatrix(Y);
  // The origin of the node is position of the camera corresponding to the
  // background texture. This is different from the position of the node
  // which has been dropped to ground level.
  node.origin = node.position.clone();
  node.origin.y += node.height;
  // The neighbors attribute contains the names of nodes adjacent to the current
  // node in the navigation graph and the visibility attribute contains the
  // names of nodes visible from the current node.
  node.neighbors = []; node.visibility = [];
  data.forEach((datum) => {
    var adj = datum['image_id'];
    if (datum['unobstructed'][ix]) node.neighbors.push(adj);
    if (datum['visible'][ix]) node.visibility.push(adj);
  });
  return node;
}

/**
 * Constructs a navigation graph from connectivity data.
 * @param {string} scanId
 * @return {!Promise<!THREE.Group>}
 */
function getGraph(scanId) {
  var loader = d3.json(
      `${CONNDIR}${scanId}_connectivity.json`, {credentials: "include"});
  return loader.then((data) => {
    var graph = new THREE.Group();
    for (var ix = 0; ix < data.length; ix++) {
      var node = getNode(data, ix);
      graph.add(node);
    }
    return graph;
  });
}

/**
 * Updates the visible property of nodes in the graph to reflect the viewpoint
 * imposed by the given panoId.
 * @param {!THREE.Group} graph
 * @param {string} panoId
 */
function setVisibility(graph, panoId) {
  graph.children.forEach((node) => {
    node.visible = node.included && node.visibility.includes(panoId);
  });
}

/**
 * Tweens the camera quaternion using spherical linear interpolation (slerp)
 * for smooth transition animation.
 * @param {!THREE.PerspectiveCamera} camera
 * @param {!THREE.Vector3} look
 * @param {number=} duration
 * @param {string=} ease
 * @return {!TWEEN.Tween}
 */
function slerp(camera, look, duration=0, ease="linear") {
  var src = new THREE.Quaternion();
  var dst = new THREE.Quaternion();
  src.copy(camera.quaternion);
  camera.lookAt(look);
  dst.copy(camera.quaternion);
  return new TWEEN.Tween({t: 0})
      .to({t: 1}, duration, ease)
      .onUpdate((obj) => {
        THREE.Quaternion.slerp(src, dst, camera.quaternion, obj.t);
      });
}

/**
 * Tweens the camera field of view to be a given percent of the original for
 * smooth transition animation. Use 0 < pct < 1 to contract the field of view
 * and pct > 1 to expand the field of view.
 * @param {!THREE.PerspectiveCamera} camera
 * @param {number} pct
 * @param {number=} duration
 * @param {string=} ease
 * @return {!TWEEN.Tween}
 */
function zoom(camera, pct, duration=0, ease="linear") {
  var fov = camera.fov * pct;
  return new TWEEN.Tween(camera)
      .to({fov}, duration, ease)
      .onUpdate(() => {
        camera.updateProjectionMatrix();
      });
}

/**
 * Returns the objects in the scene that the mouse intersects.
 * @param {!THREE.PerspectiveCamera} camera
 * @param {!THREE.Vector3} mouse
 * @param {!Array<!THREE.Object3D>} objects
 * @return {!Array<!THREE.Object3D>}
 */
function intersect(camera, mouse, objects) {
  var raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);
  var intersections = raycaster.intersectObjects(objects);
  return intersections.map((intersection) => intersection.object);
}

/**
 * The default control scheme in THREE.js is orbital. This is perceived by the
 * viewer as having the camera orbit a fixed point in space. For navigation we
 * need a first-person perspective. This can be achieved by positioning the
 * orbital target very close in the direction the camera is facing and updating
 * this position every time the camera direction changes. Call this function
 * right before the scene is rendered.
 * @param {!THREE.PerspectiveCamera} camera
 * @param {!THREE.OrbitControls} controls
 */
function setFirstPersonPerspective(camera, controls) {
  var look = new THREE.Vector3();
  camera.getWorldDirection(look);
  look.multiplyScalar(1e-5);
  controls.target.copy(camera.position);
  controls.target.add(look);
}

/**
 * This is the main function that starts the simulator. Generally you call
 * this function in your html file.
 */
function main() {

  // Variables used by THREE.js to manipulate the scene.
  var scene, renderer, camera, controls, mouse;
  // Variables that define your position in the navigation graph.
  var graphPromise, scanId, panoId;
  // Variable used to move between nodes of the graph.
  var intersected;

  // Initialize and animate the scene.
  init();
  animate();

  function init() {
    // Construct a new document container to add the WebGL canvas to.
    var div = document.createElement('div');
    document.body.appendChild(div);
    document.body.style.margin = 0;
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    div.appendChild(renderer.domElement);
    // Initialize the camera and controls with a 70% field of view.
    var asp = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(70, asp, 1, 10000);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableZoom = false;
    mouse = new THREE.Vector2();
    // Add directional and ambient lighting so that the nodes of the navigation
    // graph look natural in the scene.
    scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xAAAAAA));
    scene.add(new THREE.DirectionalLight(0xFFFFFF, 1));
    // Optionally provide url parameters to initialize the starting position
    // in the navigation graph. Defaults are given below.
    var url = new URL(window.location.href);
    scanId = url.searchParams.get('scanId') || '17DRP5sb8fy';
    panoId = (
        url.searchParams.get('panoId') || '5e9f4f8654574e699480e90ecdd150c8');
    // Render the background texture and visible nodes to the scene.
    scene.background = getTexture(scanId, panoId);
    graphPromise = getGraph(scanId);
    graphPromise.then((graph) => {
      scene.add(graph);
      setVisibility(graph, panoId);
      var node = graph.getObjectByName(panoId);
      camera.position.copy(node.origin);
    });
    // Add document listeners to interface with THREE.js.
    document.addEventListener('mousemove', onDocumentMouseMove, false);
    document.addEventListener('click', onDocumentClick, false);
    window.addEventListener('resize', onWindowResize, false);
    window.addEventListener('keydown', onKeyDown, false);
  }

  function onDocumentMouseMove(event) {
    // Update the mouse variable to reflect the mouse vector. This vector is
    // used to determine objects that are being intersected, which can be used
    // to navigate between nodes of the navigation graph or in the future
    // manimulate objects.
    event.preventDefault();
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  }

  function onDocumentClick(event) {
    // If the mouse clicks on an object in the scene, move to the node in the
    // navigation graph that the object corresponds to. This triggers a
    // transition animation.
    if (intersected) move(intersected.name);
  }

  function onWindowResize() {
    // Updates the camera aspect ratio to reflect the window size.
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function onKeyDown(event) {
    // This can be used as a keylogger to maintain a revision history of a
    // user annotation.
    console.log(event.keyCode);
  }

  function move(nextPanoId) {
    // Triggers an slerp and zoom animation that transitions the user between
    // nodes in the navigation graph.
    graphPromise.then((graph) => {
      var texture = getTexture(scanId, nextPanoId);
      var node = graph.getObjectByName(nextPanoId);
      var tween = slerp(camera, node.origin, 1000)
        .chain(zoom(camera, 0.8, 1000)
          .onComplete(() => {
            camera.fov /= 0.8;
            camera.updateProjectionMatrix();
            panoId = nextPanoId;
            scene.background = texture;
            camera.position.copy(node.origin);
            setVisibility(graph, panoId);
          }));
      tween.start();
    });
  }

  function render() {
    // Check if the mouse intersects any of the objects in the scene. If any
    // objects are intersected, render them in green.
    graphPromise.then((graph) => {
      intersects = intersect(camera, mouse, graph.children);
      if (intersects.length > 0) {
        if (intersected != intersects[0]) {
          if (intersected) intersected.material.color.setHex(0xFFFFFF);
          intersected = intersects[0];
          intersected.material.color.setHex(0x44aa88);
        }
      } else {
        if (intersected) intersected.material.color.setHex(0xFFFFFF);
        intersected = null;
      }
    });
    // Update the controls to maintain a first-person perspective.
    setFirstPersonPerspective(camera, controls);
    controls.update();
    TWEEN.update();
    renderer.render(scene, camera);
  }

  function animate() {
    requestAnimationFrame(animate);
    render();
  }

}
