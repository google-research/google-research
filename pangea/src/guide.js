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
 * @fileoverview Navigation annotation plugin (i.e., guide task).
 */

{
  // The envlogger sampling frequency. The minimum time between entries.
  const MIN_TIME_DELTA = 100;  // Milliseconds.

  // The data collection API (e.g., Amazon Mechanical Turk).
  const api = new API();

  document.addEventListener('DOMContentLoaded', () => {
    api.init(async args => {
      var tmp = {};
      args.attributes.forEach(a => tmp[a.key] = a.value);

      // Select the path to be annotated (provided by args).
      var scanID = tmp.scan;
      var path = tmp.path.split(',');
      var directory = tmp.directory;

      // Initialize the three.js environment.
      var graph = new Matterport3D(scanID, directory);
      await graph.isInitialized();
      var environment = new Environment(graph);

      // Initialize the environment to the first node in the path.
      environment.initialize(path[0]);

      // Initialize the environment to a random heading angle.
      var azimuth = Math.random() * 2 * Math.PI;
      environment.camera.rotation.y = azimuth;

      // Add lighting to the scene.
      var light = new THREE.PointLight(0xffffff, 1);
      light.position.set(10, 50, -20);
      environment.scene.add(light);
      environment.scene.add(new THREE.AmbientLight(0x888888));

      // Add markers to indicate nodes in the path. These markers are
      // graded in color from yellow to green.
      var yellow = new THREE.Color(0xaa8844);
      var green = new THREE.Color(0x44aa88);
      var objects = path.map((node, i) => {
        var geometry = new THREE.CylinderGeometry(0.1, 0.1, 0.1, 4);
        geometry.vertices.forEach(vertex => {
          vertex.x += Math.random() * 0.1;
          vertex.y += Math.random() * 0.1;
          vertex.z += Math.random() * 0.1;
        });

        var color = yellow.clone().lerp(green, i / path.length);
        var material = new THREE.MeshPhongMaterial({

          color,
          transparent: true,
          opacity: 0.9,
          flatShading: true,

        });

        var object = new THREE.Mesh(geometry, material);
        object.applyMatrix(graph.getCameraMatrix(node));
        object.position.z -= graph.getCameraHeight(node);
        environment.addDynamicGraphObject(object);
        return object;
      });

      // Render the scene to fill the entire frame.
      document.body.appendChild(environment.canvas);
      environment.setRendererSize(window.innerWidth, window.innerHeight);
      window.addEventListener('resize', () => {
        environment.setRendererSize(window.innerWidth, window.innerHeight);
      });

      // Add a envlogger for the environment.
      var envlogger = {times: [], states: []};
      envlogger.times.push(performance.now());
      envlogger.states.push(environment.getState());
      var animate = t => {
        requestAnimationFrame(animate);
        environment.updateAndRender();

        var lastState = envlogger.states[envlogger.states.length - 1];
        var lastTime = envlogger.times[envlogger.times.length - 1];
        var timeDelta = performance.now() - lastTime;
        if (

            timeDelta > MIN_TIME_DELTA && !environment.isSameState(lastState)

        ) {
          envlogger.times.push(performance.now());
          envlogger.states.push(environment.getState());
        }

        if (!initialized) {
          initialized = true;
          environment._cameraObjects.children.forEach(

              cameraObject => {
                cameraObject.material.forEach(material => {
                  if (material.map.image == undefined) {
                    initialized = false;
                  }
                });
              });
        }

        // Spin the markers for better visibility.
        objects.forEach((object, i) => {
          var speed = 1 + i * 0.1;
          var rot = t * speed * 0.001;
          object.rotation.x = rot;
          object.rotation.y = rot;
          object.rotation.z = rot;
        });
      };
      requestAnimationFrame(animate);

      // DOM elements.
      var input = document.getElementById('input');
      var blocker = document.getElementById('blocker');
      var instructions = document.getElementById('instructions');
      var blockerButton = document.getElementById('blocker-button');

      var initialized = false;  // If environment textures are loaded.
      instructions.addEventListener('click', () => {
        if (initialized) {
          instructions.style.display = 'none';
          blocker.style.display = 'none';
          input.focus();
        }
      });

      // Add a keylogger to track annotation keystrokes.
      var keylogger = {times: [], states: []};
      input.addEventListener('keyup', e => {
        keylogger.times.push(performance.now());
        keylogger.states.push({code: e.code, value: input.value});
      });

      // Add a escape key listener and control + enter submit.
      document.addEventListener('keydown', e => {
        if (e.keyCode == 27) {  // Escape key.

          blocker.style.display = 'block';
          instructions.style.display = '';
          input.blur();
        }

        if (e.ctrlKey && e.keyCode === 13) {  // Control + enter.

          api.submitAnswer({azimuth, keylogger, envlogger});
        }
      });

      // Add a button to show the instructions.
      blockerButton.addEventListener('click', () => {
        blocker.style.display = 'block';
        instructions.style.display = '';
        input.blur();
      });
    });
  });
}
