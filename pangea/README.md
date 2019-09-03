### Pangea

Pangea is a lightweight platform for annotating grounded environmental actions
in photo-realistic environment simulators (e.g., Matterport3D). Pangea can be
adapted to your data crowd-sourcing service of choice (e.g., Amazon mechanical
turk).

#### Documentation

Pangea requires several third-party dependencies that can be installed using the
node package manager (npm), downloaded [here](https://nodejs.org/en/).

```
npm install
```

Bundle these dependencies using the browserfy command line tool that can also be
installed using the npm.

```
npm install -g browserify
browserify dependencies.js > dependencies.min.js
```

Create two local directories:

* `connectivity` contains the navigation graph, downloaded
[here](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
* `scans` contains the background textures, downloaded
[here](https://niessner.github.io/Matterport/).

Optionally this data can be served remotely by modifying `index.js` such that
`CONNDIR` and `SCANDIR` to point to a remote directory. Finally, start an local
http server.

```
python -m SimpleHTTPServer
```
