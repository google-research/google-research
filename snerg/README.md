# Sparse Neural Radiance Grids

This directory contains the JavaScript viewer code for the Sparse Neural
Radiance Grid representation (SNeRG), from the paper:
"Baking Neural Radiance Fields for Real-Time View Synthesis"
Hedman et al. 2021

*Please note that this is not an officially supported Google product.*

The SNeRG representation enables rendering of scenes reconstructed as Neural
Radiance Fields (NeRF, Mildenhall et al. 2020) in real-time. SNeRG uses a sparse
voxel grid representation to store precomputed scene geometry, but keeps
storage requirements reasonable by representing view-dependent appearance
using a neural network.

Rendering is accelerated by evaluating the view-dependent shading network only
on visible parts of the scene, achieving over 30 frames per second on a laptop
GPU for typical NeRF scenes.

You can run this viewer code by uploading it to your own web-server and pointing
it to a SNeRG directory, e.g.
http://my.web.server.com/snerg/index.html?dir=scene_directory_on_server

This code repository is shared with all of Google Research, so it's not very
useful for reporting or tracking bugs. If you have any issues using this code,
please do not open an issue, and instead just email peter.j.hedman@gmail.com.

