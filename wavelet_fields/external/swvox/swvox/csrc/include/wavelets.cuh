#include "bior22_constants.cuh"
#include "db2_constants.cuh"

// 3D Haar wavelet. The constants where created with wavelets/haar_wavelet.py
__device__ __constant__ const float LLH[] = {
    -0.3535533845424652, 0.3535533845424652,  -0.3535533845424652,
    0.3535533845424652,  -0.3535533845424652, 0.3535533845424652,
    -0.3535533845424652, 0.3535533845424652};

__device__ __constant__ const float LHL[] = {
    -0.3535533845424652, -0.3535533845424652, 0.3535533845424652,
    0.3535533845424652,  -0.3535533845424652, -0.3535533845424652,
    0.3535533845424652,  0.3535533845424652};

__device__ __constant__ const float LHH[] = {
    0.3535533845424652,  -0.3535533845424652, -0.3535533845424652,
    0.3535533845424652,  0.3535533845424652,  -0.3535533845424652,
    -0.3535533845424652, 0.3535533845424652};

__device__ __constant__ const float HLL[] = {
    -0.3535533845424652, -0.3535533845424652, -0.3535533845424652,
    -0.3535533845424652, 0.3535533845424652,  0.3535533845424652,
    0.3535533845424652,  0.3535533845424652};

__device__ __constant__ const float HLH[] = {
    0.3535533845424652,  -0.3535533845424652, 0.3535533845424652,
    -0.3535533845424652, -0.3535533845424652, 0.3535533845424652,
    -0.3535533845424652, 0.3535533845424652};

__device__ __constant__ const float HHL[] = {
    0.3535533845424652,  0.3535533845424652,  -0.3535533845424652,
    -0.3535533845424652, -0.3535533845424652, -0.3535533845424652,
    0.3535533845424652,  0.3535533845424652};

__device__ __constant__ const float HHH[] = {
    -0.3535533845424652, 0.3535533845424652, 0.3535533845424652,
    -0.3535533845424652, 0.3535533845424652, -0.3535533845424652,
    -0.3535533845424652, 0.3535533845424652};

// Position should be 3D vector normalized from 0 to 1. returns the 7
// coefficients of the haar wavelet at that position
template <typename scalar_t>
__device__ __inline__ void evaluate_haar(const scalar_t* __restrict__ pos,
                                         scalar_t* __restrict__ out,
                                         const bool add = false) {
  int round_x = (int)(pos[0] * 2);
  int round_y = (int)(pos[1] * 2);
  int round_z = (int)(pos[2] * 2);

  if (round_x > 1) {
    printf("problem when evaluating haar as pos_x is %f, and should be {0,1}\n",
           pos[0]);
    return;
  }
  if (round_y > 1) {
    printf("problem when evaluating haar as pos_y is %f, and should be {0,1}\n",
           pos[1]);
    return;
  }
  if (round_z > 1) {
    printf("problem when evaluating haar as pos_z is %f, and should be {0,1}\n",
           pos[2]);
    return;
  }

  if (add) {
    out[0] += LLH[round_x * 4 + round_y * 2 + round_z];
    out[1] += LHL[round_x * 4 + round_y * 2 + round_z];
    out[2] += LHH[round_x * 4 + round_y * 2 + round_z];
    out[3] += HLL[round_x * 4 + round_y * 2 + round_z];
    out[4] += HLH[round_x * 4 + round_y * 2 + round_z];
    out[5] += HHL[round_x * 4 + round_y * 2 + round_z];
    out[6] += HHH[round_x * 4 + round_y * 2 + round_z];

  } else {
    out[0] = LLH[round_x * 4 + round_y * 2 + round_z];
    out[1] = LHL[round_x * 4 + round_y * 2 + round_z];
    out[2] = LHH[round_x * 4 + round_y * 2 + round_z];
    out[3] = HLL[round_x * 4 + round_y * 2 + round_z];
    out[4] = HLH[round_x * 4 + round_y * 2 + round_z];
    out[5] = HHL[round_x * 4 + round_y * 2 + round_z];
    out[6] = HHH[round_x * 4 + round_y * 2 + round_z];
  }
}

// 3D trilinear "wavelet""

// Position should be 3D vector normalized from 0 to 1. Returns the 8 "wavelet"
// evaluations of a trilinear interpolation inside a unit size cube Eeach
// coefficient that multiplies the corresponding to a corner of the cube
template <typename scalar_t>
__device__ __inline__ void evaluate_trilinear(const scalar_t* __restrict__ pos,
                                              scalar_t* __restrict__ out,
                                              const bool add = false) {
  if (add) {
    out[0] += (1.f - pos[0]) * (1.f - pos[1]) * (1.f - pos[2]);
    out[1] += (1.f - pos[0]) * (1.f - pos[1]) * (0.f + pos[2]);
    out[2] += (1.f - pos[0]) * (0.f + pos[1]) * (1.f - pos[2]);
    out[3] += (1.f - pos[0]) * (0.f + pos[1]) * (0.f + pos[2]);
    out[4] += (0.f + pos[0]) * (1.f - pos[1]) * (1.f - pos[2]);
    out[5] += (0.f + pos[0]) * (1.f - pos[1]) * (0.f + pos[2]);
    out[6] += (0.f + pos[0]) * (0.f + pos[1]) * (1.f - pos[2]);
    out[7] += (0.f + pos[0]) * (0.f + pos[1]) * (0.f + pos[2]);
  } else {
    // The same as above, but replacing the values
    out[0] = (1.f - pos[0]) * (1.f - pos[1]) * (1.f - pos[2]);
    out[1] = (1.f - pos[0]) * (1.f - pos[1]) * (0.f + pos[2]);
    out[2] = (1.f - pos[0]) * (0.f + pos[1]) * (1.f - pos[2]);
    out[3] = (1.f - pos[0]) * (0.f + pos[1]) * (0.f + pos[2]);

    out[4] = (0.f + pos[0]) * (1.f - pos[1]) * (1.f - pos[2]);

    out[5] = (0.f + pos[0]) * (1.f - pos[1]) * (0.f + pos[2]);
    out[6] = (0.f + pos[0]) * (0.f + pos[1]) * (1.f - pos[2]);
    out[7] = (0.f + pos[0]) * (0.f + pos[1]) * (0.f + pos[2]);
  }
}

// Position should be 3D vector normalized from 0 to 1. returns the 8
// coefficients of the db2 wavelet at that position
template <typename scalar_t>
__device__ __inline__ void evaluate_db2(const scalar_t* __restrict__ pos,
                                        scalar_t* __restrict__ out,
                                        const bool add = false) {
  // dimensionality of each axis, filter size is dim ** 3
  // needs to be changed per filter
  int dim = 4;

  int round_x = (int)(pos[0] * dim);
  int round_y = (int)(pos[1] * dim);
  int round_z = (int)(pos[2] * dim);

  if (round_x >= dim) {
    printf("problem when evaluating haar as pos_x is %f, and should be {0,1}\n",
           pos[0]);
    return;
  }
  if (round_y >= dim) {
    printf("problem when evaluating haar as pos_y is %f, and should be {0,1}\n",
           pos[1]);
    return;
  }
  if (round_z >= dim) {
    printf("problem when evaluating haar as pos_z is %f, and should be {0,1}\n",
           pos[2]);
    return;
  }

  if (add) {
    out[0] += db2LLL[round_x * dim * dim + round_y * dim + round_z];
    out[1] += db2LLH[round_x * dim * dim + round_y * dim + round_z];
    out[2] += db2LHL[round_x * dim * dim + round_y * dim + round_z];
    out[3] += db2LHH[round_x * dim * dim + round_y * dim + round_z];
    out[4] += db2HLL[round_x * dim * dim + round_y * dim + round_z];
    out[5] += db2HLH[round_x * dim * dim + round_y * dim + round_z];
    out[6] += db2HHL[round_x * dim * dim + round_y * dim + round_z];
    out[7] += db2HHH[round_x * dim * dim + round_y * dim + round_z];

  } else {
    out[0] = db2LLL[round_x * dim * dim + round_y * dim + round_z];
    out[1] = db2LLH[round_x * dim * dim + round_y * dim + round_z];
    out[2] = db2LHL[round_x * dim * dim + round_y * dim + round_z];
    out[3] = db2LHH[round_x * dim * dim + round_y * dim + round_z];
    out[4] = db2HLL[round_x * dim * dim + round_y * dim + round_z];
    out[5] = db2HLH[round_x * dim * dim + round_y * dim + round_z];
    out[6] = db2HHL[round_x * dim * dim + round_y * dim + round_z];
    out[7] = db2HHH[round_x * dim * dim + round_y * dim + round_z];
  }
}

// Position should be 3D vector normalized from 0 to 1. returns the 8
// coefficients of the db2 wavelet at that position
template <typename scalar_t>
__device__ __inline__ void evaluate_bior22(const scalar_t* __restrict__ pos,
                                           scalar_t* __restrict__ out,
                                           const bool add = false) {
  int dim = 6;

  int round_x = (int)(pos[0] * dim);
  int round_y = (int)(pos[1] * dim);
  int round_z = (int)(pos[2] * dim);

  if (round_x >= dim) {
    printf(
        "problem when evaluating bior22 as pos_x is %f, and should be {0,1}\n",
        pos[0]);
    return;
  }
  if (round_y >= dim) {
    printf(
        "problem when evaluating bior22 as pos_y is %f, and should be {0,1}\n",
        pos[1]);
    return;
  }
  if (round_z >= dim) {
    printf(
        "problem when evaluating bior22 as pos_z is %f, and should be {0,1}\n",
        pos[2]);
    return;
  }

  if (add) {
    out[0] += bior22LLL[round_x * dim * dim + round_y * dim + round_z];
    out[1] += bior22LLH[round_x * dim * dim + round_y * dim + round_z];
    out[2] += bior22LHL[round_x * dim * dim + round_y * dim + round_z];
    out[3] += bior22LHH[round_x * dim * dim + round_y * dim + round_z];
    out[4] += bior22HLL[round_x * dim * dim + round_y * dim + round_z];
    out[5] += bior22HLH[round_x * dim * dim + round_y * dim + round_z];
    out[6] += bior22HHL[round_x * dim * dim + round_y * dim + round_z];
    out[7] += bior22HHH[round_x * dim * dim + round_y * dim + round_z];

  } else {
    out[0] = bior22LLL[round_x * dim * dim + round_y * dim + round_z];
    out[1] = bior22LLH[round_x * dim * dim + round_y * dim + round_z];
    out[2] = bior22LHL[round_x * dim * dim + round_y * dim + round_z];
    out[3] = bior22LHH[round_x * dim * dim + round_y * dim + round_z];
    out[4] = bior22HLL[round_x * dim * dim + round_y * dim + round_z];
    out[5] = bior22HLH[round_x * dim * dim + round_y * dim + round_z];
    out[6] = bior22HHL[round_x * dim * dim + round_y * dim + round_z];
    out[7] = bior22HHH[round_x * dim * dim + round_y * dim + round_z];
  }
}

// For visualization purposes, puts 1's on the corners of the voxels, else 0.
// It only has one coefficient
const float C = 0.05f;

template <typename scalar_t>
__device__ __inline__ void evaluate_side(const scalar_t* __restrict__ pos,
                                         scalar_t* __restrict__ out,
                                         const bool add = false) {
  scalar_t val = 0;

  // Check each edge along the three axes
  for (int i = 0; i < 3; ++i) {
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    // Check edges where one coordinate is 0 or 1, and the other coordinates
    // vary
    for (scalar_t a : {0.0f, 1.0f}) {
      for (scalar_t b : {0.0f, 1.0f}) {
        scalar_t d = (pos[i] - a) * (pos[i] - a) +
                     (pos[j] - pos[j]) * (pos[j] - pos[j]) +
                     (pos[k] - b) * (pos[k] - b);
        if (sqrt(d) <= C) {
          val = 1;
          break;
        }
      }
      if (val) break;
    }
    if (val) break;
  }

  // Add or replace the value depending on the 'add' parameter
  if (add) {
    out[0] += val;
    out[1] += 1 - val;
  } else {
    out[0] = val;
    out[1] = 1 - val;
  }
}
