
#include <math.h>
#include <torch/extension.h>

#include <array>
#include <cstring>
#include <iostream>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INT32(x)                                \
  TORCH_CHECK(x.type().scalarType() == torch::kInt32, \
              #x " must be an int32 tensor")
#define CHECK_FLOAT32(x)                                \
  TORCH_CHECK(x.type().scalarType() == torch::kFloat32, \
              #x " must be a float32 tensor")
#define CHECK_UINT8(x)                                \
  TORCH_CHECK(x.type().scalarType() == torch::kUInt8, \
              #x " must be a uint8 tensor")

#define CHECK_CUDA_OK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_CUDA_STILL_OK() \
  { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %i %s %s %d\n", code, cudaGetErrorString(code),
            file, line);
    if (abort) exit(code);
  }
}

struct TriplaneCoordinates {
  int voxel_idx_x;
  int voxel_idx_y;
  int voxel_idx_z;
  float xy_plane_x;
  float xy_plane_y;
  float xz_plane_x;
  float xz_plane_y;
  float yz_plane_x;
  float yz_plane_y;
};

struct TriplaneGridMetadata {
  float grid_lower_corner_x;
  float grid_lower_corner_y;
  float grid_lower_corner_z;
  float voxel_size_x;
  float voxel_size_y;
  float voxel_size_z;
  int grid_res_x;
  int grid_res_y;
  int grid_res_z;
  int triplane_pixel_height;
  int triplane_pixel_width;

  __device__ __host__ TriplaneCoordinates
  SplatPoint(const float* world_position) const {
    // First translate so [0, 0, 0] is the lower corner of the grid:
    float pt[3];
    pt[0] = world_position[0] - grid_lower_corner_x;
    pt[1] = world_position[1] - grid_lower_corner_y;
    pt[2] = world_position[2] - grid_lower_corner_z;

    // Now rescale so we are in voxel coordinates:
    pt[0] /= voxel_size_x;
    pt[1] /= voxel_size_y;
    pt[2] /= voxel_size_z;
    TriplaneCoordinates out;
    out.voxel_idx_x = static_cast<int>(floor(pt[0]));
    out.voxel_idx_y = static_cast<int>(floor(pt[1]));
    out.voxel_idx_z = static_cast<int>(floor(pt[2]));

    // Now that we know which voxel we are in, let's figure out where
    // we are in that voxel:
    pt[0] -= floor(pt[0]);
    pt[1] -= floor(pt[1]);
    pt[2] -= floor(pt[2]);
    // Now we should range between [0, 1) where the value is the fraction
    // of the voxel the point is at. So let's just rescale to [0, res x/y),
    // and account for the convention that 0,0 is the top left of the image.
    // Important note: The ab plane is assumed to be the plane where
    // a is columns and b is rows (e.g., xy -> xy, not yx).
    // xy plane: pos x is right,  pos y is up:
    out.xy_plane_x = pt[0] * triplane_pixel_width;
    // Note that the flipping should really happen last but it is equivalent
    // Also, we have one minor issue: if the value was exactly 0.0 it will
    // now be exactly triplane_pixel_height, which is bad, because floor-ing
    // that will give you the next triplane. Even though it is extremely
    // unlikely we need to bump it under 1.0
    out.xy_plane_y =
        (1.0f - pt[1]) *
        triplane_pixel_height;  // Flip so y grows from top of image
    out.xy_plane_y = std::min(out.xy_plane_y, triplane_pixel_height - 1e-5f);
    // xz plane: pos x is right, pos z is up:
    out.xz_plane_x = pt[0] * triplane_pixel_width;
    out.xz_plane_y = (1.0f - pt[2]) * triplane_pixel_height;  // Flip as above
    out.xz_plane_y = std::min(out.xz_plane_y, triplane_pixel_height - 1e-5f);
    // yz plane: pos y is right, pos z is up:
    out.yz_plane_x = pt[1] * triplane_pixel_width;
    out.yz_plane_y = (1.0f - pt[2]) * triplane_pixel_height;  // Flip as above
    out.yz_plane_y = std::min(out.yz_plane_y, triplane_pixel_height - 1e-5f);
    // Final important note: we use the opengl style convention that image
    // coords are [0.0, 0.0] for the TOP LEFT of the image, and increase down
    // and to the right. Further, the storage order is set such that the top
    // left pixel is at index [0, 0]  and storage is row-major. So, pixel (x,y)
    // ends up belonging at index ri=int(floor(y)) and ci=int(floor(x)) which is
    // at location ri * width + ci.
    return out;
  }

  __host__ __device__ int NVoxels() const {
    return grid_res_z * grid_res_y * grid_res_x;
  }

  __host__ __device__ int NPixelsPerImage() const {
    return triplane_pixel_height * triplane_pixel_width;
  }

  __host__ __device__ int N2DImages() const {
    return NVoxels() * 3;  // 3 images per triplane, one triplane per voxel.
  }

  __host__ __device__ int64_t NPixelsTotal() const {
    int64_t n = N2DImages();
    return n * NPixelsPerImage();
  }

  __host__ __device__ int64_t NBytesRequired() const {
    int64_t n = NPixelsTotal();
    return n * 4;  // Bytes per int
  }

  __host__ __device__ int VoxelValStride() const {
    return 3 * NPixelsPerImage();
  }

  __host__ __device__ bool VoxelIdxInBounds(int vxi, int vyi, int vzi) const {
    return vxi >= 0 && vxi < grid_res_x && vyi >= 0 && vyi < grid_res_y &&
           vzi >= 0 && vzi < grid_res_z;
  }

  // Important: xi,yi,zi input!
  __host__ __device__ int FlattenVoxelIndex(int vxi, int vyi, int vzi) const {
    return vzi * (grid_res_y * grid_res_x) + vyi * grid_res_x + vxi;
  }

  __host__ __device__ int PixelCoordsToIdx(float x, float y) const {
    return static_cast<int>(floor(y)) * triplane_pixel_width +
           static_cast<int>(floor(x));
  }
};

__device__ void SetRed(uint8_t* pixel) {
  pixel[0] = 255;
  pixel[1] = 0;
  pixel[2] = 0;
}
__device__ void SetGreen(uint8_t* pixel) {
  pixel[0] = 0;
  pixel[1] = 255;
  pixel[2] = 0;
}
__device__ void SetBlue(uint8_t* pixel) {
  pixel[0] = 0;
  pixel[1] = 0;
  pixel[2] = 255;
}
__device__ void SetYellow(uint8_t* pixel) {
  pixel[0] = 255;
  pixel[1] = 255;
  pixel[2] = 0;
}
__device__ void SetMagenta(uint8_t* pixel) {
  pixel[0] = 255;
  pixel[1] = 0;
  pixel[2] = 255;
}
__device__ void SetCyan(uint8_t* pixel) {
  pixel[0] = 0;
  pixel[1] = 255;
  pixel[2] = 255;
}
__device__ void SetCoral(uint8_t* pixel) {
  pixel[0] = 255;
  pixel[1] = 128;
  pixel[2] = 128;
}
__device__ void SetBlack(uint8_t* pixel) {
  pixel[0] = 0;
  pixel[1] = 0;
  pixel[2] = 0;
}
__device__ void SetColor(uint8_t* pixel, float r, float g, float b) {
  uint8_t r8 = min(1.0, max(0.0, r)) * 255;
  uint8_t g8 = min(1.0, max(0.0, g)) * 255;
  uint8_t b8 = min(1.0, max(0.0, b)) * 255;
  pixel[0] = r8;
  pixel[1] = g8;
  pixel[2] = b8;
}

__device__ void Mat4Vec4Mul(const float* m, float* v, float* out) {
  // Matrix is stored row-major:
  int ri = 0;
  for (int ri = 0; ri < 4; ++ri) {
    out[ri] = 0.0f;
    for (int ci = 0; ci < 4; ++ci) {
      out[ri] += v[ci] * m[ri * 4 + ci];
    }
  }
}

__device__ void Mat3Vec3Mul(const float* m, float* v, float* out) {
  // Matrix is stored row-major:
  int ri = 0;
  for (int ri = 0; ri < 3; ++ri) {
    out[ri] = 0.0f;
    for (int ci = 0; ci < 3; ++ci) {
      out[ri] += v[ci] * m[ri * 3 + ci];
    }
  }
}

__device__ void NormalizeVec3(float* v, float* out) {
  float norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  float norm = sqrt(norm_sq + 1e-8f);
  out[0] = v[0] / norm;
  out[1] = v[1] / norm;
  out[2] = v[2] / norm;
}

__device__ float AngleWithZAxis(float* pos_cam) {
  // Note that in nuScenes camera space is +z axis. So we assume towards in cam
  // space is +z. If that is not true then this code needs to change:
  float cos_sim = pos_cam[2];
  float angle = acos(cos_sim);  // We could just use cos sim...
  return angle;
}

__global__ void Images2TriplanesKernel(
    const float* points, int n_points, const uint8_t* images, int n_images,
    int im_height, int im_width, const float* extrinsics,
    const float* intrinsics, const float* grid_lower_corner,
    const float* voxel_size, const int* grid_res, const int* triplane_res,
    uint8_t* triplane_grid_out, int64_t n_pixels_out) {
  // Algorithm is simple: initialize everything to invalid, then loop over all
  // points, splatting them onto the grid.
  // Like scatter, in the case of collisions some valid point will end up in
  // the result, but there are no guarantees about which point it will be.

  int64_t block_idx = gridDim.x * blockIdx.y + blockIdx.x;
  int64_t thread_idx = threadIdx.x;
  int64_t pixel_idx = block_idx * blockDim.x + thread_idx;
  // TODO(kgenova) We can reduce this work by a factor of three... for each
  // of the planes in the triplane, the center of the pixel should be
  // at exactly the same 3D location (right?). Thus, we can do the
  // math once and set 3 RGB values for each 3D point.
  if (pixel_idx >= n_pixels_out) return;  // dangling threads in last warp
  uint8_t* pixel_out = triplane_grid_out + 3 * pixel_idx + 0;

  // Step 0) Get the indices for the pixel in question.
  // The pixel unpacks to [dz, dy, dx, 3, th, tw]. Then the
  // triplane also stores 3 values per pixel.
  int plane_height = triplane_res[0];
  int plane_width = triplane_res[1];
  int grid_res_x = grid_res[0];
  int grid_res_y = grid_res[1];
  int grid_res_z = grid_res[2];
  // TODO(kgenova) Confirm this matches the training code convention.
  int voxel_size_x = voxel_size[0];
  int voxel_size_y = voxel_size[1];
  int voxel_size_z = voxel_size[2];
  float grid_lower_corner_x = grid_lower_corner[0];
  float grid_lower_corner_y = grid_lower_corner[1];
  float grid_lower_corner_z = grid_lower_corner[2];
  //
  int64_t pixels_per_triplane = 3 * plane_height * plane_width;
  int64_t triplanes_per_grid = grid_res_x * grid_res_y * grid_res_z;
  int64_t pixels_per_grid = triplanes_per_grid * pixels_per_triplane;
  if (n_pixels_out != pixels_per_grid) {
    SetRed(pixel_out);
    return;
  }

  int voxel_idx = pixel_idx / pixels_per_triplane;
  int triplane_pixel_idx = pixel_idx % pixels_per_triplane;

  int vxi = voxel_idx % grid_res_x;
  int vzi = voxel_idx / (grid_res_y * grid_res_x);
  int vyi = (voxel_idx % (grid_res_y * grid_res_x)) / grid_res_x;

  int plane_idx = triplane_pixel_idx / (plane_height * plane_width);
  int plane_ci = triplane_pixel_idx % plane_width;
  int plane_ri =
      (triplane_pixel_idx % (plane_height * plane_width)) / plane_width;

  // Step 1) Get the 3D point of the center of this pixel.

  // The 3D point can be imagined as if there were one high-res triplane of res
  // n_voxels_per_dim * res_along_axis. Then just get the pixel coordinate for
  // that.

  // TODO(kgenova) Support non-square triplanes:
  // Just assume for now:
  if (plane_height != plane_width) {
    SetRed(pixel_out);
    return;
  }
  int plane_res = plane_height;

  // This would be plane-dependent (ugh, ignore that case for now)
  int overall_res_x = grid_res_x * plane_res;
  int overall_res_y = grid_res_y * plane_res;
  int overall_res_z = grid_res_z * plane_res;

  int overall_triplane_base_xi = plane_res * vxi;
  int overall_triplane_base_yi = plane_res * vyi;
  int overall_triplane_base_zi = plane_res * vzi;

  int overall_xi = overall_triplane_base_xi;
  int overall_yi = overall_triplane_base_yi;
  int overall_zi = overall_triplane_base_zi;
  if (plane_idx == 0) {  // xy plane:
    // pos x is right, pos y is up:
    overall_xi += plane_ci;
    overall_yi += plane_res - 1 - plane_ri;  // flip due to image ud flip
  } else if (plane_idx == 1) {               // xz plane:
    // pos x is right, pos z is up:
    overall_xi += plane_ci;
    overall_zi += plane_res - 1 - plane_ri;  // flip due to image ud flip
  } else if (plane_idx == 2) {               // yz plane:
    // pos y is right, pos z is up:
    overall_yi += plane_ci;
    overall_zi += plane_res - 1 - plane_ri;  // flip due to image ud flip
  }

  float overall_coords_x = static_cast<float>(overall_xi) + 0.5f;
  float overall_coords_y = static_cast<float>(overall_yi) + 0.5f;
  float overall_coords_z = static_cast<float>(overall_zi) + 0.5f;

  float overall_grid_size_x = voxel_size_x * grid_res_x;
  float overall_grid_size_y = voxel_size_y * grid_res_y;
  float overall_grid_size_z = voxel_size_z * grid_res_z;

  // Finally we have the x/y/z position in world space:
  float x_global = (overall_coords_x / overall_res_x) * overall_grid_size_x +
                   grid_lower_corner_x;
  float y_global = (overall_coords_y / overall_res_y) * overall_grid_size_y +
                   grid_lower_corner_y;
  float z_global = (overall_coords_z / overall_res_z) * overall_grid_size_z +
                   grid_lower_corner_z;

  if (false) {  // Visualize coordinates:
    SetColor(pixel_out, overall_coords_x / overall_res_x,
             overall_coords_y / overall_res_y,
             overall_coords_z / overall_res_z);
    return;
  }

  float pos_global[4];
  pos_global[0] = x_global;
  pos_global[1] = y_global;
  pos_global[2] = z_global;
  pos_global[3] = 1.0f;

  // Step 2) For each of the images, compare the towards vector to the vector
  //           from the center of projection to the 3D point. Pick the image
  //           with the best angle.
  int chosen_cam_idx = -1;
  float max_cos_sim = -2.0f;
  for (int cam_idx = 0; cam_idx < 6; ++cam_idx) {
    float pos_cam[4];
    const float* world2cam = &extrinsics[cam_idx * 4 * 4];
    Mat4Vec4Mul(world2cam, pos_global, pos_cam);
    // This is a shorthand for computing which camera has the best
    // angle viewing this point. That is because we assume in nuscenes
    // that the +z axis is the towards vector. If that's not true for
    // another dataset, this code has to change to handle arbitrary
    // towards vectors:
    float pos_cam_normalized[3];
    NormalizeVec3(pos_cam, pos_cam_normalized);
    if (false) {  // Visualize normalized position:
      SetColor(pixel_out, pos_cam_normalized[0], pos_cam_normalized[1],
               pos_cam_normalized[2]);
      return;
    }
    float cos_sim_with_z_axis = pos_cam_normalized[2];
    if (cos_sim_with_z_axis > max_cos_sim) {
      chosen_cam_idx = cam_idx;
      max_cos_sim = cos_sim_with_z_axis;
    }
  }

  // Shouldn't happen:
  if (chosen_cam_idx == -1) {
    SetBlack(pixel_out);
    return;
  }
  if (false) {  // Visualize chosen camera:
    if (chosen_cam_idx == 0) {
      SetRed(pixel_out);
      return;
    }
    if (chosen_cam_idx == 1) {
      SetGreen(pixel_out);
      return;
    }
    if (chosen_cam_idx == 2) {
      SetBlue(pixel_out);
      return;
    }
    if (chosen_cam_idx == 3) {
      SetCyan(pixel_out);
      return;
    }
    if (chosen_cam_idx == 4) {
      SetMagenta(pixel_out);
      return;
    }
    if (chosen_cam_idx == 5) {
      SetYellow(pixel_out);
      return;
    }
    SetBlack(pixel_out);
    return;
  }

  // Step 3) Apply the extrinsics and then intrinsics to get an exact pixel
  // index.

  float pos_cam[4];
  const float* world2cam = &extrinsics[chosen_cam_idx * 4 * 4];
  const float* K = &intrinsics[chosen_cam_idx * 3 * 3];
  Mat4Vec4Mul(world2cam, pos_global, pos_cam);
  if (pos_cam[3] < 1.0f - 1e-3 || pos_cam[3] > 1.0f + 1e-3) {
    SetBlue(pixel_out);
    return;
  }
  float im_coords[3];
  Mat3Vec3Mul(K, pos_cam, im_coords);
  // Perspective divide:
  im_coords[0] /= im_coords[2];
  im_coords[1] /= im_coords[2];

  // Just visualize the pixel coordinates directly:
  if (false) {
    float x_ndc = im_coords[0] / im_width * 255.0;
    float y_ndc = im_coords[1] / im_width * 255.0;
    uint8_t x_col = static_cast<uint8_t>(min(max(0.0f, x_ndc), 255.0f));
    uint8_t y_col = static_cast<uint8_t>(min(max(0.0f, y_ndc), 255.0f));
    pixel_out[0] = x_col;
    pixel_out[1] = y_col;
    pixel_out[2] = 0;
    return;
  }

  // Step 4) Bilinearly resample the image at the location in order to get an
  //         RGB value for the 3D point. Set it to the RGB for all three
  //         triplanes

  int im_x_lower = static_cast<int>(floor(im_coords[0]));
  int im_y_lower = static_cast<int>(floor(im_coords[1]));
  if (im_x_lower < 0 || im_y_lower < 0) {
    SetBlack(pixel_out);
    return;
  }

  const uint8_t* image = &images[chosen_cam_idx * im_height * im_width * 3];

  // Enable for debug visualization:
  if (false) {
    if (im_x_lower >= im_width || im_y_lower >= im_height) {
      SetGreen(pixel_out);
      return;
    }
    pixel_out[0] = image[im_y_lower * im_width * 3 + im_x_lower * 3 + 0];
    pixel_out[1] = image[im_y_lower * im_width * 3 + im_x_lower * 3 + 1];
    pixel_out[2] = image[im_y_lower * im_width * 3 + im_x_lower * 3 + 2];
    return;
  }

  int im_x_upper = im_x_lower + 1;
  int im_y_upper = im_y_lower + 1;
  if (im_x_upper >= im_width || im_y_upper >= im_height) {
    SetBlack(pixel_out);
    return;
  }
  float x_interp_frac = im_coords[0] - floor(im_coords[0]);
  float y_interp_frac = im_coords[1] - floor(im_coords[1]);
  // TODO(kgenova) Flip ud to correspond to grid points?
  int idx_y_lower = im_y_lower;  // im_height - im_y_lower;
  int idx_y_upper = im_y_upper;  // im_height - im_y_upper;
  const uint8_t* pixel_ul = &image[idx_y_lower * im_width * 3 + im_x_lower * 3];
  const uint8_t* pixel_ur = &image[idx_y_lower * im_width * 3 + im_x_upper * 3];
  const uint8_t* pixel_ll = &image[idx_y_upper * im_width * 3 + im_x_lower * 3];
  const uint8_t* pixel_lr = &image[idx_y_upper * im_width * 3 + im_x_upper * 3];

  float rightness = x_interp_frac;
  float lowerness = y_interp_frac;
  for (int ci = 0; ci < 3; ++ci) {
    float pixel_lower = (1.0f - rightness) * static_cast<float>(pixel_ll[ci]) +
                        rightness * static_cast<float>(pixel_lr[ci]);
    float pixel_upper = (1.0f - rightness) * static_cast<float>(pixel_ul[ci]) +
                        rightness * static_cast<float>(pixel_ur[ci]);
    float v = lowerness * pixel_lower + (1.0f - lowerness) * pixel_upper;
    v = min(max(0.0f, v), 255.0f);
    pixel_out[ci] = static_cast<uint8_t>(v);
  }

  return;
}

torch::Tensor images2triplanes_forward(
    torch::Tensor points, torch::Tensor images, torch::Tensor extrinsics,
    torch::Tensor intrinsics, torch::Tensor grid_lower_corner,
    torch::Tensor voxel_size, torch::Tensor grid_res,
    torch::Tensor triplane_res) {
  CHECK_INPUT(points);
  CHECK_FLOAT32(points);
  CHECK_INPUT(images);
  CHECK_UINT8(images);
  CHECK_INPUT(extrinsics);
  CHECK_FLOAT32(extrinsics);
  CHECK_INPUT(intrinsics);
  CHECK_FLOAT32(intrinsics);

  CHECK_INPUT(grid_lower_corner);
  CHECK_FLOAT32(grid_lower_corner);
  CHECK_INPUT(voxel_size);
  CHECK_FLOAT32(voxel_size);
  CHECK_INPUT(grid_res);
  CHECK_INT32(grid_res);
  CHECK_INPUT(triplane_res);
  CHECK_INT32(triplane_res);

  int n_images = images.size(0);
  int image_height = images.size(1);
  int image_width = images.size(2);

  if (extrinsics.size(0) != n_images) {
    std::cout << "Bad intrinsics shape dim 0: " << extrinsics.size(0)
              << std::endl;
  }

  int n_points = points.size(0);
  int grid_res_x = grid_res.index({0}).item<int32_t>();
  int grid_res_y = grid_res.index({1}).item<int32_t>();
  int grid_res_z = grid_res.index({2}).item<int32_t>();
  // TODO(kgenova) For now it's height, width to specify the res, since
  // that is the typical image convention, but we could maybe
  // do width, height.
  int triplane_res_height = triplane_res.index({0}).item<int32_t>();
  int triplane_res_width = triplane_res.index({1}).item<int32_t>();
  auto uint8_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto triplanes_out = torch::full({grid_res_z, grid_res_y, grid_res_x, 3,
                                    triplane_res_height, triplane_res_width, 3},
                                   0, uint8_options);
  CHECK_CUDA_STILL_OK();

  int64_t total_n_planes = grid_res_z * grid_res_y * grid_res_x * 3;
  int64_t pixels_per_plane = triplane_res_height * triplane_res_width;
  int64_t total_n_pixels = total_n_planes * pixels_per_plane;
  int64_t total_n_threads =
      total_n_pixels;  // One thread per output pixel (not channel)
  constexpr int threads_per_block = 512;  // Reasonable CUDA default
  constexpr int max_blocks_per_grid_dim =
      512;  // Spins up this many extra blocks in the worst case...
  int64_t blocks_per_grid_total = total_n_threads / threads_per_block + 1;
  int64_t blocks_per_grid_x = max_blocks_per_grid_dim;
  int64_t blocks_per_grid_y =
      blocks_per_grid_total / max_blocks_per_grid_dim + 1;
  int64_t blocks_per_grid_z = 1;
  dim3 blocks_per_grid_vec(blocks_per_grid_x, blocks_per_grid_y,
                           blocks_per_grid_z);
  dim3 threads_per_block_vec(threads_per_block, 1, 1);
  Images2TriplanesKernel<<<blocks_per_grid_vec, threads_per_block_vec, 0>>>(
      points.data_ptr<float>(), n_points, images.data_ptr<uint8_t>(), n_images,
      image_height, image_width, extrinsics.data_ptr<float>(),
      intrinsics.data_ptr<float>(), grid_lower_corner.data_ptr<float>(),
      voxel_size.data_ptr<float>(), grid_res.data_ptr<int>(),
      triplane_res.data_ptr<int>(), triplanes_out.data_ptr<uint8_t>(),
      total_n_pixels);
  CHECK_CUDA_STILL_OK();
  return triplanes_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &images2triplanes_forward,
        "Images2Triplanes forward (CUDA)");
}

// A CPU implementation. Does not use the GPU at all.
void Images2TriplanesCPU(const float* points, int n_points,
                         TriplaneGridMetadata metadata,
                         int* triplane_grid_out) {
  // Algorithm is simple: initialize everything to invalid, then loop over all
  // points, splatting them onto the grid.
  // Like scatter, in the case of collisions some valid point will end up in
  // the result, but there are no guarantees about which point it will be.
  for (int64_t i = 0; i < metadata.NPixelsTotal(); ++i) {
    triplane_grid_out[i] = -1;
  }

  for (int pi = 0; pi < n_points; ++pi) {
    const float* point = points + 3 * pi;
    TriplaneCoordinates coords = metadata.SplatPoint(point);
    //  Coordinates might be out of bounds:
    // TODO(kgenova) Take in the entire coords struct, check all fields?
    if (!metadata.VoxelIdxInBounds(coords.voxel_idx_x, coords.voxel_idx_y,
                                   coords.voxel_idx_z)) {
      continue;
    }

    // Now just translate the coords to the three correct grid indices:
    int64_t vi = metadata.FlattenVoxelIndex(
        coords.voxel_idx_x, coords.voxel_idx_y, coords.voxel_idx_z);
    int64_t voxel_start = vi * 3 * metadata.NPixelsPerImage();

    // xy plane first:
    int64_t xy_plane_start = voxel_start;
    int64_t xy_plane_idx =
        xy_plane_start +
        metadata.PixelCoordsToIdx(coords.xy_plane_x, coords.xy_plane_y);
    triplane_grid_out[xy_plane_idx] = pi;

    // Then xz plane:
    int64_t xz_plane_start = voxel_start + metadata.NPixelsPerImage();
    int64_t xz_plane_idx =
        xz_plane_start +
        metadata.PixelCoordsToIdx(coords.xz_plane_x, coords.xz_plane_y);
    triplane_grid_out[xz_plane_idx] = pi;

    // Finally yz plane:
    int64_t yz_plane_start = voxel_start + 2 * metadata.NPixelsPerImage();
    int64_t yz_plane_idx =
        yz_plane_start +
        metadata.PixelCoordsToIdx(coords.yz_plane_x, coords.yz_plane_y);
    triplane_grid_out[yz_plane_idx] = pi;
  }
}
