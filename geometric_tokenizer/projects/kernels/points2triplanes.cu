
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

__global__ void Points2TriplanesKernel(
    const float* points, int n_points, const float* grid_lower_corner,
    const float* voxel_size, const int* grid_res, const int* triplane_res,
    int* triplane_grid_out, int* point_voxel_inds_out,
    float* point_triplane_coords_out) {
  // Algorithm is simple: initialize everything to invalid, then loop over all
  // points, splatting them onto the grid.
  // Like scatter, in the case of collisions some valid point will end up in
  // the result, but there are no guarantees about which point it will be.

  int block_idx = gridDim.x * blockIdx.y + blockIdx.x;
  int thread_idx = threadIdx.x;
  int pi = block_idx * blockDim.x + thread_idx;
  if (pi >= n_points) return;  // dangling threads in last warp

  // It would be marginally more efficient to work from the input
  // fields directly, or maybe to create the struct and pass that
  // to GPU rather than doing this here. But it's likely irrelevant
  // compared to materializing a dense grid of triplanes, worry
  // about that first before anything else.
  TriplaneGridMetadata metadata;
  metadata.grid_lower_corner_x = grid_lower_corner[0];
  metadata.grid_lower_corner_y = grid_lower_corner[1];
  metadata.grid_lower_corner_z = grid_lower_corner[2];
  metadata.voxel_size_x = voxel_size[0];
  metadata.voxel_size_y = voxel_size[1];
  metadata.voxel_size_z = voxel_size[2];
  metadata.grid_res_x = grid_res[0];
  metadata.grid_res_y = grid_res[1];
  metadata.grid_res_z = grid_res[2];
  // Note it matches the pytorch wrapper below, change both if
  // changing either:
  metadata.triplane_pixel_height = triplane_res[0];
  metadata.triplane_pixel_width = triplane_res[1];

  const float* point = points + 3 * pi;
  TriplaneCoordinates coords = metadata.SplatPoint(point);

  // Note that we set this as xi, yi, zi, but xi varies most rapidly in storage
  // order! Also note these are global writes but they are collision free so it
  // is fine. They could be coalesced better, the 6x stride messes that up.
  point_voxel_inds_out[3 * pi + 0] = coords.voxel_idx_x;
  point_voxel_inds_out[3 * pi + 1] = coords.voxel_idx_y;
  point_voxel_inds_out[3 * pi + 2] = coords.voxel_idx_z;
  point_triplane_coords_out[6 * pi + 0] = coords.xy_plane_x;
  point_triplane_coords_out[6 * pi + 1] = coords.xy_plane_y;
  point_triplane_coords_out[6 * pi + 2] = coords.xz_plane_x;
  point_triplane_coords_out[6 * pi + 3] = coords.xz_plane_y;
  point_triplane_coords_out[6 * pi + 4] = coords.yz_plane_x;
  point_triplane_coords_out[6 * pi + 5] = coords.yz_plane_y;

  //  Coordinates might be out of bounds:
  // TODO(kgenova) Take in the entire coords struct, check all fields?
  bool in_bounds = metadata.VoxelIdxInBounds(
      coords.voxel_idx_x, coords.voxel_idx_y, coords.voxel_idx_z);
  if (!in_bounds) {
    return;
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
  // Global write, really important: could collide with others, and there is a
  // race condition with the other plane indices. We could do an atomic min
  // here, instead, but that is not random anymore unless the points are
  // shuffled in the input. Currently we give up determinism here, but the
  // alternative would be to refactor the calling code to ensure everything is
  // always shuffled, which is probably not worth it.
  triplane_grid_out[xy_plane_idx] = pi;

  // Then xz plane:
  int64_t xz_plane_start = voxel_start + metadata.NPixelsPerImage();
  int64_t xz_plane_idx =
      xz_plane_start +
      metadata.PixelCoordsToIdx(coords.xz_plane_x, coords.xz_plane_y);
  // Another global write, see above
  triplane_grid_out[xz_plane_idx] = pi;

  // Finally yz plane:
  int64_t yz_plane_start = voxel_start + 2 * metadata.NPixelsPerImage();
  int64_t yz_plane_idx =
      yz_plane_start +
      metadata.PixelCoordsToIdx(coords.yz_plane_x, coords.yz_plane_y);
  // Final global write.
  triplane_grid_out[yz_plane_idx] = pi;
}

std::vector<torch::Tensor> points2triplanes_forward(
    torch::Tensor points, torch::Tensor grid_lower_corner,
    torch::Tensor voxel_size, torch::Tensor grid_res,
    torch::Tensor triplane_res) {
  CHECK_INPUT(points);
  CHECK_FLOAT32(points);
  CHECK_INPUT(grid_lower_corner);
  CHECK_FLOAT32(grid_lower_corner);
  CHECK_INPUT(voxel_size);
  CHECK_FLOAT32(voxel_size);
  CHECK_INPUT(grid_res);
  CHECK_INT32(grid_res);
  CHECK_INPUT(triplane_res);
  CHECK_INT32(triplane_res);

  int n_points = points.size(0);
  int grid_res_x = grid_res.index({0}).item<int32_t>();
  int grid_res_y = grid_res.index({1}).item<int32_t>();
  int grid_res_z = grid_res.index({2}).item<int32_t>();
  // TODO(kgenova) For now it's height, width to specify the res, since
  // that is the typical image convention, but we could maybe
  // do width, height.
  int triplane_res_height = triplane_res.index({0}).item<int32_t>();
  int triplane_res_width = triplane_res.index({1}).item<int32_t>();
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto triplanes_out = torch::full({grid_res_z, grid_res_y, grid_res_x, 3,
                                    triplane_res_height, triplane_res_width},
                                   -1, int_options);
  CHECK_CUDA_STILL_OK();
  auto point_voxel_inds_out = torch::empty({n_points, 3}, int_options);
  CHECK_CUDA_STILL_OK();
  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto point_triplane_coords_out =
      torch::empty({n_points, 3, 2}, float_options);
  CHECK_CUDA_STILL_OK();

  int64_t total_n_threads = n_points;
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
  Points2TriplanesKernel<<<blocks_per_grid_vec, threads_per_block_vec, 0>>>(
      points.data_ptr<float>(), n_points, grid_lower_corner.data_ptr<float>(),
      voxel_size.data_ptr<float>(), grid_res.data_ptr<int>(),
      triplane_res.data_ptr<int>(), triplanes_out.data_ptr<int>(),
      point_voxel_inds_out.data_ptr<int>(),
      point_triplane_coords_out.data_ptr<float>());
  CHECK_CUDA_STILL_OK();
  return {triplanes_out, point_voxel_inds_out, point_triplane_coords_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &points2triplanes_forward,
        "Points2Triplanes forward (CUDA)");
}

// A CPU implementation. Does not use the GPU at all.
void Points2TriplanesCPU(const float* points, int n_points,
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
