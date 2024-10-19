#import helpers

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read_write> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_bins: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> projected_splats: array<helpers::ProjectedSplat>;

#ifdef RASTER_U32
    @group(0) @binding(4) var<storage, read_write> out_img: array<u32>;
#else
    @group(0) @binding(4) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(5) var<storage, read_write> final_index : array<u32>;
#endif

var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;
var<workgroup> tile_bins_wg: vec2u;

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_WIDTH, helpers::TILE_WIDTH, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let background = uniforms.background;
    let img_size = uniforms.img_size;

    // Get index of tile being drawn.
    //let tile_id = workgroup_id.x + workgroup_id.y * uniforms.tile_bounds.x;
    let pix_id = global_id.x + global_id.y * img_size.x;

    let tile_id = global_id.x / helpers::TILE_WIDTH + global_id.y / helpers::TILE_WIDTH * uniforms.tile_bounds.x;
    let pixel_coord = vec2f(global_id.xy) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = global_id.x < img_size.x && global_id.y < img_size.y;
    var done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    if local_idx == 0u {
        tile_bins_wg = tile_bins[tile_id];
    }
    // Hack to work around issues with non-uniform data.
    // See https://github.com/tracel-ai/burn/issues/1996
    let range = workgroupUniformLoad(&tile_bins_wg);

    let num_batches = helpers::ceil_div(range.y - range.x, helpers::TILE_SIZE);
    // current visibility left to render
    var T = 1.0;

    var pix_out = vec3f(0.0);

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    var t = 0u;
    var final_idx = 0u;

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var b = 0u; b < num_batches; b++) {
        let batch_start = range.x + b * helpers::TILE_SIZE;

        // Wait for all in flight threads.
        workgroupBarrier();

        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        if local_idx < remaining {
            let load_isect_id = batch_start + local_idx;
            local_batch[local_idx] = projected_splats[compact_gid_from_isect[load_isect_id]];
        }
        // Wait for all writes to complete.
        workgroupBarrier();

        for (var t = 0u; t < remaining && !done; t++) {
            let projected = local_batch[t];

            let xy = vec2f(projected.x, projected.y);
            let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);

            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let vis = exp(-sigma);
            let alpha = min(0.999f, projected.a * vis);

            if sigma >= 0.0 && alpha >= 1.0 / 255.0 {
                let next_T = T * (1.0 - alpha);

                if next_T <= 1e-4f {
                    done = true;
                    break;
                }

                let fac = alpha * T;
                pix_out += vec3f(projected.r, projected.g, projected.b) * fac;
                T = next_T;

                let isect_id = batch_start + t;
                final_idx = isect_id;
            }
        }
    }

    if inside {
        let final_color = vec4f(pix_out + T * background.xyz, 1.0 - T);
        #ifdef RASTER_U32
            let colors_u = vec4u(clamp(final_color * 255.0, vec4f(0.0), vec4f(255.0)));
            let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
            out_img[pix_id] = packed;
        #else
            out_img[pix_id] = final_color;
            final_index[pix_id] = final_idx;
        #endif
    }
}
