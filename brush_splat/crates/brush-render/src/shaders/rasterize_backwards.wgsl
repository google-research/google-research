#import helpers;

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read_write> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_bins: array<vec2u>;

@group(0) @binding(3) var<storage, read_write> projected_splats: array<helpers::ProjectedSplat>;

@group(0) @binding(4) var<storage, read_write> final_index: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> v_output: array<vec4f>;

#ifdef HARD_FLOAT
    @group(0) @binding(7) var<storage, read_write> v_xy: array<atomic<f32>>;
    @group(0) @binding(8) var<storage, read_write> v_conics: array<atomic<f32>>;
    @group(0) @binding(9) var<storage, read_write> v_colors: array<atomic<f32>>;
#else
    @group(0) @binding(7) var<storage, read_write> v_xy: array<atomic<u32>>;
    @group(0) @binding(8) var<storage, read_write> v_conics: array<atomic<u32>>;
    @group(0) @binding(9) var<storage, read_write> v_colors: array<atomic<u32>>;
#endif


const MIN_WG_SIZE: u32 = 8u;
const BATCH_SIZE = helpers::TILE_SIZE;

// Gaussians gathered in batch.
var<workgroup> local_batch: array<helpers::ProjectedSplat, BATCH_SIZE>;
var<workgroup> local_id: array<u32, BATCH_SIZE>;

// Current queue of gradients to be flushed.
// Target count of #gradients to gather per go.
const GATHER_GRADS_MEM = BATCH_SIZE;
var<workgroup> grad_count: atomic<i32>;
var<workgroup> gather_grads: array<helpers::ProjectedSplat, GATHER_GRADS_MEM>;
var<workgroup> gather_grad_id: array<u32, GATHER_GRADS_MEM>;

// Tile bin workaround.
var<workgroup> tile_bins_wg: vec2u;

fn add_bitcast(cur: u32, add: f32) -> u32 {
    return bitcast<u32>(bitcast<f32>(cur) + add);
}

fn write_grads_atomic(grads: helpers::ProjectedSplat, id: u32) {
#ifdef HARD_FLOAT
    atomicAdd(&v_xy[id * 2 + 0], grads.x);
    atomicAdd(&v_xy[id * 2 + 1], grads.y);

    atomicAdd(&v_conics[id * 3 + 0], grads.conic_x);
    atomicAdd(&v_conics[id * 3 + 1], grads.conic_y);
    atomicAdd(&v_conics[id * 3 + 2], grads.conic_z);

    atomicAdd(&v_colors[id * 4 + 0], grads.r);
    atomicAdd(&v_colors[id * 4 + 1], grads.g);
    atomicAdd(&v_colors[id * 4 + 2], grads.b);
    atomicAdd(&v_colors[id * 4 + 3], grads.a);
#else
    // Writing out all these CAS loops individually is terrible but wgsl doesn't have a mechanism to
    // turn this into a function as ptr<storage> can't be passed to a function...
    // v_xy.x
    var old_value = atomicLoad(&v_xy[id * 2 + 0]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_xy[id * 2 + 0], old_value, add_bitcast(old_value, grads.x));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_xy.y
    old_value = atomicLoad(&v_xy[id * 2 + 1]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_xy[id * 2 + 1], old_value, add_bitcast(old_value, grads.y));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }

    // v_conic.x
    old_value = atomicLoad(&v_conics[id * 3 + 0]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_conics[id * 3 + 0], old_value, add_bitcast(old_value, grads.conic_x));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_conic.y
    old_value = atomicLoad(&v_conics[id * 3 + 1]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_conics[id * 3 + 1], old_value, add_bitcast(old_value, grads.conic_y));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_conic.z
    old_value = atomicLoad(&v_conics[id * 3 + 2]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_conics[id * 3 + 2], old_value, add_bitcast(old_value, grads.conic_z));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }

    // v_color.r
    old_value = atomicLoad(&v_colors[id * 4 + 0]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 0], old_value, add_bitcast(old_value, grads.r));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_color.g
    old_value = atomicLoad(&v_colors[id * 4 + 1]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 1], old_value, add_bitcast(old_value, grads.g));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_color.b
    old_value = atomicLoad(&v_colors[id * 4 + 2]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 2], old_value, add_bitcast(old_value, grads.b));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
    // v_color.a
    old_value = atomicLoad(&v_colors[id * 4 + 3]);
    loop {
        let cas = atomicCompareExchangeWeak(&v_colors[id * 4 + 3], old_value, add_bitcast(old_value, grads.a));
        if cas.exchanged { break; } else { old_value = cas.old_value; }
    }
#endif
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let background = uniforms.background.xyz;
    let img_size = uniforms.img_size;
    let tile_bounds = uniforms.tile_bounds;

    let tile_id = global_id.x / helpers::TILE_SIZE;
    let tile_loc = vec2u(tile_id % tile_bounds.x, tile_id / tile_bounds.x);
    let pixel_coordi = tile_loc * helpers::TILE_WIDTH + vec2u(local_idx % helpers::TILE_WIDTH, local_idx / helpers::TILE_WIDTH);
    let pix_id = pixel_coordi.x + pixel_coordi.y * img_size.x;
    let pixel_coord = vec2f(pixel_coordi) + 0.5;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    let inside = pixel_coordi.x < img_size.x && pixel_coordi.y < img_size.y;

    // this is the T AFTER the last gaussian in this pixel
    let T_final = 1.0 - output[pix_id].w;

    // Have all threads in tile process the same gaussians in batches
    // first collect gaussians between bin_start and bin_final in batches
    // which gaussians to look through in this tile
    if local_idx == 0u {
        tile_bins_wg = tile_bins[tile_id];
    }
    // Hack to work around issues with non-uniform data.
    // See https://github.com/tracel-ai/burn/issues/1996
    var range = workgroupUniformLoad(&tile_bins_wg);

    let num_batches = helpers::ceil_div(range.y - range.x, BATCH_SIZE);

    // current visibility left to render
    var T = T_final;

    var final_isect = 0u;
    var buffer = vec3f(0.0);

    if inside {
        final_isect = final_index[pix_id];
    }

    // df/d_out for this pixel
    var v_out = vec4f(0.0);
    if inside {
        v_out = v_output[pix_id];
    }

    // Make sure all groups start with empty gradient queue.
    atomicStore(&grad_count, 0);

    let sg_per_tile = helpers::ceil_div(helpers::TILE_SIZE, subgroup_size);
    let microbatch_size = helpers::TILE_SIZE / sg_per_tile;

    for (var b = 0u; b < num_batches; b++) {
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        let batch_end = range.y - b * BATCH_SIZE;
        let remaining = min(BATCH_SIZE, batch_end - range.x);

        // Gather N gaussians.
        var load_compact_gid = 0u;
        if local_idx < remaining {
            let load_isect_id = batch_end - 1u - local_idx;
            load_compact_gid = compact_gid_from_isect[load_isect_id];
            local_id[local_idx] = load_compact_gid;
            local_batch[local_idx] = projected_splats[load_compact_gid];
        }

        // wait for all threads to have collected the gaussians.
        workgroupBarrier();

        for (var tb = 0u; tb < remaining; tb += microbatch_size) {
            for(var tt = 0u; tt < microbatch_size; tt++) {
                let t = tb + tt;

                if t >= remaining {
                    break;
                }

                let isect_id = batch_end - 1u - t;

                var v_xy = vec2f(0.0);
                var v_conic = vec3f(0.0);
                var v_colors = vec4f(0.0);

                var splat_active = false;

                if inside && isect_id <= final_isect {
                    let projected = local_batch[t];

                    let xy = vec2f(projected.x, projected.y);
                    let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
                    let color = vec4f(projected.r, projected.g, projected.b, projected.a);

                    let delta = xy - pixel_coord;
                    let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                    let vis = exp(-sigma);
                    let alpha = min(0.99f, color.w * vis);

                    // Nb: Don't continue; here - local_idx == 0 always
                    // needs to write out gradients.
                    // compute the current T for this gaussian
                    if (sigma >= 0.0 && alpha >= 1.0 / 255.0) {
                        splat_active = true;

                        let ra = 1.0 / (1.0 - alpha);
                        T *= ra;
                        // update v_colors for this gaussian
                        let fac = alpha * T;

                        // contribution from this pixel
                        var v_alpha = dot(color.xyz * T - buffer * ra, v_out.xyz);
                        v_alpha += T_final * ra * v_out.w;
                        // contribution from background pixel
                        v_alpha -= dot(T_final * ra * background, v_out.xyz);

                        // update the running sum
                        buffer += color.xyz * fac;

                        let v_sigma = -color.w * vis * v_alpha;

                        v_xy = v_sigma * vec2f(
                            conic.x * delta.x + conic.y * delta.y,
                            conic.y * delta.x + conic.z * delta.y
                        );

                        v_conic = vec3f(0.5f * v_sigma * delta.x * delta.x,
                                                        v_sigma * delta.x * delta.y,
                                                        0.5f * v_sigma * delta.y * delta.y);

                        v_colors = vec4f(fac * v_out.xyz, vis * v_alpha);
                    }
                }

                // Queue a new gradient if this subgroup has any.
                // The gradient is sum of all gradients in the subgroup.
                if subgroupAny(splat_active) {
                    var v_xy_sum = subgroupAdd(v_xy);
                    var v_conic_sum = subgroupAdd(v_conic);
                    var v_colors_sum = subgroupAdd(v_colors);

                    // First thread of subgroup writes the gradient. This should be a
                    // subgroupBallot() when it's supported.
                    if subgroup_invocation_id == 0 {
                        let grad_idx = atomicAdd(&grad_count, 1);
                        gather_grads[grad_idx] = helpers::ProjectedSplat(
                            v_xy_sum.x,
                            v_xy_sum.y,

                            v_conic_sum.x,
                            v_conic_sum.y,
                            v_conic_sum.z,

                            v_colors_sum.x,
                            v_colors_sum.y,
                            v_colors_sum.z,
                            v_colors_sum.w,
                        );
                        gather_grad_id[grad_idx] = local_id[t];
                    }
                }
            }

            // Make sure all threads are done, and flush a batch of gradients.
            workgroupBarrier();
            if local_idx < u32(grad_count) {
                write_grads_atomic(gather_grads[local_idx], gather_grad_id[local_idx]);
            }
            workgroupBarrier();
            atomicStore(&grad_count, 0);
        }
    }
}
