#define UNIFORM_WRITE

#import helpers;

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;

@group(0) @binding(1) var<storage, read_write> means: array<helpers::PackedVec3>;
@group(0) @binding(2) var<storage, read_write> log_scales: array<helpers::PackedVec3>;
@group(0) @binding(3) var<storage, read_write> quats: array<vec4f>;

@group(0) @binding(4) var<storage, read_write> global_from_compact_gid: array<u32>;
@group(0) @binding(5) var<storage, read_write> depths: array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let global_gid = global_id.x;

    if global_gid >= uniforms.total_splats {
        return;
    }

    // Project world space to camera space.
    let mean = helpers::as_vec(means[global_gid]);

    let img_size = uniforms.img_size;
    let viewmat = uniforms.viewmat;
    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let p_view = W * mean + viewmat[3].xyz;

    if p_view.z <= 0.01 {
        return;
    }

    // compute the projected covariance
    let scale = exp(helpers::as_vec(log_scales[global_gid]));
    let quat = quats[global_gid];

    let cov2d = helpers::calc_cov2d(uniforms.focal, uniforms.img_size, uniforms.pixel_center, viewmat, p_view, scale, quat);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;

    if det == 0.0 {
        return;
    }

    // Calculate ellipse conic.
    let conic = helpers::cov_to_conic(cov2d);

    // compute the projected mean
    let xy = helpers::project_pix(uniforms.focal, p_view, uniforms.pixel_center);

    // TODO: Include opacity here or is this ok?
    let radius = helpers::radius_from_conic(conic, 1.0);

    let tile_minmax = helpers::get_tile_bbox(xy, radius, uniforms.tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    if (tile_max.x - tile_min.x) == 0u || (tile_max.y - tile_min.y) == 0u {
        return;
    }

    // Now write all the data to the buffers.
    let write_id = atomicAdd(&uniforms.num_visible, 1u);
    global_from_compact_gid[write_id] = global_gid;
    depths[write_id] = p_view.z;
}
